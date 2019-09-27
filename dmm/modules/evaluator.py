""" Evaluation engine for rvos + matching model 
support only batchsize == n GPU and nGPU > 1
"""
from dmm.misc.config_youtubeVOS import cfg as cfg_youtube
from dmm.misc.config_youtubeVOS import get_db_path, get_anno_path, get_img_path 
from dmm.utils.utils import make_dir, ohw_mask2boxlist
from dmm.utils.checker import * 
import dmm.utils.eval_helper as eval_helper
import yaml
import torch
import torch.nn as nn
import numpy as np
import logging
import time
import os
import json
from PIL import Image
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

class Evaler(nn.Module):
    """ engine/container for encoder, decoder and all DMM modules:
        match_layer
        feature_extractor
    """
    def __init__(self, DMM, encoder, decoder, args, dmmcfgs): 
        super(Evaler, self).__init__()
        self.meters =  MetricLogger(delimiter=" ") 
        self.decoder = decoder 
        self.encoder = encoder 
        self.DMM = DMM
        if args.load_proposals and not args.load_proposals_dataset:
            logging.info('load %s'%args.pred_offline_meta)
            self.pred_offline_meta = json.load(open(args.pred_offline_meta, 'r')) 
            if 'vidfid2index' in self.pred_offline_meta: 
                self.pred_offline_meta = self.pred_offline_meta['vidfid2index']
        if args.use_gpu:
            self.encoder.cuda()
            self.decoder.cuda()
            self.DMM.cuda()
        timestr = time.strftime('%m-%d')
        model_name = args.model_name.strip('/')
        model_id = model_name.split('/')[-2]+'/epo'+model_name.split('/')[-1].split('epo')[-1]
        self.eval_output_root = '%s/eval/%s/%s/L%s_%d_s%s/'%(args.models_root, timestr, model_id,
                                 args.eval_flag, args.test_image_h, args.eval_split)
        self.eval_output_root = self.eval_output_root.replace('__', '_')
        timestr = time.strftime('%m-%d-%H-%M')
        save_config_dir = '%s/save_config/%s/'%(self.eval_output_root, timestr)
        isrank0 = args.local_rank == 0
        if isrank0:
           if not os.path.exists(save_config_dir):
                make_dir(save_config_dir)
           yaml.dump(args,    open(os.path.join(save_config_dir,'eval_args.yaml'),'w'))
           yaml.dump(dmmcfgs, open(os.path.join(save_config_dir,'eval_dmm_config.yaml'),'w'))
        json_file = open(get_db_path(args.eval_split), 'r')
        self.seq_dir = get_img_path(args.eval_split)
        self.anno_dir = get_anno_path(args.eval_split)
        self.data = json.load(json_file)
        if isrank0: logging.info('num vid %d'%len(self.data['videos']))
        self.DMM.eval() 
        self.encoder.eval()
        self.decoder.eval()

    def forward(self,batch_idx,inputs,imgs_names,targets,seq_name, args, proposals_input):
        """ Evaluation 
        forward a batch of clip
        """
        if args.pad_video:
            CHECK4D(targets) # B,len,O,HW 
            CHECK5D(inputs)  # B,len,O,H,W 
        device_id = torch.cuda.current_device()
        if args.batch_size == 1:
            if not args.distributed: seq_name = [seq_name[device_id]]
        else:
            batch_size_device = int(args.batch_size / args.ngpus)
            if not args.distributed:
                seq_name = seq_name[device_id*batch_size_device:(1+device_id)*batch_size_device]
        CHECKEQ(len(seq_name), len(inputs))
        njpgs_batch, img_shape, frame_names_batch = self.prepare_frame_names(seq_name)
        # send batch to GPU
        prev_thid_list = None
        B,nframe,O,H,W = inputs.shape 
        max_length_clip = min(nframe,args.length_clip)
        for frame_idx in range(max_length_clip):
            tic = time.time()
            extra_frame =  [njpgs <= frame_idx for njpgs in njpgs_batch] 
            proposal_cur, predid_cur_frames = None, None
            if args.load_proposals and proposals_input is None:
                predid = [0]*len(seq_name)
                for b, seq_n in enumerate(seq_name):
                    if extra_frame[b]:
                        predid[b] = len(self.encoder.pred_offline)-1
                    else:
                        frame_name = imgs_names[b][frame_idx] # tuple + b name 
                        predid[b] = int(self.pred_offline_meta[seq_n][frame_name])
                predid_cur_frames = predid 
            elif proposals_input is not None:
                proposal_cur = []
                for b in range(B):
                    if len(proposals_input[b]) > frame_idx: 
                        proposal_cur.append(proposals_input[b][frame_idx])
                    else:
                        proposal_cur.append(proposals_input[b][-1])
            x = inputs[:, frame_idx] # B,1->0,O,H,W, select 1 from clip len 
            Bx, Cx, Hx, Wx = x.shape 
            # targets shape: B,len,O,H*W
            # input shape:   B,len,3(Cx),H,W
            y_mask = targets[:, frame_idx][:,:,:-1].float() 
            CHECKEQ(Hx*Wx, y_mask.shape[-1])
            CHECKEQ(Bx, y_mask.shape[0])
            B,O,HW = CHECK3D(y_mask) 
            CHECKEQ(Bx, B)
            if frame_idx == 0:
                mask_hist = None
                tplt_dict, tplt_valid_batch, proposals = \
                   self.forward_timestep_init(args, x, y_mask, predid_cur_frames, proposal_cur)
                prev_thid_list, thid_list = None, None
                prev_mask = y_mask.view(B,O,HW) 
                outs  = y_mask 
                init_pred_inst = y_mask.view(B,O,H,W)
                infos = {'args':args, 'shape':img_shape, 'extra_frame':extra_frame, 'valid':tplt_valid_batch,
                        'predid':predid_cur_frames}
                _, prev_thid_list, _, _, _  = self.inference_timestep(infos, tplt_dict, x, y_mask, \
                    prev_thid_list=prev_thid_list, prev_mask=prev_mask, mask_hist=mask_hist, proposal_cur=proposal_cur)
            else:
                # ---- start inference of current batch ----    
                infos = {'args':args, 'shape':img_shape, 'extra_frame':extra_frame, 'valid':tplt_valid_batch,
                        'predid':predid_cur_frames}
                outs, thid_list, init_pred_inst, proposals, mask_hist = self.inference_timestep(infos, tplt_dict, x, y_mask,  
                    prev_thid_list=prev_thid_list, prev_mask=prev_mask, mask_hist=mask_hist, proposal_cur=proposal_cur)
                self.meters.update(ft=time.time()-tic)
                prev_mask =  outs.view(B,O,HW) 
                if args.only_spatial == False:
                    prev_thid_list = thid_list
                prev_mask = outs.view(B,O,HW) if frame_idx > 0 else y_mask
            # ---------------- save merged mask ----------------------
            for b in range(B):
                if extra_frame[b]: continue # skip the extra frames
                saved_name = self.eval_output_root+'merged/%s/%s.png'%(seq_name[b], frame_names_batch[b][frame_idx])
                obj_index = tplt_valid_batch[b].sum() 
                refine_mask = outs[b, :obj_index].view(-1, H*W)
                refine_bg = 1 - refine_mask.max(0)[0]
                refine_fbg = torch.cat([refine_bg.view(1,H,W), refine_mask.view(-1,H,W)], dim=0) 
                max_v, max_i = refine_fbg.max(0)
                eval_helper.plot_scores_map(max_i.float(), saved_name) 
            # ---------------- save outs to mask  ----------------------
            del outs, thid_list, x, y_mask, init_pred_inst  
        if (batch_idx % 10 == 0 and args.local_rank == 0):
            logging.info('save at {}'.format(self.eval_output_root)) 
            logging.info(self.meters)

    def inference_timestep(self, infos, tplt_dict, x, y_mask, prev_thid_list, prev_mask, mask_hist, proposal_cur): 
        r""" inference for frames at current image step, 
        argument:
            infos: 'args','shape','extra_frame','valid','predid'}
                   img_shape: list, len=B, element: [h,w]
            x: shape: B,3,H W
            y_mask: B,O,H W
        return 
            init_pred_inst: BOHW, prediction from the mask branch, without refine  
            tplt_dict, 
            proposals, mask_hist_new #4,5,6,7,8    
        """
        args = infos['args']
        img_shape = infos['shape']
        extra_frame, tplt_valid_batch = infos['extra_frame'], infos['valid']
        hidden_spatial = None
        out_masks = []
        assert(isinstance(x, torch.Tensor))
        features, proposals, _ = self.encoder(args, x, predid_cur_frames=infos['predid'], proposals=proposal_cur) 
        bone_feat = features['backbone_feature'] # B,Lev,(D,H,W); 
        B,D,H,W = x.shape
        thid_list = []
        B,O,HW = CHECK3D(prev_mask)
        if mask_hist is None: 
            mask_hist = prev_mask.view(B,O,H,W)
        assert('mask' in proposals[0].fields())
        init_pred_inst, tplt_dict, match_loss, mask_hist_new \
            = self.DMM.inference(infos, proposals, bone_feat, mask_hist, tplt_dict )
        valid_num_obj_max = max(1, (tplt_valid_batch.sum(0) > 0).sum()) # shape: 1, O
        for t in range(0, valid_num_obj_max):
            if prev_thid_list is not None:
                hidden_temporal = prev_thid_list[t]
                if args.only_temporal:
                    hidden_spatial = None
            else:
                hidden_temporal = None
            mask_lstm = []
            maxpool = nn.MaxPool2d((2, 2),ceil_mode=True)
            prev_m_inst = torch.cat([prev_mask[:,t,:].view(B,1,H*W), y_mask[:,t,:].view(B,1,H*W),
                                     init_pred_inst[:,t].view(B,1,H*W)], dim=2).view(B,3,H,W) # cat along new dim
            prev_m_inst = maxpool(prev_m_inst)
            for _ in range(len(features['refine_input_feat'])):
                prev_m_inst = maxpool(prev_m_inst)
                mask_lstm.append(prev_m_inst)
            mask_lstm = list(reversed(mask_lstm))
            out_mask, hidden = self.decoder(features['refine_input_feat'], mask_lstm, hidden_spatial, hidden_temporal)
            hidden_tmp = [hidden[ss][0] for ss in range(len(hidden))]
            hidden_spatial = hidden
            thid_list.append(hidden_tmp)
            upsample_match = nn.UpsamplingBilinear2d(size=(x.size()[-2], x.size()[-1]))
            out_mask = upsample_match(out_mask)
            for b in range(B): # should behave differently for differnet vid; 
                is_template_valid_cur_b = tplt_valid_batch[b, t] # current batch 
                if not is_template_valid_cur_b: continue 
                mask_hist_new[b,t:t+1,:,:] = torch.sigmoid(out_mask[b])  # shape: B,O,H,W and B,1,H,W
            out_mask = out_mask.view(out_mask.size(0), -1)
            out_masks.append(out_mask)
            del mask_lstm, hidden_temporal, hidden_tmp, prev_m_inst, out_mask
        out_masks = torch.cat(out_masks,1).view(out_masks[0].size(0), len(out_masks), -1) # B,O,HW
        outs = torch.sigmoid(out_masks)
        outs_pad = outs.new_zeros(B, O, HW)
        outs_pad[:,:valid_num_obj_max,:] = outs 
        return outs_pad, thid_list, init_pred_inst, proposals, mask_hist_new 

    def forward_timestep_init(self, args, x, y_mask, predid_cur_frames, proposal_cur):
        features, proposals, cocoloss = self.encoder(args, x, predid_cur_frames=predid_cur_frames, proposals=proposal_cur) 
        B,D,H,W = CHECK4D(x)
        tplt_valid_batch = []
        for b in range(B):
            prop, template_valid = ohw_mask2boxlist(y_mask[b].view(-1,H,W)) # OHW 
            tplt_valid_batch.append(template_valid) # append O
            proposals[b] = prop
        tplt_valid_batch = torch.stack(tplt_valid_batch, dim=0)
        tplt_dict = self.DMM.fill_template_dict(args, proposals, features,y_mask,tplt_valid_batch)
        return tplt_dict, tplt_valid_batch, proposals 

    def prepare_frame_names(self, seq_name):
        njpgs_batch       = []
        img_shape         = []
        frame_names_batch = [] 
        for inx, seq_name_b in enumerate(seq_name):
            frame_names = np.sort(os.listdir(self.seq_dir + '/' + seq_name_b))
            frame_names = [os.path.splitext(os.path.basename(fullname))[0] for fullname  in frame_names]
            vid_img = np.array(Image.open(self.seq_dir+ '/' + seq_name_b + '/%s.jpg'%frame_names[0])) 
            img_h, img_w,_ = vid_img.shape
            img_shape.append([img_h, img_w])
            seq_info = self.data['videos'][seq_name_b]['objects']
            frame_names_has_obj = []
            for obj_id in seq_info.keys(): # loop over all objects 
                for frame_name in seq_info[obj_id]['frames']: 
                    if frame_name not in frame_names_has_obj: # add if this a new frames 
                        frame_names_has_obj.append(frame_name)
            start_annotation_frame = frame_names_has_obj[0]
            id_start = frame_names.index(start_annotation_frame) 
            if id_start != 0:
                logging.warning('find a video annotation not start from the first frame in ' + \
                                'rgb images :{}; {}'.format(seq_name_b,frame_names[0])) 
                frame_names = frame_names[id_start:]
            frame_names_batch.append(frame_names)
            njpgs = len(frame_names)
            njpgs_batch.append(njpgs)
        return njpgs_batch, img_shape, frame_names_batch
