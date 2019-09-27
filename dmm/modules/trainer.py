""" the largest container for the training 
s 1. trainer (its decoder and encoder)
 encoder and decoder 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
import logging
import time
import os
from PIL import Image
import pickle
import random
import yaml
import json
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from .match_model import compute_iou_binary_mask_2D
from dmm.misc.config_youtubeVOS import cfg as cfg_youtube
from dmm.misc.config_youtubeVOS import phase as PHASE
from dmm.utils.checker import *
from dmm.utils.utils import ohw_mask2boxlist
from dmm.utils.objectives import softIoULoss
DEBUG = 0 
# global refine_io
# refine_io = {}

class Trainer(nn.Module):
    """ container for encoder, decoder and all DMM modules:
        match_layer
        feature_extractor
    """
    def __init__(self, DMM, encoder, decoder, args):
        super(Trainer, self).__init__()
        self.count = 0
        self.meters =  MetricLogger(delimiter=" ") 
        self.decoder = decoder 
        self.encoder = encoder 
        self.DMM = DMM 
        if args.load_proposals and not args.load_proposals_dataset:
            self.pred_offline_meta = json.load(open(args.pred_offline_meta, 'r')) 
            if 'vidfid2index' in self.pred_offline_meta: 
                self.pred_offline_meta = self.pred_offline_meta['vidfid2index']
        mask_siou = softIoULoss()
        if args.use_gpu:
            self.encoder.to(args.device)
            if self.decoder is not None:
                self.decoder.to(args.device)
            self.DMM.to(args.device)
            mask_siou.to(args.device)
        if args.train_split == PHASE.TRAIN.value:
            json_file = cfg_youtube.FILES.DB_INFO_TRAIN
        elif args.train_split == PHASE.TRAINTESTDEVOT.value:
            json_file = cfg_youtube.FILES.DB_INFO_TRAINTESTDEVOT
        #elif args.train_split == PHASE.DAVISTRAIN.value:
        #    json_file = cfg_youtube.FILES.DB_INFO_TRAIN_DAVIS
        else:
            raise AttributeError('not support %s'%args.train_split)
        assert(os.path.exists(json_file)), json_file
        json_data = open(json_file)
        self.data = json.load(json_data)
        if args.local_rank == 0: logging.info('load from json_data; num vid %d'%len(self.data['videos']))
        self.crits = mask_siou

    def forward(self,batch_idx,inputs,imgs_names,targets,seq_name,starting_frame, split, args, proposals):
        """ forward a batch of clip
            a clip has batchsize B; length_clip: L; Ninstance O; H; W
        first loop over length;
            loop over Ninstance 
                loop over batch
        targets: list of gt mask for each frame: 
            len == length_clip
            targets[0]: BOHW for one time steps; all videos
            x: input images (N consecutive frames from M different sequences)
            y_mask: gt annotations (some of them are zeros to have a fixed 
                     length in number of object instances)
            sw_mask: this mask indicates which masks from y_mask are valid
        """
        if split == args.train_split:
            self.train(True)
            self.encoder.train(True)
            self.DMM.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval() 
            self.DMM.eval()

        prev_thid_list = None # previous hidden temporal list 
        loss = [] 
        max_length_clip = min(len(inputs),args.length_clip)                      
        # loop over the current clip
        CHECKEQ(len(targets), max_length_clip)
        assert(type(targets) == list)
        for frame_idx in range(max_length_clip):
            x = inputs[frame_idx]
            y_mask = targets[frame_idx][:,:,:-1].float() # y_mask shape: B,O,HW?
            ref_mask = targets[0][:,:,:-1].float()
            sw_mask = targets[frame_idx][:,:,-1] 
            proposal_cur, predid_cur_frames=None,None
            if args.load_proposals and proposals is None:
                predid = [0]*len(seq_name) # np.zeros((len(seq_name)))  
                for b, seq_n in enumerate(seq_name):
                   frame_name = imgs_names[frame_idx][b] # f tuple + b name 
                   predid[b] = int(self.pred_offline_meta[seq_n][frame_name])
                predid_cur_frames = predid #.astype(np.uint8)
            elif proposals is not None:
                proposal_cur=proposals[frame_idx] 
                CHECKEQ(len(proposal_cur), x.shape[0])

            if frame_idx == 0:
                tplt_dict, tplt_valid_batch = self.forward_timestep_init(args, x, y_mask,
                     mode=split, predid_cur_frames=predid_cur_frames, proposal_cur=proposal_cur)
                prev_mask = y_mask 
                loss_timestep, losses_print, _, prev_thid_list, _ \
                    = self.forward_timestep(frame_idx, args, tplt_dict, x, y_mask, sw_mask, tplt_valid_batch, ref_mask=ref_mask, mode=split,
                    prev_thid_list=prev_thid_list, prev_mask=prev_mask, predid_cur_frames=predid_cur_frames, proposal_cur=proposal_cur) 
            else:
                loss_timestep, losses_print, outs, thid_list, tplt_dict \
                    = self.forward_timestep(frame_idx, args, tplt_dict, x, y_mask, sw_mask, tplt_valid_batch, ref_mask=ref_mask, mode=split,
                    prev_thid_list=prev_thid_list, prev_mask=prev_mask, predid_cur_frames=predid_cur_frames, proposal_cur=proposal_cur) 
                prev_thid_list = thid_list # update hidden temporal list !
                if args.sample_inference_mask and random.random() > 0.5: 
                    prev_mask = outs 
                else:
                    prev_mask = y_mask 
            loss.append(loss_timestep)
        loss = sum(loss)
        return loss, losses_print 
 
    def forward_timestep(self, frame_idx, args, tplt_dict, x, y_mask, sw_mask, tplt_valid_batch, mode='train', \
            prev_thid_list=None, ref_mask=None, prev_mask=None, predid_cur_frames=None, proposal_cur=None):
        """
        Runs forward a batch
        if timestep == 0;
            prev_mask is the provided gt mask at 0 
        else:
            prev_mask is the predicted mask at last timestep
            shape of prev: B,1,H,W? 
        """
        mask_siou = self.crits
        max_num_obj = args.maxseqlen
        CHECKEQ(tplt_valid_batch.shape[1], max_num_obj)
        if args.skip_empty_starting_frame:
            valid_num_obj = (tplt_valid_batch.sum(0) > 0).sum() # shape: 1, O
            if valid_num_obj < args.maxseqlen:
                logging.debug('reduce max_num_obj: {} - {};'.format(valid_num_obj, args.maxseqlen))
        else: 
            valid_num_obj = max_num_obj
        assert(isinstance(x, torch.Tensor))
        # ============================ forward encoder ========================================
        if args.load_proposals:
            if mode == args.eval_split:
                with torch.no_grad(): # feature tuple: x5 x4 x3 x2 x1
                    features, proposals, _ = self.encoder(args, x, predid_cur_frames=predid_cur_frames, proposals=proposal_cur) 
            else:
                features, proposals, _ = self.encoder(args, x, targets=y_mask, predid_cur_frames=predid_cur_frames, proposals=proposal_cur) 
        else:
            if mode == args.eval_split:
                with torch.no_grad(): # feature tuple: x5 x4 x3 x2 x1
                    features, proposals, cocoloss = self.encoder(args, x)
            else:
                features, proposals, cocoloss =self.encoder(args,x,targets=y_mask) 
        loss = []
        losses_print = {}
        B,D,H,W = x.shape 
        B,O,HW = prev_mask.shape
        assert(H*W == HW), 'x {}; prev_mask {}'.format(x.shape, prev_mask.shape)
        mask_last_occur = prev_mask.view(B,O,H,W)

        assert(len(proposals) == x.shape[0]), '{} {}'.format(len(proposals), x.shape)
        dmm_target = y_mask.view(B,O,H,W) 
        # B, max_num_obj, along B, at every obj_index, only select part of B 
        #if DEBUG:
        #    dmm_io = [features['backbone_feature'], mask_last_occur, tplt_dict, tplt_valid_batch]
        #    check_io = torch.load('../../drvos/src/debug/dmm_io_%d.pth'%frame_idx)
        #    logging.info('load debug file')
        #    CHECKDEBUG(dmm_io, check_io)

        # ===========================  forward DMM ===========================================
        init_pred_inst, tplt_dict, match_loss, mask_last_occur = self.DMM(args, proposals=proposals, backbone_feature=features['backbone_feature'],  mask_last_occurence=mask_last_occur,  tplt_dict=tplt_dict, tplt_valid_batch=tplt_valid_batch, targets=dmm_target)
        CHECK4D(init_pred_inst) # BOHW 
        CHECK3D(prev_mask) # B,O,HW 
        CHECKEQ(init_pred_inst.shape[0], B)
        CHECKEQ(init_pred_inst.shape[1], O)
        
        with torch.no_grad(): # we want to compute on the valid template; 
            num_template_not_empty = tplt_valid_batch.sum()
            hard_iou1 = compute_iou_binary_mask_2D(y_mask.view(-1,H*W),init_pred_inst.view(-1,H*W)).view(B,O)*tplt_valid_batch.float() 
            # set those empty template iou as 0; not added to sum 
            if num_template_not_empty == 0:
                logging.info('get num_template_not_empty equal to 0 at rank %d'%args.local_rank)
            # y_mask: B, O, H*W; init_pred_inst: B,O,H,W; hard_iou1: B*O
            hard_iou1 = hard_iou1.sum() / (num_template_not_empty + 1e-6) \
                        if num_template_not_empty > 0 else torch.zeros_like(hard_iou1).sum()  
        assert(type(match_loss) == list), type(match_loss)
        if len(match_loss) > 0:
            match_loss_mean = sum(match_loss) / len(match_loss)
            losses_print.update({'match_loss':match_loss_mean})
            loss.append(match_loss_mean * args.loss_weight_match) 

        losses_print.update({'hard_iou_raw': hard_iou1})
        # ============================= compute matched-mask-loss ============================
        loss_mask_iou_raw = mask_siou( y_mask.view(-1,H*W), (init_pred_inst).view(-1,H*W), sw_mask.view(-1,1), need_sigmoid=0)
        loss_mask_iou_raw = torch.mean(loss_mask_iou_raw)
        losses_print.update({'loss_miou_raw': loss_mask_iou_raw})
        loss.append(loss_mask_iou_raw * args.loss_weight_iouraw) 
        loss_list = loss
        loss = sum(loss)
        # ============================= do refine  ============================
        shapes = {'BOHWXhXw':[B,O,H,W, x.shape[-2], x.shape[-1]], 'valid_num_obj':valid_num_obj}
        if num_template_not_empty == 0: 
            losses_print.update({'hard_iou0': init_pred_inst.sum()*0, 'loss_miou': init_pred_inst.sum()*0})
            refine_output = init_pred_inst.view(B,O,HW) 
            refine_temporal_hidden_list = prev_thid_list
        else:
            refine_loss, refine_loss_pred, refine_output, refine_temporal_hidden_list \
                = self.refine(args, shapes, tplt_dict, features['refine_input_feat'], prev_thid_list, init_pred_inst, \
                prev_mask, sw_mask, y_mask, tplt_valid_batch, ref_mask)
            loss += refine_loss 
            losses_print.update(refine_loss_pred)
        losses_print.update({'loss':loss})
        if DEBUG:
            refine_io.update({'after_refine': [shapes, tplt_dict, features['refine_input_feat'], prev_thid_list, init_pred_inst, prev_mask, sw_mask, y_mask, tplt_valid_batch, ref_mask]})
            # , refine_loss, refine_loss_pred, refine_output, refine_temporal_hidden_list, loss_list, losses_print, loss]})
            refine_io.update({'a-refloss': refine_loss, 'a-ref-loss-pred': refine_loss_pred, 'a-refout': refine_output, 'a-ref-this-list': refine_temporal_hidden_list, 
                                  'a-loss-list': loss_list, 'a-loss-print': losses_print, 'a-loss': loss})
            prev_io = torch.load('../../drvos/src/debug/refine_io_count%d.pth'%self.count)
            self.count += 1
            logging.info('check debug file: %d'%self.count)
            CHECKDEBUG(refine_io, prev_io)

        return loss, losses_print, refine_output, refine_temporal_hidden_list, tplt_dict

    def refine(self, args, shapes, tplt_dict, feats, prev_thid_list, init_pred_inst, prev_mask, \
                            sw_mask, y_mask, tplt_valid_batch, ref_mask):
        """ refine all instances in current frames, whole batch 
        arguments: 
            tplt_dict (dict, len=batchsize)
            feats (tensor, ): features of current frames; for whole image 
        """
        valid_num_obj = shapes['valid_num_obj']
        B, O, H, W, xshape_h, xshape_w = shapes['BOHWXhXw']
        hidden_spatial = None
        thid_list = []
        out_masks = []
        for obj_index in range(0, valid_num_obj): # add background ??? 
            if prev_thid_list is not None:
                CHECKEQ(len(prev_thid_list), valid_num_obj) 
                hidden_temporal = prev_thid_list[obj_index] # different scale, different Batch []
                if args.only_temporal:
                    hidden_spatial = None
            else:
                hidden_temporal = None
            mask_lstm = []
            maxpool = nn.MaxPool2d((2, 2),ceil_mode=True)
            prev_m_inst = torch.cat([prev_mask[:, obj_index].view(B,1,H*W), ref_mask[:, obj_index].view(B,1,H*W),
                                     init_pred_inst[:,obj_index].view(B,1,H*W)], dim=2).view(B,3,H,W) # cat along new dim 
            prev_m_inst = maxpool(prev_m_inst)
            for ii in range(len(feats)):
                prev_m_inst = maxpool(prev_m_inst) 
                mask_lstm.append(prev_m_inst)
                
            mask_lstm = list(reversed(mask_lstm))
            #if DEBUG:
            #    refine_io.update({'decoder-input-%d'%obj_index: {'feats':feats, 'min':mask_lstm, 'hid':hidden_spatial, 'hidt':hidden_temporal, 'prev':prev_m_inst}})
            out_mask, hidden = self.decoder(feats, mask_lstm, hidden_spatial, hidden_temporal)

            #if DEBUG:
            #    refine_io.update({'decoder-output-%d'%obj_index: [out_mask, hidden]})
            hidden_tmp = [hidden[ss][0] for ss in range(len(hidden))]
            thid_list.append(hidden_tmp)
            hidden_spatial = hidden 
            out_mask = F.interpolate(out_mask, (xshape_h, xshape_w)).view(out_mask.size(0), -1)  
            out_masks.append(out_mask)

        total_obj = len(out_masks)
        out_masks = torch.cat(out_masks,1).view(B, total_obj, -1)
        total_obj = out_masks.shape[1] 
        sw_mask = torch.from_numpy(sw_mask.data.cpu().numpy()[:,0:total_obj]).contiguous().float()
        y_mask_valid = y_mask[:, :total_obj].contiguous()
        CHECKEQ(sw_mask.shape[1], out_masks.shape[1]) 
        sw_mask = sw_mask.to(args.device)
        
        HW = y_mask.size()[-1]
        out_masks = torch.sigmoid(out_masks)
        Ov = out_masks.shape[1]
        loss_mask_iou = self.crits(y_mask_valid.view(-1,HW), out_masks.view(-1, HW), sw_mask.view(-1,1), need_sigmoid=0)

        if DEBUG:
            refine_io.update({'refine-y_mask_valid': y_mask_valid, 'refine-out_masks': out_masks, 'refine-sw_mask': sw_mask, 'refine-loss': loss_mask_iou})
        loss_mask_iou = torch.mean(loss_mask_iou)
        losses_print = {} 
        loss = args.iou_weight * loss_mask_iou 
        with torch.no_grad():
            num_template_not_empty = tplt_valid_batch.sum()
            hard_iou0 = compute_iou_binary_mask_2D( y_mask_valid.view(-1, HW), out_masks.view(-1, HW)).view(B, total_obj)
            hard_iou0 = hard_iou0.sum() / (num_template_not_empty + 1e-6) \
                            if num_template_not_empty > 0 else torch.zeros_like(hard_iou0).sum() 
            losses_print.update({'hard_iou': hard_iou0})
        # pad the outs to B,O,HW shape 
        outs_pad = out_masks.new_zeros(B, O, HW)
        outs_pad[:,:out_masks.shape[1],:] = out_masks
        losses_print.update({'loss_miou':loss_mask_iou})
        return loss, losses_print, outs_pad, thid_list

    def forward_timestep_init(self, args, x, y_mask, mode='train', predid_cur_frames=None, proposal_cur=None):
        """ Runs forward a batch """
        mask_siou = self.crits
        assert(isinstance(x, torch.Tensor))
        # feature tuple: x5 x4 x3 x2 x1
        features,proposals,_=self.encoder(args, x, targets=y_mask, predid_cur_frames=predid_cur_frames, proposals=proposal_cur) 
        B,D,H,W = x.shape 
        tplt_valid_batch = []
        for b in range(B):
            prop, template_valid = ohw_mask2boxlist(y_mask[b].view(-1,H,W)) # OHW 
            tplt_valid_batch.append(template_valid) # append O
            proposals[b] = prop
        tplt_valid_batch = torch.stack(tplt_valid_batch, dim=0)
        tplt_dict = self.DMM.fill_template_dict(args, proposals, features, y_mask, tplt_valid_batch)
        if DEBUG:
            refine_io.update({'init_tplt': {
                                            'input':x,
                                            'features': features, 'prop':[p.bbox for p in proposals], 'y_mask':y_mask, 'dict':tplt_dict
                                            }})
        return tplt_dict, tplt_valid_batch 
