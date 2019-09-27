import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
import numpy as np
from scipy.misc import imresize
from PIL import Image
import time
import random
from .transforms.transforms import Affine
import glob
import json
import logging
from args import get_parser
from maskrcnn_benchmark.structures.bounding_box import BoxList
from dmm.utils.masker import make_roi_mask_post_processor 
import dmm.utils.utils as utils
from dmm.utils.checker import *
# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()
DEBUG = 0 

if args.dataset == 'youtube':
    from dmm.misc.config_youtubeVOS import cfg as cfg_youtube
    from dmm.misc.config_youtubeVOS import phase as PHASE
    from dmm.misc.config_youtubeVOS import get_db_path, get_anno_path, get_img_path 
else:
    from dmm.misc.config import cfg

class MyDataset(data.Dataset):
    """
    impliment the __getitem__; which visit the underlying dataset
    """

    def __init__(self,
                 args,
                 transform=None,
                 target_transform=None,
                 augment=False,
                 split = 'train',
                 resize = False,
                 inputRes = None,
                 video_mode = True,
                 use_prev_mask = False):
        assert(not args.my_augment)
        self.max_seq_len = args.gt_maxseqlen
        self._length_clip = args.length_clip
        self.classes = []
        self.augment = augment
        self.split = split
        self.inputRes = inputRes
        self.video_mode = video_mode
        self.dataset = args.dataset
        self.use_prev_mask = use_prev_mask
        self.args = args 
        self.load_proposals = args.load_proposals
        self.load_proposals_dataset = args.load_proposals_dataset
        self.local_rank = args.local_rank
        self.get_root_dir()
        self.init_meta_data() 
        if self.args.load_proposals_dataset:
            if split == args.train_split:
                self.pred_offline_path = args.pred_offline_path[0] # list to str 
            elif split == args.eval_split:
                self.pred_offline_path = args.pred_offline_path_eval[0] # list to str 
            else:
                logging.error('not support %s'%split)

            if not os.path.isfile(self.pred_offline_path):
                self.offline_pred = self.pred_offline_path
                self.countobj = json.load(open(self.offline_pred+'/countobj.json', 'r'))
            else:
                self.offline_pred = {}
                self.allparts_pred = [self.pred_offline_path] 
                self.init_load_proposals() 
            self.mask_processor = make_roi_mask_post_processor(padding=1, mask_threshold=0.4)

    def init_meta_data(self):
        if self.dataset == 'youtube':
            db_path = get_db_path(self.split)
            json_data = open(db_path)
            self.json_data = json.load(json_data)

    def get_root_dir(self):
        if self.dataset == 'youtube':
            annot_root_dir = get_anno_path(self.split) 
            img_root_dir   = get_img_path(self.split) 
        else:
            img_root_dir = cfg.PATH.SEQUENCES
            annot_root_dir = cfg.PATH.ANNOTATIONS
        self.img_root_dir = img_root_dir 
        self.annot_root_dir = annot_root_dir

    def process_mask(self, proposals):
        mask_prob = [p.get_field('mask') for p in proposals]
        proposals_softm = self.mask_processor.forward_mask_prop(mask_prob, proposals) 
        return proposals_softm

    def get_classes(self):
        return self.classes

    def get_raw_sample(self,index):
        """
        Returns sample data in raw format (no resize)
        return edict: {
            'images': image_obj has attribute: 
                1. name = seq_name 2. starting_frame = name of first frame, e.g. 00000 3. 
            'annotations': }
        """
        img, ins, seg = [],[],[]
        return img, ins, seg

    def init_load_proposals(self, vid=None):
        tic = time.time() 
        self.countobj = {}
        if self.local_rank == 0: 
            logging.info('[{}] loading offline from {}; Nf {}'.format(self.split, self.pred_offline_path, self.allparts_pred))
        num_toload = len(self.allparts_pred)
        for p in range(num_toload): # self.allparts_pred: 
            parts_pred = self.allparts_pred.pop()
            ttic = time.time()
            new_parts = torch.load(parts_pred)
            logging.info('+new_parts {}: {}'.format(len(new_parts), time.time()-ttic))
            self.offline_pred.update(new_parts)
            if vid is not None and vid in self.offline_pred:
                break 
        for vid in self.offline_pred.keys():
            self.countobj[vid] = {}
            for f in self.offline_pred[vid].keys():
                self.countobj[vid][f] = len(self.offline_pred[vid][f])
        if self.local_rank == 0: logging.info('load offline use %.2f | len %d'%(time.time()-tic, len(self.offline_pred)))

    def _need_init(self):
        if not self.load_proposals or not self.load_proposals_dataset:
            return 0 
        else: # load_proposals
            if len(self.allparts_pred) == 0: 
                return 0
            elif vid is not None and vid in self.offline_pred:
                return 0
            else:
                return 1

    def __getitem__(self, index):

        assert(self.video_mode)

        # not supporting the else case
        edict = self.get_raw_sample_clip(index)
        img = edict['images']
        annot = edict['annotations']

        seq_name = img.name
        if self.args.load_proposals_dataset:
            if os.path.isfile(self.pred_offline_path): 
                vid_prop = self.offline_pred[seq_name]
            else:
                vid_prop = torch.load(self.pred_offline_path + '/%s.pth'%seq_name) 
        img_seq_dir = os.path.join(self.img_root_dir, seq_name)
        annot_seq_dir = os.path.join(self.annot_root_dir, annot.name)
        assert(img.name == annot.name), '{} should be same as {}'.format(img.name, annot.name)
        starting_frame = img.starting_frame

        imgs, imgs_names = [], []
        targets, frame_props = [], []
        flip_clip = (random.random() < 0.5)
        # Check if img._files are ustrings or strings
        if type(img._files[0]) == str:
            images = [f for f in img._files]
        else:
            images = [str(f.decode()) for f in img._files]
        frame_img = os.path.join(img_seq_dir,'%05d.jpg' % starting_frame)
        assert(frame_img in images), 'images {}; frame_img {};'.format(images, frame_img)

        starting_frame_idx = images.index(frame_img) # images is a list; find the index of frame_img
        max_ii = min(self._length_clip, len(images)-starting_frame_idx) # substract starting_frame_idx in case starting_frame_idx is 1
        frame_idx_first = 0
        for ii in range(max_ii):
            frame_idx = starting_frame_idx + ii # starting_frame_idx offset by ii
            # some of the video has the init annotation/png not the first frame in images/jpg
            assert(frame_idx < len(images)), '{} out of len {}; at {}; dir: {}; starting idx {}'.format(frame_idx, len(images), ii, images[0], starting_frame_idx) 
            if args.random_select_frames and ii>0 and ('train' in self.split and random.random()>0.5) or ( 'ot' in self.split):
                # select one from frame_idx to len(images) 
                offset = int(random.random() * (len(images)))
                frame_idx = min(offset, len(images)-1)
            frame_idx = int(os.path.splitext(os.path.basename(images[frame_idx]))[0])
            if ii == 0:
                frame_idx_first = frame_idx 
            frame_img = os.path.join(img_seq_dir,'%05d.jpg' % frame_idx)
            imgs_names.append('%05d'%frame_idx)
            if self.args.load_proposals_dataset:
                proposals = vid_prop[imgs_names[-1]].resize((self.inputRes[1], self.inputRes[0]))
                proposals_softm = self.process_mask([proposals])[0]
            if 'train' in self.split and not self.args.test:
                frame_annot = os.path.join(annot_seq_dir,'%05d.png' % frame_idx)
            else:
                assert(self.split == PHASE.VAL.value or self.split == PHASE.TRAINVAL.value)
                frame_annot = os.path.join(annot_seq_dir,'%05d.png' % frame_idx_first) 
            assert(os.path.exists(frame_annot) and os.path.exists(frame_img)), '{} & {}'.format(frame_annot, frame_img)
            img   = Image.open(frame_img)
            annot = Image.open(frame_annot)

            if DEBUG:
                logging.info('open img {}: {}'.format(frame_img, np.array(img).sum()))
            if self.inputRes is not None:
                img   = imresize(img, self.inputRes) 
                annot = imresize(annot, self.inputRes, interp='nearest') 
            if self.transform is not None:
                img = self.transform(img)
            annot = np.expand_dims(annot, axis=0)
        
            if flip_clip and self.flip:
                img = np.flip(img.numpy(),axis=2).copy()
                img = torch.from_numpy(img)
                annot = np.flip(annot,axis=2).copy()
        
            annot = torch.from_numpy(annot)
            annot = annot.float()

            if self.augmentation_transform is not None and (self.split == PHASE.TRAIN.value or self.split == PHASE.TRAINTESTDEVOT.value): 
                # if train on online training video, we apply different augmentation to the data 
                # define tf_function to use PIL: Image as input 
                tf_matrix = self.augmentation_transform(img)
                tf_function = Affine(tf_matrix,interp='nearest')
                img, annot = tf_function(img,annot) # input is tensor: HW3, HW 
                if args.load_proposals_dataset:
                    proposal_mask = proposals_softm.get_field('mask') 
                    N,_,H,W = proposal_mask.shape
                    proposal_masks = []
                    for pi in range(proposal_mask.shape[0]):
                        m = proposal_mask[pi][0]
                        bin_mask = tf_function(m.view(1,H,W))[0] # .float() 
                        proposal_masks.append(bin_mask.view(1,1,H,W))
                        bin_mask = (bin_mask > 0.5).float()
                        if bin_mask.sum() == 0: continue
                        proposals_softm.bbox[pi] = torch.FloatTensor(utils.binmask_to_bbox_xyxy_pt((bin_mask).float()))
                        assert(proposals_softm.mode == 'xyxy'), proposals_softm[0]
                    proposal_masks =torch.cat(proposal_masks, dim=0).view(N,1,H,W)
                    proposals_softm.add_field('mask', proposal_masks)
                # here we expect img shape: 3HW 
                # annot shape: 1HW
                # after tf_function: 3,3,224 and 1,1,224,416; the max value is also changed 

            annot = annot.numpy().squeeze()   
            target, ori_shape = self.sequence_from_masks(seq_name,annot)
            
            if self.target_transform is not None:
                target = self.target_transform(target)
            imgs.append(img)
            targets.append(target)
            if self.load_proposals_dataset:
                frame_props.append(proposals_softm)

            if DEBUG:
                check_data = torch.load('../../drvos/src/debug/data_%d_ii%d.pth'%(index, ii))
                gen_data = [img, annot, proposals_softm]
                for id_item, (ic, ig) in enumerate(zip(check_data, gen_data)):
                    logging.info('checking %d'%id_item)
                    if id_item <= 1:
                        ic = ic.sum() 
                        ig = ig.sum()
                        CHECKEQ(ic, ig)
                    else:
                        CHECKEQ(ic.bbox.sum(), ig.bbox.sum())
                        CHECKEQ(ic.get_field('mask').sum(), ig.get_field('mask').sum())


                # torch.save([img, annot, proposals_softm], 'debug/data_%d.pth'%index)
        boxt = utils.binmask_to_bbox_xyxy(targets[0][:,:-1].sum(0).reshape(ori_shape[0], ori_shape[1])) 
        short_side = min(boxt[3]-boxt[1], boxt[2]-boxt[0]) if len(boxt) > 0 else 0 # boxt can be empty
        if args.skip_empty_starting_frame and short_side < 3 and 'val' not in self.split:
            new_index = random.random() * self.__len__() 
            new_index = int(new_index)
            logging.info('box {}'.format(boxt))
            logging.warning('[training][dataset] split %s; skip_empty_starting_frame found one empty start frame %d:%s| get_item id %d; reduce to %d'%(
                self.split, frame_idx_first, seq_name, index, new_index
                ))
            return self.__getitem__(new_index)

        if args.load_proposals_dataset: 
            return imgs, frame_props, targets, seq_name, starting_frame 
        else:
            return imgs, imgs_names, targets, seq_name, starting_frame

    def __len__(self):
        return len(self.sequence_clips)

    def get_sample_list(self):
        return self.sequence_clips
    def sequence_from_masks(self, seq_name, annot):
        """
        Reads segmentation masks and outputs sequence of binary masks and labels
        """
        if self.dataset == 'youtube':
            self.init_meta_data()
            instance_ids_str = self.json_data['videos'][seq_name]['objects'].keys()
            instance_ids = []
            for id in instance_ids_str:
                instance_ids.append(int(id))
        else:
            instance_ids = np.unique(annot)[1:]
            #In DAVIS 2017, some objects not present in the initial frame are annotated in some future frames with ID 255. We discard any id with value 255.
            if len(instance_ids) > 0:
                    instance_ids = instance_ids[:-1] if instance_ids[-1]==255 else instance_ids

        h = annot.shape[0]
        w = annot.shape[1]
        total_num_instances = len(instance_ids)
        max_instance_id = 0
        if total_num_instances > 0:
            max_instance_id = int(np.max(instance_ids))
        num_instances = max(self.max_seq_len,max_instance_id)

        gt_seg = np.zeros((num_instances, h*w))
        size_masks = np.zeros((num_instances,)) # for sorting by size
        sample_weights_mask = np.zeros((num_instances,1))

        for i in range(total_num_instances):
            id_instance = int(instance_ids[i])
            aux_mask = np.zeros((h, w))
            aux_mask[annot==id_instance] = 1
            gt_seg[id_instance-1,:] = np.reshape(aux_mask,h*w)
            size_masks[id_instance-1] = np.sum(gt_seg[id_instance-1,:])
            sample_weights_mask[id_instance-1] = 1

        gt_seg = gt_seg[:][:self.max_seq_len]
        sample_weights_mask = sample_weights_mask[:][:self.max_seq_len]
        targets = np.concatenate((gt_seg,sample_weights_mask),axis=1)

        return targets, (h,w)
