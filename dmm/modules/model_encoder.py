""" container for coco model, serve as a API
which can load prediction.pth 
"""
import torch
import torch.nn as nn
import sys
import time
import logging
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.modeling.roi_heads.mask_head.roi_mask_feature_extractors import make_roi_mask_feature_extractor
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from dmm.utils.masker import make_roi_mask_post_processor 
from dmm.utils.boxlist_ops import filter_results 
from dmm.utils.utils import get_skip_dims, load_DMM_config
from dmm.utils.checker import *
from .base import FeatureExtractorBase
from .vision import VGG16, ResNet34, ResNet50, ResNet101
from torchvision import transforms, models 
DEBUG = 0 

class FeatureExtractor(FeatureExtractorBase):
    '''
    Returns base network to extract visual features from image
    '''
    def __init__(self, args, pretrained = 1):
        super().__init__(args, pretrained)
        self.count = 0
        skip_dims_in                = get_skip_dims(args.base_model)
        self.args                   = args
        self.local_rank             = args.local_rank
        self.isrank0 = self.local_rank == 0
        if args.load_proposals:
            self.pred_offline_path = args.pred_offline_path 
            self.pred_offline = None
        else:
            self.pred_offline_path = None

        if args.base_model == 'resnet34':
            self.base = ResNet34()
        elif args.base_model == 'resnet50':
            self.base = ResNet50()
            self.base.load_state_dict(models.resnet50(pretrained=True).state_dict())
        elif args.base_model == 'resnet101':
            self.base = ResNet101()
        elif args.base_model == 'vgg16':
            self.base = VGG16()
        else:
            raise Exception("The base model you chose is not supported ! {}".format(args.base_model))
       
        self.mask_processor = make_roi_mask_post_processor(padding=1, mask_threshold=0.4)
        self.cfgs = load_DMM_config(args.config_train, args.local_rank)
        self.pred_offline = []
    def init_pred_offline(self):
        tic = time.time() 
        if self.isrank0: logging.info('loading offline from {}'.format(self.pred_offline_path))
        for fpath in self.pred_offline_path:
            self.pred_offline.extend(torch.load(fpath)) # self.pred_offline_path) 
        if self.isrank0: logging.info('load offline use %.2f'%(time.time()-tic))

    def need_init(self):
        if self.args.load_proposals_dataset:
            return False
        elif len(self.pred_offline) > 0:
            return False
        elif self.args.load_proposals:
            return True
        return False 
   
    def get_optimizer(self, optim_name, lr, args, weight_decay = 0, momentum = 0.9):
        """
        customize optimizer for encoder 
        weight_decay and weight_decay_cnn has the same default value, not add here, but 
        it can be append to list
        """
        # projection parameters:
        opt_list = []
        opt_list.append({'params':self.get_skip_params(), 'lr':args.lr})
        opt_list.append({'params':self.get_backbone_para(), 'lr':args.lr_cnn})
        assert(optim_name == 'adam')
        opt = torch.optim.Adam(opt_list, lr=lr, weight_decay = weight_decay)
        return opt

    def forward_base(self, coco_input, args):
        """ backbone: body """
        x5,x4,x3,x2,x1 = self.base(coco_input)
        body_feat = x2,x3,x4,x5  
        fpn_feature = None
        CHECK4D(coco_input)
        assert(len(coco_input.shape) == 4 and coco_input.shape[1] == 3), coco_input.shape 
        lcoco_input = [c.squeeze(0) for c in coco_input.split(1)]
        coco_input = to_image_list(lcoco_input, 32)
        return fpn_feature, body_feat, coco_input

    def forward(self,args,coco_input,predid_cur_frames=None,targets=None,proposals=None):
        """ given proposals or proposal id  
        Arguments: 
            args: 
            coco_input: f,3,h,w 
            targets:    f,O,h,w
        Return:
            proposals: list[BoxList], len=f, BoxListLen=num_proposal 
            return_feature: 
                    image feature for the downsteam layer;
                    dict of feature tuple;
        """
        if self.need_init():
            self.init_pred_offline() 
        cur_device = coco_input.device
        # backbone_feature, 
        fpn_feature, body_feature, coco_input = self.forward_base(coco_input, args)
        """ RPN """
        if predid_cur_frames is not None and not args.load_proposals_dataset:
            assert(proposals is None)
            proposals = [self.pred_offline[pi].resize((coco_input.image_sizes[ki][1],
                                                      coco_input.image_sizes[ki][0])
                                                      ) for ki, pi in enumerate(predid_cur_frames)]
            for ki, p in enumerate(proposals):
                if len(p) == 0: # empty proposals 
                    logging.warning('get empty proposal {}'.format(p))
                    proposals[ki] = self.pred_offline[-1].resize((coco_input.image_sizes[ki][1], coco_input.image_sizes[ki][0]))
            proposals = [p.to(cur_device) for p in proposals] 
            mask_prob = [p.get_field('mask') for p in proposals]
            proposals_softm = self.mask_processor.forward_mask_prop(mask_prob, proposals) 
        elif args.load_proposals_dataset: 
            assert(proposals is not None) 
            proposals_softm = proposals
        else:
            raise ValueError('need to provide predid_cur_frames if load proposals in encoder ')
        score_field = 'scores' if 'scores' in proposals[0].fields() else 'objectness'
        proposals = filter_results(proposals_softm, nms_thresh=self.cfgs['encoder']['nms_thresh'], 
                max_proposals=self.cfgs['sort_max_num'], score_field=score_field) 
        input_shape = coco_input.tensors.shape[-2:]
        body_x2, body_x3, body_x4, body_x5 = body_feature # return_feature['body']# body_feat
        x5_skip = self.bn5(self.sk5(body_x5))
        x4_skip = self.bn4(self.sk4(body_x4))
        x3_skip = self.bn3(self.sk3(body_x3))
        x2_skip = self.bn2(self.sk2(body_x2))
        p5 = self.prop5(body_x5)
        p4 = self.prop4(body_x4)
        p3 = self.prop3(body_x3)
        p2 = self.prop2(body_x2)
        backbone_feature = p2,p3,p4,p5
        refine_input_feature = (x5_skip, x4_skip, x3_skip, x2_skip)
        if DEBUG:
            feat = {
                    'para_prop3': list(self.prop3.parameters()), 
                    'para_prop4': list(self.prop4.parameters()), 
                    'body': body_feature, 'x_skip':refine_input_feature, 'back_bone': backbone_feature, 'prop':[ p.bbox for p in proposals]}
            feat_check = torch.load('../../drvos/src/debug/encoder_%d.pth'%self.count)
            logging.info('check %d'%self.count) 
            self.count += 1
            CHECKDEBUG(feat, feat_check)

        features = {'backbone_feature': backbone_feature, 
                    'refine_input_feat': refine_input_feature,
                    'body_feature': body_feature
                    }
        cocoloss = {}
        return features, proposals, cocoloss

