import torch 
import torch.nn as nn
from maskrcnn_benchmark.modeling.poolers import Pooler
from dmm.utils.checker import *
class FeatureExtractor(nn.Module):
    """ Heads for FPN for classification """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.init_pooler()

    def init_pooler(self):
        """ build roi pooler """
        scales = (0.25, 0.125, 0.0625, 0.03125) # benchmark_cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = 2  # benchmark_cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        resolution = 14      # benchmark_cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        self.collector = Pooler( output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio,)
        self.num_levels = len(self.collector.poolers) 
        self.output_size = (resolution, resolution)

    def forward(self, backbone_feature, proposals):
        """ do ROI pooing 
        input: 
            backbone_feature: list of feature [backbone_feature, ...]
            proposals 
        output:
            feature_extracted, 
        """
        result_alllevel, rois = self.roi_pooling(backbone_feature, proposals)
        result_alllevel = result_alllevel.mean(4).mean(3).view(len(rois), -1)
        return result_alllevel

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat( [ torch.full((len(b), 1), i, dtype=dtype, device=device) for i, b in enumerate(boxes) ], dim=0,)
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def roi_pooling(self, backbone_feature, proposals):
        """ perform pooling in all level 
        return feature in shape: 
            Tensor shape: num_levels, num_rois, num_channels, H, W_resolution
        """
        rois = self.convert_to_roi_format(proposals)
        num_rois = len(rois) #num_rois for all frames + all proposals
        num_channels = backbone_feature[0].shape[1]
        output_size = self.output_size[0]
        dtype, device = backbone_feature[0].dtype, backbone_feature[0].device
        result_alllevel = torch.zeros( (num_rois, self.num_levels, num_channels, self.output_size[0], self.output_size[1]), dtype=dtype, device=device,)
        for level, (per_level_feature, pooler) in enumerate(zip(backbone_feature, self.collector.poolers)):
             result_alllevel[:, level] = pooler(per_level_feature, rois)
        return result_alllevel, rois

def make_roi_mask_feature_extractor():
    return FeatureExtractor()

def cat(tensors, dim=0):
    """ Efficient version of torch.cat that avoids a copy if there is only a single element in a list """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)
