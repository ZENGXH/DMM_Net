import logging
import os
import sys 
import numpy as np
import torch
from torch.nn import functional as F
from .checker import *
from dmm.modules.submodules.relax_match import relax_matching 
def compute_iou_binary_mask_2D(annotation, segmentation):
    """
    input: NxM, NxM 
    output: N
    """
    CHECK2D(annotation)
    CHECK2D(segmentation)
    annotation = annotation > 0.5
    segmentation = segmentation > 0.5
    annotation = annotation.byte()
    segmentation = segmentation.byte()
    sys.stdout.flush()
    with torch.no_grad():
        union = (annotation | segmentation).float()
        inter = (annotation & segmentation).float()

        union = union.sum(1) + 1e-6
        inter = inter.sum(1)
        iou = inter / union
    return iou.detach()

def compute_matching_loss(proposed_mask, targets, similarity_matrix, cfgs):
     CHECK3D(proposed_mask)
     CHECK3D(targets)
     CHECKEQ(proposed_mask.shape[-1], targets.shape[-1])
     bin_proposed_mask = proposed_mask > 0.5 
     num_template = targets.shape[0] 
     num_proposal = bin_proposed_mask.shape[0] 
     targets_kq               = targets.contiguous().view(num_template, 1, -1).expand(-1, num_proposal, -1)
     bin_proposed_mask_expand = bin_proposed_mask.view(num_proposal, -1).expand(num_template, -1, -1)
     targets_kq               = targets_kq.contiguous().view(num_template*num_proposal, -1)
     bin_proposed_mask_expand = bin_proposed_mask_expand.contiguous().view(num_template*num_proposal, -1)
     CHECKEQ(bin_proposed_mask_expand.shape, targets_kq.shape)
     gt_cost_matrix           = compute_iou_binary_mask_2D(bin_proposed_mask_expand, targets_kq).view(num_template, num_proposal)
     similarity_matrix_sub = similarity_matrix
     gt_matched, _, _, _ = relax_matching(-gt_cost_matrix, max_iter=0, proj_iter=0, lr=0) # use greedy 

     """ compute loss """
     # cost_loss = None
     cost_loss = F.mse_loss(similarity_matrix_sub, gt_matched) 
     return cost_loss
   
def get_cosine_score(query_feature, key_feature, cfgs=None):
    """ compute cosine score 
    Arguments:
        query_feature (Tensor): Shape: O,D
        key_feature (Tensor):   Shape: P,D
    """
    num_template = query_feature.shape[0] 
    num_proposal = key_feature.shape[0] 
    query_feature = query_feature.unsqueeze(2).expand(-1,-1,num_proposal) # ODP  
    key_feature   = key_feature.permute(1,0).expand(num_template, -1,-1) # PD->DP->ODP
    CHECKEQ(query_feature.shape, key_feature.shape)
    
    similarity_matrix = F.cosine_similarity(query_feature, key_feature, dim=1) # query_feature ODP, key_feature ODP 
    return similarity_matrix

