""" DMM model: 
    0. extract feature 
    1. do matching
"""
import logging
import torch
import torch.nn as nn
from .feature_extractor import make_roi_mask_feature_extractor
from .match_model import MatchModel
from dmm.utils.checker import * 
class DMM_Model(nn.Module):
    r""" container for all DMM modules: 
        match_layer
        feature_extractor
    """
    def __init__(self, cfgs, is_test=0):
        super(DMM_Model, self).__init__()
        self.match_layer = MatchModel(cfgs, is_test)
        self.feature_extractor = make_roi_mask_feature_extractor()
        self.match_algo = cfgs['matching']['algo']

    def fill_template_dict(self, args, proposals, features, y_mask, tplt_valid_batch):
        r""" one frame of each video in the batch 
            match current frame's proposals for all instances; 
        Arguments:
            proposals: BoxList
            backbone_feature: tuple of feature: B,Lev,(D,H,W); 
            mask_last_occurence: binary: B x [OHW]
            y_mask: reference mask, y_mask: B,max_num_obj,HW 
            tplt_feature: Bx[OHW]
        Return:
            tplt_dict: dict, len=B, 
                tplt_dict[b]: list of tplt features for current video 
                tplt_dict: { batch_id: { 'feat': [], 'refine_input_feat': [], (img feat) 
        """
        backbone_feature = features['backbone_feature']
        refine_input_feat= features['refine_input_feat']
        boxes_per_image = [len(box) for box in proposals]
        result_alllevel = self.feature_extractor(backbone_feature, proposals)
        MRC_feature_list = result_alllevel.split(boxes_per_image, dim=0)
        tplt_dict = {}
        for b, feat in enumerate(MRC_feature_list): # loop over B
            tplt_dict[b] = {}
            tplt_dict[b]['feat'] = [feat] # O,L,D,H,W
            tplt_dict[b]['refine_input_feat'] = [tuple([f[b] for f in refine_input_feat])] # 1x[L(D,H,W)]
        return tplt_dict 

    def inference(self, infos, proposals, backbone_feature, mask_last_occurence, tplt_dict, target=None):
        args = infos['args']
        img_shape = infos['shape']
        extra_frame, tplt_valid_batch = infos['extra_frame'], infos['valid']

        hidden_spatial = None
        B,F,H,W = CHECK4D(mask_last_occurence)
        boxes_per_image = [len(box) for box in proposals]
        result_alllevel = self.feature_extractor(backbone_feature, proposals)
        prop_feat = result_alllevel.split(boxes_per_image, dim=0) # Bx[P.D.H.W]
        prop_m = [prop.get_field('mask').squeeze(1) for prop in proposals] # B[p1hw] -> BPHW
        output_mask = []
        out_mask_last = []

        for bid in range(len(prop_m)): # iterating over videos in current batch  
            tplt_valid_vid  = tplt_valid_batch[bid] # F,1 
            O = int(tplt_valid_vid.sum().item())
            CHECKEQ(prop_m[bid].shape[-2:], mask_last_occurence[bid].shape[-2:])
            if O == 0 or extra_frame[bid]: # have zero targets 
                output_mask.append(mask_last_occurence.new_zeros(F,H,W)) 
                out_mask_last.append(mask_last_occurence[bid]) 
                continue 
            tplt_feat_valid, OF_matrix = self.prepare_tplt_feature(tplt_valid_vid, tplt_dict, bid) 
            CHECKEQ(len(proposals[bid]), prop_feat[bid].shape[0]) # n proposal
            prop_score = proposals[bid].get_field('objectness') if 'objectness' in proposals[bid].fields() \
                                else proposals[bid].get_field('scores') 
            mask_occur_valid = mask_last_occurence[bid,:O,:,:].view(O,H,W)
            full_outmask_valid, match_score, det_score, mask_occur_newvalid,  \
                loss = self.match_layer(prop_feat[bid], prop_m[bid], tplt_feat_valid, mask_occur_valid, 
                                            prop_score, targets=None if target is None else target[bid][:O])
            FO_matrix = OF_matrix.t()
            full_output_masks       = torch.mm(FO_matrix, full_outmask_valid.view(O,-1)).view(F,H,W)
            mask_last_occurence_new = torch.mm(FO_matrix, mask_occur_newvalid.view(O,-1)).view(F,H,W) 
            output_mask.append(full_output_masks) 
            out_mask_last.append(mask_last_occurence_new)
        output_mask = torch.stack(output_mask, dim=0)
        out_mask_last = torch.stack(out_mask_last, dim=0)
        match_loss = [] 
        return output_mask, tplt_dict, match_loss, out_mask_last

    def forward(self, args, proposals, backbone_feature, mask_last_occurence, tplt_dict, tplt_valid_batch, targets):
        r""" one frame of each video in the batch 
            prepare and call match_layer, 
            match current frame's proposals for all instances; 
        Arguments:
            proposals: BoxList
            backbone_feature: tuple of feature: B 
            mask_last_occurence: binary: B x [OHW] # B
            tplt_dict: dict, len == batch_size, 
            tplt_valid_batch: 2D binary tensor, BxO
            targets: targets mask: BOHW
        Returns:
            full_output_masks:
            output_mask, 
            tplt_dict 
            match_loss 
            out_mask_last, 
        """
        B,F,H,W = CHECK4D(mask_last_occurence)
        boxes_per_image = [len(box) for box in proposals]
        result_alllevel = self.feature_extractor(backbone_feature, proposals)
        prop_feat = result_alllevel.split(boxes_per_image, dim=0)
        # prepare the input of self.match()
        prop_m = [prop.get_field('mask').squeeze(1) for prop in proposals] # B[p1hw] -> BPHW
        output_mask = []
        out_mask_last = []
        match_loss = []
        for bid in range(len(prop_m)): 
            tplt_valid_vid  = tplt_valid_batch[bid] # F,1 
            O = int(tplt_valid_vid.sum().item())
            if O == 0: 
                output_mask.append(mask_last_occurence.new_zeros(F,H,W)) 
                out_mask_last.append(mask_last_occurence[bid]) 
                match_loss.append(prop_feat[bid].sum() * 0) 
                continue 
            CHECKEQ(prop_m[bid].shape[-2:], mask_last_occurence[bid].shape[-2:])
            mask_occur_valid = mask_last_occurence[bid, :O].view(O,H,W) 
            prop_score = proposals[bid].get_field('objectness') if 'objectness' in proposals[bid].fields() \
                                else proposals[bid].get_field('scores') 
            tplt_feat_valid, OF_matrix = self.prepare_tplt_feature(tplt_valid_vid, tplt_dict, bid) 
            assert(targets is not None) # FHW, training mode, must have targetd 
            targets_valid = targets[bid, :O].view(O,H,W) 
            full_outmask_valid, match_score, det_score, mask_occur_newvalid,  \
                loss = self.match_layer(prop_feat[bid], prop_m[bid], tplt_feat_valid, 
                                        mask_occur_valid, prop_score, targets=targets_valid)
            FO_matrix = OF_matrix.t()
            full_output_masks       = torch.mm(FO_matrix, full_outmask_valid.view(O,-1)).view(F,H,W) 
            mask_last_occurence_new = torch.mm(FO_matrix, mask_occur_newvalid.view(O,-1)).view(F,H,W) 
            if len(loss) > 0:
                match_loss.append(loss['cost_loss'])
            output_mask.append(full_output_masks) 
            out_mask_last.append(mask_last_occurence_new)
        output_mask = torch.stack(output_mask, dim=0)
        out_mask_last = torch.stack(out_mask_last, dim=0)
        return output_mask, tplt_dict, match_loss, out_mask_last

    def prepare_tplt_feature(self, tplt_valid_vid, tplt_dict, bid): #tplt_prop_all_class):
        r""" select valid tplt feature 
            F: full, maximum num of proposall; 
            O: number of real instances in current vid 
        """
        O = int(tplt_valid_vid.sum().item())
        tplt_feat    = tplt_dict[bid]['feat'] 
        FF_matrix = torch.diag(tplt_valid_vid).float() 
        OF_matrix = FF_matrix[:O, :] # OF
        assert(type(tplt_feat) == list)
        self.tplt_feat_shape = tplt_feat[0].shape
        F,D1 = tplt_feat[0].shape 
        tplt_feat_valid=[torch.mm(OF_matrix, tplt_feat_idx.view(F, -1)).view(O,D1) \
                          for tplt_feat_idx in tplt_feat]
        return tplt_feat_valid, OF_matrix
