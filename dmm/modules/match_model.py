"""
MatchModel: perform matching 
"""
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from .submodules.relax_match import relax_matching, hungarian_matching 
from dmm.utils.checker import *
from dmm.utils.match_helper import compute_iou_binary_mask_2D
import dmm.utils.match_helper as match_helper

class MatchModel(nn.Module):
    def __init__(self, cfgs={}, is_test=0):
        super(MatchModel, self).__init__()
        self.cfgs = cfgs
        self.is_test = is_test 
        self.match_algo = cfgs['matching']['algo']
        self.max_iter = self.cfgs['relax_max_iter']
        self.proj_iter = self.cfgs['relax_proj_iter']
        self.relax_lr = self.cfgs['relax_learning_rate']
        assert(self.match_algo == 'relax' or self.match_algo == 'hun')
 
    def forward(self, proposed_feature, proposed_mask, template_feature, mask_last_occurence, proposal_score, targets=None):
        """ matching layer compute the cost matrix and perform matching 
        Arguments"
            proposed_feature: p,256; PHW  
            proposed_mask: p,h.w: PHW
            template_feature: list of template_feature [o,256]: 
            mask_last_occurence: PHW 
            proposal_score
            targets: OHW
        return:
        """
        features = {'proposed': proposed_feature,  # PXX
                    "template":template_feature}   # OXX 
        mask     = {'proposed': proposed_mask,     # PHW 
                    'template':mask_last_occurence}# OHW
        scores   = {'proposal_score':proposal_score}
        # loss = {}
        # compute Cosine Distance 
        target_sim_matrix, num_K, num_Q, match_loss = self.compute_cost_matrix(features, mask, scores, targets)
        n_prop = proposed_mask.shape[0] # P,H,W
        n_tplt = template_feature[0].shape[0] # O,D
        full_outmask, match_score, det_score, logic_mask, assign_matrix = self.match_with_first_frame( 
                target_sim_matrix, n_prop, n_tplt, proposed_mask.float(), proposal_score, mask_last_occurence)
        return full_outmask, match_score, det_score, full_outmask, match_loss 

    def compute_cost_matrix(self, features, mask, scores, targets=None):
        """ compute_cost_matrix for one frame 
        filling the score matrix
        Arguments:
            proposed_feature:   p,256, or [p,5,256,h,w] if spatial_mean == 3
            template_feature:   list of template_feature: [o,256] or N_template_feature x [O,L,D,H,W] 
            proposed_mask: p,h.w
            mask_last_occurence: ohw, same as unet input shape
            proposal_score: 
            targets: OHW 
        return sim_matrix, n_prop, n_tplt, match_loss
        """
        proposed_feature = features['proposed']
        template_feature = features['template']
        proposed_mask = mask['proposed']
        mask_last_occurence = mask['template'] 
        proposal_score = scores['proposal_score']

        match_loss = {}
        CHECK3D(proposed_mask)
        n_prop = proposed_mask.shape[0]
        n_tplt = template_feature[0].shape[0] # len(look_up_table)
        N_template_feature = len(template_feature) 
        feature_sim = proposed_feature.new_zeros(n_tplt, n_prop)
        for t in range(N_template_feature):
            query_feature = template_feature[t] # Q, D
            feature_sim += self.compute_feature_score(proposed_feature, query_feature)
        feature_sim /= N_template_feature
        sim_matrix = feature_sim
        # expand features to be (Q,D,K)
        if targets is not None:
            cost_loss = match_helper.compute_matching_loss(proposed_mask, targets, sim_matrix, self.cfgs)
            match_loss.update({'cost_loss': cost_loss})
        # assert(len(match_loss) > 0), 'targets: {}; loss {}'.format(targets, match_loss)
        O, H, W = mask_last_occurence.shape 
        proposed_mask = proposed_mask.view(n_prop, -1).expand(n_tplt, -1, -1) 
        proposed_mask = proposed_mask.contiguous().view(n_tplt*n_prop, -1) 
        mask_last_occurence_kq = mask_last_occurence.contiguous().view(n_tplt, 1, -1).expand(-1, n_prop, -1)
        mask_last_occurence_kq = mask_last_occurence_kq.contiguous().view(n_tplt*n_prop, -1)
        CHECKEQ(proposed_mask.shape, mask_last_occurence_kq.shape)
        iou_premask_prop = compute_iou_binary_mask_2D( proposed_mask, mask_last_occurence_kq).view(n_tplt, n_prop) # OP
        sim_matrix=sim_matrix*(1-self.cfgs['score_weight'])+iou_premask_prop*self.cfgs['score_weight']  
        return sim_matrix, n_prop, n_tplt, match_loss

    def match_with_first_frame(self, sim_matrix, n_prop, n_tplt, proposed_mask, proposal_score, mask_last_occurence):
        """ input sim_matrix to matching algorithm 
        output matching decision matrix, 
        Q: query, Look-up-table/features_each_instances: 
        K: key, P, D: mean pool to be 
        Arguments:
            sim_matrix: O P
        Return:
            full_outmask: shape O,H,W 
            match_score: O 
            det_score: O
            logic_mask, 
            binary_Ridx_matched 
        """
        table_return = {}
        need_pad = 0
        if sim_matrix.shape[1] <= sim_matrix.shape[0]: 
            # 0: template, 1: proposal; require: # proposal > # template
            sim_matrix_pad = sim_matrix.new_zeros((sim_matrix.shape[0], sim_matrix.shape[0]+1))
            sim_matrix_pad[:, :sim_matrix.shape[1]] = sim_matrix
            need_pad = sim_matrix_pad.shape[1] - sim_matrix.shape[1]
        else: 
            sim_matrix_pad = sim_matrix
        cost_matrix = -sim_matrix_pad
        """ compute assignment matrix """
        if self.match_algo == 'relax':
            Ridx_matched, cost, X_list, _ = relax_matching( cost_matrix, max_iter=self.max_iter, 
                                                            proj_iter=self.proj_iter, lr=self.relax_lr)
            Ridx_matched = sum(X_list) / len(X_list)
        elif self.match_algo == 'hun':
            Ridx_matched, _,_,_ = hungarian_matching(cost_matrix)
        # R*K
        maxv, maxi = Ridx_matched.max(dim=1, keepdim=True)
        if self.is_test:
            logic_mask = (Ridx_matched == maxv).float() 
        else:
            logic_mask = (Ridx_matched > 0.01).float() 
        binary_Ridx_matched = Ridx_matched.float()* logic_mask 
        H,W = proposed_mask.shape[-2], proposed_mask.shape[-1]
        n_prop = proposed_mask.shape[0]
        proposed_mask2d = proposed_mask.view(n_prop, -1) # P H*W
        if need_pad:
            pad_proposed_mask2d = proposed_mask2d.new_zeros((n_prop+need_pad, proposed_mask2d.shape[1]))
            pad_proposed_mask2d[:n_prop] = proposed_mask2d 
            pad_proposal_score = proposal_score.new_zeros((n_prop+need_pad))
            pad_proposal_score[:n_prop] = proposal_score
            CHECKEQ(proposal_score.shape[0], n_prop)
        else:
            pad_proposed_mask2d = proposed_mask2d
            pad_proposal_score = proposal_score
        # binary_Ridx_matched: O P
        full_outmask=torch.mm(binary_Ridx_matched,pad_proposed_mask2d).view(-1,H,W) # OHW
        # match_score: O H W
        match_score, _ = (Ridx_matched.clamp(0,1) * (- cost_matrix)).max(1) # higher the more confideny 
        det_score = (pad_proposal_score.view(1, -1).expand(n_tplt, -1) * binary_Ridx_matched).sum(1)
        return full_outmask, match_score, det_score, logic_mask, binary_Ridx_matched 

    def compute_feature_score(self, key_feature, query_feature): 
        sim_matrix = match_helper.get_cosine_score(query_feature, key_feature, self.cfgs) 
        return sim_matrix 
