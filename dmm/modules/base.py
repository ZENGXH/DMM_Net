""" container for coco model, serve as a API
"""
import torch
import torch.nn as nn
from .clstm import ConvLSTMCell, ConvLSTMCellMask
import torch.nn.functional as F 
import sys
import time
import logging
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.data import transforms as T
from dmm.utils.checker import *
from dmm.utils.utils import get_skip_dims 

class FeatureExtractorBase(nn.Module):
    '''
    Returns base network to extract visual features from image
    '''
    def __init__(self, args, pretrained = 1):
        super(FeatureExtractorBase,self).__init__()
        self.coco = None
        self.feature_extractor = None

        skip_dims_in  = get_skip_dims(args.base_model)
        self.hidden_size = args.hidden_size
        self.kernel_size = args.kernel_size
        self.padding = 0 if self.kernel_size == 1 else 1
        self.args = args
        hid_sz = int(self.hidden_size)
        pad_sz = int(self.padding)
        ker_sz = int(self.kernel_size)
        self.sk5 = nn.Conv2d(skip_dims_in[0],hid_sz,ker_sz,padding=pad_sz)
        self.sk4 = nn.Conv2d(skip_dims_in[1],hid_sz,ker_sz,padding=pad_sz)
        self.sk3 = nn.Conv2d(skip_dims_in[2],int(hid_sz/2),ker_sz,padding=pad_sz)
        self.sk2 = nn.Conv2d(skip_dims_in[3],int(hid_sz/4),ker_sz,padding=pad_sz)
        self.bn5 = nn.BatchNorm2d(hid_sz)
        self.bn4 = nn.BatchNorm2d(hid_sz)
        self.bn3 = nn.BatchNorm2d(int(hid_sz/2))
        self.bn2 = nn.BatchNorm2d(int(hid_sz/4))
        self.prop5 = nn.Sequential( nn.Conv2d(skip_dims_in[0],hid_sz,ker_sz,padding=pad_sz),
                        nn.BatchNorm2d(hid_sz), nn.ReLU(),
                        nn.Conv2d(hid_sz, hid_sz,ker_sz,padding=pad_sz), nn.BatchNorm2d(hid_sz),)
        self.prop4 = nn.Sequential( nn.Conv2d(skip_dims_in[1],hid_sz,ker_sz,padding=pad_sz),
                        nn.BatchNorm2d(hid_sz), nn.ReLU(),
                        nn.Conv2d(hid_sz, hid_sz,ker_sz,padding=pad_sz), nn.BatchNorm2d(hid_sz),)
        self.prop3 = nn.Sequential( nn.Conv2d(skip_dims_in[2],int(hid_sz/2),ker_sz,padding=pad_sz),
                        nn.BatchNorm2d(int(hid_sz/2)), nn.ReLU(),
                        nn.Conv2d(int(hid_sz/2), hid_sz,ker_sz,padding=pad_sz), nn.BatchNorm2d(hid_sz),)
        self.prop2 = nn.Sequential( nn.Conv2d(skip_dims_in[3],int(hid_sz/4),ker_sz,padding=pad_sz),
                        nn.BatchNorm2d(int(hid_sz/4)), nn.ReLU(),
                        nn.Conv2d(int(hid_sz/4), hid_sz,ker_sz,padding=pad_sz), nn.BatchNorm2d(hid_sz))
    def get_backbone_para(self):
        for name, param in self.base.named_parameters():
            if param.requires_grad:
                assert(param.is_leaf)
                yield param

    def get_skip_params(self):
        """ parameters of the neck parameters connecting body and matching. decoder 
        """
        plist = []
        param = [self.sk2,self.sk3,self.sk4,self.sk5, self.bn2,self.bn3,self.bn4,self.bn5, \
                self.prop5,self.prop4,self.prop3,self.prop2]
        for p in param:
            plist.extend(list(p.parameters()))
        return plist

class RSISMask(nn.Module):
    """
    The recurrent decoder
    """

    def __init__(self, args):

        super(RSISMask,self).__init__()
        self.hidden_size = args.hidden_size
        self.kernel_size = args.kernel_size
        padding = 0 if self.kernel_size == 1 else 1

        self.dropout = args.dropout
        self.skip_mode = args.skip_mode

        # convlstms have decreasing dimension as width and height increase
        skip_dims_out = [self.hidden_size, int(self.hidden_size/2),
                         int(self.hidden_size/4),int(self.hidden_size/8)]

        # initialize layers for each deconv stage
        self.clstm_list = nn.ModuleList()
        # 4 is the number of deconv steps that we need to reach image size in the output
        for i in range(len(skip_dims_out)):
            if i == 0:
                clstm_in_dim = self.hidden_size
            else:
                clstm_in_dim = skip_dims_out[i-1]
                if self.skip_mode == 'concat':
                    clstm_in_dim*=2

            clstm_i = ConvLSTMCellMask(args, clstm_in_dim, skip_dims_out[i],self.kernel_size, padding = padding)
            self.clstm_list.append(clstm_i)
            del clstm_i

        self.conv_out = nn.Conv2d(skip_dims_out[-1], 1,self.kernel_size, padding = padding)

        # calculate the dimensionality of classification vector
        # side class activations are taken from the output of the convlstm
        # therefore we need to compute the sum of the dimensionality of outputs
        # from all convlstm layers
        fc_dim = 0
        for sk in skip_dims_out:
            fc_dim+=sk
  
    def forward(self, skip_feats, prev_mask_list, prev_state_spatial, prev_hidden_temporal):     
        prev_mask_n2 = [p[:,2:,:,:] for p in prev_mask_list]
        prev_maskn   = [p[:,1:2,:,:] for p in prev_mask_list]
        prev_mask    = [p[:,0:1,:,:] for p in prev_mask_list]
        #The decoder receives two hidden state variables: 
        # 1. hidden_spatial (a tuple, with hidden_state and cell_state) which refers to the
        #    hidden state from the previous object instance from the same time instant, and
        # 2. hidden_temporal which refers to the hidden state from the same
        #    object instance from the previous time instant.
        #input: feats: (feature_at_scale_1(shape: B,D,H,W), feature_at_scale_2, ..)
        clstm_in = skip_feats[0]
        skip_feats = skip_feats[1:]
        hidden_list = []

        for i in range(len(skip_feats)+1):

            # hidden states will be initialized the first time forward is called
            if prev_state_spatial is None:
                if prev_hidden_temporal is None:
                    state = self.clstm_list[i](clstm_in, prev_mask[i], None, None) 
                    state1 = self.clstm_list[i](clstm_in, prev_mask_n2[i], None, None) 
                    state2 = self.clstm_list[i](clstm_in, prev_maskn[i], None, None)
                else:
                    state=self.clstm_list[i](clstm_in, prev_mask[i], None, prev_hidden_temporal[i]) 
                    state1 =  self.clstm_list[i](clstm_in, prev_mask_n2[i], None, prev_hidden_temporal[i]) 
                    state2 = self.clstm_list[i](clstm_in, prev_maskn[i], None, prev_hidden_temporal[i])
            else:
                # else we take the ones from the previous step for the forward pass
                if prev_hidden_temporal is None:
                    state = self.clstm_list[i](clstm_in, prev_mask[i], prev_state_spatial[i], None)
                    state1 = self.clstm_list[i](clstm_in, prev_mask_n2[i], prev_state_spatial[i], None)
                    state2 = self.clstm_list[i](clstm_in, prev_maskn[i], prev_state_spatial[i], None)
                else:
                    state = self.clstm_list[i](clstm_in, prev_mask[i], prev_state_spatial[i], prev_hidden_temporal[i]) 
                    state1 = self.clstm_list[i](clstm_in, prev_mask_n2[i], prev_state_spatial[i], prev_hidden_temporal[i]) 
                    state2 =  self.clstm_list[i](clstm_in, prev_maskn[i], prev_state_spatial[i], prev_hidden_temporal[i])

            state[0] = (state[0]+state1[0]+state2[0])/3
            state[1] = (state[1]+state1[1]+state2[1])/3

            hidden_list.append(state)
            hidden = state[0]

            if self.dropout > 0:
                hidden = nn.Dropout2d(self.dropout)(hidden)

            # apply skip connection
            if i < len(skip_feats):

                skip_vec = skip_feats[i]
                upsample = nn.UpsamplingBilinear2d(size = (skip_vec.size()[-2],skip_vec.size()[-1]))
                hidden = upsample(hidden)
                # skip connection
                if self.skip_mode == 'concat':
                    clstm_in = torch.cat([hidden,skip_vec],1)
                elif self.skip_mode == 'sum':
                    clstm_in = hidden + skip_vec
                elif self.skip_mode == 'mul':
                    clstm_in = hidden*skip_vec
                elif self.skip_mode == 'none':
                    clstm_in = hidden
                else:
                    raise Exception('Skip connection mode not supported !')
            else:
                self.upsample = nn.UpsamplingBilinear2d(size = (hidden.size()[-2]*2,hidden.size()[-1]*2))
                hidden = self.upsample(hidden)
                clstm_in = hidden
            del hidden

        out_mask = self.conv_out(clstm_in)
        
        del clstm_in, skip_feats

        return out_mask, hidden_list


