r""""Contains definitions of the methods used by the _DataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""
import sys
import numpy as np
import torch
import re
from torch._six import container_abcs, string_classes, int_classes

from dmm.utils.checker import *
import logging
_use_shared_memory = False
r"""Whether to use shared memory in default_collate"""

np_str_obj_array_pattern = re.compile(r'[SaUO]')

error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def eval_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size
    batch: [ (imgs, targets, seq_name, starting_frame), # for batch 1
            (....), 
    ]
    imgs: list of Tensor 
    imgs of different batch may has different length
    """
    #elem_type = type(batch[0])
    transposed = list(zip(*batch))
    imgs = transposed[0] # expect: list of list of Tensor 
    targets = transposed[2]
    #for im in imgs:
    #    logging.info('len {}'.format(len(im)))

    assert(type(imgs) == tuple), 'type: {}, len {} BatchContent {}; shape {}'.format(type(imgs), len(imgs), len(transposed), len(imgs[0]))
    imgs = list(imgs)
    assert(type(imgs[0]) == list), type(imgs[0])
    B = len(batch)
    CHECKEQ(B, len(imgs)) 

    #if not args.pad_video:
    #    return default_collate(batch)
    
    max_len = [len(vid) for vid in imgs]
    max_len = np.array(max_len).max() 
    # create empty frames 
    input_shape = imgs[0][0].shape 
    empty_frame = imgs[0][0].new_zeros(input_shape)
    
    targets = list(targets)
    targets = [[torch.from_numpy(tar_cur_frame) for tar_cur_frame in target_cur_vid] for target_cur_vid in targets]
    empty_target = targets[0][0].new_zeros(targets[0][0].shape)
    for b in range(B):
        while len(imgs[b]) < max_len:
            imgs[b].append(empty_frame)
            targets[b].append(empty_target)
        imgs[b] = torch.stack(imgs[b]) # Len, D, H, W 
        targets[b] = torch.stack(targets[b])
    batch_imgs = torch.stack(imgs) # B, Len, D, H, W
    batch_tar = torch.stack(targets)

    CHECK5D(batch_imgs)

    return [batch_imgs, transposed[1], batch_tar, transposed[3], transposed[4]]

def cocotrain_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size
    batch: [ (imgs, targets, seq_name, starting_frame), # for batch 1
            (....), 
    ]
    imgs: list of Tensor 
    imgs of different batch may has different length
    """
    #elem_type = type(batch[0])
    transposed = list(zip(*batch))
    imgs = transposed[0] # expect: list of list of Tensor 
    targets = transposed[1]
    #for im in imgs:
    #    logging.info('len {}'.format(len(im)))

    assert(type(imgs) == tuple), \
            'type: {}, len {} BatchContent {}; shape {}'.format(type(imgs), len(imgs), len(transposed), len(imgs[0]))
    imgs = list(imgs)
    assert(type(imgs[0]) == list), type(imgs[0])
    B = len(batch)
    CHECKEQ(B, len(imgs)) 

    #if not args.pad_video:
    #    return default_collate(batch)
    # max_len = [len(vid) for vid in imgs]
    # max_len = np.array(max_len).max() 
    # create empty frames 
    # input_shape = imgs[0][0].shape 
    # empty_frame = imgs[0][0].new_zeros(input_shape)
    
    targets = list(targets)
    targets = [[ torch.from_numpy(tar_cur_frame) for tar_cur_frame in target_cur_vid[0]] for target_cur_vid in targets]
    # empty_target = targets[0][0].new_zeros(targets[0][0].shape)
    for b in range(B):
        #while len(imgs[b]) < max_len:
        #    imgs[b].append(empty_frame)
        #    targets[b].append(empty_target)
        imgs[b] = torch.stack(imgs[b]) # Len, D, H, W 
        targets[b] = torch.stack(targets[b])
    batch_imgs = torch.stack(imgs) # B, Len, D, H, W
    batch_tar = torch.stack(targets)

    CHECK5D(batch_imgs)

    return [batch_imgs, batch_tar, transposed[2], transposed[3]]



