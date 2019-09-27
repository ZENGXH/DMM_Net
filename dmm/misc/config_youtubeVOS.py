#!/usr/bin/env python
""" Configuration file."""
import os
import json
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
from enum import Enum

class phase(Enum):
    TRAIN    = 'train'
    VAL      = 'valid' # change to eval if needed
    TRAINVAL = 'trainval'
    TRAINTESTDEVOT  =  'train_testdev_ot'

__C = edict()

# Public access to configuration settings
cfg = __C

# Number of CPU cores used to parallelize evaluation.
__C.N_JOBS = 32

# Paths to dataset folders
__C.PATH = edict()

__C.PHASE = phase.TRAIN

# Multiobject evaluation (Set to False only when evaluating DAVIS 2016)
__C.MULTIOBJECT = True

# Root folder of project
code_path=osp.abspath('../')
__C.PATH.CODE_PATH=code_path
__C.PATH.ROOT = osp.abspath('%s/DMM_Net/'%code_path)

# Data folder
__C.PATH.DATA = osp.abspath('%s/datasets/youtubeVOS/'%__C.PATH.ROOT)

__C.PATH.DAVIS_ROOT  = 'datasets/DAVIS/'

# Path to input images
__C.PATH.SEQUENCES_TRAIN    = osp.join(__C.PATH.DATA,phase.TRAIN.value,  "JPEGImages")
__C.PATH.SEQUENCES_TRAINTESTDEVOT  = osp.join(__C.PATH.DATA, 'train_testdev_ot/JPEGImages') 
__C.PATH.SEQUENCES_TRAINVAL = osp.join(__C.PATH.DATA,phase.TRAIN.value,  "JPEGImages")
__C.PATH.SEQUENCES_VAL      = osp.join(__C.PATH.DATA,phase.VAL.value,    "JPEGImages")

# Path to annotations
__C.PATH.ANNOTATIONS_TRAIN    = osp.join(__C.PATH.DATA,phase.TRAIN.value,  "Annotations")
__C.PATH.ANNOTATIONS_TRAINVAL = osp.join(__C.PATH.DATA,phase.TRAIN.value,  "Annotations")
__C.PATH.ANNOTATIONS_TRAINTESTDEVOT  = osp.join(__C.PATH.DATA, 'train_testdev_ot/Annotations') 
__C.PATH.ANNOTATIONS_VAL      = osp.join(__C.PATH.DATA,phase.VAL.value,    "Annotations")

# Color palette
__C.PATH.PALETTE = osp.abspath(osp.join(__C.PATH.ROOT, 'dmm/dataloader/palette.txt'))

# Paths to files
__C.FILES = edict()

# Path to property file, holding information on evaluation sequences.
__C.FILES.DB_INFO_TRAIN    = 'data/ytb_vos/splits_813_3k_trainvaltest/meta_train3k.json'
__C.FILES.DB_INFO_TRAINVAL = 'data/ytb_vos/splits_813_3k_trainvaltest/meta_val200.json'
__C.FILES.DB_INFO_VAL      = osp.abspath(osp.join(__C.PATH.DATA,phase.VAL.value,   "meta.json"))
__C.FILES.DB_INFO_TRAINOT  = 'datasets/youtubeVOS/trainvalot/meta.json'
__C.FILES.DB_INFO_TRAINTESTDEVOT  = 'datasets/youtubeVOS/train_testdev_ot/meta.json'

# Measures and Statistics
__C.EVAL = edict()

# Metrics: J: region similarity, F: contour accuracy, T: temporal stability
__C.EVAL.METRICS = ['J','F']

# Statistics computed for each of the metrics listed above
__C.EVAL.STATISTICS= ['mean','recall','decay']
def db_read_sequences_train_testdev_ot():
  """ Read list of sequences. """
  if not os.path.exists(__C.FILES.DB_INFO_TRAINTESTDEVOT):
      return []
  json_data = open(__C.FILES.DB_INFO_TRAINTESTDEVOT)
  data = json.load(json_data)
  sequences = data['videos'].keys()
  return sequences

def db_read_sequences_train():
  """ Read list of sequences. """
  json_data = open(__C.FILES.DB_INFO_TRAIN)
  data = json.load(json_data)
  sequences = data['videos'].keys()
  return sequences
  
def db_read_sequences_val():
  """ Read list of sequences. """
  json_data = open(__C.FILES.DB_INFO_VAL)
  data = json.load(json_data)
  sequences = data['videos'].keys()
  return sequences
 
def db_read_sequences_trainval():
  """ Read list of sequences. """
  json_data = open(__C.FILES.DB_INFO_TRAINVAL)
  data = json.load(json_data)
  sequences = data['videos'].keys()
  return sequences

__C.SEQUENCES_TRAIN          = db_read_sequences_train()
__C.SEQUENCES_TRAINTESTDEVOT = db_read_sequences_train_testdev_ot()
__C.SEQUENCES_VAL            = db_read_sequences_val()
__C.SEQUENCES_TRAINVAL       = db_read_sequences_trainval()
def get_img_path(split):
    if split == phase.TRAIN.value:
        seq = cfg.PATH.SEQUENCES_TRAIN 
    elif split == phase.VAL.value:
        seq = cfg.PATH.SEQUENCES_VAL
    elif split == phase.TRAINVAL.value:
        seq = cfg.PATH.SEQUENCES_TRAINVAL
    elif split == phase.TRAINTESTDEVOT.value:
        seq =  cfg.PATH.SEQUENCES_TRAINTESTDEVOT
    else:
        raise ValueError('not support %s'%split)
    return seq 


def get_anno_path(split):
    if split == phase.TRAIN.value:
        seq = cfg.PATH.ANNOTATIONS_TRAIN 
    elif split == phase.VAL.value:
        seq = cfg.PATH.ANNOTATIONS_VAL
    elif split == phase.TRAINVAL.value:
        seq = cfg.PATH.ANNOTATIONS_TRAINVAL
    elif split == phase.TRAINTESTDEVOT.value:
        seq =  cfg.PATH.ANNOTATIONS_TRAINTESTDEVOT
    else:
        raise ValueError('not support %s'%split)
    return seq 

def get_db_path(split):
    if split == phase.TRAIN.value:
        seq = cfg.FILES.DB_INFO_TRAIN 
    elif split == phase.VAL.value:
        seq = cfg.FILES.DB_INFO_VAL
    elif split == phase.TRAINVAL.value:
        seq = cfg.FILES.DB_INFO_TRAINVAL
    elif split == phase.TRAINTESTDEVOT.value:
        seq =  cfg.FILES.DB_INFO_TRAINTESTDEVOT
    else:
        raise ValueError('not support %s'%split)
    return seq 
__C.palette = np.loadtxt(__C.PATH.PALETTE,dtype=np.uint8).reshape(-1,3)
