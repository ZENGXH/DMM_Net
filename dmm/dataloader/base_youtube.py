import functools
import os.path as osp

import numpy as np

from PIL import Image
from skimage.io import ImageCollection

from dmm.misc.config_youtubeVOS import cfg, phase, get_anno_path
from dmm.misc.io_aux import imread_indexed,imwrite_indexed
import logging

#################################
# HELPER FUNCTIONS
#################################

def _load_annotation(filename,single_object):
  """ Load image given filename."""

  annotation,_ = imread_indexed(filename)

  if single_object:
    annotation = (annotation != 0).astype(np.uint8)

  return annotation

def _get_num_objects(annotation):
  """ Count number of objects from segmentation mask"""

  ids = sorted(np.unique(annotation))

  # Remove unknown-label
  ids = ids[:-1] if ids[-1] == 255 else ids

  # Handle no-background case
  ids = ids if ids[0] else ids[1:]

  return len(ids)

#################################
# LOADER CLASSES
#################################
def get_seq_path(split):
    if split == phase.TRAIN.value:
        seq = cfg.PATH.SEQUENCES_TRAIN 
    elif split == phase.VAL.value:
        seq = cfg.PATH.SEQUENCES_VAL
    elif split == phase.TRAINVAL.value:
        seq = cfg.PATH.SEQUENCES_TRAINVAL
    #elif split == phase.TRAINOT.value:
    #    seq =  cfg.PATH.SEQUENCES_TRAINOT
    #elif split == phase.TEST.value:
    #    seq =  cfg.PATH.SEQUENCES_TEST
    elif split == phase.TRAINTESTDEVOT.value:
        seq =  cfg.PATH.SEQUENCES_TRAINTESTDEVOT
    #elif split == phase.DAVISTRAIN.value:
    #    seq = cfg.PATH.SEQUENCES_TRAIN_DAVIS
    #elif split == phase.DAVISVAL.value:
    #    seq = cfg.PATH.SEQUENCES_VAL_DAVIS
    else:
        raise ValueError('not support %s'%split)
    return seq 


def get_seq(split):
    if split == phase.TRAIN.value:
        seq = cfg.SEQUENCES_TRAIN 
    elif split == phase.VAL.value:
        seq = cfg.SEQUENCES_VAL
    elif split == phase.TRAINVAL.value:
        seq = cfg.SEQUENCES_TRAINVAL
    #elif split == phase.TRAINOT.value:
    #    seq =  cfg.SEQUENCES_TRAINOT
    #elif split == phase.TEST.value:
    #    seq =  cfg.SEQUENCES_TEST
    elif split == phase.TRAINTESTDEVOT.value:
        seq =  cfg.SEQUENCES_TRAINTESTDEVOT
    #elif split == phase.DAVISTRAIN.value:
    #    seq = cfg.SEQUENCES_TRAIN_DAVIS 
    #elif split == phase.DAVISVAL.value:
    #    seq = cfg.SEQUENCES_VAL_DAVIS
    else:
        raise ValueError('not support %s'%split)
    return seq 

class BaseLoader(ImageCollection):
  """
  Base class to load image sets (inherit from skimage.ImageCollection).
  count and collect all the image file under path; and assign attribute 'split' to the object 
  Arguments:
    path      (string): path to sequence folder.
    regex     (string): regular expression to define image search pattern.
    load_func (func)  : function to load image from disk (see skimage.ImageCollection).
    split     (string): attribute of current Sequence 
  """
  def __init__(self, split, path, regex, load_func=None, lmdb_env=None, files=None):
    
    if files is not None:
        super(BaseLoader, self).__init__(files, load_func=load_func)
    elif not lmdb_env == None:
        key_db = osp.basename(path)
        with lmdb_env.begin() as txn:
            _files_vec = txn.get(key_db.encode()).decode().split('|')
            _files = [bytes(osp.join(path, f).encode()) for f in _files_vec]
        super(BaseLoader, self).__init__(_files, load_func=load_func)
    else:  
        assert(osp.exists(path)), 'not exist %s'%path
        super(BaseLoader, self).__init__(
            osp.join(path + '/' + regex),load_func=load_func)
    # Sequence name
    self.name = osp.basename(path)
    self.split = split
    if not self.name in get_seq(split):
        raise Exception("Sequence name \'{}\' not found.".format(self.name))
  def __str__(self):
    return "< class: '{}' name: '{}', frames: {} >".format(
        type(self).__name__,self.name,len(self))

class Sequence(BaseLoader):
  """
  Load image sequences.
  Arguments:
    name  (string): sequence name.
    regex (string): regular expression to define image search pattern.

  """
  def __init__(self,split,name,regex="*.jpg", lmdb_env=None, files=None):
    super(Sequence, self).__init__(split, 
                                    osp.join(get_seq_path(split), name), 
                                    regex, 
                                    lmdb_env=lmdb_env,
                                    files=files)

class SequenceClip_simple:
    """
    Load image sequences.

    Arguments:
      name  (string): sequence name.
      regex (string): regular expression to define image search pattern.

    """

    def __init__(self, seq, starting_frame):
        if seq.split == phase.TRAIN.value:
            self.__dict__.update(seq.__dict__)
        else:
            self.__dict__.update(seq.__dict__)
        self.starting_frame = starting_frame

    def __str__(self):
        return "< class: '{}' name: '{}', startingframe: {}, frames: {} >".format(
            type(self).__name__, self.name, self.starting_frame, len(self))

class SequenceClip(BaseLoader):

  """
  Load image sequences.

  Arguments:
    name  (string): sequence name.
    regex (string): regular expression to define image search pattern.

  """

  def __init__(self,split,name,starting_frame,regex="*.jpg", lmdb_env=None):
    super(SequenceClip, self).__init__(split, 
                                        osp.join(get_seq_path(split), name), 
                                        regex, 
                                        lmdb_env=lmdb_env)
    self.starting_frame = starting_frame
    
  def __str__(self):
    return "< class: '{}' name: '{}', startingframe: {}, frames: {} >".format(
        type(self).__name__,self.name,self.starting_frame,len(self))

class Segmentation(BaseLoader):

  """
  Load image sequences.

  Arguments:
    path          (string): path to sequence folder.
    single_object (bool):   assign same id=1 to each object.
    regex         (string): regular expression to define image search pattern.

  """

  def __init__(self, split, path, single_object, regex="*.png", lmdb_env=None):
    super(Segmentation, self).__init__(split, 
            path, 
            regex,
            functools.partial(_load_annotation, 
                                single_object=single_object
                                ), 
            lmdb_env=lmdb_env)
    self.n_objects = _get_num_objects(self[0])

  def iter_objects_id(self):
    """
    Iterate over objects providing object id for each of them.
    """
    for obj_id in range(1,self.n_objects+1):
      yield obj_id

  def iter_objects(self):
    """
    Iterate over objects providing binary masks for each of them.
    """

    for obj_id in self.iter_objects_id():
      bn_segmentation = [(s==obj_id).astype(np.uint8) for s in self]
      yield bn_segmentation

class Annotation(Segmentation):

  """
  Load ground-truth annotations.

  Arguments:
    name          (string): sequence name.
    single_object (bool):   assign same id=1 to each object.
    regex         (string): regular expression to define image search pattern.

  """

  def __init__(self,split,name,single_object,regex="*.png", lmdb_env=None):
    super(Annotation, self).__init__(split,osp.join(get_anno_path(split), name),single_object,regex, lmdb_env=lmdb_env)

class AnnotationClip_simple:
    """
    Load ground-truth annotations.

    Arguments:
      name          (string): sequence name.
      single_object (bool):   assign same id=1 to each object.
      regex         (string): regular expression to define image search pattern.
    # xh: AnnotationClip_simple is define by xiaohui, since I want to skip the AnnotationClip init every time I run experiment 
    # which cost a lot of time !!!
    """
    
    def __init__(self, annot, starting_frame):
        self.__dict__.update(annot.__dict__)
        self.starting_frame = starting_frame

class AnnotationClip(Segmentation):
  """
  Load ground-truth annotations.

  Arguments:
    name          (string): sequence name.
    single_object (bool):   assign same id=1 to each object.
    regex         (string): regular expression to define image search pattern.

  """

  def __init__(self,split,name,starting_frame,single_object,regex="*.png", lmdb_env=None):
    super(AnnotationClip, self).__init__(split,osp.join(get_anno_path(split), name), single_object, regex, lmdb_env=lmdb_env)
    self.starting_frame = starting_frame
    
