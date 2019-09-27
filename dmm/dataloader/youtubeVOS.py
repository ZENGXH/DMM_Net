from collections import namedtuple

from torchvision.transforms import transforms as T
import numpy as np
import pickle
from PIL import Image
from .base_youtube import Sequence, SequenceClip, Annotation, AnnotationClip, BaseLoader, Segmentation, SequenceClip_simple, AnnotationClip_simple
from dmm.misc.config_youtubeVOS import cfg, phase 
from dmm.misc.config_youtubeVOS import get_db_path, get_anno_path, get_img_path 
# db_read_sequences_train,db_read_sequences_val, db_read_sequences_test, db_read_sequences_trainval, db_read_sequences_trainot, db_read_sequences_train_testdev_ot

from .transforms.transforms import RandomAffine
import os.path as osp
import os
import glob
import lmdb
import logging
import time
import json 

import functools
#import utils.data_helper as data_helper
from easydict import EasyDict as edict

from .dataset import MyDataset

class YoutubeVOSLoader(MyDataset):
  """
  Helper class for accessing the DAVIS dataset.

  Arguments:
    phase         (string): dataset set eg. train, val. (See config.phase)
    single_object (bool):   assign same id (==1) to each object.

  Members:
    sequences (list): list of 'Sequence' objects containing RGB frames.
    annotations(list): list of 'Annotation' objects containing ground-truth segmentations.
  """
  def __init__(self,
                 args,
                 transform=None,
                 target_transform=None,
                 augment=False,
                 split='train',
                 resize=False,
                 inputRes=None,
                 video_mode=True,
                 use_prev_mask=False):
    super().__init__(args, split=split, inputRes=inputRes)

    self.isrank0 = args.local_rank == 0
    self._phase = split
    self._single_object = args.single_object
    self._length_clip = args.length_clip
    self.transform = transform
    self.target_transform = target_transform
    self.split = split
    self.inputRes = inputRes
    self.video_mode = video_mode
    self.max_seq_len = args.gt_maxseqlen
    self.dataset = args.dataset
    self.flip = augment
    self.use_prev_mask = use_prev_mask
    #if args.train_coco:
    #    self.sequence_from_masks = functools.partial(data_helper.sequence_cocotarget_from_masks, dataset=self.dataset, split=self.split)
    if augment:
        # self.my_augment = args.my_augment
        self.augmentation_transform = RandomAffine(rotation_range=args.rotation, translation_range=args.translation,
                                                        shear_range=args.shear, zoom_range=(args.zoom,max(args.zoom*2,1.0)),
                                                        interp = 'nearest', lazy = True)
    else:
        self.augmentation_transform = None
    # overwrite if read train set during evaluation
    self._phase_read = self._phase 
    start = time.time()
    # _db_sequences: list of names of video 
    self._meta_data = json.load(open(get_db_path(self._phase_read), 'r')) 
    self._db_sequences = self._meta_data['videos'].keys()

    if 'DEBUG' in args.model_name: 
        self._db_sequences = list(self._db_sequences)
        self._db_sequences = self._db_sequences[:5]

    if self.isrank0: logging.info('[dataset] phase read {}; len of db seq {}'.format(self._phase_read, len(self._db_sequences)))
    # Check lmdb existance. If not proceed with standard dataloader.
    
    lmdb_env_seq_dir   = osp.join(cfg.PATH.DATA, 'lmdb_seq')
    lmdb_env_annot_dir = osp.join(cfg.PATH.DATA, 'lmdb_annot')
    pickle_cache_seq   = osp.join(os.path.dirname(cfg.FILES.DB_INFO_TRAIN), 'dmm_cached_%s.pkl'%self._phase_read)
    
    if osp.isdir(lmdb_env_seq_dir) and osp.isdir(lmdb_env_annot_dir) and (self._phase_read == phase.TRAIN.value or self._phase_read == phase.TRAINVAL.value):
        # only works for 'train' or 'trainval' phase 
        lmdb_env_seq = lmdb.open(lmdb_env_seq_dir,
                                    max_readers=4, readonly=True, lock=False,
                                    readahead=False, meminit=False)
        lmdb_env_annot = lmdb.open(lmdb_env_annot_dir,
                                    max_readers=4, readonly=True, lock=False,
                                    readahead=False, meminit=False)
        if self.isrank0: logging.info('load LMDB from %s and %s'%(lmdb_env_annot_dir, lmdb_env_seq_dir))
    else:
        lmdb_env_seq = None
        lmdb_env_annot = None
        if self.isrank0: logging.info('LMDB not found. This could affect the data loading time. It is recommended to use LMDB.')
    rank = args.local_rank
    if pickle_cache_seq is not None and os.path.isfile(pickle_cache_seq) and args.cache_data:
        # found existing pickle file, load it 
        if rank == 0: logging.info('try to load in {}'.format(pickle_cache_seq))
        with open(pickle_cache_seq, 'rb') as pkl_fid:
            cache = pickle.load(pkl_fid)
        self.sequences = cache['sequences']
        self.annotations = cache['annotations']
        del cache
    else:
        # even if lmdb_env_seq is None, we can still cache pkl
        if rank == 0: logging.info('no cache data found at %s; it will take a while to cache the data '%pickle_cache_seq)
        self.sequences = [Sequence(self._phase_read,s,lmdb_env=lmdb_env_seq) for s in self._db_sequences]

        # Load annotations
        self.annotations = [Annotation(self._phase_read,s,self._single_object,lmdb_env=lmdb_env_annot) for s in self._db_sequences]
        if args.local_rank == 0 and args.cache_data:
            if rank == 0: logging.info('try to dump in {}'.format(pickle_cache_seq))
            with open(pickle_cache_seq, 'wb') as pkl_fid:
                # we may be able to sort the length of sequences? 
                pickle.dump({'sequences': self.sequences, 'annotations': self.annotations}, pkl_fid)
    if rank == 0: logging.info('load lmdb %.2f'%(time.time() - start))
    # Load sequences
    self.sequence_clips = []
    start = time.time()
    # fill self.annotation_clips and self.sequence_clips 
    # 1. skip those empty template 

    # Load sequences
    self.annotation_clips = []
    skip_number = 0
    if not self.use_prev_mask: 
        """ training """
        #if args.load_proposals and 'train' in self._phase and args.load_proposals_countobj is not None: 
        #    # filter proposals that are empty: by reading countobj.json  
        #    load_proposals_countobj = json.load(open(args.load_proposals_countobj, 'r')) 
        #    filter_out = 0
        #    for seqid, (seq, dbname) in enumerate(zip(self.sequences, self._db_sequences)):
        #        images = seq.files 
        #        images = [osp.splitext(osp.basename(img))[0].decode("utf-8") for img in images]
        #        images_valid = [fname for img, fname in zip(images, seq.files) if load_proposals_countobj[dbname][img] > 0 ] 
        #        if len(images_valid) < len(images):
        #            filter_out += len(images) - len(images_valid)
        #            self.sequences[seqid] = Sequence(self._phase_read,dbname,files=images_valid)
        #    if rank == 0: logging.info('filtered images out -> {} for #vid {}'.format(filter_out, len(self.sequences)))
        if args.load_proposals and args.load_proposals_dataset and 'train' in self._phase:
            filter_out = 0
            for seqid, (seq, dbname) in enumerate(zip(self.sequences, self._db_sequences)):
                images = seq.files 
                images = [osp.splitext(osp.basename(img))[0] for img in images]
                if not type(images[0]) == str:
                    images = [img.decode("utf-8") for img in images]

                images_valid = [fname for img, fname in zip(images, seq.files) if self.countobj[dbname][img] > 0 ] 
                if len(images_valid) < len(images):
                    filter_out += len(images) - len(images_valid)
                    self.sequences[seqid] = Sequence(self._phase_read,dbname,files=images_valid)
            if rank == 0: logging.info('filtered images out -> {} for #vid {}'.format(filter_out, len(self.sequences)))

        for seq, annot, s in zip(self.sequences, self.annotations, self._db_sequences):
            images = seq.files
            starting_frame_idx = 0
            starting_frame = int(osp.splitext(osp.basename(images[starting_frame_idx]))[0])
            if args.skip_empty_starting_frame:
                # first compute how many instance for each frame: 
                seq_info = self._meta_data['videos'][s]['objects']
                frame_names_has_obj = []
                for obj_id in seq_info.keys(): # loop over all objects 
                    for frame_name in seq_info[obj_id]['frames']: # loop over all frames containing current objects 
                        if frame_name not in frame_names_has_obj: # add if this a new frames 
                            frame_names_has_obj.append(int(frame_name))
            if args.skip_empty_starting_frame and starting_frame not in frame_names_has_obj:
                skip_number += 1
            else:
                self.sequence_clips.append(SequenceClip_simple(seq, starting_frame))
                self.annotation_clips.append(AnnotationClip_simple(annot, starting_frame))
            # num_frames = self.sequence_clips[-1]._numframes
            num_frames = seq._numframes
            num_clips = int(num_frames / self._length_clip)
            for idx in range(num_clips - 1):
                starting_frame_idx += self._length_clip
                # starting_frame is consider based on the first frame in 'jpg' folder
                starting_frame = int(osp.splitext(osp.basename(images[starting_frame_idx]))[0])
                if args.skip_empty_starting_frame and starting_frame not in frame_names_has_obj: 
                    # here we need to check how many objects at the annotation of the starting_frame 
                    # and annot.n_objects == 0:
                    skip_number += 1
                    continue 
                    # assert(annot.n_objects > 0)
                # skip if the starting_frame is empty 
                self.sequence_clips.append(SequenceClip_simple(seq, starting_frame))
                self.annotation_clips.append(AnnotationClip_simple(annot, starting_frame))
        if self._phase_read == phase.TRAINVAL.value: 
            if args.max_eval_iter > 0:
                self.sequence_clips = self.sequence_clips[:args.max_eval_iter] 
                self.annotation_clips = self.annotation_clips[:args.max_eval_iter] 
    else:
        """ evaluation 
        start frame is the first annotations frame
        """
        for seq, s in zip(self.sequences, self._db_sequences):
            # use_prev_mask only during evaluation 
            #if s not in ['e90c10fc4c', 'e98eda8978']: 
            #    continue 
            annot_seq_dir = osp.join(get_anno_path(self._phase), s)
            annotations = glob.glob(osp.join(annot_seq_dir,'*.png'))
            annotations.sort()
            # We only consider the first frame annotated to start the inference mode with such a frame
            if self._phase == 'trainval':
                seq_info = self._meta_data['videos'][s]['objects']
                frame_names_has_obj = []
                for obj_id in seq_info.keys(): # loop over all objects 
                    for frame_name in seq_info[obj_id]['frames']: 
                        # loop over all frames containing current objects 
                        if frame_name not in frame_names_has_obj: # add if this a new frames 
                            frame_names_has_obj.append(int(frame_name))
                images_idx_list = [int(osp.splitext(osp.basename(annotations[k]))[0]) for k in range(len(annotations))]
                starting_frame = frame_names_has_obj[0]
            else:
                starting_frame = int(osp.splitext(osp.basename(annotations[0]))[0])
            self.sequence_clips.append(SequenceClip(self._phase_read, s, starting_frame, lmdb_env=lmdb_env_seq))

        for annot, s in zip(self.annotations, self._db_sequences):
            images = annot.files
            #if s not in ['e90c10fc4c', 'e98eda8978']: 
            #    continue 
            if self._phase == 'trainval':
                seq_info = self._meta_data['videos'][s]['objects']
                frame_names_has_obj = []
                for obj_id in seq_info.keys(): # loop over all objects 
                    for frame_name in seq_info[obj_id]['frames']: 
                        # loop over all frames containing current objects 
                        if frame_name not in frame_names_has_obj: # add if this a new frames 
                            frame_names_has_obj.append(int(frame_name))
                images_idx_list = [int(osp.splitext(osp.basename(images[k]))[0]) for k in range(len(images))]
                starting_frame = frame_names_has_obj[0]
                starting_frame_idx = images_idx_list.index(starting_frame)
            else:
                starting_frame_idx = 0
                starting_frame = int(osp.splitext(osp.basename(images[starting_frame_idx]))[0])
            self.annotation_clips.append(AnnotationClip_simple(annot, starting_frame))
            num_frames = self.annotation_clips[-1]._numframes
            num_clips = int(num_frames / self._length_clip)
            for idx in range(num_clips - 1):
                starting_frame_idx += self._length_clip
                starting_frame = int(osp.splitext(osp.basename(images[starting_frame_idx]))[0])
                self.annotation_clips.append(AnnotationClip_simple(annot, starting_frame))
    if self.isrank0: logging.info('[init][data][youtube][load clips] load anno %.2f; cliplen %d| annotation clip %d(skip %d)| videos %d'%(time.time() - start, 
                self._length_clip,
                len(self.annotation_clips), skip_number, len(self._db_sequences)))
    start = time.time()
    self._keys = dict(zip([s for s in self.sequences],
      range(len(self.sequences))))
      
    self._keys_clips = dict(zip([s.name+str(s.starting_frame) for s in self.sequence_clips],
      range(len(self.sequence_clips))))
    try:
      self.color_palette = np.array(Image.open(
        self.annotations[0].files[0]).getpalette()).reshape(-1,3)
    except Exception as e:
      self.color_palette = np.array([[0,255,0]])
    if self.isrank0: logging.info('load keys %.2f'%(time.time() - start))
    start = time.time()

  def get_raw_sample(self, key):
    """ Get sequences and annotations pairs."""
    if isinstance(key,str):
      sid = self._keys[key]
    elif isinstance(key,int):
      sid = key
    else:
      raise InputError()

    return edict({
      'images'  : self.sequences[sid],
      'annotations': self.annotations[sid]
      })
      
  def get_raw_sample_clip(self, key):
    """ Get sequences and annotations pairs."""
    if isinstance(key,str):
      sid = self._keys_clips[key]
    elif isinstance(key,int):
      sid = key
    else:
      raise InputError()

    return edict({
      'images'  : self.sequence_clips[sid],
      'annotations': self.annotation_clips[sid]
      })

  def sequence_name_to_id(self,name):
    """ Map sequence name to index."""
    return self._keys[name]
    
  def sequence_name_to_id_clip(self,name):
    """ Map sequence name to index."""
    return self._keys_clips[name]

  def sequence_id_to_name(self,sid):
    """ Map index to sequence name."""
    return self._db_sequences[sid]
    
  def sequence_id_to_name_clip(self,sid):
    """ Map index to sequence name."""
    return self.sequence_clips[sid]

  def iternames(self):
    """ Iterator over sequence names."""
    for s in self._db_sequences:
      yield s

  def iternames_clips(self):
    """ Iterator over sequence names."""
    for s in self.sequence_clips:
      yield s

  def iteritems(self):
    return self.__iter__()
