r""" load the extracted prediction and convert to dict format, 
with key as vid and imageid 
usage python ./tools/reduce_pth_size_by_videos.py PATH/TO/PREDICTION.pth 'trainval'
example: 
     srun -p gpuc --mem=18G python tools/reduce_pth_size_by_videos.py  ../../experiments/maskrcnn-models/ytb_pred_train/inference/youtubevos_train3k_meta/predictions.pth train 50 
     srun -p gpuc --mem=18G python tools/reduce_pth_size_by_videos.py  ../../experiments/maskrcnn-models/ytb_pred_train_thre00_d50/inference/youtubevos_val200_meta/predictions.pth trainval 50
"""
import torch 
import logging
import json
import os
import numpy as np 
import sys 
import time 
from tqdm import tqdm 
import multiprocessing 
import functools
sys.path.append('.')
import dmm.misc.config_youtubeVOS as cfg_ytb 
from dmm.misc.config_youtubeVOS import get_img_path, get_db_path

saved_name = sys.argv[1]
split = sys.argv[2] 
top = int(sys.argv[3]) 

print('get: %s; split: %s'%(saved_name, split))
assert(split in ['trainval', 'val', 'train', 'train_testdev_ot', 'davis_train', 'davis_val']), 'not support {}'.format(split) 

meta = json.load(open(get_db_path(split)))['videos']
data_root = get_img_path(split)

basename = os.path.basename(saved_name)
target_dir = os.path.dirname(saved_name) + '/asdict_%d/'%top
print('save at ', target_dir)

if not os.path.isdir(target_dir):
    os.makedirs(target_dir)
else:
    raise ValueError('target_dir exists: may lead to overwritting. are you sure? - %s'%target_dir)

count = 0
counts_2_vidframe = []
vid_frame_2_predid = dict()
for k in meta.keys():
    vid_dir = data_root + '/' + k
    vid_frame_2_predid[k] = dict()
    frames = np.sort(os.listdir(vid_dir))
    for f in frames:
        vid_frame_2_predid[k][f.split('.')[0]] = count 
        count += 1
        counts_2_vidframe.append(tuple([k, f.split('.')[0]]))
print('- total frame {}'.format(count)) 

all_videos = list(meta.keys())
tic = time.time()
f = torch.load(saved_name)
print('- load %s takes %.2f'%(saved_name, time.time()-tic))


### save for training 

if len(all_videos) > 200: 
    # if the number of videos is too large, 
    # we save one file per video 
    def save_videos(newf, target_dir):
        vid = newf['vid']
        newf = newf['pred']
        torch.save(newf, target_dir + '%s.pth'%vid)

    newf = []

    vid2index = {}
    countobj = {}
    videos = all_videos
    print('- num of videos  %d'%len(videos))

    for k, predictions in enumerate(tqdm(f)):
        vid, fid = counts_2_vidframe[k] 
        if vid not in videos:
            continue
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        if len(idx) > top:
            idx = idx[:top]
        subp = predictions[idx]
        if vid not in vid2index: 
            vid2index[vid] = len(vid2index)
            newf.append({'vid':vid, 'pred':{}})
            countobj[vid] = {}
        newf[vid2index[vid]]['pred'][fid] = subp 
        countobj[vid][fid] = len(subp)

    newf1 = newf[:1500]
    newf2 = newf[1500:]

    vids = vid2index.keys() 

    target_dir = target_dir + '/videos/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    json.dump(countobj, open('%s/countobj.json'%target_dir, 'w')) 
    p_eval_vid = functools.partial(save_videos, target_dir=target_dir) #, newf=newf) 
    for n in tqdm(newf):
        p_eval_vid(n)
else:
    newf = {} 
    vid2index = {}
    countobj = {}
    videos = all_videos
    print('- num of videos  %d: N predictions: %d'%(len(videos), len(f)))
    for k, predictions in enumerate(tqdm(f)):
        vid, fid = counts_2_vidframe[k] 
        if vid not in videos:
            continue
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        if len(idx) > top:
            idx = idx[:top]
        subp = predictions[idx]
        if vid not in vid2index: 
            vid2index[vid] = len(vid2index)
            countobj[vid] = {}
            newf[vid] = {}
        newf[vid][fid] = subp
        countobj[vid][fid] = len(subp)
    torch.save(newf, target_dir + 'pred_DICT.pth')
