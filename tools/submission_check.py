"""
check if the number of file for each video is correct before submission 
"""
import argparse
import sys
from PIL import Image

import os
import numpy as np 
import json 

import multiprocessing #import Pool
import functools


def check_seq_results(seq_name, result_dir, json_data, anno_dir, strict):
    frame_names_expect = []
    seq_info = json_data['videos'][seq_name]['objects']
    for obj_id in seq_info.keys():
        for frame_name in seq_info[obj_id]['frames']: 
            if frame_name not in frame_names_expect:
                frame_names_expect.append(frame_name)
    frame_names_expect = list(np.sort(frame_names_expect))
    frame_names_get = np.sort(os.listdir(result_dir + '/%s'%seq_name))
    frame_names_get = [f.split('.')[0] for f in frame_names_get] 
    if len(frame_names_get) != len(frame_names_expect):
        print('length not match for %s: %d and %d, start with %s %s'%(
            seq_name, len(frame_names_get), len(frame_names_expect), 
            frame_names_get[0], frame_names_expect[0]))
        print(frame_names_get)
        print(frame_names_expect)
        if not strict:
            frames_add = [f for f in frame_names_expect if f not in frame_names_get]
            img_shape = np.array(Image.open('%s/%s/%s.png'%(result_dir, seq_name, frame_names_get[0])))
            for f_expect in frames_add:
                target_file = '%s/%s/%s.png'%(result_dir, seq_name, f_expect)
                a = Image.fromarray(np.zeros_like(img_shape))
                a.save(target_file)
                print('save empty %s'%target_file)
                
    else:
        frame_names_get.reverse()
        frame_names_expect.reverse()
        rename_whole_seq = 0
        img_gets = []
        #print('check %s'%seq_name)
        for n, (f_get, f_expect) in enumerate(zip(frame_names_get, frame_names_expect)):
            if f_get != f_expect:
                print('.')
                rename_whole_seq = 1 
                
            src_file = '%s/%s/%s.png'%(result_dir, seq_name, f_get)
            img_gets.append(Image.open(src_file))
        
        if rename_whole_seq:
            for n, (f_get, f_expect) in enumerate(zip(frame_names_get, frame_names_expect)):
                src_file = '%s/%s/%s.png'%(result_dir, seq_name, f_get)
                img_get = img_gets[n]
                target_file = '%s/%s/%s.png'%(result_dir, seq_name, f_expect)
        #gt_file_names = os.listdir(anno_dir+'/' +seq_name)
        #for gt_file_id in gt_file_names:
        #    gt_file_name = '%s/%s/%s'%(anno_dir, seq_name, gt_file_id)
        #    img_get = Image.open(gt_file_name)
        #    target_file = '%s/%s/%s'%(result_dir, seq_name, gt_file_id)
        #    img_get.save(target_file) # save as target file 
        
if __name__ == "__main__":
     
    parser = argparse.ArgumentParser(description='check if submission is valid')
    parser.add_argument('-p', dest='path', help='path to the xxx/merged/')
    parser.add_argument('-s', dest='strict', default=1, type=int, help='not add empty mask') 

    args = parser.parse_args()
    
    json_data = open('../../datasets/youtubeVOS/val/meta.json')
    data = json.load(json_data)
    sequences = data['videos'].keys()
     
    #for seq_name in sequences:
    #    check_seq_results(seq_name, args.path, data) 
    assert(len(data['videos']) == len(os.listdir(args.path))), 'get len {} vs len {}'.format(len(data['videos']), len(os.listdir(args.path)))
    anno_dir = '../../datasets/youtubeVOS/val/Annotations/'
    #for seq_name in sequences:
    #    anno_dir = '../../datasets/youtubeVOS/val/Annotations/%s/'%seq_name
    #    anno = os.listdir(anno_dir)
    #    anno = np.sort(anno)
    #    if len(anno) > 1:
    #        print(anno_dir, anno)
    #        anno = np.array(Image.open(anno_dir+anno[-1]))
    #        print(np.unique(anno))

    p_save = functools.partial(check_seq_results, result_dir=args.path, json_data=data, anno_dir=anno_dir, strict=args.strict)

    pool = multiprocessing.Pool()
    pool.map(p_save, sequences)
    
    pool.close()
    pool.join()

    print('done check %s '%args.path)

    #cmd = 
    #print(cmd)
    #os.system(cmd) 

