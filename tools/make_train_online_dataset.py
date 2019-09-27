import json 
import os, sys 
from PIL import Image 

#datapath = '../../datasets/youtubeVOS/train/'
#targetpath = '../../datasets/youtubeVOS/trainvalot/' 
#datapath_png = datapath + 'Annotations/480p/'
#datapath_jpg = datapath + 'JPEGImages/480p/'
#meta_json = json.load(open('../data/ytb_vos/splits_813_3k_trainvaltest/meta_val200.json', 'r'))['videos']
datapath = '../../datasets/youtubeVOS/val/'
targetpath = '../../datasets/youtubeVOS/train_testdev_ot/' 
datapath_png = datapath + 'Annotations/'
datapath_jpg = datapath + 'JPEGImages/'
meta_json = json.load(open('../../datasets/youtubeVOS/val/meta.json', 'r'))['videos']

if not os.path.exists(datapath):
    raise ValueError('cant found %s'%datapath)

if not os.path.exists(targetpath):
    print('create ', targetpath) 
    os.makedirs(targetpath)
else:
    raise ValueError('target dir exists %s'%targetpath) 

targetpath_png = targetpath + 'Annotations/'
targetpath_jpg = targetpath + 'JPEGImages/'

anno_names_all = {}
meta_json_new = {'videos':{}}
for vids in meta_json.keys():
    objs = meta_json[vids]['objects'] 
    meta_json_new['videos'][vids] = {'objects':{}}
    init_frames = []
    for obj in objs:
        meta_json_new['videos'][vids]['objects'][obj] = {'frames':[]}
        init_frames.append(meta_json[vids]['objects'][obj]['frames'][0]) 
    init_frames = list(set(init_frames))
    anno_names_all[vids] = init_frames

# for vid in os.listdir(datapath_png): 
for vid in meta_json.keys():
    targetpath_png_vid = targetpath_png + vid 
    targetpath_jpg_vid  = targetpath_jpg + vid 
    if not os.path.exists(targetpath_png_vid):
        os.makedirs(targetpath_png_vid)
        os.makedirs(targetpath_jpg_vid)
    vid_path = datapath_png + vid
    anno_names = anno_names_all[vid]
    # anno_imgs = os.listdir(vid_path)[0] 
    # anno_names = [anno.split('.png')[0] for anno in anno_imgs] 
    print('vid ', vid, 'len', len(anno_names))
    for names_new in range(10):
        data_name = anno_names[names_new % len(anno_names)]
        jpg_img = datapath_jpg + vid + '/%s.jpg'%data_name 
        jpg_img = Image.open(jpg_img)
        jpg_img.save(targetpath_jpg_vid + '/%05d.jpg'%names_new)

        png_img = datapath_png + vid + '/%s.png'%data_name 
        png_img = Image.open(png_img)
        png_img.save(targetpath_png_vid + '/%05d.png'%names_new)
        
        for obj in meta_json[vid]['objects']:
            if data_name in meta_json[vid]['objects'][obj]['frames']:
                meta_json_new['videos'][vid]['objects'][obj]['frames'].append('%05d'%names_new)
    print(meta_json_new['videos'][vid])
    print('done', targetpath_png_vid)
json.dump(meta_json_new, open(targetpath+ 'meta.json', 'w')) 
