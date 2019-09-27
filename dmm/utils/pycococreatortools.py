#!/usr/bin/env python3

import os
import re
import datetime
import numpy as np
import json
from itertools import groupby
from skimage import measure
from PIL import Image
from pycocotools import mask

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info

def create_annotation_info(
                           annotation_id, 
                           image_id, 
                           category_info, 
                           binary_mask, 
                           image_size=None, 
                           tolerance=2, 
                           bounding_box=None
                           ):

    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask, image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    if category_info["is_crowd"]:
        is_crowd = 1
        segmentation = binary_mask_to_rle(binary_mask)
    else :
        is_crowd = 0
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)
        if not segmentation:
            return None

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    } 

    return annotation_info

def create_youtube_json(data_path='../../datasets/youtubeVOS/train/'):
    d_imgset = data_path + '/ImageSets/' 
    imgsets = os.listdir(d_imgset)
    image_id = 0
    annotation_id = 0
    class_id = {}
    categories = [] #{"supercategory": "vehicle", "id": 2, "name": "bicycle"}
    anno_list = []
    image_list = []
    
    for imgset in imgsets:
        if 'json' in imgset:
            continue

        setname = imgset.split('.txt')[0]
        meta = json.load(open(data_path + 'train-%s-meta.json'%setname, 'r'))['videos']

        setfile = open(d_imgset + imgset, 'r')
        setvids = setfile.readlines()
        setvids = [vid.rstrip('\n') for vid in setvids]
        for vid in setvids:
            dirvid = data_path + 'JPEGImages/480p/' + vid 
            frames = np.sort(os.listdir(dirvid)) 
            meta_vid = meta[vid]
            classes_vid = meta_vid['objects']
            classes_vid_ = {}
            classes_names = {} 
            for it in classes_vid.items():
                idx, class_frame = it 
                class_name = class_frame['category']
                class_img = class_frame['frames']
                classes_vid_[idx] = class_img 
                classes_names[idx] = class_name

            for k, c in classes_names.items():
                if c not in class_id.keys():
                    class_id[c] = len(class_id) #assign a classid to class c
                    categories.append({"supercategory": "youtube", "id":class_id[c], "name": c})
            # print(meta_vid)
            for frame in frames:
                file_name = vid + '/' + frame 
                full_path = dirvid + '/' + frame 
                image_size = Image.open(full_path).size 
                img_info = create_image_info(image_id, file_name, image_size)
                image_idx = frame.split('.jpg')[0]
                for k, v in classes_vid_.items(): #.items():
                    # idx = k 
                    # class name: 
                    class_name_to_id = classes_names[k]
                    if image_idx in v:
                        binary_mask = np.array(Image.open(full_path.replace('jpg', 'png').replace('JPEGImages', 'Annotations')))  == int(k)
                        binary_mask = binary_mask.astype(np.uint8)
                        if not int(k) - 1 < len(classes_names):
                            print(k, len(classes_names), vid)
                        anno = create_annotation_info(
                           annotation_id=annotation_id, 
                           image_id=image_id, 
                           category_info={'id': class_id[class_name_to_id], #classes_names[int(k)-1] , 
                                            'is_crowd': 0}, 
                           binary_mask=binary_mask)
                        anno_list.append(anno)
                        annotation_id += 1
                image_list.append(img_info)

                image_id += 1
        json.dump(
                    {'images': image_list, 'annotations': anno_list, 'categories': categories, },
                    open('%s_youtube.json'%setname, 'w') 
                    )
if __name__ == '__main__':
    create_youtube_json()
