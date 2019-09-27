# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# Adapted from DAVIS 2016 (Federico Perazzi)
# ----------------------------------------------------------------------------

""" Compute Jaccard Index. """

import numpy as np
from PIL import Image 

def db_eval_iou(annotation,segmentation):

    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
    Return:
        jaccard (float): region similarity
 """

    annotation   = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if annotation.shape[-1] != segmentation.shape[-1] or annotation.shape[0] != segmentation.shape[0]:
        segmentation = Image.fromarray(segmentation.astype(np.uint8))
        h, w = annotation.shape
        segmentation = segmentation.resize((w, h), Image.NEAREST)
        segmentation = np.array(segmentation)
    if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation),dtype=np.float32)
