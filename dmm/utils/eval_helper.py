import torch
import numpy as np
import os
from .checker import *
from PIL import Image
palette = Image.open('dmm/utils/bear/00000.png').getpalette()
def compute_iou_binary_mask(annotation,segmentation):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
    Return:
        jaccard (float): region similarity
     """
    annotation   = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)
    if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation),dtype=np.float32)
def plot_scores_map(mask, fname):
    """
    outs: OHW. binary  
    scores_map: OHW
    """
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname): 
        os.makedirs(dirname)
    if len(mask.shape) == 3 and mask.shape[0] == 1: 
        mask = mask[0]
    CHECK2D(mask)
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = mask.astype(np.uint8)
    result = Image.fromarray(mask.astype(np.uint8), 'P')
    result.putpalette(palette)
    result.save(fname)
    #fname_check = fname.replace('Lytbcheck', 'Lytb')
    #result_c = np.array(Image.open(fname_check))
    #assert((mask - result_c).sum() == 0)
