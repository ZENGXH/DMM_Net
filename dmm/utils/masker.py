# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from torch import nn
import logging
from maskrcnn_benchmark.layers.misc import interpolate
from maskrcnn_benchmark.structures.bounding_box import BoxList
from .checker import *

# TODO check if want to return a single BoxList or a composite
# object
class MaskPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, masker=None):
        super(MaskPostProcessor, self).__init__()
        self.masker = masker

    def forward_mask_prop(self, mask_prob, boxes): 
        """
        Arguments:
            x (list[Tensor]): the mask prop, already the score at tge labels dim 
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        if self.masker:
            mask_prob, tight_box = self.masker(mask_prob, boxes)

        results = []
        for prob, box, tbox in zip(mask_prob, boxes, tight_box):
            # bbox = BoxList(box.bbox, box.size, mode="xyxy")
            bbox = BoxList(tbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("mask", prob)
            results.append(bbox)
        return results



    def forward(self, x, boxes):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        mask_prob = x.sigmoid()

        # select masks coresponding to the predicted classes
        num_masks = x.shape[0]
        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_masks, device=labels.device)
        mask_prob = mask_prob[index, labels][:, None]

        boxes_per_image = [len(box) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)

        if self.masker:
            mask_prob, tight_box = self.masker(mask_prob, boxes)

        results = []
        for prob, box, tbox in zip(mask_prob, boxes, tight_box):
            # bbox = BoxList(box.bbox, box.size, mode="xyxy")
            bbox = BoxList(tbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("mask", prob)
            results.append(bbox)
        return results


# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily gor paste_mask_in_image
def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale)[0]
    box = box.to(dtype=torch.int32)

    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = mask.to(torch.float32)
    mask = interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]
    im_mask = mask.new_zeros((im_h, im_w))

    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
    ]
    new_box = binmask_to_box(im_mask > thresh)
    if new_box.sum() == 0:
        print(im_mask.max(), im_mask.min())

    #print(new_box, box, im_mask.shape)
    
    return im_mask, new_box

def binmask_to_box(mask):
    """
    mask: HxW
    """
    inds = (mask > 0).nonzero()
    if inds.shape[0] < 1:
        # print('binmask_to_box get empty mask !!!, give up align')
        return torch.tensor([0,0,mask.shape[0],mask.shape[1]]).to(mask.device) #mask.shape[0],mask.shape[1] # mask.shape[0]-1,0,mask.shape[1]-1
    if len(inds.shape) == 1:
        inds = inds.unsqueeze(0)
    c1_min=inds[:,0].min()
    c1_max=inds[:,0].max()
    c2_min=inds[:,1].min()
    c2_max=inds[:,1].max()
    # assert(inds.shape[0] < mask.shape[0]*mask.shape[1]), np.unique(mask.cpu().numpy())
    return torch.tensor([c2_min.item(),c1_min.item(), c2_max.item(),c1_max.item()]).to(mask.device)


class Masker(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold, padding=1):
        self.threshold = threshold
        self.padding = padding

    def forward_single_image(self, masks, boxes):
        """
        masks: Nbox,1,28,28 
        boxes: BoxList with Nbox.
        """
        boxes = boxes.convert("xyxy")
        im_w, im_h = boxes.size
        res  = []
        resb = []
        for mask, box in zip(masks, boxes.bbox):
            im_mask, new_box = \
                    paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding)
            res.append(im_mask)
            resb.append(new_box) 
        #res = [
        #    paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding)
        #    for mask, box in zip(masks, boxes.bbox)
        #]

        if len(res) > 0:
            res = torch.stack(res, dim=0)[:, None]
            resb = torch.stack(resb, dim=0) # N, 4
        else:
            res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
            resb = boxes.bbox
            # assert(False), boxes 
            logging.warning('get empty box {}'.format(boxes))
        return res, resb

    def __call__(self, masks, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]

        # Make some sanity check
        assert len(boxes) == len(masks), "Masks and boxes should have the same length."

        # TODO:  Is this JIT compatible?
        # If not we should make it compatible.
        results = []
        results_box = []
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box), "Number of objects should be the same."
            result, newbox = self.forward_single_image(mask, box)
            results_box.append(newbox)
            results.append(result)
        return results, results_box


def make_roi_mask_post_processor(padding=1, mask_threshold=0.5):
    # masker project mask to the whole image size
    masker = Masker(padding=1, threshold=mask_threshold)
    # sigmoid and select by label
    mask_post_processor = MaskPostProcessor(masker)
    return mask_post_processor
