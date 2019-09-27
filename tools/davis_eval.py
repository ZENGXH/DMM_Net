import numpy as np
import logging

import sys
sys.path.append('.')
#sys.path.append('dmm')

import os.path as osp
import itertools
import yaml
from prettytable import PrettyTable
import time
from tabulate import tabulate

def davis_toolbox_evaluation(output_dir, eval_split, skip_F=1, title=None):
    """
    Use default DAVIS toolbox to evaluate the performance (J, F)
    Print the result on the table
    Save the result into a yaml file

    Args:
        output_dir: the folder includes all final annotations
    """
    from dmm.dataloader import db_eval, Segmentation  
    from dmm.dataloader.youtubeVOS import YoutubeVOSLoader 
    from dmm.misc.config_youtubeVOS import phase, cfg
    from args import get_parser

    phase = phase.VAL

    parser = get_parser()
    args = parser.parse_args()
    db = YoutubeVOSLoader(args, split=eval_split) 

    print('Loading video segmentations from: {}'.format(output_dir))
    # Load segmentation
    segmentations = [Segmentation('trainval',
        osp.join(output_dir, s), False) for s in db.iternames()]
    # Evaluate results
    if skip_F:
        evaluation = db_eval(db, segmentations, ['J'])
        JF = ['J']
    else:
        evaluation = db_eval(db, segmentations, ['J', 'F'])
        JF = ['J', 'F']
    # Print results
    table = PrettyTable(['Method'] + [p[0] + '_' + p[1] for p in
                                      itertools.product(JF, ['mean', 'recall', 'decay'])])
    table.add_row([osp.basename(output_dir)] + ["%.3f" % np.round(
        evaluation['dataset'][metric][statistic], 3) for metric, statistic
                                                in itertools.product(JF, ['mean', 'recall', 'decay'])])
    print(str(table) + "\n")
    # Save results into yaml file
    with open(osp.join(output_dir, 'davis_eval_results.yaml'), 'w') as f:
        yaml.dump(evaluation, f)
    tim = '{}'.format(time.strftime('%m-%d-%H'))
    print(tim) 

    headers = ['Method'] + [p[0] + '_' + p[1] for p in
                                      itertools.product(JF, ['mean', 'recall', 'decay'])]
    if title is None:
        title = output_dir
    tables = ["{}".format(title)] + ["%.3f" % np.round(
        evaluation['dataset'][metric][statistic], 3) for metric, statistic
                                                in itertools.product(JF, ['mean', 'recall', 'decay'])]
    print(tables)
    info = '\n{}\n'.format(tabulate([tables], headers, tablefmt="github"))
    print(info)
    f = open("EXPERIMENT.md", "a")
    f.write(info)
    f.close()

def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the IoU for the two boxes.
    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou


def calculateTpFpFn(gt_bbox, predict_bbox, IOU_threshold):
    """
        Calculates number of true_pos, false_pos, false_neg from single batch of boxes in one image.
            Returns:
                dict: true positives (int), false positives (int), false negatives (int)
    """
    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(predict_bbox):
        for igb, gt_box in enumerate(gt_bbox):
            iou = calc_iou_individual(pred_box, gt_box)
            if iou > IOU_threshold:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(predict_bbox)
        fn = len(gt_bbox)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(predict_bbox) - len(pred_match_idx)
        fn = len(gt_bbox) - len(gt_match_idx)
    return tp, fp, fn


def calc_precision_recall(true_pos, false_pos, false_neg):

    try:
        precision = true_pos / (true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos / (true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)


def evaluate_bbox(prediction_boxes_list, gt_boxes_list, IOU_threshold = 0.75):

    tp, fp, fn = calculateTpFpFn(gt_boxes_list, prediction_boxes_list, IOU_threshold=IOU_threshold )
    precision, recall = calc_precision_recall(tp, fp, fn)
    return precision, recall 

if __name__ == '__main__':
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    all_test = [args.davis_eval_folder]
    davis_toolbox_evaluation(args.davis_eval_folder, args.eval_split, skip_F=0, title=None)
