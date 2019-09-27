- scripts used to extract prediction of testdev split, online-trained proposal network:
```
    python \
    -m torch.distributed.launch --nproc_per_node=$NGPUS \
    ../maskrcnn-benchmark/tools/test_net.py \
    --config-file "../maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml" \
    MODEL.ROI_BOX_HEAD.NUM_CLASSES 2 \
    TEST.IMS_PER_BATCH 28 DATALOADER.ASPECT_RATIO_GROUPING False  \
    OUTPUT_DIR './experiments/proposals/ytb_ot/' \
    MODEL.WEIGHT './experiments/propnet/online_ytb/model_0225000.pth' \
    MODEL.ROI_HEADS.SCORE_THRESH 0.0 \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 50 \
    DATASETS.TEST "'youtubevos_testdev_meta', "    
```
