xport NGPUS=1
srun -p max12hours --gres=gpu:$NGPUS --mem=26G -c 8 -x ~/.exclude \
python \
    -m torch.distributed.launch --nproc_per_node=$NGPUS \
    ../maskrcnn-benchmark/tools/test_net.py \
    --config-file "../maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml" \
    MODEL.ROI_BOX_HEAD.NUM_CLASSES 81 \
    TEST.IMS_PER_BATCH 28 DATALOADER.ASPECT_RATIO_GROUPING False  \
    OUTPUT_DIR './experiments/proposals/coco81/' \
    MODEL.ROI_HEADS.SCORE_THRESH 0.0 \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 50 \
    DATASETS.TEST "'youtubevos_val200_meta', "

# srun -p t4 --gres=gpu:$NGPUS --mem=45G -c 8 -x ~/.exclude \
python \
    ../maskrcnn-benchmark/tools/test_net.py \
    --config-file "../maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml" \
    MODEL.ROI_BOX_HEAD.NUM_CLASSES 81 \
    TEST.IMS_PER_BATCH 4 DATALOADER.ASPECT_RATIO_GROUPING False  \
    OUTPUT_DIR '../experiments/proposals/coco81/' \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 50 \
    DATASETS.TEST "'youtubevos_train3k_meta', "
    # MODEL.ROI_HEADS.SCORE_THRESH 0.01 \
    # -m torch.distributed.launch --nproc_per_node=$NGPUS \

# srun -p p100  --gres=gpu:$NGPUS --mem=26G -c 8 -x ~/.exclude \
python \
    ../maskrcnn-benchmark/tools/test_net.py \
    --config-file "../maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml" \
    MODEL.ROI_BOX_HEAD.NUM_CLASSES 81 \
    TEST.IMS_PER_BATCH 8 DATALOADER.ASPECT_RATIO_GROUPING False  \
    OUTPUT_DIR './experiments/proposals/coco81/' \
    MODEL.ROI_HEADS.SCORE_THRESH 0.0 \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 50 \
    DATASETS.TEST "'youtubevos_testdev_online_meta', "
