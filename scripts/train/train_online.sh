BS=4
NGPU=4
train_h=255 
train_w=448
len_clip=3
Nworkers=5

model_root='./experiments/models/' # path to saved the model 
MODEL_NAME=online_ytb 
    # --resume -resume_path './experiments/dmmnet/ytb_255_b44l3_101/epo01_iter01640/' -epoch_resume 2 \

# srun -p interactive --gres=gpu:$NGPU --mem=49G -c 8 -x ~/.exclude \
    python \
    -m torch.distributed.launch --nproc_per_node=$NGPU \
    train.py \
    --resume -resume_path '/scratch/ssd001/home/xiaohui/code/experiments/models/REPEAT_sep22_ytb_tsf_255_b44l3_prev_mask/epo01_iter01640/' -epoch_resume 2 \
    -pred_offline_path 'experiments/proposals/coco81/inference/youtubevos_testdev_online_meta/asdict_90/videos/' \
    -pred_offline_path_eval './experiments/proposals/coco81/inference/youtubevos_val200_meta/asdict_50/pred_DICT.pth' \
    -load_proposals_dataset 1 \
    -load_proposals 1 \
    -distributed 1 \
    -train_split 'train_testdev_ot' \
    -eval_split 'trainval' \
    -finetune_after=3 \
    -skip_empty_starting_frame 1 -random_select_frames 1 \
    -model_name=$MODEL_NAME \
    -train_h=$train_h \
    -train_w=$train_w \
    -loss_weight_match=1 \
    -loss_weight_iouraw=18 \
    -num_workers=$Nworkers \
    -lr 0.0001 \
    -lr_cnn 0.00001 \
    -config_train='./dmm/configs/train.yaml' \
    -dataset=youtube \
    -batch_size=$BS \
    -length_clip=$len_clip \
    -base_model=resnet101 \
    -max_epoch=101 \
    --resize \
    -models_root=$model_root \
    --log_term \
    -max_eval_iter=0 \
    --augment \
    -ngpus $NGPU
