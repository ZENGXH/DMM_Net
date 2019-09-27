BS=4
NGPU=4
train_h=255
train_w=448 
seqlen=5   # max n obj
len_clip=3 # N frames 
Nworkers=4

model_root='../../experiments/models/' # path to saved the model
MODEL_NAME=ytb_train_x101_w11

srun -p max12hours --gres=gpu:$NGPU --mem=45G -c 8 \
    -x ~/.exclude \
    python \
    -m torch.distributed.launch --nproc_per_node=$NGPU \
     train.py \
    -pred_offline_path_eval 'experiments/proposals/coco81/inference/youtubevos_val200_meta/asdict_50/pred_DICT.pth' \
    -pred_offline_path './experiments/proposals/coco81/inference/youtubevos_train3k_meta/asdict_50/videos/' \
    -load_proposals_dataset 1 \
    -load_proposals 1 \
    -distributed 1 \
    -save_every 3000 \
    -train_split 'train' \
    -eval_split 'trainval' \
    -loss_weight_match=1 \
    -loss_weight_iouraw=1 \
    -finetune_after 3 \
    -skip_empty_starting_frame 1 -random_select_frames 1 \
    -model_name $MODEL_NAME \
    -train_h $train_h \
    -train_w $train_w \
    -num_workers $Nworkers \
    -lr 0.0001 \
    -lr_cnn 0.00001 \
    -config_train 'dmm/configs/train.yaml' \
    -batch_size=$BS \
    -length_clip=$len_clip \
    -max_epoch=2 \
    --resize \
    -base_model 'resnet101' \
    -models_root=$model_root \
    -max_eval_iter=800 \
    --augment \
    -ngpus $NGPU
