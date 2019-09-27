#!/bin/bash 

eval_flag=ytb_testdev_online
MODEL='experiments/dmmnet/online_ytb/epo90_iter00088/' # epo117_iter00088' 
# MODEL='../../experiments/models/ytb_train_x101_online/epo101_iter00088/'
MODEL='../../experiments/models/ytb_train_x101_online_from11/epo50_iter00088/' # epo89_iter00088/'
MODEL='../../experiments/models/REPEAT_ONLINE_sep25_prev_mask/epo75_iter00088/'
pred_offline_meta='./data/folder_data/meta_val474-CACHE_maskben_folderdata_index_474.json'
path_to_prediction_pth='./experiments/proposals/ytb_ot/inference/youtubevos_testdev_meta/predictions.pth'
dataset=youtube

# change: 
NGPU=12
ranks=$(expr $NGPU - 1)
batch_size=1
input_h=255
input_w=448
part=max12hours
# part=p100

for i in $(seq 0 ${ranks}) 
do
    rank=$i
    echo ` \
    srun -p $part --mem=10G --gres=gpu:1 \
        -x ~/.exclude -J $eval_flag \
    python \
    eval.py \
    -eval_flag ${eval_flag} \
    -pred_offline_path ${path_to_prediction_pth} \
    -pred_offline_meta ${pred_offline_meta} \
    -load_proposals 1 \
    -distributed_manully=1 \
    -distributed_manully_Nrep=${NGPU} \
    -distributed_manully_rank=${rank} \
    --local_rank=${rank} \
    -test=1 \
    -test_image_h=$input_h \
    -test_image_w=$input_w \
    -num_workers=4 \
    -model_name=$MODEL \
    -config_train='./dmm/configs/eval.yaml' \
    -pad_video=1 \
    -base_model=resnet101 \
    -maxseqlen=5 \
    -gt_maxseqlen=5 \
    -prev_mask_d=1 \
    -dataset=${dataset} \
    -eval_split=valid \
    -batch_size=$batch_size \
    -length_clip=100 \
    -ngpu 1 `  &
 done
 exit
