path_to_prediction_pth='experiments/proposals/ytb_train/inference/youtubevos_val200_meta/predictions.pth'
eval_flag=ytb_max
NGPU=3
ranks=$(expr $NGPU - 1)
rank=0
batch_size=1
input_h=255 
input_w=448 
MODEL='./experiments/models/ytb_train_x101_w11_mf/best_0.777_epo01/' # /epo01_iter01640/'
MODEL='./experiments/models/ytb_train_x101/best_0.776_epo01/' # best_0.777_epo01/' # /epo01_iter01640/'
MODEL='../../experiments/models/ytb_train_x101_w18/best_0.772_epo01/'
# MODEL='../../experiments/models/ytb_train_x101_w11/best_0.772_epo01/'
part=p100


for i in $(seq 0 ${ranks}) 
do
    rank=$i
    echo ` \
    srun -p $part --mem=10G --gres=gpu:1 \
        -x ~/.exclude -J $eval_flag \
    python \
    eval.py \
    -eval_flag ${eval_flag} \
    -pred_offline_path_eval ${path_to_prediction_pth} \
    -pred_offline_path ${path_to_prediction_pth} \
    -pred_offline_meta 'data/folder_data/meta_val200-CACHE_maskben_folderdata_index_200.json' \
    -load_proposals 1 \
    -load_proposals_dataset 0 \
    -cache_data=1 \
    -distributed_manully=1 \
    -distributed_manully_Nrep=${NGPU} \
    -distributed_manully_rank=$rank \
    --local_rank=$rank \
    -base_model 'resnet101' \
    -test=1 \
    -test_image_h=$input_h \
    -test_image_w=$input_w \
    -batch_size=$batch_size \
    -num_workers=4 \
    -model_name=$MODEL \
    -config_train='dmm/configs/eval.yaml' \
    -pad_video=1 \
    -prev_mask_d 1 \
    -eval_split=trainval \
    -length_clip=100 \
    -ngpu 1 \
    `  &
 done
 exit 
