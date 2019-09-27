srun -p p100 --mem=18G python tools/reduce_pth_size_by_videos.py  experiments/proposals/coco81/inference/youtubevos_train3k_meta/predictions.pth train 50 
srun -p p100 --mem=18G python tools/reduce_pth_size_by_videos.py  experiments/proposals/coco81/inference/youtubevos_val200_meta/predictions.pth  trainval 50 
