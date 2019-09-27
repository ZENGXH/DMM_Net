srun -p p100 -x ~/.exclude --mem=7G \
    python tools/davis_eval.py -dataset youtube -davis_eval_folder $1 -eval_split 'trainval'
