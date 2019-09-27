model_path=$1

DATE=`date "+%m%d-%H%M%S"`
model=${model_path}
key=`echo $model| sed 's/\.//g'`
key=`echo $key| sed 's/\//_/g'`
key=${DATE}${key}
# check 
mkdir -p tmp 

python tools/submission_check.py -p ${model} -s 0 && echo ${model} && cd tmp/ \
    && mkdir ${DATE} && cd ${DATE} && ln -s ../../${model} Annotations &&  echo "let zip it." \
    && zip -rq As_${key}.zip Annotations \
    && rm Annotations && echo "As_${key}.zip from ${model}" >> ../submission.md  \
    && echo "submit file saved at tmp/${DATE}/As_${key}.zip"
    
cd ../../
