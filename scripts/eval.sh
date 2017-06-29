set -x
set -e
# ./scripts/eval.sh 1 icdar2013 train ckpt

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
SPLIT=$3
CHECKPOINT_PATH=$4


#dataset
if [ $DATASET == 'synthtext' ]
then
    DATA_PATH=SynthText
elif [ $DATASET == 'scut' ]
then
    DATA_PATH=SCUT
elif [ $DATASET == 'icdar2013' ]
then
    DATA_PATH=ICDAR
else
    echo invalid dataset: $DATASET
    exit
fi

DATASET_DIR=$HOME/dataset/SSD-tf/${DATA_PATH}

python eval_seglink.py \
			--checkpoint_path=${CHECKPOINT_PATH} \
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=${DATASET} \
            --dataset_split_name=$SPLIT \
			--eval_image_width=384 \
			--eval_image_height=384 \
			--gpu_memory_fraction=0.3 &

python eval_seglink.py \
			--checkpoint_path=${CHECKPOINT_PATH} \
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=${DATASET} \
            --dataset_split_name=$SPLIT \
			--eval_image_width=512 \
			--eval_image_height=512 \
			--gpu_memory_fraction=0.3
			
			
			
			
			
