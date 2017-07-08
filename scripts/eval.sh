set -x
set -e
# ./scripts/eval.sh 1 icdar2013 train 384 384 ckpt 
# ./scripts/eval.sh 1 icdar2013 train 512 512 ckpt 

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
SPLIT=$3
WIDTH=$4
HEIGHT=$5
CHECKPOINT_PATH=$6

if [ $DATASET == 'synthtext' ]
then
    DATA_PATH=SynthText
elif [ $DATASET == 'scut' ]
then
    DATA_PATH=SCUT
elif [ $DATASET == 'icdar2013' ]
then
    DATA_PATH=ICDAR
elif [ $DATASET == 'icdar2015' ]
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
			--eval_image_width=${WIDTH} \
			--eval_image_height=${HEIGHT} \
			--gpu_memory_fraction=0.4 \
			--do_grid_search=$7 \
			--using_moving_average=0

			
			
			
