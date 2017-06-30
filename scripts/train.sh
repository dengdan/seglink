set -x
set -e
# ./scripts/train.sh 0 18 synthtext
export CUDA_VISIBLE_DEVICES=$1
IMG_PER_GPU=$2
DATASET=$3

#TRAIN_DIR=${HOME}/models/seglink/seglink_synthtext
TRAIN_DIR=${HOME}/temp/no-use/seglink_debug3


# get the number of gpus
OLD_IFS="$IFS" 
IFS="," 
gpus=($CUDA_VISIBLE_DEVICES) 
IFS="$OLD_IFS"
NUM_GPUS=${#gpus[@]}

# batch_size = num_gpus * IMG_PER_GPU
BATCH_SIZE=`expr $NUM_GPUS \* $IMG_PER_GPU`

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

python train_seglink.py \
			--train_dir=${TRAIN_DIR} \
			--num_gpus=${NUM_GPUS} \
			--learning_rate=0.0001 \
			--gpu_memory_fraction=-1 \
			--train_image_width=384 \
			--train_image_height=384 \
			--batch_size=${BATCH_SIZE}\
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=${DATASET} \
            --dataset_split_name=train \
			--checkpoint_path=${HOME}/models/ssd-pretrain/seglink
