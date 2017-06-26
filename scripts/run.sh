set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
IMG_PER_GPU=$2

OLD_IFS="$IFS" 
IFS="," 
gpus=($CUDA_VISIBLE_DEVICES) 
IFS="$OLD_IFS"
num_gpus=${#gpus[@]}

# get the number of gpus
OLD_IFS="$IFS" 
IFS="," 
gpus=($CUDA_VISIBLE_DEVICES) 
IFS="$OLD_IFS"
num_gpus=${#gpus[@]}

# batch_size = num_gpus * IMG_PER_GPU
BATCH_SIZE=`expr $num_gpus \* $IMG_PER_GPU`

python train_seglink.py \
			--train_dir=${HOME}/temp/no-use/seglink_debug \
			--num_gpus=${num_gpus} \
			--learning_rate=0.001 \
			--train_image_width=384 \
			--train_image_height=384 \
			--gpu_memory_fraction=-1 \
			--batch_size=${BATCH_SIZE}\
			--checkpoint_path=${HOME}/models/ssd-pretrain/seglink