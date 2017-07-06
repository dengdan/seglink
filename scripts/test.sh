set -x
set -e
# ./scripts/test.sh 1 icdar2013 train 384 384 ckpt 
# ./scripts/test.sh 1 icdar2013 train 512 512 ckpt 
# ./scripts/test.sh 1 icdar2015 train 1280 768 ckpt 

export CUDA_VISIBLE_DEVICES=$1
CHECKPOINT_PATH=$2


DATASET_DIR=$HOME/dataset/SSD-tf/${DATA_PATH}

python test_seglink.py \
			--checkpoint_path=${CHECKPOINT_PATH} \
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=${DATASET} \
            --dataset_split_name=$SPLIT \
			--eval_image_width=${WIDTH} \
			--eval_image_height=${HEIGHT} \
			--gpu_memory_fraction=-1 \
			--seg_conf_threshold=0.1 \
			--link_conf_threshold=0.5

			
			
			
			
			
