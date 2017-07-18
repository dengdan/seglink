set -x
set -e
# ./scripts/test.sh 1 icdar2013 train 384 384 ckpt 
# ./scripts/test.sh 1 icdar2013 train 512 512 ckpt 
# ./scripts/test.sh 1 icdar2015 train 1280 768 ckpt 

export CUDA_VISIBLE_DEVICES=$1
CHECKPOINT_PATH=$2
DATASET_DIR=$3



python test_seglink.py \
			--checkpoint_path=${CHECKPOINT_PATH} \
			--gpu_memory_fraction=-1 \
			--seg_conf_threshold=0.8 \
			--link_conf_threshold=0.5 \
            --dataset_dir=${DATASET_DIR}
			
			
			
			
			
