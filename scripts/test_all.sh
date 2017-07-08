set -x
set -e
# ./scripts/test.sh 1 icdar2013 train 384 384 ckpt 
# ./scripts/test.sh 1 icdar2013 train 512 512 ckpt 
# ./scripts/test.sh 1 icdar2015 train 1280 768 ckpt 

export CUDA_VISIBLE_DEVICES=$1
CHECKPOINT_PATH=$2


python test_seglink_all.py \
			--checkpoint_path=${CHECKPOINT_PATH} \
			--gpu_memory_fraction=0.25

			
			
			
			
			
