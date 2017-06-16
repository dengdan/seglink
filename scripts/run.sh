set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
python train_seglink.py

