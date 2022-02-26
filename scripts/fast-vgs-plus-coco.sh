#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate my_env
export CUDA_VISIBLE_DEVICES=0,1

data_root=$1
raw_audio_base_path=$2
fb_w2v2_weights_fn=$3
exp_dir=$4

python \
../run_spokencoco.py \
--data_root ${data_root} \
--raw_audio_base_path ${raw_audio_base_path} \
--fb_w2v2_weights_fn ${fb_w2v2_weights_fn} \
--exp_dir ${exp_dir} \
--batch_size 64 \
--val_batch_size 100 \
--val_cross_batch_size 8000 \
--n_epochs 20 \
--n_print_steps 1 \
--n_val_steps 2 \
--lr 0.0001 \
--warmup_fraction 0.1 \
--normalize \
--xtrm_layers 2 \
--trm_layers 6 \
--fine_matching_weight 1.0 \
--coarse_matching_weight 0.1 \
--libri_w2v2_weight 1.0 \
--caption_w2v2_weight 1.0 \
--coarse_to_fine_retrieve \
--feature_grad_mult 0. \
--layer_use 7