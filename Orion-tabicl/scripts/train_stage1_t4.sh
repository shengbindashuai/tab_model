##!/bin/bash

#tabicl
# This script is used to train TabICL for the first stage of the curriculum learning

# ----------------------------------
# Generate prior datasets on the fly
# ----------------------------------


torchrun --standalone --nproc_per_node=8 /vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/tab_model/Orion-tabicl/src/orion_msp/train/run.py \
            --wandb_log True \
            --wandb_project TabICL \
            --wandb_name ldm_s1_2 \
            --wandb_dir ./wandb/dir \
            --wandb_mode offline \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 160000 \
            --batch_size 512 \
            --micro_batch_size 4 \
            --lr 1e-4 \
            --scheduler cosine_with_restarts \
            --warmup_proportion 0.02 \
            --cosine_lr_end 2e-5 \
            --gradient_clipping 1.0 \
            --prior_type mix_scm \
            --prior_device cpu \
            --batch_size_per_gp 8 \
            --min_features 2 \
            --max_features 100 \
            --max_classes 10 \
            --max_seq_len 1024 \
            --min_train_size 0.1 \
            --max_train_size 0.9 \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 4 \
            --col_num_inds 128 \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --icl_num_blocks 12 \
            --icl_nhead 4 \
            --ff_factor 2 \
            --norm_first True \
            --checkpoint_dir ./tab/stage1_t4/checkpoint/dir \
            --save_temp_every 50 \
            --save_perm_every 5000