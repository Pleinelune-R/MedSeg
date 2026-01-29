#!/bin/bash
nohup python train.py \
  --data_path ./data/preprocessed_train \
  --save_dir ./checkpoints_supervised_3d \
  --max_epochs 100 \
  --batch_size 2 \
  --devices_numbers 2 \
  --embed_dim 768 \
  --patch_size 8 \
  --dropout 0.1 \
  --learning_rate 1e-5 \
  --pretrained_path ./checkpoints_mae_pretrain/mae_3d_best_model.ckpt \
  > output_train.log 2>&1 &
