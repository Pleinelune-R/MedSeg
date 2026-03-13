#!/bin/bash
nohup python run_train.py \
  --data_path ./data/preprocessed_train \
  --save_dir ./checkpoints_train \
  --max_epochs 100 \
  --batch_size 2 \
  --devices_numbers 2 \
  --embed_dim 768 \
  --patch_size 8 \
  --dropout 0.1 \
  --learning_rate 1e-5 \
  > output_train.log 2>&1 &
