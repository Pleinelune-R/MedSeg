#!/bin/bash
# Script to restart MAE pretraining with full volume dimensions (matching training data)
# Uses data/preprocessed_mae which contains uncropped (resized) volumes.

nohup python pretrain.py   \
  --data_path ./data/preprocessed_pretrain   \
  --save_dir ./checkpoints_mae_pretrain   \
  --volume_size 48 256 256   \
  --patch_size 8   \
  --embed_dim 768   \
  --decoder_embed_dim 576   \
  --decoder_num_heads 16   \
  --batch_size 4   \
  --mask_ratio 0.75   \
  --learning_rate 1e-4   \
  --dropout 0.1   \
  --crop_ratio 0.4   \
  --max_epochs 200   \
  --devices_numbers 0   \
  > output_pretrain.log 2>&1 &

echo "Pretraining started in background. Check output_pretrain.log for progress."
