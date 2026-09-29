#!/usr/bin/env bash
# convtasnet-lrs2-swap_matrix-layer19
# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/convtasnet/lrs2/swap_matrix/layer19.yaml \
  --tag convtasnet-lrs2-swap_matrix-layer19
