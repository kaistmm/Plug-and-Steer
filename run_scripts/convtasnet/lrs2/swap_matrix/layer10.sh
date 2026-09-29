#!/usr/bin/env bash
# convtasnet-lrs2-swap_matrix-layer10
# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/convtasnet/lrs2/swap_matrix/layer10.yaml \
  --tag convtasnet-lrs2-swap_matrix-layer10
