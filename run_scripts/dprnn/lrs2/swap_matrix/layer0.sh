#!/usr/bin/env bash
# dprnn-lrs2-swap_matrix-layer0
# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/dprnn/lrs2/swap_matrix/layer0.yaml \
  --tag dprnn-lrs2-swap_matrix-layer0
