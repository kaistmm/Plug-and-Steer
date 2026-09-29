#!/usr/bin/env bash
# dprnn-libri2mix100-swap_matrix-layer1
# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/dprnn/libri2mix100/swap_matrix/layer1.yaml \
  --tag dprnn-libri2mix100-swap_matrix-layer1
