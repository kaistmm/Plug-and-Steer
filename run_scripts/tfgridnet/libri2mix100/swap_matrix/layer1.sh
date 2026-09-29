#!/usr/bin/env bash
# tfgridnet-libri2mix100-swap_matrix-layer1
# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/tfgridnet/libri2mix100/swap_matrix/layer1.yaml \
  --tag tfgridnet-libri2mix100-swap_matrix-layer1
