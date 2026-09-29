#!/usr/bin/env bash
# mossformer2-libri2mix100-swap_matrix-layer9
# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/mossformer2/libri2mix100/swap_matrix/layer9.yaml \
  --tag mossformer2-libri2mix100-swap_matrix-layer9
