#!/usr/bin/env bash
# mossformer2-lrs2-swap_matrix-layer15
# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/mossformer2/lrs2/swap_matrix/layer15.yaml \
  --tag mossformer2-lrs2-swap_matrix-layer15
