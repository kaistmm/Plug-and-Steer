#!/usr/bin/env bash
# dprnn-lrs2-av_swap-layer5
# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/dprnn/lrs2/av_swap/layer5.yaml \
  --tag dprnn-lrs2-av_swap-layer5
