#!/usr/bin/env bash
# mossformer2-clearer-av_res-layer23
# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/mossformer2/clearer/av_res/layer23.yaml \
  --tag mossformer2-clearer-av_res-layer23
