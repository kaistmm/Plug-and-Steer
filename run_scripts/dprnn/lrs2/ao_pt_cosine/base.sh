#!/usr/bin/env bash
# dprnn-lrs2-ao_pt_cosine-base
# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --config ./configs/dprnn/lrs2/ao_pt_cosine/base.yaml \
  --tag dprnn-lrs2-ao_pt_cosine-base
