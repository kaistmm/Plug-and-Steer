#!/usr/bin/env bash
# dprnn-libri3mix360-ao_pt-base
# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --config ./configs/dprnn/libri3mix360/ao_pt/base.yaml \
  --tag dprnn-libri3mix360-ao_pt-base
