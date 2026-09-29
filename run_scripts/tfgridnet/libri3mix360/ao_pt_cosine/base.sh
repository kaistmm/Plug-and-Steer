#!/usr/bin/env bash
# tfgridnet-libri3mix360-ao_pt_cosine-base
# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --config ./configs/tfgridnet/libri3mix360/ao_pt_cosine/base.yaml \
  --tag tfgridnet-libri3mix360-ao_pt_cosine-base
