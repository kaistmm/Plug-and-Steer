#!/usr/bin/env bash
# convtasnet-libri2mix100-ao_pt_cosine-base
# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/convtasnet/libri2mix100/ao_pt_cosine/base.yaml \
  --tag convtasnet-libri2mix100-ao_pt_cosine-base
