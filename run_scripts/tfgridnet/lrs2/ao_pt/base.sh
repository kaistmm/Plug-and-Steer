#!/usr/bin/env bash
# tfgridnet-lrs2-ao_pt-base
# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --config ./configs/tfgridnet/lrs2/ao_pt/base.yaml \
  --tag tfgridnet-lrs2-ao_pt-base
