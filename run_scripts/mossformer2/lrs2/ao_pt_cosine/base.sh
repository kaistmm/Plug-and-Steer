#!/usr/bin/env bash
# mossformer2-lrs2-ao_pt_cosine-base
# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --config ./configs/mossformer2/lrs2/ao_pt_cosine/base.yaml \
  --tag mossformer2-lrs2-ao_pt_cosine-base
