#!/usr/bin/env bash
# mossformer2-libri3mix100-ao_pt-base
# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --config ./configs/mossformer2/libri3mix100/ao_pt/base.yaml \
  --tag mossformer2-libri3mix100-ao_pt-base
