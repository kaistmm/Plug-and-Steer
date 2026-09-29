# Paper rows

Every row of the paper maps to one experiment id, and that id is the same string in
two places:

    configs/{backbone}/{corpus}/{setting}/{variant}.yaml
    run_scripts/{backbone}/{corpus}/{setting}/{variant}.sh

Split an id on `-` and you have the config path. This release includes those
configs and scripts. The checkpoints and decoded audio behind the tables are not
included. The one exception is the best AV-TSE checkpoint, linked from the README.

## Table 1 — LRS2-2mix, backbone pre-trained on Libri2Mix train-100

| row | experiment id |
|---|---|
| Conv-TasNet | `convtasnet-libri2mix100-ao_pt-base` |
| Conv-TasNet + Residual (unfrozen) | `convtasnet-libri2mix100-av_fulltune-layer23` |
| Conv-TasNet + Residual (frozen) | `convtasnet-libri2mix100-av_res-layer23` |
| Conv-TasNet + LSM | `convtasnet-libri2mix100-av_swap-layer23` |
| DPRNN | `dprnn-libri2mix100-ao_pt-base` |
| DPRNN + Residual (unfrozen) | `dprnn-libri2mix100-av_fulltune-layer5` |
| DPRNN + Residual (frozen) | `dprnn-libri2mix100-av_res-layer5` |
| DPRNN + LSM | `dprnn-libri2mix100-av_swap-layer5` |
| TF-GridNet | `tfgridnet-libri2mix100-ao_pt-base` |
| TF-GridNet + Residual (unfrozen) | `tfgridnet-libri2mix100-av_fulltune-layer5` |
| TF-GridNet + Residual (frozen) | `tfgridnet-libri2mix100-av_res-layer5` |
| TF-GridNet + LSM | `tfgridnet-libri2mix100-av_swap-layer5` |
| MossFormer2 | `mossformer2-libri2mix100-ao_pt-base` |
| MossFormer2 + Residual (unfrozen) | `mossformer2-libri2mix100-av_fulltune-layer23` |
| MossFormer2 + Residual (frozen) | `mossformer2-libri2mix100-av_res-layer23` |
| MossFormer2 + LSM | `mossformer2-libri2mix100-av_swap-layer23` |

The four AV-TSE baseline rows are released ClearerVoice models, not runs of ours.

## Table 2 — LRS2-2mix with the 107 h MossFormer2

| row | experiment id |
|---|---|
| MossFormer2 (AO) | released `MossFormer2_SS_16K` |
| + Residual (unfrozen) | `mossformer2-clearer-av_fulltune-layer23` |
| + Residual (frozen) | `mossformer2-clearer-av_res-layer23` |
| + LSM | `mossformer2-clearer-av_swap-layer23` |

## Table 3 — LRS2-2mix in-domain

| row | experiment id |
|---|---|
| Conv-TasNet | `convtasnet-lrs2-ao_pt-base` |
| Conv-TasNet + Residual (unfrozen) | `convtasnet-lrs2-av_fulltune-layer23` |
| Conv-TasNet + Residual (frozen) | `convtasnet-lrs2-av_res-layer23` |
| Conv-TasNet + LSM | `convtasnet-lrs2-av_swap-layer23` |
| DPRNN | `dprnn-lrs2-ao_pt-base` |
| DPRNN + Residual (unfrozen) | `dprnn-lrs2-av_fulltune-layer5` |
| DPRNN + Residual (frozen) | `dprnn-lrs2-av_res-layer5` |
| DPRNN + LSM | `dprnn-lrs2-av_swap-layer5` |
| TF-GridNet | `tfgridnet-lrs2-ao_pt-base` |
| TF-GridNet + Residual (unfrozen) | `tfgridnet-lrs2-av_fulltune-layer5` |
| TF-GridNet + Residual (frozen) | `tfgridnet-lrs2-av_res-layer5` |
| TF-GridNet + LSM | `tfgridnet-lrs2-av_swap-layer5` |
| MossFormer2 | `mossformer2-lrs2-ao_pt-base` |
| MossFormer2 + Residual (unfrozen) | `mossformer2-lrs2-av_fulltune-layer23` |
| MossFormer2 + Residual (frozen) | `mossformer2-lrs2-av_res-layer23` |
| MossFormer2 + LSM | `mossformer2-lrs2-av_swap-layer23` |

## Table 4 — routing objective

| objective | experiment id |
|---|---|
| L_BCE | `mossformer2-lrs2-av_swap-layer23_bce` |
| L_SI-SNR + LSM | `mossformer2-lrs2-av_swap-layer23_sisnr` |
| L_BCE + L_SI-SNR + LSM | `mossformer2-lrs2-av_swap-layer23` |

The post-hoc SyncNet rows (LSE-C / LSE-D) were produced outside this repository.

---

## Reproducing a row

Train the three stages in order. A stage that resumes from an earlier run reads it
through `model.config.ckpt_path`. The configs name `exps/in26/...`, and those files
are not in this release. Before starting a stage, point `ckpt_path` at the
`exps/<tag>/ckpt/best.pt` it should load: `swap_matrix`, `av_res` and `av_fulltune`
load the audio-only backbone, and `av_swap` loads the `swap_matrix` checkpoint.

```bash
bash run_scripts/mossformer2/lrs2/ao_pt/base.sh            # 1. audio-only backbone
bash run_scripts/mossformer2/lrs2/swap_matrix/layer23.sh   # 2. LSM, backbone frozen
bash run_scripts/mossformer2/lrs2/av_swap/layer23.sh       # 3. visual steering
```

Score the run you just trained:

```bash
python tools/evaluate.py --exp_dir ./exps/mossformer2-lrs2-av_swap-layer23 --gpu_id 0
```

## Two corrections to the paper

- **`λ`.** Section 3.2 gives `λ = 0.1` in `L_total = L_BCE + λ·L_SI-SNR`. Every run
  behind Tab. 1 and Tab. 3 used **1.0**. The `λ = 0.1` runs were not reported.
- **Full fine-tuning.** Section 3.3's 5e-5 backbone learning rate was implemented in a
  separate entry point that did not survive the rewrite. It is `optim.param_groups` in
  the config now, so the current `av_fulltune` configs follow the paper. They do not
  reproduce the original `av_fulltune` runs exactly.

## What has no config here

- The four **AV-TSE baselines** in Tab. 1 and the baseline row of Tab. 2 are released
  ClearerVoice models.
- The **post-hoc SyncNet** rows of Tab. 4 (LSE-C, LSE-D) were produced with code that
  was never ported into this repository.
- Tab. 2's **AO row** is the released `MossFormer2_SS_16K` checkpoint.
