# Training guide

How experiments are named, what each training stage does, and which config
fields matter. Setting up the environment and the dataset manifests is covered
in the [README](../README.md#setup) and assumed here.

Every component — dataset, loss, model, trainer, optimizer, scheduler — is
selected by **name** in a YAML config and imported dynamically, so a new
experiment is a new config rather than a new branch of code.

| see also | |
|---|---|
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | every paper row → its config |

---

## One id, two places

A tag is the config's own path with hyphens between the directory levels:

```
configs/{backbone}/{corpus}/{setting}/{variant}.yaml
run_scripts/{backbone}/{corpus}/{setting}/{variant}.sh
```

<!-- Levels never contain a hyphen — underscores stay inside a level (`av_swap`,
`layer23_bce`) — so splitting a tag gives back the config path and the run script. -->

| token | values |
|---|---|
| `backbone` | `convtasnet` `dprnn` `mossformer2` `tfgridnet` |
| `corpus` | `libri2mix100` `libri2mix360` `libri3mix100` `libri3mix360` `lrs2` `clearer` |
| `setting` | `ao_pt` `ao_pt_cosine` `swap_matrix` `av_swap` `av_res` `av_fulltune` |
| `variant` | `base` for the AO stages, otherwise `layer{N}` (+ a loss suffix) |

`corpus` means different things by stage: for `ao_pt` and `swap_matrix` it is
**the data the run trains on**; for the `av_*` stages it is **the corpus the
backbone was pre-trained on**. AV training is always LRS2, the only corpus with
video, so it cannot distinguish those rows. Read
`configs/dprnn/libri2mix100/av_swap/layer5.yaml` as *a Libri2Mix backbone steered
by LRS2 video*.

## Where results live

Training writes to `exps/<tag>/`: `ckpt/best.pt`, the last few numbered
checkpoints, TensorBoard events and a copy of the resolved config. Change the
root with `train.py --output_path` (default `./exps`).

`tools/inference.py` writes decoded audio to `exps/<tag>/inferences/` unless you
pass `--save_dir`.

A later stage loads an earlier one through `model.config.ckpt_path`. Point that
field at a checkpoint you trained (`exps/<tag>/ckpt/best.pt`) before launching
the script. Which stage loads which checkpoint is in
[Reproducing a row](EXPERIMENTS.md#reproducing-a-row).


## Settings

| setting | what it trains | trainer |
|---|---|---|
| `ao_pt` | audio-only backbone from scratch, plateau schedule | `basic_separation_trainer` |
<!-- | `ao_pt_cosine` | the same, on a fixed 200k-step cosine budget | `basic_separation_trainer` | -->
| `swap_matrix` | LSM at separator block *i*, backbone frozen | `basic_separation_trainer_swap` |
| `av_swap` | visual steering module, backbone + LSM frozen | `av_gate_trainer` |
| `av_res` | residual visual adapter, backbone frozen | `av_trainer` |
| `av_fulltune` | residual visual adapter, backbone trained at a lower lr | `av_trainer` |

<!-- `ao_pt` reproduces the paper. `ao_pt_cosine` is the set for new work: a
4 backbone × 5 corpus matrix on one fixed budget, so runs are comparable across
corpora. The two sets differ **only** by the schedule and the fixed-epoch knobs;
everything else was aligned deliberately, so keep them that way. -->


## Configuration notes

**Model classes.** One pattern per backbone, no exceptions:

```
{Backbone}          audio-only
{Backbone}_swap     LSM
{Backbone}_av       residual adapter — `freeze: true/false` picks res or fulltune
{Backbone}_av_swap  visual gate
```

**res vs fulltune.** Both load the AO backbone; never point them at an LSM
checkpoint. They differ by `freeze` and by the learning rate:

```yaml
optim:
  config: {lr: 5.0e-4}
  param_groups:                 # av_fulltune only
    - name: adapter
      modules: [visual_module, vis_module, compress, decompress]
      lr: 5.0e-4
    - name: backbone            # no `modules`: everything not claimed above
      lr: 5.0e-5
```

Omit `param_groups` and every parameter goes into one group, which is what all
the other configs do. This only works because the schedulers are `LambdaLR`-based
and scale each group from its own base lr — a scheduler that writes one lr into
every group would silently flatten it.

**Loss weighting for the steering module** (`av_gate_trainer`). Tab. 4's ablation
is configuration, not separate code:

| objective | `sisnr_loss_weight` | `gate_loss_weight` |
|---|---|---|
| `L_BCE + L_SI-SNR` (the paper) | `1.0` | `> 0` |
| `L_BCE` only | `0.0` | `> 0` |
| `L_SI-SNR` only | `1.0` | `0.0` |

**`C_r`** is `model.config.compress_dim`, default 16, for DPRNN and TF-GridNet.
The other two backbones have 1D latents and no reduction. It is a config field,
never part of a name.

**Checkpoint retention.** `trainer.keep_last_ckpt` (10 everywhere) bounds the
numbered checkpoints; `best.pt` is never pruned. Set it to `0` to keep everything
— a large backbone reaches tens of GB that way.

**Fixed-size epochs.** `num_per_epoch` makes an epoch a fixed number of items
instead of a full pass, so validation and checkpoint cadence is the same whatever
the corpus size. It does not change how long training runs. Items come from a
rotating partition, so coverage is complete after one sweep. Validation should
always pair it with `resample_per_epoch: false`.

**Budget and stopping.** `trainer.epoch` is 2000 everywhere and `early_stop`
matches it, so a run never exits on its own. That count is not the training
budget. The stopping rule is in the config:

| stage | setting | what ends training |
|---|---|---|
| audio-only backbone | `ao_pt` | `ReduceLROnPlateau` — no step budget |
| LSM | `swap_matrix` | `num_training_steps` ≈ 10k, warmup 0 |
| visual steering | `av_swap`, `av_res`, `av_fulltune` | `num_training_steps` ≈ 100k, warmup 1,000 |

Exact step counts are under `scheduler.config`. On the cosine schedules
`num_cycles` is 1, so past `num_training_steps` the learning rate is exactly 0.
`ao_pt` reproduces the paper. `ao_pt_cosine` is a separate 200k-step schedule
with a single cosine cycle and a 6,000-step warmup.

**Reading a saved config.** `exps/<tag>/config.yaml` is written by ruamel and must
be read back with `tools/config_io.load_config`, not `yaml.safe_load` — ruamel
emits floats as `1e-05`, which PyYAML parses as a *string*.

## Tools

```bash
python tools/validate_configs.py --check-paths     # every config resolves; files exist
python tools/gen_run_scripts.py                    # rebuild run_scripts/ from configs/

python tools/evaluate.py  --exp_dir ./exps/convtasnet-libri2mix100-av_swap-layer23 --gpu_id 0
python tools/inference.py --exp_dir ./exps/convtasnet-libri2mix100-av_swap-layer23 --gpu_id 0 --save_dir ./out
```

`--exp_dir` is the checkpoint folder itself — `exps/<tag>` for a run trained
here, and any path outside `exps/` works too.
Run `validate_configs.py` after touching configs
and `gen_run_scripts.py` after adding or removing one — it also deletes scripts
whose config is gone.

Run scripts are generated *from* configs, so hand-edits to a `.sh` are lost on the
next regenerate. To change how many GPUs a run asks for, set `trainer.gpus_hint`
in the config and regenerate. `CUDA_VISIBLE_DEVICES` is taken as given —
`train.py` never rewrites it.

## Adding a module

Drop a file in the matching `src/` directory exposing the expected symbol, then
name it in a config. Each directory has a `skeleton.py` showing the contract:

| directory | symbol |
|---|---|
| `src/model/` | `Main_Model(**config)` |
| `src/loss/` | `Loss(ests, egs, **config)` — function or class |
| `src/datasets/dataset/` | `Datasets(**config)` |
| `src/datasets/sampler/` | `Sampler(dataset, **config, **proc)` |
| `src/optimization/optimizer/` | `Optimizer(params, **config)` |
| `src/optimization/scheduler/` | `Scheduler(optimizer, **config)` |

A trainer subclasses `DDP_Trainer_Skeleton` and implements `train()` and
`validate()`. `run_train_epoch(compute_loss)` supplies the accumulation loop, DDP
sync suppression, collective NaN handling, clipping and logging, so a trainer
usually only describes its own forward pass. Changing the skeleton touches every
trainer — override in the trainer instead.

## Data

`LRS2-2mix` and `LibriMix` are read from `.scp` lists under `data/`; the
audio-visual dataset additionally reads facial videos at 25 FPS. Paths are set
per config under `datasets.{train,validation}.dataset.config`. Building those
lists in the first place is covered in the
[README](../README.md#prepare-datasets).

