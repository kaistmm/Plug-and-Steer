"""Generate the ao_pt_cosine config set.

One config per (backbone, corpus). Everything that is not the backbone or the
corpus is held identical, so the set is a clean matrix.
"""

import os

# ---------------------------------------------------------------- constants

BUDGET = 200_000          # optimizer steps; the LR schedule ends here
WARMUP = 6_000            # max(2000, 3% of budget)
EFFECTIVE_BATCH = 4       # per the paper
GPUS = 2                  # what the run scripts assume
UPD_PER_EPOCH = 2_000     # -> 100 epochs for every corpus, uniform cadence
TRAIN_PER_EPOCH = UPD_PER_EPOCH * EFFECTIVE_BATCH   # 8000 items
VAL_PER_EPOCH = 1_000
BASE_PORT = 24100

# ---------------------------------------------------------------- backbones

BACKBONES = {
    'convtasnet': dict(
        model='ConvTasnet', batch=2, precision='bf16-mixed', amp_infer='true',
        params=lambda n: f"""    num_sources: {n}
    enc_kernel_size: 16
    enc_num_feats: 512
    msk_kernel_size: 3
    msk_num_feats: 128
    msk_num_hidden_feats: 512
    msk_num_layers: 8
    msk_num_stacks: 3
    msk_activate: 'relu'"""),
    'dprnn': dict(
        # DPRNN has no working mixed-precision path; fp32 is not a choice.
        model='DPRNN', batch=2, precision='fp32', amp_infer='false',
        params=lambda n: f"""    in_channels: 256
    out_channels: 64
    hidden_channels: 128
    kernel_size: 2
    rnn_type: LSTM
    norm: ln
    dropout: 0
    bidirectional: true
    num_layers: 6
    K: 250
    num_spks: {n}"""),
    'mossformer2': dict(
        model='MossFormer2', batch=1, precision='fp32', amp_infer='false',
        params=lambda n: f"""    in_channels: 512
    out_channels: 512
    num_blocks: 24
    kernel_size: 16
    norm: "ln"
    num_spks: {n}
    skip_around_intra: true
    use_global_pos_enc: true
    max_length: 20000"""),
    'tfgridnet': dict(
        model='TFGridNetV1', batch=2, precision='bf16-mixed', amp_infer='true',
        params=lambda n: f"""    n_srcs: {n}
    n_fft: 512
    stride: 128
    window: "hann"
    n_imics: 1
    n_layers: 6
    lstm_hidden_units: 256
    attn_n_head: 4
    attn_approx_qk_dim: 512
    emb_dim: 32
    emb_ks: 8
    emb_hs: 1
    activation: "prelu"
    eps: 0.00001
    use_builtin_complex: false
    ref_channel: -1"""),
}

# ------------------------------------------------------------------ corpora
# n_train is measured from the scp files and only feeds the D/N comment.

CORPORA = {
    'libri2mix_100': dict(
        group='librimix', dataset='Librimix', spks=2, n_train=13_900,
        root='./data/librimix', train='tr', val='cv',
        note="Libri2Mix train-100 (~58 h). The clean, studio-quality "
             "pre-training the paper contrasts against LRS2-2mix."),
    'libri2mix': dict(
        group='librimix', dataset='Librimix', spks=2, n_train=50_800,
        root='./data/librimix/libri2mix', train='tr', val='cv',
        note="Libri2Mix train-360. 3.7x the data of train-100 at the same "
             "step budget, so this isolates corpus size from budget."),
    'libri3mix': dict(
        group='librimix', dataset='Librimix', spks=3, n_train=33_900,
        root='./data/librimix/libri3mix', train='tr', val='cv',
        note="Libri3Mix train-360, three speakers."),
    'libri3mix_100': dict(
        group='librimix', dataset='Librimix', spks=3, n_train=9_300,
        root='./data/librimix/libri3mix_100', train='tr', val='cv',
        note="Libri3Mix train-100, three speakers. The 3-speaker counterpart "
             "of libri2mix_100."),
    'lrs2_2mix': dict(
        group='lrs2', dataset='LRS2mix', spks=2, n_train=40_000,
        root='./data/lrs2_2mix', train='tr', val='cv',
        note="LRS2-2mix, the in-the-wild corpus. Pre-training here gives the "
             "in-domain setting of Tab. 3; pre-training on Libri2Mix and "
             "adapting here gives Tab. 1."),
}

HEADER = """#### {model} audio-only pre-training on {corpus}
#
# ao_pt_cosine: fixed-budget cosine, as opposed to the `ao_pt` set, which
# reproduces the paper's ReduceLROnPlateau schedule. Same model, same data,
# same optimizer -- only the learning-rate schedule and its stopping rule
# differ.
#
#   budget      {budget:,} optimizer steps, single cosine cycle, {warmup:,}-step warmup
#   batch       {batch} per GPU x {gpus} GPUs x accum {accum} = {eff} effective (paper: 4)
#   corpus      N = {n_train:,} utterances  ->  D/N = {dn:.0f} effective passes
#   epoch       {upd:,} updates ({train_per:,} items), so {epochs} epochs cover the budget
#
# {note}
"""


def render(bb_name, bb, corpus_name, corpus, port):
    n = corpus['spks']
    root = corpus['root']
    tr, cv = corpus['train'], corpus['val']
    tgt_tr = ', '.join(f'{root}/{tr}_s{i}.scp' for i in range(1, n + 1))
    tgt_cv = ', '.join(f'{root}/{cv}_s{i}.scp' for i in range(1, n + 1))

    # accumulate up to the paper's effective batch of 4 whatever fits in memory
    accum = max(1, EFFECTIVE_BATCH // (bb['batch'] * GPUS))
    eff = bb['batch'] * GPUS * accum
    D = BUDGET * eff

    head = HEADER.format(
        model=bb['model'], corpus=corpus_name, budget=BUDGET, warmup=WARMUP,
        batch=bb['batch'], gpus=GPUS, accum=accum, eff=eff,
        n_train=corpus['n_train'], dn=D / corpus['n_train'],
        upd=UPD_PER_EPOCH, train_per=TRAIN_PER_EPOCH,
        epochs=BUDGET // UPD_PER_EPOCH, note=corpus['note'])

    extra = ''
    if corpus['dataset'] == 'LRS2mix':
        extra = '\n        return_video: false'

    return f"""{head}
name: {bb['model']}
num_spks: {n}

### DDP settings; node1 is master node
DDP:
  port: {port}
  backend: nccl
  devices:
    node1:
      ip: localhost
      gpus: []          # empty -> use whatever CUDA_VISIBLE_DEVICES exposes
      batch_size: {bb['batch']}


#### datasets
datasets:
  train:
    dataset:
      name: {corpus['dataset']}
      config:
        mix_scp: {root}/{tr}_mix.scp
        target_scp: [{tgt_tr}]

        sample_rate: 16000
        chunk_size: 48000        # 3 s, as in the paper
        debug_mode: false
        normalize: true{extra}

        # A fixed-size epoch. This does NOT change how much training happens --
        # the budget is the step count above. It makes the validation and
        # checkpoint cadence identical across corpora of different sizes.
        # Items are drawn as a rotating partition of a shuffled corpus, so
        # every utterance is seen once per sweep (see epoch_subset.py).
        num_per_epoch: {TRAIN_PER_EPOCH}
        resample_per_epoch: true

    sampler:
      name: DDP_sampler
      config:
        shuffle: true

    loader:
      shuffle: false
      num_workers: 4          # per GPU
      pin_memory: true
      drop_last: true         # keeps every accumulation window full

  validation:
    dataset:
      name: {corpus['dataset']}
      config:
        mix_scp: {root}/{cv}_mix.scp
        target_scp: [{tgt_cv}]

        sample_rate: 16000
        chunk_size: -1           # full utterance
        debug_mode: false
        normalize: true{extra}

        # Fixed subset, never resampled: the validation curve has to measure the
        # model, not which utterances this epoch happened to draw.
        num_per_epoch: {VAL_PER_EPOCH}
        resample_per_epoch: false

    sampler:
      name: DDP_sampler
      config:
        shuffle: false

    loader:
      shuffle: false
      num_workers: 4
      pin_memory: true


#### network structures
model:
  name: {bb['model']}
  config:
{bb['params'](n)}


#### training settings
trainer:
  name: basic_separation_trainer
  seed: 1234
  precision_mode: "{bb['precision']}"
  precision_inference: {bb['amp_infer']}
  # The cosine schedule defines the budget; `epoch` only has to be large enough
  # that the job never exits on its own. Past {BUDGET:,} steps the lr is 0, so the
  # extra epochs hold the GPU reservation without moving the weights.
  epoch: 2000
  early_stop: 2000
  clip_norm: 5
  grad_accum_num: {accum}
  keep_last_ckpt: 10
  loss:
    name: SISNR_PIT
    config:


#### Optimizer settings
optim:
  name: Adam
  config:
    lr: 0.0005
    weight_decay: 0


#### scheduler settings
scheduler:
  name: CosineAnnealingLR
  loss_pass: false
  call_step: iteration
  config:
    num_warmup_steps: {WARMUP}
    num_training_steps: {BUDGET}
    num_cycles: 1            # one decay to zero; never restart


#### Resume training settings
checkpoint:
  resume:
    enable: true
    type: last
    only_model: false


#### logging by tensorboard
logging:
  name: logger
  log_level: INFO
  log_freq: 100
  tensorboard: true
"""


def main():
    port = BASE_PORT
    made = []
    for bb_name, bb in BACKBONES.items():
        for corpus_name, corpus in CORPORA.items():
            path = os.path.join('configs', bb_name, corpus['group'],
                                'ao_pt_cosine', corpus_name + '.yaml')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(render(bb_name, bb, corpus_name, corpus, port))
            made.append((path, port, f'{bb_name}_{corpus_name}_cos200k'))
            port += 1
    for p, prt, tag in made:
        print(f"  {prt}  {tag:44s} {p}")
    print(f"\n{len(made)} configs, ports {BASE_PORT}-{port - 1}")


if __name__ == '__main__':
    main()
