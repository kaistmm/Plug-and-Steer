"""Static check that every config resolves against the code.

Catches the drift that accumulates when configs and trainers are edited apart:
a renamed module, a trainer key nobody sets any more, a checkpoint path that no
longer exists. Import-only -- nothing is built on the GPU.

    python tools/validate_configs.py [--configs configs] [--check-paths]
"""

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.config_io import load_config  # noqa: E402

MODULE_DIRS = {
    'trainer':   'src/trainer',
    'model':     'src/model',
    'loss':      'src/loss',
    'dataset':   'src/datasets/dataset',
    'sampler':   'src/datasets/sampler',
    'optimizer': 'src/optimization/optimizer',
    'scheduler': 'src/optimization/scheduler',
}

# Keys each trainer reads without a default.
REQUIRED_TRAINER_KEYS = {
    'basic_separation_trainer':      ['epoch', 'early_stop', 'clip_norm', 'loss'],
    'basic_separation_trainer_swap': ['epoch', 'early_stop', 'clip_norm', 'loss'],
    'av_trainer':                    ['epoch', 'early_stop', 'clip_norm', 'loss'],
    'av_gate_trainer':               ['epoch', 'early_stop', 'clip_norm', 'loss',
                                      'gate_loss_weight'],
}


def check(path, check_paths=False):
    problems = []
    try:
        c = load_config(path)
    except Exception as e:
        return [f'unparseable: {type(e).__name__}: {e}']
    if not isinstance(c, dict) or 'trainer' not in c:
        return []

    t = c['trainer']
    refs = [('trainer', t.get('name')),
            ('loss', t.get('loss', {}).get('name')),
            ('optimizer', c.get('optim', {}).get('name')),
            ('scheduler', c.get('scheduler', {}).get('name'))]
    if 'model' in c:
        refs.append(('model', c['model'].get('name')))
    for split in ('train', 'validation'):
        d = c.get('datasets', {}).get(split, {})
        refs.append(('dataset', d.get('dataset', {}).get('name')))
        refs.append(('sampler', d.get('sampler', {}).get('name')))

    for kind, name in refs:
        if name is None:
            problems.append(f'missing {kind}.name')
        elif not os.path.exists(os.path.join(MODULE_DIRS[kind], name + '.py')):
            problems.append(f'{kind} module not found: {MODULE_DIRS[kind]}/{name}.py')

    for key in REQUIRED_TRAINER_KEYS.get(t.get('name'), []):
        if key not in t:
            problems.append(f'trainer.{key} is required by {t["name"]} but absent')

    if check_paths:
        ckpt = c.get('model', {}).get('config', {}).get('ckpt_path')
        if ckpt and not os.path.exists(ckpt):
            problems.append(f'ckpt_path does not exist: {ckpt}')
        for split in ('train', 'validation'):
            dc = c.get('datasets', {}).get(split, {}).get('dataset', {}).get('config', {})
            for key in ('mix_scp', 'target_scp'):
                val = dc.get(key)
                for p in ([val] if isinstance(val, str) else (val or [])):
                    if p and not os.path.exists(p):
                        problems.append(f'{split}.{key} missing: {p}')
    return problems


def warnings_for(path):
    """Non-fatal notes: valid configs whose behaviour is worth being aware of."""
    notes = []
    try:
        c = load_config(path)
    except Exception:
        return notes
    if not isinstance(c, dict) or 'trainer' not in c:
        return notes
    t = c['trainer']
    if t.get('early_stop', 0) < t.get('epoch', 0):
        notes.append(f"early_stop ({t.get('early_stop')}) < epoch ({t.get('epoch')}): "
                     f"the run can terminate before `epoch` and release its GPU")
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', default='configs')
    ap.add_argument('--check-paths', action='store_true',
                    help='also verify scp / checkpoint files exist on this machine')
    ap.add_argument('--warnings', action='store_true',
                    help='also list non-fatal notes')
    args = ap.parse_args()

    # configs/_backup_pre_reorg/ is a snapshot taken before the 2026-08-26
    # reorganisation. It is kept for recovery, not for running, so it is not validated.
    files = [f for f in sorted(glob.glob(os.path.join(args.configs, '**', '*.yaml'), recursive=True))
             if os.sep + '_backup_pre_reorg' + os.sep not in f]
    bad, warned = 0, 0
    for f in files:
        problems = check(f, args.check_paths)
        notes = warnings_for(f) if args.warnings else []
        if problems or notes:
            print(f'\n{f}')
        for p in problems:
            print(f'    ERROR   {p}')
        for n in notes:
            print(f'    warning {n}')
        bad += bool(problems)
        warned += bool(notes)

    print(f'\n{len(files) - bad}/{len(files)} configs OK'
          + (f', {bad} with errors' if bad else '')
          + (f', {warned} with warnings' if warned else ''))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
