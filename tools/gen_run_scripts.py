"""Regenerate run_scripts/ from configs/.

One script per config, mirroring the tree, with the tag derived from the path so
scripts and experiment directories cannot drift apart. Re-run after adding
configs:

    python tools/gen_run_scripts.py
"""

import os
import glob
import argparse

from ruamel.yaml import YAML

_yaml = YAML(typ='safe', pure=True)

def make_tag(rel):
    """The tag is the config's own path, one hyphen per directory level.

        configs/{backbone}/{corpus}/{setting}/{variant}.yaml
            ->  {backbone}-{corpus}-{setting}-{variant}

    Levels are separated by hyphens and never contain one, so splitting a tag on '-'
    recovers the config path -- and with it the run script and the inference tree,
    which mirror the same four levels. Checkpoints are stored flat under this name.
    """
    parts = rel[len('configs/'):-len('.yaml')].split('/')
    if len(parts) != 4:
        return None
    if any('-' in p for p in parts):
        raise ValueError(f'path level contains a hyphen, which would break tag parsing: {rel}')
    return '-'.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', default='configs')
    args = ap.parse_args()

    made = skipped = 0
    written = set()
    for p in sorted(glob.glob(os.path.join(args.configs, '**', '*.yaml'), recursive=True)):
        # configs/_backup_pre_reorg/ is a pre-reorganisation snapshot, recovery only.
        if os.sep + '_backup_pre_reorg' + os.sep in p:
            continue
        c = _yaml.load(open(p))
        tag = make_tag(p)
        if not isinstance(c, dict) or 'trainer' not in c or tag is None:
            skipped += 1
            continue

        node = c['DDP']['devices']['node1']
        # An explicit `gpus` list in the config wins. Otherwise a config may state
        # its GPU count as trainer.gpus_hint -- the ao_pt set carries the count the
        # archived run actually used, so a regenerate does not silently drop it.
        # With neither, ao_pt_cosine documents its own count in the header (2) and
        # everything else was single-GPU historically.
        default_gpu = 2 if '/ao_pt_cosine/' in p else 1
        n_gpu = (len(node.get('gpus') or [])
                 or c['trainer'].get('gpus_hint')
                 or default_gpu)
        devices = ','.join(str(i) for i in range(n_gpu))

        sh = p.replace(args.configs + '/', 'run_scripts/', 1).replace('.yaml', '.sh')
        os.makedirs(os.path.dirname(sh), exist_ok=True)
        with open(sh, 'w') as f:
            f.write('#!/usr/bin/env bash\n')
            f.write(f'# {tag}\n')
            f.write('# Edit CUDA_VISIBLE_DEVICES to pick GPUs; train.py takes it as-is.\n')
            f.write(f'CUDA_VISIBLE_DEVICES={devices} python train.py \\\n')
            f.write(f'  --config ./{p} \\\n')
            f.write(f'  --tag {tag}\n')
        os.chmod(sh, 0o755)
        written.add(os.path.normpath(sh))
        made += 1

    # A script whose config was retired would otherwise linger and still be runnable,
    # pointing at a config that no longer exists.
    removed = 0
    for sh in glob.glob(os.path.join('run_scripts', '**', '*.sh'), recursive=True):
        if os.path.normpath(sh) not in written:
            os.remove(sh)
            removed += 1
    for root, _, _ in sorted(os.walk('run_scripts'), reverse=True):
        if root != 'run_scripts' and not os.listdir(root):
            os.rmdir(root)

    print(f'run scripts generated: {made}  (skipped {skipped} non-experiment configs'
          + (f', removed {removed} orphaned' if removed else '') + ')')


if __name__ == '__main__':
    main()
