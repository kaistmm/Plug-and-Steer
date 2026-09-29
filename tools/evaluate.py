"""
Evaluation tool: load a trained model, run on test set, compute metrics.

Metrics: SI-SNR, SI-SNRi, PESQ (optional), DNSMOS (optional), NISQA (optional)

Usage:
    python tools/evaluate.py --exp_dir ./exps/tfgridnet-libri2mix100-ao_pt-base --gpu_id 0
    python tools/evaluate.py --exp_dir ./exps/tfgridnet-libri2mix100-ao_pt-base --gpu_id 0 --metrics sisnr sisnri pesq
    python tools/evaluate.py --exp_dir ./exps/tfgridnet-libri2mix100-ao_pt-base --gpu_id 0 --dataset_config ./configs/tfgridnet/libri3mix360/ao_pt/base.yaml
    python tools/evaluate.py --exp_dir ./exps/tfgridnet-libri2mix100-ao_pt-base --gpu_id 0 --save_json

`--exp_dir` is any folder holding `config.yaml` and `ckpt/`.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.config_io import load_config  # noqa: E402
import torch
import numpy as np
import json
from itertools import permutations
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def sisnr(x, s, eps=1e-8):
    """
    SI-SNR between estimate x and reference s.
    input:  x (T,), s (T,)  — 1-D tensors
    output: scalar
    """
    x_zm = x - torch.mean(x)
    s_zm = s - torch.mean(s)

    dot = torch.sum(x_zm * s_zm)
    s_energy = torch.sum(s_zm ** 2) + eps

    proj = dot * s_zm / s_energy
    noise = x_zm - proj

    return 10 * torch.log10(
        torch.sum(proj ** 2) / (torch.sum(noise ** 2) + eps) + eps
    )


def sisnri(mix, est, ref, eps=1e-8):
    """SI-SNR improvement: SI-SNR(est, ref) - SI-SNR(mix, ref)."""
    return sisnr(est, ref, eps) - sisnr(mix, ref, eps)


def pit_match(ests, refs, metric_fn):
    """
    Find best PIT permutation and return per-source metric values.
    ests: list of (T,) tensors
    refs: list of (T,) tensors
    metric_fn: callable(est, ref) -> scalar
    returns: best_scores (list of scalars), best_perm (tuple)
    """
    num_spks = len(refs)
    best_total = float('-inf')
    best_perm = None
    best_scores = None

    for perm in permutations(range(num_spks)):
        scores = [metric_fn(ests[s], refs[t]) for s, t in enumerate(perm)]
        total = sum(s.item() for s in scores)
        if total > best_total:
            best_total = total
            best_perm = perm
            best_scores = [s.item() for s in scores]

    return best_scores, best_perm


# ──────────────────────────────────────────────
# Model / data loading (same pattern as inference.py)
# ──────────────────────────────────────────────

def resolve_exp_dir(exp_dir):
    """Absolute path of a checkpoint folder holding `config.yaml` and `ckpt/`."""
    exp_dir = os.path.abspath(exp_dir)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f'Checkpoint folder not found: {exp_dir}')
    if not os.path.isfile(os.path.join(exp_dir, 'config.yaml')):
        raise FileNotFoundError(f'No config.yaml in checkpoint folder: {exp_dir}')
    return exp_dir


def load_model(exp_dir, gpu_id=0, ckpt='best'):
    exp_dir = resolve_exp_dir(exp_dir)
    config_path = os.path.join(exp_dir, 'config.yaml')
    if ckpt == 'best':
        ckpt_path = os.path.join(exp_dir, 'ckpt', 'best.pt')
    elif ckpt == 'last':
        ckpt_path = os.path.join(exp_dir, 'ckpt', 'last.pt')
    else:
        ckpt_path = os.path.join(exp_dir, 'ckpt', ckpt)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    device = f'cuda:{gpu_id}'

    config = load_config(config_path)

    _model = importlib.import_module(
        'src.model.' + config['model']['name']
    ).__getattribute__('Main_Model')
    model = _model(**config['model']['config']).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    logger.info(f'Loaded model from {ckpt_path}')
    return model, config


def load_dataloader(config_path):
    config = load_config(config_path)

    data_config = config['datasets']
    data_config['validation']['dataset']['config']['chunk_size'] = -1

    _valset = importlib.import_module(
        'src.datasets.dataset.' + data_config['validation']['dataset']['name']
    ).__getattribute__('Datasets')
    valset = _valset(**data_config['validation']['dataset']['config'])

    test_loader = DataLoader(
        valset,
        batch_size=1,
        **data_config['validation']['loader'],
    )

    logger.info(f'Loaded dataset: {len(valset)} samples')
    return test_loader


# ──────────────────────────────────────────────
# Optional metric loaders (lazy import to avoid hard dependency)
# ──────────────────────────────────────────────

def get_pesq_fn(sr, device):
    from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
    return PerceptualEvaluationSpeechQuality(sr, 'wb').to(device)


def get_dnsmos_fn(sr, device):
    from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
    return DeepNoiseSuppressionMeanOpinionScore(sr, False, device=device)


def get_nisqa_fn(sr, device):
    from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment
    return NonIntrusiveSpeechQualityAssessment(sr).to(device)


def main():
    parser = ArgumentParser(description='Evaluate a trained separation model')
    parser.add_argument('--exp_dir', required=True, type=str,
                        help='Checkpoint folder holding config.yaml and ckpt/')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--ckpt', default='best', type=str)
    parser.add_argument('--dataset_config', default=None, type=str)
    parser.add_argument('--metrics', nargs='+', default=['sisnr', 'sisnri'],
                        choices=['sisnr', 'sisnri', 'pesq', 'dnsmos', 'nisqa'],
                        help='Metrics to compute')
    parser.add_argument('--max_samples', default=-1, type=int)
    parser.add_argument('--save_json', action='store_true',
                        help='Save per-sample results to JSON')
    parser.add_argument('--sample_rate', default=16000, type=int)
    args = parser.parse_args()

    device = f'cuda:{args.gpu_id}'

    try:
        args.exp_dir = resolve_exp_dir(args.exp_dir)
    except FileNotFoundError as err:
        sys.exit(str(err))

    model, config = load_model(args.exp_dir, args.gpu_id, args.ckpt)

    dataset_config = args.dataset_config or os.path.join(args.exp_dir, 'config.yaml')
    test_loader = load_dataloader(dataset_config)

    # Initialize optional metrics
    metric_fns = {}
    if 'pesq' in args.metrics:
        metric_fns['pesq'] = get_pesq_fn(args.sample_rate, device)
    if 'dnsmos' in args.metrics:
        metric_fns['dnsmos'] = get_dnsmos_fn(args.sample_rate, device)
    if 'nisqa' in args.metrics:
        metric_fns['nisqa'] = get_nisqa_fn(args.sample_rate, device)

    # Accumulators
    totals = {m: 0.0 for m in args.metrics}
    steps = 0
    per_sample_results = {}

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, dynamic_ncols=True)):
            if args.max_samples > 0 and idx >= args.max_samples:
                break

            mix = batch[0].to(device)
            refs = [r.to(device) for r in batch[1]]

            output = model(mix)

            # Squeeze batch dim: (1, T) -> (T,)
            mix_1d = mix.squeeze(0)
            ests_1d = [o.squeeze(0) for o in output]
            refs_1d = [r.squeeze(0) for r in refs]

            # PIT-aligned SI-SNR
            scores, best_perm = pit_match(ests_1d, refs_1d, sisnr)
            aligned_ests = [ests_1d[s] for s, _ in enumerate(best_perm)]

            sample_result = {}

            if 'sisnr' in args.metrics:
                avg_sisnr = np.mean(scores)
                totals['sisnr'] += avg_sisnr
                sample_result['sisnr'] = avg_sisnr

            if 'sisnri' in args.metrics:
                sisnri_scores = [
                    sisnri(mix_1d, ests_1d[s], refs_1d[t]).item()
                    for s, t in enumerate(best_perm)
                ]
                avg_sisnri = np.mean(sisnri_scores)
                totals['sisnri'] += avg_sisnri
                sample_result['sisnri'] = avg_sisnri

            # Per-source optional metrics (computed on first aligned source)
            if 'pesq' in args.metrics:
                pesq_val = metric_fns['pesq'](
                    aligned_ests[0].unsqueeze(0), refs_1d[best_perm[0]].unsqueeze(0)
                ).item()
                totals['pesq'] += pesq_val
                sample_result['pesq'] = pesq_val

            if 'dnsmos' in args.metrics:
                dnsmos_val = metric_fns['dnsmos'](aligned_ests[0].unsqueeze(0))[-1].item()
                totals['dnsmos'] += dnsmos_val
                sample_result['dnsmos'] = dnsmos_val

            if 'nisqa' in args.metrics:
                nisqa_val = metric_fns['nisqa'](aligned_ests[0].unsqueeze(0))[0].item()
                totals['nisqa'] += nisqa_val
                sample_result['nisqa'] = nisqa_val

            per_sample_results[f'sample_{idx:05d}'] = sample_result
            steps += 1

    # Print summary
    logger.info(f'===== Results ({steps} samples) =====')
    for m in args.metrics:
        avg = totals[m] / steps if steps > 0 else 0
        logger.info(f'  {m:>8s}: {avg:.4f}')

    if args.save_json:
        json_path = os.path.join(args.exp_dir, 'eval_results.json')
        summary = {m: totals[m] / steps for m in args.metrics}
        save_data = {'summary': summary, 'per_sample': per_sample_results}
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        logger.info(f'Results saved to {json_path}')


if __name__ == '__main__':
    main()
