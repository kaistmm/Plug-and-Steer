"""
Inference tool: load a trained model, run separation on test set, save wav files.

Usage:
    python tools/inference.py --exp_dir ./exps/tfgridnet-libri2mix100-ao_pt-base --gpu_id 0
    python tools/inference.py --exp_dir ./exps/tfgridnet-libri2mix100-ao_pt-base --gpu_id 0 --ckpt best
    python tools/inference.py --exp_dir ./exps/tfgridnet-libri2mix100-ao_pt-base --gpu_id 0 --dataset_config ./configs/tfgridnet/libri3mix360/ao_pt/base.yaml
    python tools/inference.py --exp_dir ./exps/tfgridnet-libri2mix100-ao_pt-base --gpu_id 0 --max_samples 100

`--exp_dir` is any folder holding `config.yaml` and `ckpt/`.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.config_io import load_config  # noqa: E402
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def resolve_exp_dir(exp_dir):
    """Absolute path of a checkpoint folder holding `config.yaml` and `ckpt/`."""
    exp_dir = os.path.abspath(exp_dir)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f'Checkpoint folder not found: {exp_dir}')
    if not os.path.isfile(os.path.join(exp_dir, 'config.yaml')):
        raise FileNotFoundError(f'No config.yaml in checkpoint folder: {exp_dir}')
    return exp_dir


def load_model(exp_dir, gpu_id=0, ckpt='best'):
    """Load model from a checkpoint folder (`config.yaml` + `ckpt/*.pt`)."""
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
    # Skip training-time backbone preload; full weights load from ckpt below.
    model_config = dict(config['model']['config'])
    model_config['ckpt_path'] = None
    model = _model(**model_config).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    logger.info(f'Loaded model from {ckpt_path}')
    return model, config


def load_dataloader(config_path):
    """Load test dataloader from config yaml."""
    config = load_config(config_path)

    data_config = config['datasets']
    # Force full utterance for inference
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


def peak_normalize(wav):
    """Zero-mean and peak normalize a numpy waveform."""
    wav = wav - wav.mean()
    max_abs = np.max(np.abs(wav))
    if max_abs > 0:
        wav = wav / max_abs
    return wav


def main():
    parser = ArgumentParser(description='Run inference and save separated audio')
    parser.add_argument('--exp_dir', required=True, type=str,
                        help='Checkpoint folder holding config.yaml and ckpt/')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--ckpt', default='best', type=str,
                        help='Checkpoint: "best", "last", or filename (e.g., model00010.pt)')
    parser.add_argument('--dataset_config', default=None, type=str,
                        help='Override dataset config yaml (default: use experiment config)')
    parser.add_argument('--save_dir', default=None, type=str,
                        help='Output directory (default: {exp_dir}/inferences/)')
    parser.add_argument('--max_samples', default=-1, type=int,
                        help='Max samples to process (-1 for all)')
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

    if args.save_dir is None:
        save_dir = os.path.join(args.exp_dir, 'inferences')
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f'Saving outputs to {save_dir}')

    processed = 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, dynamic_ncols=True)):
            if args.max_samples > 0 and idx >= args.max_samples:
                break

            # Support both (mix, ref) and (mix, ref, ...) formats
            mix = batch[0].to(device)

            output = model(mix)  # list of (1, T) tensors

            # Get filename from dataset
            try:
                mix_path = test_loader.dataset.mix_list[idx]
                base_name = os.path.splitext(os.path.basename(mix_path))[0]
            except (AttributeError, IndexError):
                base_name = f'sample_{idx:05d}'

            sample_dir = os.path.join(save_dir, base_name)
            os.makedirs(sample_dir, exist_ok=True)

            for spk_idx, est in enumerate(output):
                wav = est.squeeze().cpu().numpy()
                wav = peak_normalize(wav)
                sf.write(
                    os.path.join(sample_dir, f'pred_{spk_idx + 1}.wav'),
                    wav,
                    args.sample_rate,
                )

            processed += 1

    logger.info(f'Done. Processed {processed} samples.')


if __name__ == '__main__':
    main()
