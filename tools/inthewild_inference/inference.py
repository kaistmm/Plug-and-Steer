"""
In-the-wild AV target speaker extraction.

Takes a `run_pipeline.py` output folder (face-crop mp4s + `audio.wav`), runs a
trained AV model once per face track, and writes the steered target (and the
remaining separated channel) under `{output_dir}/{ref}/`.

Usage (from `tools/inthewild_inference`, conda env `clearer`):
    conda activate clearer
    cd tools/inthewild_inference
    python inference.py \
        --input_video ./work/preprocessed/sample_6_debate \
        --output_dir ./work/result \
        --gpu_id 0
"""

import importlib
import logging
import os
import subprocess
import sys
from argparse import ArgumentParser

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
# Repo root on path so `tools` / `src` resolve when launched from this folder
sys.path.insert(0, _REPO_ROOT)

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from tools.config_io import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

VIDEO_EXTS = {'.mp4', '.avi', '.mkv', '.mov'}

# Checkpoint folder used when --exp_dir is not given
DEFAULT_EXP_DIR = os.path.join(
    _REPO_ROOT, 'exps', '_release', 'mossformer2-clearer-av_swap-layer23'
)

# Same visual frontend stats as LRS2mix training
_ROI_SIZE = 112
_VIS_MEAN = 0.4161
_VIS_STD = 0.1688

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


def load_raw_video(visual_path):
    """Load face video to (T_v, 112, 112) grayscale, training-normalized."""
    capture = cv2.VideoCapture(visual_path)
    roi_sequence = []
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayed = cv2.resize(grayed, (_ROI_SIZE * 2, _ROI_SIZE * 2))
        half = _ROI_SIZE // 2
        grayed = grayed[
            _ROI_SIZE - half:_ROI_SIZE + half,
            _ROI_SIZE - half:_ROI_SIZE + half,
        ]
        roi_sequence.append(grayed)
    capture.release()

    if not roi_sequence:
        raise RuntimeError(f'No frames read from {visual_path}')

    visual = np.asarray(roi_sequence, dtype=np.float32) / 255.0
    visual = (visual - _VIS_MEAN) / _VIS_STD
    return visual


def normalize_numpy_wav(wav, eps=1e-8, std=None):
    mean = np.mean(wav, axis=-1, keepdims=True)
    if std is None:
        std = np.std(wav, axis=-1, keepdims=True)
    return (wav - mean) / (std + eps)


def peak_normalize(wav):
    wav = wav - wav.mean()
    max_abs = np.max(np.abs(wav))
    if max_abs > 0:
        wav = wav / max_abs
    return wav


def align_video_to_audio(visual, audio_len, sample_rate, fps):
    """Pad/truncate visual frames so T_v == round(T_audio / sr * fps)."""
    target_frames = max(1, int(round(audio_len / sample_rate * fps)))
    cur = visual.shape[0]
    if cur < target_frames:
        visual = np.pad(visual, ((0, target_frames - cur), (0, 0), (0, 0)))
    elif cur > target_frames:
        visual = visual[:target_frames]
    return visual


def collect_videos(input_video):
    """Return a sorted list of video paths from a file or directory."""
    if os.path.isfile(input_video):
        ext = os.path.splitext(input_video)[1].lower()
        if ext not in VIDEO_EXTS:
            raise ValueError(f'Unsupported video file: {input_video}')
        return [input_video]

    if os.path.isdir(input_video):
        videos = [
            os.path.join(input_video, name)
            for name in sorted(os.listdir(input_video))
            if os.path.splitext(name)[1].lower() in VIDEO_EXTS
            and os.path.isfile(os.path.join(input_video, name))
        ]
        if not videos:
            raise FileNotFoundError(f'No video files found in {input_video}')
        return videos

    raise FileNotFoundError(f'--input_video not found: {input_video}')


def resolve_audio_path(input_video):
    """Load mix wav from the same folder as run_pipeline crop outputs."""
    if os.path.isdir(input_video):
        audio_dir = input_video
    elif os.path.isfile(input_video):
        audio_dir = os.path.dirname(input_video)
    else:
        raise FileNotFoundError(f'--input_video not found: {input_video}')

    audio_path = os.path.join(audio_dir, 'audio.wav')
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f'Expected mix audio at {audio_path} (same folder as face video(s))'
        )
    return audio_path


def default_ref(input_video):
    base = os.path.basename(os.path.normpath(input_video))
    stem, ext = os.path.splitext(base)
    if ext.lower() in VIDEO_EXTS:
        return stem
    return base


def mux_target_video(face_video, target_wav, outfile):
    """Face crop video with the extracted target speaker as its only audio."""
    command = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', face_video, '-i', target_wav,
        '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac',
        '-shortest', outfile,
    ]
    subprocess.run(command, check=True)
    logger.info('Written %s', outfile)


def main():
    parser = ArgumentParser(
        description='In-the-wild AV inference on face-crop video(s) + mix wav'
    )
    parser.add_argument(
        '--input_video', type=str, default='./work/preprocessed/sample_6_debate',
        help='Single mp4 file, or a directory of face-track videos '
             '(audio.wav is read from the same folder; default: ./work/preprocessed)',
    )
    parser.add_argument(
        '--output_dir', type=str, default='./work/result',
        help='Root directory for outputs (default: ./work/result)',
    )
    parser.add_argument(
        '--ref', type=str, default=None,
        help='Output subfolder name under output_dir '
             '(default: basename of --input_video)',
    )
    parser.add_argument(
        '--exp_dir', type=str, default=DEFAULT_EXP_DIR,
        help='Checkpoint folder holding config.yaml and ckpt/ '
             f'(default: {DEFAULT_EXP_DIR})',
    )
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument(
        '--ckpt', default='best', type=str,
        help='Checkpoint: "best", "last", or filename (e.g., model00010.pt)',
    )
    parser.add_argument('--sample_rate', default=16000, type=int)
    parser.add_argument('--fps', default=25, type=int)
    parser.add_argument(
        '--save_videos', action='store_true',
        help='Also save each face crop video with its extracted target '
             'speaker as the audio track',
    )
    args = parser.parse_args()

    # Resolve I/O against the launch cwd, then move to repo root so relative
    # checkpoint / visual-frontend paths inside the model resolve correctly.
    args.input_video = os.path.abspath(args.input_video)
    args.output_dir = os.path.abspath(args.output_dir)
    args.exp_dir = resolve_exp_dir(args.exp_dir)
    input_audio = resolve_audio_path(args.input_video)
    os.chdir(_REPO_ROOT)

    ref = args.ref if args.ref is not None else default_ref(args.input_video)
    save_dir = os.path.join(args.output_dir, ref)
    os.makedirs(save_dir, exist_ok=True)

    video_paths = collect_videos(args.input_video)
    logger.info(f'Found {len(video_paths)} face video(s)')
    logger.info(f'Using mix audio: {input_audio}')
    logger.info(f'Saving outputs to {save_dir}')

    device = f'cuda:{args.gpu_id}'
    model, _ = load_model(args.exp_dir, args.gpu_id, args.ckpt)

    mix_raw, _ = librosa.load(input_audio, sr=args.sample_rate)
    sf.write(
        os.path.join(save_dir, 'mix.wav'),
        peak_normalize(mix_raw.copy()),
        args.sample_rate,
    )

    mix = normalize_numpy_wav(mix_raw)
    mix_t = torch.from_numpy(mix).float().unsqueeze(0).to(device)  # (1, T)

    with torch.no_grad():
        for video_path in tqdm(video_paths, dynamic_ncols=True):
            track_name = os.path.splitext(os.path.basename(video_path))[0]
            track_dir = os.path.join(save_dir, track_name)
            os.makedirs(track_dir, exist_ok=True)

            visual = load_raw_video(video_path)
            visual = align_video_to_audio(
                visual, mix.shape[0], args.sample_rate, args.fps
            )
            visual_t = torch.from_numpy(visual).float().unsqueeze(0).to(device)

            output = model(mix_t, visual_t)  # list of (1, T)

            for spk_idx, est in enumerate(output):
                wav = est.squeeze().detach().cpu().numpy()
                wav = peak_normalize(wav)
                sf.write(
                    os.path.join(track_dir, f'pred_{spk_idx + 1}.wav'),
                    wav,
                    args.sample_rate,
                )

            logger.info(
                f'{track_name}: saved {len(output)} streams '
                f'(pred_1 = visually steered target)'
            )

            if args.save_videos:
                mux_target_video(
                    video_path,
                    os.path.join(track_dir, 'pred_1.wav'),
                    os.path.join(save_dir, f'{track_name}.mp4'),
                )

    logger.info(f'Done. Processed {len(video_paths)} face track(s) → {save_dir}')


if __name__ == '__main__':
    main()
