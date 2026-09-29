#!/usr/bin/env python3
"""
End-to-end in-the-wild AV target speaker extraction.

Merges `run_pipeline.py` (scene detection, face tracking, face crops + mix wav)
and `inference.py` (AV model pass per face track) into one command, so a raw
video needs a single argument:

    conda activate clearer
    cd tools/inthewild_inference
    python run_full_pipeline.py --videofile ./work/sample/sample_6_debate.mp4

Preprocessed face crops land in `{data_dir}/{ref}_scene###/` and separated
audio in `{output_dir}/{ref}_scene###/{track}/pred_*.wav`. With `--save_videos`,
each face crop is also remuxed with its extracted target speaker as
`{output_dir}/{ref}_scene###/{track}.mp4`.
"""

import argparse
import glob
import logging
import os
import pickle
import subprocess
import sys
from shutil import rmtree

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
from scenedetect import ContentDetector, SceneManager, open_video
from scipy import signal
from scipy.interpolate import interp1d
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
# Script dir on path so `detectors` / `inference` resolve when imported as a module
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from detectors import S3FD  # noqa: E402
from inference import (  # noqa: E402
    DEFAULT_EXP_DIR,
    align_video_to_audio,
    collect_videos,
    load_model,
    load_raw_video,
    mux_target_video,
    normalize_numpy_wav,
    peak_normalize,
    resolve_audio_path,
    resolve_exp_dir,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

VIDEO_EXTS = {'.mp4', '.avi', '.mkv', '.mov'}


def default_ref(videofile):
    base = os.path.basename(os.path.normpath(videofile))
    stem, ext = os.path.splitext(base)
    if ext.lower() in VIDEO_EXTS:
        return stem
    return base


# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========

def bb_intersection_over_union(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


def track_shot(opt, scenefaces):

    iouThres = 0.5  # Minimum IOU between consecutive face detections
    tracks = []

    while True:
        track = []
        for framefaces in scenefaces:
            for face in framefaces:
                if track == []:
                    track.append(face)
                    framefaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        framefaces.remove(face)
                        continue
                else:
                    break

        if track == []:
            break
        elif len(track) > opt.min_track:

            framenum = np.array([f['frame'] for f in track])
            bboxes = np.array([np.array(f['bbox']) for f in track])

            frame_i = np.arange(framenum[0], framenum[-1] + 1)

            bboxes_i = []
            for ij in range(0, 4):
                interpfn = interp1d(framenum, bboxes[:, ij])
                bboxes_i.append(interpfn(frame_i))
            bboxes_i = np.stack(bboxes_i, axis=1)

            if mean_face_size(bboxes_i) > opt.min_face_size:
                tracks.append({'frame': frame_i, 'bbox': bboxes_i})

    return tracks


def mean_face_size(bboxes):
    return max(
        np.mean(bboxes[:, 2] - bboxes[:, 0]),
        np.mean(bboxes[:, 3] - bboxes[:, 1]),
    )


def keep_largest_tracks(tracks, max_n_track):
    """Keep the `max_n_track` biggest faces, in their original track order."""

    if max_n_track is None or len(tracks) <= max_n_track:
        return tracks

    sizes = [mean_face_size(t['bbox']) for t in tracks]
    keep = sorted(sorted(range(len(tracks)), key=lambda i: sizes[i], reverse=True)[:max_n_track])

    logger.info(
        'Keeping %d of %d face track(s) by face size: %s',
        len(keep), len(tracks), ', '.join('%.0f px' % sizes[i] for i in keep),
    )
    return [tracks[i] for i in keep]


# ========== ========== ========== ==========
# # VIDEO CROP AND SAVE
# ========== ========== ========== ==========

def crop_video(opt, track, cropfile):

    flist = glob.glob(os.path.join(opt.frames_dir, opt.ref, '*.jpg'))
    flist.sort()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vOut = cv2.VideoWriter(cropfile + 't.avi', fourcc, opt.frame_rate, (224, 224))

    dets = {'x': [], 'y': [], 's': []}

    for det in track['bbox']:

        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)  # crop center x
        dets['x'].append((det[0] + det[2]) / 2)  # crop center y

    # Smooth detections
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

    for fidx, frame in enumerate(track['frame']):

        cs = opt.crop_scale

        bs = dets['s'][fidx]      # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount

        image = cv2.imread(flist[frame])

        frame = np.pad(
            image, ((bsi, bsi), (bsi, bsi), (0, 0)),
            'constant', constant_values=(110, 110),
        )
        my = dets['y'][fidx] + bsi  # BBox center Y
        mx = dets['x'][fidx] + bsi  # BBox center X

        face = frame[
            int(my - bs):int(my + bs * (1 + 2 * cs)),
            int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs)),
        ]

        vOut.write(cv2.resize(face, (224, 224)))

    audiopath = cropfile + '_tmp.wav'
    audiostart = (track['frame'][0]) / opt.frame_rate
    audioend = (track['frame'][-1] + 1) / opt.frame_rate

    vOut.release()

    # ========== CROP AUDIO FILE (temp, for mux only) ==========

    logger.info('Cropping audio track for %s', cropfile)
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i",
               os.path.join(opt.tmp_dir, opt.ref, 'audio.wav'),
               "-ss", "%.3f" % audiostart, "-to", "%.3f" % audioend,
               audiopath]
    subprocess.run(command, check=True)

    # ========== COMBINE AUDIO AND VIDEO FILES ==========

    logger.info('Merging audio and video for %s', cropfile)
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", cropfile + 't.avi', "-i", audiopath,
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", cropfile + '.mp4']
    subprocess.run(command, check=True)

    logger.info('Written %s.mp4', cropfile)

    os.remove(cropfile + 't.avi')
    os.remove(audiopath)

    logger.info(
        'Mean pos: x %.2f y %.2f s %.2f',
        np.mean(dets['x']), np.mean(dets['y']), np.mean(dets['s']),
    )

    return {'track': track, 'proc_track': dets}


# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========

def detect_faces(opt, device):

    DET = S3FD(device=device)

    flist = glob.glob(os.path.join(opt.frames_dir, opt.ref, '*.jpg'))
    flist.sort()

    dets = []

    with tqdm(enumerate(flist), total=len(flist), desc='Detecting faces') as pbar:
        for fidx, fname in pbar:

            image = cv2.imread(fname)

            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])

            dets.append([])
            for bbox in bboxes:
                dets[-1].append(
                    {'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]}
                )

            pbar.set_postfix(dets=len(dets[-1]))

    savepath = os.path.join(opt.work_dir, opt.ref, 'faces.pckl')

    with open(savepath, 'wb') as fil:
        pickle.dump(dets, fil)

    return dets


# ========== ========== ========== ==========
# # SCENE DETECTION
# ========== ========== ========== ==========

def scene_detect(opt):

    video_path = os.path.join(opt.avi_dir, opt.ref, 'video.avi')
    video = open_video(video_path)

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()

    savepath = os.path.join(opt.work_dir, opt.ref, 'scene.pckl')

    if not scene_list:
        scene_list = [(video.base_timecode, video.base_timecode + video.duration)]

    with open(savepath, 'wb') as fil:
        pickle.dump(scene_list, fil)

    logger.info('%s - scenes detected %d', video_path, len(scene_list))

    return scene_list


# ========== ========== ========== ==========
# # PREPROCESSING
# ========== ========== ========== ==========

def preprocess(opt, device):
    """Face-crop the input video per scene. Returns the scene folder paths."""

    # ========== DELETE EXISTING DIRECTORIES ==========

    for d in [opt.work_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
        path = os.path.join(d, opt.ref)
        if os.path.exists(path):
            if not opt.overwrite:
                sys.exit(
                    f"Output directory already exists: {path}. Use --overwrite to overwrite."
                )
            logger.warning('Overwriting existing directory: %s', path)
            rmtree(path)

    existing_scene_dirs = sorted(glob.glob(os.path.join(opt.data_dir, opt.ref + '_scene*')))
    if existing_scene_dirs:
        if not opt.overwrite:
            sys.exit(
                f"Output directory already exists: {existing_scene_dirs[0]}. "
                "Use --overwrite to overwrite."
            )
        for path in existing_scene_dirs:
            logger.warning('Overwriting existing directory: %s', path)
            rmtree(path)

    # ========== MAKE NEW DIRECTORIES ==========

    for d in [opt.work_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
        os.makedirs(os.path.join(d, opt.ref), exist_ok=True)

    # ========== CONVERT VIDEO AND EXTRACT FRAMES ==========

    logger.info('Converting video to 25fps: %s', opt.videofile)
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", opt.videofile,
               "-qscale:v", "2", "-async", "1", "-r", "25",
               os.path.join(opt.avi_dir, opt.ref, 'video.avi')]
    subprocess.run(command, check=True)

    logger.info('Extracting frames from video')
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i",
               os.path.join(opt.avi_dir, opt.ref, 'video.avi'),
               "-qscale:v", "2", "-threads", "1", "-f", "image2",
               os.path.join(opt.frames_dir, opt.ref, '%06d.jpg')]
    subprocess.run(command, check=True)

    logger.info('Extracting audio from video')
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i",
               os.path.join(opt.avi_dir, opt.ref, 'video.avi'),
               "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
               os.path.join(opt.tmp_dir, opt.ref, 'audio.wav')]
    subprocess.run(command, check=True)

    # ========== FACE DETECTION ==========

    faces = detect_faces(opt, device)

    # ========== SCENE DETECTION ==========

    scene = scene_detect(opt)

    # ========== FACE TRACKING + CROP ==========

    scene_dirs = []

    for scene_idx, shot in enumerate(scene, start=1):

        if shot[1].get_frames() - shot[0].get_frames() < opt.min_track:
            continue

        tracks = track_shot(opt, faces[shot[0].get_frames():shot[1].get_frames()])
        if not tracks:
            continue

        tracks = keep_largest_tracks(tracks, opt.max_n_track)

        scene_dir = os.path.join(opt.data_dir, '%s_scene%03d' % (opt.ref, scene_idx))
        os.makedirs(scene_dir, exist_ok=True)

        for track_idx, track in enumerate(tracks):
            cropfile = os.path.join(scene_dir, '%05d' % track_idx)
            crop_video(opt, track, cropfile)

        # One shared mix wav per scene folder (same source for all face tracks).
        frame_start = min(int(t['frame'][0]) for t in tracks)
        frame_end = max(int(t['frame'][-1]) for t in tracks)
        audiostart = frame_start / opt.frame_rate
        audioend = (frame_end + 1) / opt.frame_rate
        audiopath = os.path.join(scene_dir, 'audio.wav')
        logger.info('Writing shared mix audio %s (%.3f-%.3f s)', audiopath, audiostart, audioend)
        command = ["ffmpeg", "-y", "-loglevel", "error", "-i",
                   os.path.join(opt.tmp_dir, opt.ref, 'audio.wav'),
                   "-ss", "%.3f" % audiostart, "-to", "%.3f" % audioend,
                   audiopath]
        subprocess.run(command, check=True)

        scene_dirs.append(scene_dir)

    # ========== CLEANUP INTERMEDIATE OUTPUTS ==========

    for d in [opt.work_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
        if os.path.exists(d):
            logger.info('Removing intermediate directory: %s', d)
            rmtree(d)

    return scene_dirs


# ========== ========== ========== ==========
# # INFERENCE
# ========== ========== ========== ==========

def separate_scene(model, scene_dir, opt, device):
    """Run the AV model on every face track of one scene folder."""

    ref = os.path.basename(os.path.normpath(scene_dir))
    save_dir = os.path.join(opt.output_dir, ref)
    os.makedirs(save_dir, exist_ok=True)

    input_audio = resolve_audio_path(scene_dir)
    video_paths = collect_videos(scene_dir)
    logger.info(f'{ref}: {len(video_paths)} face video(s), mix audio {input_audio}')

    mix_raw, _ = librosa.load(input_audio, sr=opt.sample_rate)
    sf.write(os.path.join(save_dir, 'mix.wav'), peak_normalize(mix_raw.copy()), opt.sample_rate)

    mix = normalize_numpy_wav(mix_raw)
    mix_t = torch.from_numpy(mix).float().unsqueeze(0).to(device)  # (1, T)

    with torch.no_grad():
        for video_path in tqdm(video_paths, dynamic_ncols=True):
            track_name = os.path.splitext(os.path.basename(video_path))[0]
            track_dir = os.path.join(save_dir, track_name)
            os.makedirs(track_dir, exist_ok=True)

            visual = load_raw_video(video_path)
            visual = align_video_to_audio(visual, mix.shape[0], opt.sample_rate, opt.fps)
            visual_t = torch.from_numpy(visual).float().unsqueeze(0).to(device)

            output = model(mix_t, visual_t)  # list of (1, T)

            for spk_idx, est in enumerate(output):
                wav = peak_normalize(est.squeeze().detach().cpu().numpy())
                sf.write(
                    os.path.join(track_dir, f'pred_{spk_idx + 1}.wav'), wav, opt.sample_rate
                )

            logger.info(
                f'{ref}/{track_name}: saved {len(output)} streams '
                f'(pred_1 = visually steered target)'
            )

            if opt.save_videos:
                mux_target_video(
                    video_path,
                    os.path.join(track_dir, 'pred_1.wav'),
                    os.path.join(save_dir, f'{track_name}.mp4'),
                )

    return save_dir


# ========== ========== ========== ==========
# # MAIN
# ========== ========== ========== ==========

def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess a raw video and run in-the-wild AV inference in one pass'
    )
    parser.add_argument('--videofile', type=str, default='work/sample/sample_6_debate.mp4',
                        help='Input video file')
    parser.add_argument('--data_dir', type=str, default='work/preprocessed',
                        help='Preprocess output directory')
    parser.add_argument('--output_dir', type=str, default='work/result',
                        help='Root directory for separated audio outputs')
    parser.add_argument('--ref', type=str, default=None,
                        help='Output identifier / subfolder name '
                             '(default: basename of --videofile)')
    # Preprocessing
    parser.add_argument('--facedet_scale', type=float, default=0.25,
                        help='Scale factor for face detection')
    parser.add_argument('--crop_scale', type=float, default=0.40, help='Scale bounding box')
    parser.add_argument('--min_track', type=int, default=50, help='Minimum facetrack duration')
    parser.add_argument('--frame_rate', type=int, default=25, help='Frame rate')
    parser.add_argument('--num_failed_det', type=int, default=25,
                        help='Number of missed detections allowed before tracking is stopped')
    parser.add_argument('--min_face_size', type=int, default=200,
                        help='Minimum face size in pixels')
    parser.add_argument('--max_n_track', type=int, default=None,
                        help='Maximum number of face tracks per scene, biggest faces '
                             'first (default: no limit)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output directories')
    # Inference
    parser.add_argument('--exp_dir', type=str, default=DEFAULT_EXP_DIR,
                        help='Checkpoint folder holding config.yaml and ckpt/ '
                             f'(default: {DEFAULT_EXP_DIR})')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='best',
                        help='Checkpoint: "best", "last", or filename (e.g., model00010.pt)')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--save_videos', action='store_true',
                        help='Also save each face crop video with its extracted target '
                             'speaker as the audio track')
    return parser.parse_args()


def main():
    opt = parse_args()

    # Resolve I/O against the launch cwd; the inference stage later chdirs to
    # the repo root so relative paths inside the model resolve correctly.
    opt.videofile = os.path.abspath(opt.videofile)
    opt.data_dir = os.path.abspath(opt.data_dir)
    opt.output_dir = os.path.abspath(opt.output_dir)
    opt.ref = opt.ref if opt.ref is not None else default_ref(opt.videofile)

    if not os.path.isfile(opt.videofile):
        sys.exit(f'--videofile not found: {opt.videofile}')

    # Checked up front so a bad checkpoint path fails before preprocessing runs.
    try:
        opt.exp_dir = resolve_exp_dir(opt.exp_dir)
    except FileNotFoundError as err:
        sys.exit(f'{err}\nPass a checkpoint folder with --exp_dir')

    opt.avi_dir = os.path.join(opt.data_dir, 'pyavi')
    opt.tmp_dir = os.path.join(opt.data_dir, 'pytmp')
    opt.work_dir = os.path.join(opt.data_dir, 'pywork')
    opt.frames_dir = os.path.join(opt.data_dir, 'pyframes')

    device = f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu'
    logger.info('Using device %s', device)

    scene_dirs = preprocess(opt, device)
    if not scene_dirs:
        sys.exit(
            f'No usable face track found in {opt.videofile} '
            '(try lowering --min_track / --min_face_size)'
        )
    logger.info('Preprocessed %d scene(s): %s', len(scene_dirs),
                ', '.join(os.path.basename(d) for d in scene_dirs))

    if not torch.cuda.is_available():
        sys.exit('Inference needs a CUDA device; face crops are in ' + opt.data_dir)

    os.chdir(_REPO_ROOT)
    model, _ = load_model(opt.exp_dir, opt.gpu_id, opt.ckpt)

    save_dirs = [separate_scene(model, scene_dir, opt, device) for scene_dir in scene_dirs]

    logger.info('Done. %d scene(s) written under %s', len(save_dirs), opt.output_dir)


if __name__ == '__main__':
    main()
