# In-the-wild inference

Run AV target speaker extraction on an arbitrary video: face tracking and
cropping, then the steered separation, in one command.

## Requirements

- `ffmpeg` on `PATH`
- S3FD weights — see [`detectors/README.md`](detectors/README.md)
- A checkpoint folder holding `config.yaml` and `ckpt/best.pt`
- A CUDA device

## Usage

```bash
conda activate plugandsteer
cd tools/inthewild_inference

python run_full_pipeline.py \
    --videofile ./work/sample/sample_6_debate.mp4 \
    --exp_dir /path/to/mossformer2-clearer-av_swap-layer23
```

`--exp_dir` is any folder laid out like this — it does not have to live under
`exps/`:

```
<exp_dir>/config.yaml
<exp_dir>/ckpt/best.pt
```

Pass `--exp_dir` yourself. Use the best checkpoint linked from the README, or a
folder you trained. The script's fallback path is not part of this release.

That is the whole pipeline: the video is resampled to 25 fps, split into scenes,
every face track is cropped to 224×224, and the model runs once per track
against the scene's mix audio.

Outputs (one folder per scene, one subfolder per face track):

```
work/preprocessed/<video>_scene001/00000.mp4   # face crop, model input
work/preprocessed/<video>_scene001/audio.wav   # scene mix audio
work/result/<video>_scene001/mix.wav
work/result/<video>_scene001/00000/pred_1.wav  # target steered by 00000.mp4
work/result/<video>_scene001/00000/pred_2.wav  # remaining channel
work/result/<video>_scene001/00000.mp4         # only with --save_videos
```

`pred_1.wav` is the speaker whose face is in that track; `pred_2.wav` is the
other separated channel. `--save_videos` additionally writes one mp4 per track:
the face crop with `pred_1.wav` as its audio, which is the quickest way to check
whether the steering picked the right speaker.

## Options

Defaults are fine for most clips. The ones worth knowing:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--videofile` | `work/sample/sample_6_debate.mp4` | Input video |
| `--output_dir` | `work/result` | Where separated audio goes |
| `--data_dir` | `work/preprocessed` | Where face crops go |
| `--exp_dir` | pass explicitly | Checkpoint folder (`config.yaml` + `ckpt/`). The fallback path is not in this release |
| `--ckpt` | `best` | `best`, `last`, or a filename |
| `--gpu_id` | `0` | CUDA device |
| `--save_videos` | off | Also save `<track>.mp4` (face crop + extracted target) |
| `--overwrite` | off | Required when rerunning the same video |
| `--min_track` | `50` | Minimum face track length in frames |
| `--min_face_size` | `100` | Minimum face size in pixels |
| `--max_n_track` | no limit | Keep at most N tracks per scene, biggest faces first |

If no face track survives, lower `--min_track` and `--min_face_size`.

`python run_full_pipeline.py --help` lists the rest.

## Other scripts

`run_full_pipeline.py` merges two scripts that can also be run on their own —
useful for inspecting the crops before spending GPU time, or for reusing crops
across checkpoints:

```bash
python run_pipeline.py --videofile ./work/sample/sample_6_debate.mp4
python inference.py \
    --input_video ./work/preprocessed/sample_6_debate_scene001 \
    --exp_dir /path/to/mossformer2-clearer-av_swap-layer23
```

`download_yt.py` fetches a clip into `work/sample/` (needs a YouTube cookies
file):

```bash
python download_yt.py --url <YOUTUBE_URL> --start 00:55 --end 01:16
```

## Demo

`voxmm_result/` is a demo cut from overlap regions of VoxMM: the mixture and the separated tracks for each face.
