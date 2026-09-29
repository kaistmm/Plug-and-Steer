#!/usr/bin/env python3

import sys, time, os, argparse, pickle, subprocess, glob, cv2, logging
import numpy as np
import torch
from shutil import rmtree

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from scenedetect import open_video, SceneManager, ContentDetector
from tqdm import tqdm

from scipy.interpolate import interp1d
from scipy import signal

from detectors import S3FD

VIDEO_EXTS = {'.mp4', '.avi', '.mkv', '.mov'}


def default_ref(input_video):
  base = os.path.basename(os.path.normpath(input_video))
  stem, ext = os.path.splitext(base)
  if ext.lower() in VIDEO_EXTS:
    return stem
  return base

# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description = "FaceTracker")
parser.add_argument('--data_dir',       type=str, default='work/preprocessed', help='Preprocess output directory')
parser.add_argument('--videofile',      type=str, default='work/sample/sample_6_debate.mp4',   help='Input video file')
parser.add_argument('--ref',            type=str, default=None,
                    help='Output identifier / subfolder name (default: basename of --videofile)')
parser.add_argument('--facedet_scale',  type=float, default=0.25, help='Scale factor for face detection')
parser.add_argument('--crop_scale',     type=float, default=0.40, help='Scale bounding box')
parser.add_argument('--min_track',      type=int, default=50,  help='Minimum facetrack duration')
parser.add_argument('--frame_rate',     type=int, default=25,   help='Frame rate')
parser.add_argument('--num_failed_det', type=int, default=25,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--min_face_size',  type=int, default=100,  help='Minimum face size in pixels')
parser.add_argument('--overwrite',      action='store_true',    help='Overwrite existing output directories')
opt = parser.parse_args()

opt.ref = opt.ref if opt.ref is not None else default_ref(opt.videofile)

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'frames_dir',os.path.join(opt.data_dir,'pyframes'))

# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========

def bb_intersection_over_union(boxA, boxB):

  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interArea = max(0, xB - xA) * max(0, yB - yA)

  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

  iou = interArea / float(boxAArea + boxBArea - interArea)

  return iou

# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========

def track_shot(opt,scenefaces):

  iouThres  = 0.5     # Minimum IOU between consecutive face detections
  tracks    = []

  while True:
    track     = []
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

      framenum    = np.array([ f['frame'] for f in track ])
      bboxes      = np.array([np.array(f['bbox']) for f in track])

      frame_i   = np.arange(framenum[0],framenum[-1]+1)

      bboxes_i    = []
      for ij in range(0,4):
        interpfn  = interp1d(framenum, bboxes[:,ij])
        bboxes_i.append(interpfn(frame_i))
      bboxes_i  = np.stack(bboxes_i, axis=1)

      if max(np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1])) > opt.min_face_size:
        tracks.append({'frame':frame_i,'bbox':bboxes_i})

  return tracks

# ========== ========== ========== ==========
# # VIDEO CROP AND SAVE
# ========== ========== ========== ==========

def crop_video(opt,track,cropfile):

  flist = glob.glob(os.path.join(opt.frames_dir,opt.ref,'*.jpg'))
  flist.sort()

  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  vOut = cv2.VideoWriter(cropfile+'t.avi', fourcc, opt.frame_rate, (224,224))

  dets = {'x':[], 'y':[], 's':[]}

  for det in track['bbox']:

    dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2)
    dets['y'].append((det[1]+det[3])/2) # crop center x
    dets['x'].append((det[0]+det[2])/2) # crop center y

  # Smooth detections
  dets['s'] = signal.medfilt(dets['s'],kernel_size=13)
  dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
  dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

  for fidx, frame in enumerate(track['frame']):

    cs  = opt.crop_scale

    bs  = dets['s'][fidx]   # Detection box size
    bsi = int(bs*(1+2*cs))  # Pad videos by this amount

    image = cv2.imread(flist[frame])

    frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
    my  = dets['y'][fidx]+bsi  # BBox center Y
    mx  = dets['x'][fidx]+bsi  # BBox center X

    face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]

    vOut.write(cv2.resize(face,(224,224)))

  audiopath   = cropfile + '_tmp.wav'
  audiostart  = (track['frame'][0])/opt.frame_rate
  audioend    = (track['frame'][-1]+1)/opt.frame_rate

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
  command = ["ffmpeg", "-y", "-loglevel", "error", "-i", cropfile+'t.avi', "-i", audiopath,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", cropfile+'.mp4']
  subprocess.run(command, check=True)

  logger.info('Written %s.mp4', cropfile)

  os.remove(cropfile+'t.avi')
  os.remove(audiopath)

  logger.info('Mean pos: x %.2f y %.2f s %.2f', np.mean(dets['x']),np.mean(dets['y']),np.mean(dets['s']))

  return {'track':track, 'proc_track':dets}

# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========

def inference_video(opt):

  if torch.cuda.is_available():
    device = 'cuda:0'
  else:
    device = 'cpu'
  DET = S3FD(device=device)

  flist = glob.glob(os.path.join(opt.frames_dir,opt.ref,'*.jpg'))
  flist.sort()

  dets = []

  with tqdm(enumerate(flist), total=len(flist), desc='Detecting faces') as pbar:
    for fidx, fname in pbar:

      image = cv2.imread(fname)

      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])

      dets.append([])
      for bbox in bboxes:
        dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})

      pbar.set_postfix(dets=len(dets[-1]))

  savepath = os.path.join(opt.work_dir,opt.ref,'faces.pckl')

  with open(savepath, 'wb') as fil:
    pickle.dump(dets, fil)

  return dets

# ========== ========== ========== ==========
# # SCENE DETECTION
# ========== ========== ========== ==========

def scene_detect(opt):

  video_path = os.path.join(opt.avi_dir,opt.ref,'video.avi')
  video = open_video(video_path)

  scene_manager = SceneManager()
  scene_manager.add_detector(ContentDetector())
  scene_manager.detect_scenes(video)

  scene_list = scene_manager.get_scene_list()

  savepath = os.path.join(opt.work_dir,opt.ref,'scene.pckl')

  if not scene_list:
    scene_list = [(video.base_timecode, video.base_timecode + video.duration)]

  with open(savepath, 'wb') as fil:
    pickle.dump(scene_list, fil)

  logger.info('%s - scenes detected %d', video_path, len(scene_list))

  return scene_list


# ========== ========== ========== ==========
# # EXECUTE DEMO
# ========== ========== ========== ==========

# ========== DELETE EXISTING DIRECTORIES ==========

for d in [opt.work_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
  path = os.path.join(d, opt.ref)
  if os.path.exists(path):
    if not opt.overwrite:
      sys.exit(f"Output directory already exists: {path}. Use --overwrite to overwrite.")
    logger.warning('Overwriting existing directory: %s', path)
    rmtree(path)

existing_scene_dirs = sorted(glob.glob(os.path.join(opt.data_dir, opt.ref + '_scene*')))
if existing_scene_dirs:
  if not opt.overwrite:
    sys.exit(f"Output directory already exists: {existing_scene_dirs[0]}. Use --overwrite to overwrite.")
  for path in existing_scene_dirs:
    logger.warning('Overwriting existing directory: %s', path)
    rmtree(path)

# ========== MAKE NEW DIRECTORIES ==========

for d in [opt.work_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
  os.makedirs(os.path.join(d, opt.ref), exist_ok=True)

# ========== CONVERT VIDEO AND EXTRACT FRAMES ==========

logger.info('Converting video to 25fps: %s', opt.videofile)
command = ["ffmpeg", "-y", "-loglevel", "error", "-i", opt.videofile, "-qscale:v", "2", "-async", "1", "-r", "25",
           os.path.join(opt.avi_dir, opt.ref, 'video.avi')]
subprocess.run(command, check=True)

logger.info('Extracting frames from video')
command = ["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(opt.avi_dir, opt.ref, 'video.avi'),
           "-qscale:v", "2", "-threads", "1", "-f", "image2",
           os.path.join(opt.frames_dir, opt.ref, '%06d.jpg')]
subprocess.run(command, check=True)

logger.info('Extracting audio from video')
command = ["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(opt.avi_dir, opt.ref, 'video.avi'),
           "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
           os.path.join(opt.tmp_dir, opt.ref, 'audio.wav')]
subprocess.run(command, check=True)

# ========== FACE DETECTION ==========

faces = inference_video(opt)

# ========== SCENE DETECTION ==========

scene = scene_detect(opt)

# ========== FACE TRACKING + CROP ==========

for scene_idx, shot in enumerate(scene, start=1):

  if shot[1].get_frames() - shot[0].get_frames() < opt.min_track:
    continue

  tracks = track_shot(opt, faces[shot[0].get_frames():shot[1].get_frames()])
  if not tracks:
    continue

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
  logger.info('Writing shared mix audio %s (%.3f–%.3f s)', audiopath, audiostart, audioend)
  command = ["ffmpeg", "-y", "-loglevel", "error", "-i",
             os.path.join(opt.tmp_dir, opt.ref, 'audio.wav'),
             "-ss", "%.3f" % audiostart, "-to", "%.3f" % audioend,
             audiopath]
  subprocess.run(command, check=True)

# ========== CLEANUP INTERMEDIATE OUTPUTS ==========

for d in [opt.work_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
  if os.path.exists(d):
    logger.info('Removing intermediate directory: %s', d)
    rmtree(d)
