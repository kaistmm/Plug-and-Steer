import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset

import librosa

from .epoch_subset import EpochSubsetMixin
import cv2

def _load_raw_video(visual_path):
    captureObj = cv2.VideoCapture(visual_path)
    roiSequence = []
    roiSize = 112
    mean = 0.4161
    std = 0.1688
    while (captureObj.isOpened()):
        ret, frame = captureObj.read()
        if ret == True:
            grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayed = cv2.resize(grayed, (roiSize*2,roiSize*2))
            grayed = grayed[int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)), int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2))]
            roiSequence.append(grayed)
        else:
            break
    captureObj.release()
    visual = np.asarray(roiSequence)/255.0
    visual = (visual - mean)/std

    return visual

def read_scp(scp_path):
    file_list = []
    with open(scp_path,'r') as f:
        for line in f.readlines():
            try:
                file_list.append(line.split(' ')[1].strip())
            except:
                print(f'Invalid scp file: {line}')
    
    return file_list

def read_av_scp(scp_path):
    file_list = []
    with open(scp_path,'r') as f:
        for line in f.readlines():
            try:
                file_list.append(line.split(' ')[2].strip())
            except:
                print(f'Invalid scp file: {line}')
    
    return file_list

# Not used
def loadaudio(file_path, max_len, sample_rate, tensor=True):
    audio, sample_rate = librosa.load(file_path, sr=sample_rate)

    if max_len!=-1:
        audiosize = audio.shape[0]
        
        if audiosize < max_len:
            shortage = max_len - audiosize
            audio = np.pad(audio, (0, shortage), 'constant')
        else:
            max_audio_start = audiosize - max_len
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+max_len]
        
        if audio.shape[0]!=max_len:
            raise Exception('Length error occur while loading the audio file')
        
    return torch.FloatTensor(audio) if tensor else audio
    

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


def normalize_numpy_wav(wav_numpy, eps=1e-8, std=None):
    mean = np.mean(wav_numpy, axis=-1, keepdims=True)
    if std is None:
        std = np.std(wav_numpy, axis=-1, keepdims=True)
    return (wav_numpy - mean) / (std + eps) 

class LRS2mix_Dataset(Dataset, EpochSubsetMixin):
    def __init__(
        self, 
        mix_scp, 
        target_scp, 
        sample_rate=16000, 
        chunk_size=32000, 
        debug_mode=False, 
        normalize=True, 
        return_video=False,
        fps=25,
        num_per_epoch=None,
        resample_per_epoch=True,
        seed=0,
        **kwargs
        ):
        
        self.mix_list = read_scp(mix_scp)
        self.target_list = [read_scp(scp) for scp in target_scp]
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.debug = debug_mode
        self.normalize = normalize
        
        self.return_video = return_video
        if self.return_video:
            self.fps = fps
            self.target_video_list = [read_av_scp(scp) for scp in target_scp]
            
        
        if os.path.basename(mix_scp).startswith('tr_'):
            self.split = 'train'
        elif os.path.basename(mix_scp).startswith('cv_'):
            self.split = 'val'
        else:
            self.split = 'test'
        
        self.init_epoch_subset(len(self.mix_list), num_per_epoch, resample_per_epoch, seed)

        try:
            for i in range(len(target_scp)):
                if len(self.mix_list)!=len(self.target_list[i]):
                    raise Exception('Length of mix and gt are different')
        except Exception as e:
            #raise Exception('Length of mix and gt are different')
            print(f"Error occur while comparing the mix and target scp: {str(e)}")
            
    def __getitem__(self, index):
        if self.return_video and self.split!="train":
            list_index = self.map_index(index // 2)
        else:
            list_index = self.map_index(index)
        mix, _ = librosa.load(self.mix_list[list_index], sr=self.sample_rate)
        target = [librosa.load(path_list[list_index], sr=self.sample_rate)[0] for path_list in self.target_list]

        if self.chunk_size!=-1:
            audio_size = mix.shape[0]
            if audio_size < self.chunk_size:
                audio_start = 0
                shortage = self.chunk_size - audio_size
                mix = np.pad(mix, (0, shortage), 'constant')
                target = [np.pad(t, (0, shortage), 'constant') for t in target]
            else:
                max_audio_start = audio_size - self.chunk_size
                audio_start = random.randint(0, max_audio_start)
                mix = mix[audio_start:audio_start+self.chunk_size]
                target = [t[audio_start:audio_start+self.chunk_size] for t in target]

        if self.normalize:  
            m_std = np.std(mix)
            mix = normalize_numpy_wav(mix, std=m_std)
            target = [normalize_numpy_wav(t, std=m_std) for t in target]
        
        if not self.return_video:
            return mix, target
        else:
            ## return mix, target_audio, target_video
            if self.split == 'train':
                if random.random()>0.5:
                    target_audio = [target[0], target[1]]
                    target_video_path = self.target_video_list[0][list_index]
                else:
                    target_audio = [target[1], target[0]]
                    target_video_path = self.target_video_list[1][list_index]
            else:
                if index % 2 == 0:
                    target_audio = [target[0], target[1]]
                    target_video_path = self.target_video_list[0][list_index]
                else:
                    target_audio = [target[1], target[0]]
                    target_video_path = self.target_video_list[1][list_index]
            
            target_video = _load_raw_video(target_video_path)
            if self.chunk_size!=-1:
                video_chunk_size = round(self.chunk_size / self.sample_rate * self.fps)
            else:
                video_chunk_size = round(target[0].shape[0] / self.sample_rate * self.fps)
                audio_start = 0

            video_start = round(audio_start / self.sample_rate * self.fps)
            target_video = target_video[video_start:]
            if target_video.shape[0] < video_chunk_size:
                video_shortage = video_chunk_size - target_video.shape[0]
                target_video = np.pad(target_video, ((0, video_shortage), (0,0), (0,0)))
            else:
                target_video = target_video[:video_chunk_size]
            # print(index, list_index, mix.shape, [(x.shape, x.max(), x.min()) for x in target_audio], target_video.shape, target_video.mean())

            return mix, target_audio, target_video
    
    def __len__(self):
        n = self.epoch_length(len(self.mix_list))
        # validation with video visits each mixture twice, once per target speaker
        data_len = n * 2 if (self.return_video and self.split != "train") else n
        return min(data_len, 100) if self.debug else data_len

        

'''
def Datasets(mix_scp, target_scp, sample_rate, chunk_size, **kwargs):
    dataset = Librimix_Dataset(mix_scp, target_scp, sample_rate, chunk_size, **kwargs)
    return dataset
'''
def Datasets(**kwargs):
    dataset = LRS2mix_Dataset(**kwargs)
    return dataset
