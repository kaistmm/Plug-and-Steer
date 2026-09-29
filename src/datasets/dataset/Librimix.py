import os
import random
import numpy as np

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

import librosa

from .epoch_subset import EpochSubsetMixin

def read_scp(scp_path):
    file_list = []
    with open(scp_path,'r') as f:
        for line in f.readlines():
            try:
                file_list.append(line.split(' ')[1].strip())
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
    
class Librimix_Dataset(Dataset, EpochSubsetMixin):
    def __init__(self, mix_scp, target_scp, sample_rate=8000, chunk_size=32000, debug_mode=False, normalize=True,
                 num_per_epoch=None, resample_per_epoch=True, seed=0, **kwargs):
        self.mix_list = read_scp(mix_scp)
        self.target_list = [read_scp(scp) for scp in target_scp]
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.debug = debug_mode
        self.normalize = normalize
        self.init_epoch_subset(len(self.mix_list), num_per_epoch, resample_per_epoch, seed)
        
        try:
            for i in range(len(target_scp)):
                if len(self.mix_list)!=len(self.target_list[i]):
                    raise Exception('Length of mix and gt are different')
        except Exception as e:
            #raise Exception('Length of mix and gt are different')
            print(f"Error occur while comparing the mix and target scp: {str(e)}")
            
    def __getitem__(self, index):
        index = self.map_index(index)
        #mix = loadaudio(self.mix_list[index],self.chunk_size,self.sample_rate)        
        #scale_factor = torch.max(torch.abs(mix)) if self.normalize else 1
        #target = [loadaudio(path_list[index],self.chunk_size, self.sample_rate)/scale_factor for path_list in self.target_list]
        mix, _ = librosa.load(self.mix_list[index], sr=self.sample_rate)
        target = [librosa.load(path_list[index], sr=self.sample_rate)[0] for path_list in self.target_list]

        if self.chunk_size!=-1:
            audio_size = mix.shape[0]
            if audio_size <= self.chunk_size:
                shortage = self.chunk_size - audio_size
                mix = np.pad(mix, (0, shortage), 'constant')
                target = [np.pad(t, (0, shortage), 'constant') for t in target]
            else:
                max_audio_start = audio_size - self.chunk_size
                audio_start = np.random.randint(0, max_audio_start)
                #audio_start = 10
                mix = mix[audio_start:audio_start+self.chunk_size]
                target = [t[audio_start:audio_start+self.chunk_size] for t in target]

        if self.normalize:  
            m_std = np.std(mix)
            mix = normalize_numpy_wav(mix, std=m_std)
            target = [normalize_numpy_wav(t, std=m_std) for t in target]

        return mix, target
    
    def __len__(self):
        full = int(len(self.mix_list)*0.1) if self.debug else len(self.mix_list)
        return self.epoch_length(full)

'''
def Datasets(mix_scp, target_scp, sample_rate, chunk_size, **kwargs):
    dataset = Librimix_Dataset(mix_scp, target_scp, sample_rate, chunk_size, **kwargs)
    return dataset
'''
def Datasets(**kwargs):
    dataset = Librimix_Dataset(**kwargs)
    return dataset



