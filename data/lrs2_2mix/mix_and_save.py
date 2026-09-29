import numpy as np
import math, os, csv

import torch
import torch.nn as nn
import torch.utils.data as data
import librosa
import soundfile as sf
import cv2 as cv



def get_dataloader_lip(args, partition):
    datasets = dataset_lip(args, partition)

    sampler = None

    generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = False,
            num_workers = 0,
            persistent_workers = False,
            drop_last=False, 
            sampler=sampler,)
    
    return sampler, generator


class dataset_lip(data.Dataset):
    def __init__(self, args, partition):
        self.minibatch =[]
        self.args = args
        self.partition = partition
        self.audio_sr=args.audio_sr
        self.speaker_no=args.speaker_no

        self.mix_lst_path = args.mix_lst_path
        self.lrs2_src_dir = args.lrs2_src_dir
        lrs2_2mix_dir = args.lrs2_2mix_dir
        if partition=='train':
            self.save_audio_data_dir = os.path.join(lrs2_2mix_dir, 'tr')
        elif partition=='val':
            self.save_audio_data_dir = os.path.join(lrs2_2mix_dir, 'cv')
        else:
            self.save_audio_data_dir = os.path.join(lrs2_2mix_dir, 'tt')
        
        os.makedirs(os.path.join(self.save_audio_data_dir, 'mix'), exist_ok=True)
        os.makedirs(os.path.join(self.save_audio_data_dir, 's1'), exist_ok=True)
        os.makedirs(os.path.join(self.save_audio_data_dir, 's2'), exist_ok=True)
        
        mix_lst=open(self.mix_lst_path).read().splitlines()
        mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))#[:200]
        mix_lst = sorted(mix_lst, key=lambda data: float(data.split(',')[-1]), reverse=True)
        
        start = 0
        while True:
            end = min(len(mix_lst), start + 1)
            self.minibatch.append(mix_lst[start:end])
            if end == len(mix_lst):
                break
            start = end

    

    def _load_raw_video(self, visual_path, min_length_visual):
        captureObj = cv.VideoCapture(visual_path)
        roiSequence = []
        roiSize = 112
        mean = 0.4161
        std = 0.1688
        while (captureObj.isOpened()):
            ret, frame = captureObj.read()
            if ret == True:
                grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                grayed = cv.resize(grayed, (roiSize*2,roiSize*2))
                grayed = grayed[int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)), int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2))]
                # grayed = cv.resize(grayed, (roiSize,roiSize))
                roiSequence.append(grayed)
            else:
                break
        captureObj.release()
        visual = np.asarray(roiSequence)/255.0
        visual = (visual - mean)/std

        visual = visual[:min_length_visual,...]
        if visual.shape[0] < min_length_visual:
            visual = np.pad(visual, ((0,int(min_length_visual - visual.shape[0])),(0,0),(0,0)), mode = 'constant')
        return visual


    def _audioread(self, path, min_length_audio, sampling_rate):
        data, fs = sf.read(path, dtype='float32')    
        if fs != sampling_rate:
            data = librosa.resample(data, orig_sr=fs, target_sr=sampling_rate)
        if len(data.shape) >1:
            data = data[:, 0]    

        data = data[:min_length_audio]
        if data.shape[0] < min_length_audio:
            data = np.pad(data, (0, int(min_length_audio - data.shape[0])),mode = 'constant')
        return  data


    def __getitem__(self, index):
        mix_audios=[]
        tgt_audios=[]
        tgt_visuals=[]
        
        batch_lst = self.minibatch[index]
        min_length_second = float(batch_lst[-1].split(',')[-1])      # truncate to the shortest utterance in the batch
        min_length_audio = math.floor(min_length_second*self.audio_sr)

        for line_cache in batch_lst:
            line=line_cache.split(',')
            name = '_'.join(line[2:5]+line[6:9])
            c=0

            # read tgt audio
            tgt_audio_path=self.lrs2_src_dir+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.wav'
            a_tgt = self._audioread(tgt_audio_path, min_length_audio, self.audio_sr)
            target_power = np.linalg.norm(a_tgt, 2)**2 / a_tgt.size
            snr_0 = 10**(float(line[c*4+4])/20)
            
            
            c=1
            # read int audio
            int_audio_path=self.lrs2_src_dir+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.wav'
            a_int = self._audioread(int_audio_path, min_length_audio, self.audio_sr)
            intef_power = np.linalg.norm(a_int, 2)**2 / a_int.size
            a_int *= np.sqrt(target_power/intef_power)
            snr_1 = 10**(float(line[c*4+4])/20)

            if self.args.speaker_no == 2:
                max_snr = max(snr_0, snr_1)
                a_tgt /= max_snr
                a_int /= max_snr

                a_tgt = a_tgt * snr_0
                a_int = a_int * snr_1
                if len(a_tgt) != len(a_int):
                    min_len = min(len(a_tgt), len(a_int))
                    # print(f"lengths: {len(a_tgt)}, {len(a_int)}")
                    a_tgt = a_tgt[:min_len]
                    a_int = a_int[:min_len]
                else:
                    min_len = len(a_tgt)
                a_mix = a_tgt + a_int

            elif self.args.speaker_no == 3:
                c=2
                # read int audio
                int_audio_path_2=self.lrs2_src_dir+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.wav'
                a_int2 = self._audioread(int_audio_path_2, min_length_audio, self.audio_sr)
                intef_power_2 = np.linalg.norm(a_int2, 2)**2 / a_int2.size
                a_int2 *= np.sqrt(target_power/intef_power_2)
                snr_2 = 10**(float(line[c*4+4])/20)
                
                max_snr = max(snr_0, snr_1, snr_2)
                a_tgt /= max_snr
                a_int /= max_snr
                a_int2 /= max_snr

                a_tgt = a_tgt * snr_0
                a_int = a_int * snr_1
                a_int2 = a_int2 * snr_2

                a_mix = a_tgt + a_int + a_int2
            else:
                raise NameError('Wrong speaker_no selection')
            
            
            # audio normalization
            max_val = np.max(np.abs(a_mix))
            if max_val > 1:
                a_mix /= max_val
                a_tgt /= max_val
                a_int /= max_val
                if self.args.speaker_no == 3:
                    a_int2 /= max_val

            #### save audios
            sf.write(os.path.join(self.save_audio_data_dir, 'mix', name+'.wav'), a_mix, self.audio_sr)
            sf.write(os.path.join(self.save_audio_data_dir, 's1', name+'.wav'), a_tgt, self.audio_sr)
            sf.write(os.path.join(self.save_audio_data_dir, 's2', name+'.wav'), a_int, self.audio_sr)
            
            
        return 1

    def __len__(self):
        return len(self.minibatch)


if __name__ == '__main__':
    import yamlargparse, tqdm
    parser = yamlargparse.ArgumentParser("Settings")

    # others
    parser.add_argument('--mix_lst_path', type=str, default='./data/lrs2_2mix/mixture_data_list_2mix.csv')
    parser.add_argument('--lrs2_src_dir', type=str, required=True)
    parser.add_argument('--speaker_no', type=int, default=2)
    parser.add_argument('--audio_sr',  type=int, help='audio sampling_rate', default=16000)
    parser.add_argument('--lrs2_2mix_dir', type=str, required=True)

    args = parser.parse_args()
    
    for split in ['train', 'val', 'test']:
        print(f'Saving {split} data...')
        args.partition = split
        _, dataloader = get_dataloader_lip(args, split)
        
        for data_item in tqdm.tqdm(dataloader, dynamic_ncols=True):
            pass
    