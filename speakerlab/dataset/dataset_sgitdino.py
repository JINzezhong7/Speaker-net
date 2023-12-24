# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
import random
import torch
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset
from torchaudio import transforms
from scipy import signal

from speakerlab.utils.fileio import load_wav_scp


class SGITDINODataset(Dataset):
    def __init__(self, data, noise, reverb, local_frames,global_frames, n_mels, glb_num, local_num):
        print(data)
        self.noise = noise
        self.reverb = reverb
        self.local_frames = local_frames
        self.global_frames = global_frames
        self.n_mels = n_mels
        self.glb_num = glb_num
        self.local_num = local_num
        self.SIGPRO_MIN_RANDGAIN = -7
        self.SIGPRO_MAX_RANDGAIN = 3

        self.torchfb = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=0.0,
            f_max=8000,
            pad=0,
            n_mels=self.n_mels
        )
        self.torchlc = transforms.LFCC(
            sample_rate = 16000,
            n_lfcc = 80 ,
            speckwargs={'n_fft':512,'hop_length':160,'window_fn':torch.hamming_window} 
        )
        self.data = list(
            load_wav_scp(data).values()
            )
        self.rir = np.load("/home/jinzezhong/data/rirs/rir.npy")
        self.noisesnr = {
            'noise':[0, 15], 'speech':[13, 20], 'music':[5, 15]
            }
        self.noise_types = list(self.noisesnr.keys())
        self.noise = {}
        noise_dict = load_wav_scp(noise)
        for id, path in noise_dict.items():
            noise_type = path.split('/')[-4]
            if not noise_type in self.noise:
                self.noise[noise_type] = []
            self.noise[noise_type].append(path)

    def __getitem__(self, index):

        glb_audios, local_audios = Gener_glob_loc_audio(self.data[index], self.local_frames, self.global_frames, self.glb_num, self.local_num)

        augment_profiles = []
        glb_audios_aug = []
        local_audios_aug = []
        noise_random = [0,1,1,1,2,2]

        for ii in range(0, self.glb_num+self.local_num):
            # rir augmentation
            rir_gains = np.random.uniform(self.SIGPRO_MIN_RANDGAIN, self.SIGPRO_MAX_RANDGAIN, 1)
            rir_file = random.choice(self.rir)
            ## noise augmentation
            noise_type = random.choice(self.noise_types)
            noise_file = random.choice(self.noise[noise_type])
            noise_snr = [random.uniform(self.noisesnr[noise_type][0], self.noisesnr[noise_type][1])]
            noise_random_num = random.choice(noise_random)

            if noise_random_num == 0:
                augment_profiles.append({'add_rir':None, 'rir_gain':None, 'add_noise': None, 'noise_snr': None})
            elif noise_random_num == 1:
                if random.random() > 0.75:
                    augment_profiles.append({'add_rir':rir_file, 'rir_gain':rir_gains, 'add_noise': None, 'noise_snr': None})
                else:
                    augment_profiles.append({'add_rir':None, 'rir_gain':None, 'add_noise': noise_file, 'noise_snr': noise_snr})
            elif noise_random_num == 2:
                augment_profiles.append({'add_rir':rir_file, 'rir_gain':rir_gains, 'add_noise': noise_file, 'noise_snr': noise_snr})
            else:
                raise ValueError('Invalid augment profile')

        for i in range(self.glb_num):
            glb_audios_aug.append(self.augment_wav(glb_audios[i], augment_profiles[i], 'True'))
        for j in range(self.local_num):
            local_audios_aug.append(self.augment_wav(local_audios[j], augment_profiles[j+self.glb_num], 'False'))

        glb_audios_aug = np.concatenate(glb_audios_aug, axis=0)
        local_audios_aug = np.concatenate(local_audios_aug,axis=0)
        local_audios_aug = np.concatenate(local_audios_aug, axis=0).reshape(self.glb_num,-1)
        # audios_aug = np.concatenate((glb_audios_aug, local_audios_aug), axis=0)

        with torch.no_grad():
            # To obtain even-numbered frames, we delete 100 points for 4s segments
            glb_feat = torch.FloatTensor(glb_audios_aug[:,:63900])
            local_feat = torch.FloatTensor(local_audios_aug[:,:63900])
            glb_feat_fb = self.torchfb(glb_feat)
            glb_feat_lc = self.torchlc(glb_feat)
            local_feat = self.torchfb(local_feat)
            feat = torch.cat((glb_feat_fb,glb_feat_lc,local_feat),axis=0)
            # feat = torch.FloatTensor(audios_aug[:, :63900])
            # feat = self.torchfb(feat)
        # return local_feat, glb_feat
        return feat

    def __len__(self):
        return len(self.data)

    def augment_wav(self, audio, augment_profile, glb_flag):
        if augment_profile['add_rir'] is not None:
            audio = gene_rir_audio(audio, augment_profile['add_rir'], augment_profile['rir_gain'])

        if augment_profile['add_noise'] is not None:
            if glb_flag == 'True':
                noiseaudio = fill_split(augment_profile['add_noise'], self.global_frames, eval_mode=False)
            else:
                noiseaudio = fill_split(augment_profile['add_noise'], self.local_frames, eval_mode=False)

            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2)+1e-4)
            clean_db = 10 * np.log10(np.mean(audio ** 2)+1e-4)
            noise = np.sqrt(10 ** ((clean_db - noise_db - augment_profile['noise_snr']) / 10)) * noiseaudio
            audio = audio + noise

        else:
            audio = np.expand_dims(audio, 0)

        return audio

def gene_rir_audio(audio, rir, filterGain):
    rir = np.multiply(rir, pow(10, 0.1 * filterGain))    
    audio_rir = signal.convolve(audio, rir, mode = 'full')[ : len(audio)]  

    return audio_rir


def fill_split(filename, max_frames, eval_mode=False, num_eval=10):
    max_audio = max_frames * 160
    sample_rate, audio = wavfile.read(filename)
    if len(audio.shape) == 2:
        audio = audio[:, 0]
    audio_size = audio.shape[0]

    if audio_size <= max_audio:
        shortage = max_audio - audio_size
        audio = np.pad(audio, (0, shortage), 'constant', constant_values=0)
        audio_size = audio.shape[0]
    if eval_mode:
        start_point = np.linspace(0, audio_size - max_audio, num=num_eval)
    else:
        start_point = np.array([np.int64(random.random()*(audio_size - max_audio))])

    feat = []
    if eval_mode and max_frames == 0:
        feat.append(audio)
    else:
        for asf in start_point:
            feat.append(audio[int(asf): int(asf) + max_audio])
    feats = np.stack(feat, axis=0).astype(np.float)

    return feats


def Gener_glob_loc_audio(filename, local_frames, global_frames, glb_num, local_num):
    # Maximum audio length
    glb_audio_size = global_frames * 160
    local_audio_size = local_frames * 160
    sample_rate, audio = wavfile.read(filename)
    if len(audio.shape) == 2:
        audio = audio[:, 0]
    audio_size = audio.shape[0]

    if audio_size <= glb_audio_size:
        shortage = glb_audio_size - audio_size + glb_num
        audio = np.pad(audio, (0, shortage), 'constant', constant_values=0)
        audio_size = audio.shape[0]

    glb_rand = audio_size - glb_audio_size
    assert glb_rand >= glb_num - 1
    glb_start_point = random.sample(range(0, glb_rand), glb_num)
    glb_start_point.sort()
    np.random.shuffle(glb_start_point)
    glb_audio = []
    for asf in glb_start_point:
        glb_audio.append(audio[int(asf): int(asf) + glb_audio_size])

    local_rand = audio_size - local_audio_size
    local_start_point = random.sample(range(0, local_rand), local_num)
    local_start_point.sort()
    np.random.shuffle(local_start_point)
    local_audio = []
    for asf in local_start_point:
        local_audio.append(audio[int(asf): int(asf) + local_audio_size])

    glb_audios = np.stack(glb_audio, axis=0).astype(np.float)
    local_audios = np.stack(local_audio,axis=0).astype(np.float)

    return glb_audios, local_audios