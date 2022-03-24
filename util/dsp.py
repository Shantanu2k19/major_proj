import numpy as np
import torch

import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

from util.config import Config
from util.vocoder import get_vocoder



class Dsp():
    def __init__(self, config=None):
        self.load_config(config)
        self.build_mel_basis()

        # another modules
        self.vocoder = None
        self.s3prl = None
        self.resemblyzer = None

    def load_config(self, config):
        default = {
            'n_fft': 1024,
            'hop_length': 256,
            'win_length': 1024,
            'sample_rate': 22050,
            'n_mels': 80,
            'f_min': 0,
            'f_max': 11025,
            'trim': 20,
        }
        if config is None:
            print('Dsp config is None, use default config.')
            config = default
        self.config = Config(config)
        print(self.config)
        for k, v in default.items():
            if k not in self.config.keys():
                self.config[k] = v

    def load_wav(self, path):
        y, sr = librosa.load(path, sr=self.config.sample_rate)
        if type(self.config.trim) is int:
            y, _ = librosa.effects.trim(y, top_db=self.config.trim)
        # y = np.clip(y, -1.0, 1.0)
        return librosa.util.normalize(y)

    def save_wav(self, y, path):
        sf.write(file=path, data=y, samplerate=self.config.sample_rate)

    def build_mel_basis(self):
        self.mel_basis = librosa.filters.mel(self.config.sample_rate, self.config.n_fft,
                fmin=self.config.f_min, fmax=self.config.f_max,
                n_mels=self.config.n_mels)

    def wav2mel(self, y):
        D = np.abs(librosa.stft(y, n_fft=self.config.n_fft,
            hop_length=self.config.hop_length, win_length=self.config.win_length))
        S = np.dot(self.mel_basis, D)
        S[S == 0] = 1e-5
        log_S = np.log10(S)
        return log_S

    def mel2wav(self, mel, save=''):
        if self.vocoder is None:
            self.build_vocoder()
        return self.vocoder.mel2wav(mel, save=save)

    def build_vocoder(self):
        if torch.cuda.is_available:
            device = 'cuda'
        else:
            device = 'cpu'
        self.vocoder = get_vocoder(device=device)

    @staticmethod
    def plot_spectrogram(mag, save=''):
        librosa.display.specshow(mag, x_axis='off', cmap='viridis')
        plt.title('spectrogram')
        if save != '':
            plt.savefig(save, format='jpg')
            plt.close()
        else:
            plt.show()
