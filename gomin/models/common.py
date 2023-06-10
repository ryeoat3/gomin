import math

import numpy as np
import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram


def get_least_power2_above(x):
    return np.power(2, math.ceil(np.log2(x)))


class MelspecInversion(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 24000,
        win_length: int = 1024,
        hop_length: int = 256,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.melspec_layer = None

    @classmethod
    def from_pretrained(cls, pretrained_model_path, **config):
        model = cls(**config)
        model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"))
        return model

    def prepare_melspectrogram(self, audio):
        if self.melspec_layer is None:
            self.melspec_layer = MelSpectrogram(
                n_mels=self.n_mels,
                sample_rate=self.sample_rate,
                n_fft=get_least_power2_above(self.win_length),
                win_length=self.win_length,
                hop_length=self.hop_length,
                f_min=0.0,
                f_max=(self.sample_rate / 2.0),
                center=True,
                power=2.0,
                mel_scale="slaney",
                norm="slaney",
                normalized=True,
                pad_mode="constant",
            )
            self.melspec_layer = self.melspec_layer.to(audio.device)

        melspec = self.melspec_layer(audio)
        melspec = 10 * torch.log10(melspec + 1e-10)
        melspec = torch.clamp((melspec + 100) / 100, min=0.0)
        return melspec
