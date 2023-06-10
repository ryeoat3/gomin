# This file is modified from https://github.com/jik876/hifi-gan

# Original terms for the license are as follows:

# MIT License

# Copyright (c) 2020 Jungil Kong

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import functools
from typing import List

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MelspecInversion


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class Snake(nn.Module):
    """Periodic activation function with learned parameter"""

    def __init__(self, kernel_size):
        super().__init__()
        self.alpha = nn.Parameter(data=torch.randn(kernel_size), requires_grad=True)

    def forward(self, x):
        return x + (torch.sin(self.alpha * x) ** 2) / self.alpha


class AmpBlock(nn.Module):
    """Anti-aliased Multi-Periodicity Block with 2 Convolution per Step"""

    def __init__(self, channels, kernel_size, dilation, sample_rate):
        super().__init__()

        conv1d = functools.partial(nn.Conv1d, channels, channels, kernel_size, 1)

        # Low pass filter
        filter_ = scipy.signal.firwin(
            numtaps=12, cutoff=sample_rate / 4, width=0.6, fs=sample_rate
        )
        filter_ = np.sqrt(2) * torch.from_numpy(filter_).reshape(1, 1, -1)
        self.register_buffer("lpf", filter_.tile(channels, 1, 1).float())

        # Layer: snake activations
        self.snakes1 = nn.ModuleList([Snake([channels, 1]) for _ in dilation])
        self.snakes2 = nn.ModuleList([Snake([channels, 1]) for _ in dilation])

        self.convs1 = nn.ModuleList(
            [conv1d(dilation=d, padding=get_padding(kernel_size, d)) for d in dilation]
        )
        self.convs2 = nn.ModuleList(
            [conv1d(dilation=1, padding=get_padding(kernel_size, 1)) for _ in dilation]
        )

    def upsample(self, x):
        b, c, _ = x.shape
        x = torch.cat([x.unsqueeze(-1), torch.zeros_like(x.unsqueeze(-1))], dim=-1)
        x = x.reshape(b, c, -1)
        x = F.conv1d(x, self.lpf, padding=(self.lpf.shape[-1] // 2), groups=c)
        return x, c

    def downsample(self, x, groups):
        x = F.conv1d(x, self.lpf, padding=(self.lpf.shape[-1] // 2), groups=groups)
        x = x[..., 1:-1:2]
        return x

    def forward(self, x):
        for i in range(len(self.convs1)):
            xt, c = self.upsample(x)
            xt = self.snakes1[i](xt)
            xt = self.downsample(xt, c)
            xt = self.convs1[i](xt)
            xt = self.snakes2[i](xt)
            xt = self.convs2[i](xt)
            x = xt + x
        return x


class FreqTemporalFiLM(nn.Module):
    def __init__(self, in_channels, upsample_in_channels, upsample_strides):
        super().__init__()
        in_ch = in_channels  # initial value
        out_ch = upsample_in_channels  # for scale and shift each
        self.convs = nn.ModuleList()
        for s in upsample_strides:
            self.convs.append(
                nn.ConvTranspose1d(in_ch, out_ch, (2 * s), s, padding=(s // 2))
            )
            in_ch = out_ch
            out_ch = out_ch // 2

    def forward(self, x):
        scales, shifts = [], []
        for conv in self.convs:
            x = conv(x)
            y1, y2 = torch.tensor_split(x, 2, dim=1)
            scales.append(y1)
            shifts.append(y2)
        return scales, shifts


class GomiGAN(MelspecInversion):
    """GomiGAN: Gaudio open mel-spectrogram inversion GAN.

    Based on HiFi-GAN (Kong et al. 2020) and BigVGAN (Lee et al. 2022).

    Args:
        in_channels (int): Number of input channels (i.e. number of mel bands)
        upsample_in_channels (int): Number of channels in the first upsampling layer
        upsample_strides (List[int]): Upsampling strides
        resblock_kernel_sizes (List[int]): Kernel sizes for the resblocks
        resblock_dilations (List[List[int]]): Dilations for each resblock
        sample_rate (int): Sample rate of the audio signal
    """

    def __init__(
        self,
        in_channels: int,
        upsample_in_channels: int,
        upsample_strides: List[int],
        resblock_kernel_sizes: List[int],
        resblock_dilations: List[List[int]],
        sample_rate: int = 24000,
        **mel_config,
    ):
        super().__init__(n_mels=in_channels, sample_rate=sample_rate, **mel_config)
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_strides)
        self.upsample_in_channels = upsample_in_channels
        self.sample_rate = sample_rate

        # Initial sample rate for anti-aliased snake function
        sr = (2 * sample_rate) / np.prod(upsample_strides)

        # Layers
        self.film_generator = FreqTemporalFiLM(
            in_channels, upsample_in_channels, upsample_strides
        )
        self.conv_pre = nn.Conv1d(in_channels, upsample_in_channels, 7, 1, padding=3)

        self.snake_ups = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i, us in enumerate(upsample_strides):
            ch = upsample_in_channels // (2 ** (i + 1))
            sr *= us

            self.snake_ups.append(Snake([2 * ch, 1]))
            self.ups.append(nn.ConvTranspose1d(2 * ch, ch, 2 * us, us, us // 2))

            for rk, rd in zip(resblock_kernel_sizes, resblock_dilations):
                self.resblocks.append(AmpBlock(ch, rk, rd, sr))

        self.snake_post = Snake([ch, 1])
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)

    def forward(self, x):
        # Get FiLM weights
        scales, shifts = self.film_generator(x)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.snake_ups[i](x)
            x = self.ups[i](x)

            # Apply FiLM weights
            x = (1 + 0.01 * scales[i]) * x + shifts[i]

            xs = 0.0
            for j in range(self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.snake_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x
