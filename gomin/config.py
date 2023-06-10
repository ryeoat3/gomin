from dataclasses import dataclass, field
from typing import List


@dataclass
class MelConfig:
    n_mels: int = 128
    sample_rate: int = 24000
    win_length: int = 1024
    hop_length: int = 256


@dataclass
class DiffusionConfig:
    in_channels: int = 128
    residual_layers: int = 30
    residual_channels: int = 128
    dilation_cycle_length: int = 10
    num_diffusion_steps: int = 50

    # mel config
    sample_rate: int = 24000
    win_length: int = 1024
    hop_length: int = 256


@dataclass
class GANConfig:
    in_channels: int = 128
    upsample_in_channels: int = 1536
    upsample_strides: List[int] = field(default_factory=lambda: [4, 4, 2, 2, 2, 2])
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilations: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    sample_rate: int = 24000

    # mel config
    win_length: int = 1024
    hop_length: int = 256
