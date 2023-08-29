# GOMIN: Gaudio Open Melspectrogram Inversion Network

tl;dr - GOMIN is a general-purpose, general-source model for melspectrogram -> waveform.


## About

GOMIN is a general-purpose, general-source model for converting melspectrograms to waveforms.

We open source this code and provide pretrained models to further research in music / general audio generation.
The models have been trained on a diverse range of audio datasets, including speech signals, music stems, animal sound recordings, and foley sound stems.
To cover those various filed of audio and make it more universal and robust, some improvements are applied to its neural vocoder baseline.
This makes GOMIN suitable for various applications and research endeavors.

### Supported models

The available models are based on two state-of-the-art neural vocoder models; BigVGAN \[[Lee et al. 2022](https://arxiv.org/abs/2206.04658)\] and DiffWave \[[Kong et al. 2020](https://arxiv.org/abs/2009.09761)\].
These models have been slightly modified to improve their performance in generating general audio signals beyond just speech signals.

The GAN-based models have been enhanced with Feature-wise Linear Modulation (FiLM) \[[Perez et al. 2017](https://arxiv.org/abs/1709.07871)\] after every upsampling block.
The modulation parameters, i.e. shift and scale parameters, are calculated from the raw melspectrogram and each upsampling layer has distinct parameters, meaning that the parameters are not shared. 
This modification improves tonal consistency and leads to better sound reconstruction for general audio signals.
Note that code for this model is largely brought from [HiFi-GAN](https://github.com/jik876/hifi-gan) \[[Kong et al. 2020](https://arxiv.org/abs/2010.05646)\], not directly from [BigVGAN](https://github.com/NVIDIA/BigVGAN) repository.

For Diffusion-based models, in addition to the FiLM, we have also fine-tuned its noise schedule to better accommodate universal audio generation.
This was achieved by interpolating two popular schedules, the linear and cosine schedule.
The interpolated schedule roughly follows the linear schedule near $t=0$ and the cosine schedule near $t=T$.
This noise schedule injects more noise in earlier steps, helping the model to handle more diverse and complex data distributions. 
Research has shown that the use of noise schedules is crucial in high-resolution image generation \[[Chen 2023](https://arxiv.org/abs/2301.10972), [Hoogeboom et al. 2023](https://arxiv.org/abs/2301.11093)\].
We believe that further research in this area will lead to even better audio generation in the future.


## Requirements

- python3    >= 3.10
- torch      (tested on 1.13.1+cu116)
- torchaudio (tested on 0.13.1+cu116)
- diffusers  (tested on 0.17.0)
- librosa    >= 0.9.2
- numpy      == 1.24.3
- pyyaml
- scipy      >= 1.10.0
- soundfile  >= 0.11.0
- tqdm


## Install
You can install this package using following command.
```Shell
$ git clone https://github.com/ryeoat3/gomin.git
$ cd gomin
$ pip install -e .
```


## Pretrained checkpoint

You can download pretrained checkpoint from google drive ([gan](https://drive.google.com/file/d/1TyNCS7fdeeCJK66x_n9TeR_SurPaft4L), [diffusion](https://drive.google.com/file/d/1vkrTICKruShu_0ofM3vTc3No2Fj5rMxD)).


## Inference

### Python
```Python
>>> from gomin.models import GomiGAN, DiffusionWrapper
>>> from config import GANConfig, DiffusionConfig

# Model loading
>>> model = GomiGAN.from_pretrained(
  pretrained_model_path="CHECKPOINT PATH", **GANConfig().__dict__
)  # for GAN model
>>> model = DiffusionWrapper.from_pretrained(
  pretrained_model_path="CHECKPOINT_PATH", **DiffusionConfig().__dict__
)  # for diffusion model

# To convert your waveform into mel-spectrogram, run:
>>> assert waveform.ndim == 2
>>> melspec = model.prepare_melspectrogram(waveform)

# To reconstruct wavefrom from mel-spectrogram, run:
>>> assert melspec.ndim == 3
>>> waveform = model(melspec)
```

### CLI
```Shell
$ python -m gomin.run -m {MODEL: 'gan' | 'diffusion'} -p {MODEL_PATH} -i {INPUT_FILES} -o {OUTPUT_FILES}
```

**Arguments**

- `-m` (`--model`): model type to run process. either 'gan' or 'diffusion' is valid.
- `-p` (`--model_path`): directory path for model checkpoint. default value is `'checkpoints/'`
- `-i` (`--input_files`): list of file paths or directory path for input files.
- `-o` (`--output_files`): list of file paths or directory path for output files.

**Notes**

For CLI run, `input_files` and `output_files` option supports

- list of files (e.g. `-i a.wav b.wav c.wav -o a_out.wav b_out.wav c_out.wav`)
- directory path (e.g. `-i inputs/ -o outputs/`)
- and path with wildcard `*` (e.g. `-i inputs/**/*.wav -o outputs/**/*.pt`)

Extensions of `input_files` and `output_files` will determine which process to run:

- Analysis and synthesis
    - If both are audio files (with extension of `.wav` or `.mp3`), this program will first convert to mel-spectrogram and reconstruct it back to waveform.
- Analysis
    - If `input_files` extinsions are one of [`.wav` and `.mp3`] and `output_files` are `.pt` file, thie program will convert waveforms to mel-spectrogram and save them. Saved files will include mel-spectrogram tensor and configurations (`config.py::MelConfig`).
- Synthesis
    - If `input_files` are `.pt` files and `output_files` are audio files, this program will reconstruct mel-spectrogram into waveform.


## References
- [Lee et al. 2022](https://arxiv.org/abs/2206.04658): Sang-gil Lee, Wei Ping, Boris Ginsburg, Bryan Catanzaro, Sungroh Yoon, "BigVGAN: A Universal Neural Vocoder with Large-Scale Training"
- [Kong et al. 2020](https://arxiv.org/abs/2009.09761): Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, Bryan Catanzaro, "DiffWave: A Versatile Diffusion Model for Audio Synthesis"
- [Perez et al. 2017](https://arxiv.org/abs/1709.07871): Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville. "FiLM: Visual Reasoning with a General Conditioning Layer"
- [Chen 2023](https://arxiv.org/abs/2301.10972): Ting Chen. "On the Importance of Noise Scheduling for Diffusion Models"
- [Hoogeboom et al. 2023](https://arxiv.org/abs/2301.11093): Emiel Hoogeboom, Jonathan Heek, Tim Salimans. "simple diffusion: End-to-end diffusion for high resolution images"

## Related repos
- HiFi-GAN: https://github.com/jik876/hifi-gan
- DiffWave: https://github.com/lmnt-com/diffwave


## How to cite
If you find this work useful, please refer this:
```bibtex
# Workshop paper submitted to DCASE Workshop 2023
@misc{kang2023falle,
    title={FALL-E: A Foley Sound Synthesis Model and Strategies},
    author={Minsung Kang and Sangshin Oh and Hyeongi Moon and Kyungyun Lee and Ben Sangbae Chon},
    year={2023},
    eprint={2306.09807},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}

# or technical report for DCASE Challenge 2023 Task7
@techreport{ChonGLI2023,
    Author = "Kang, Minsung and Oh, Sangshin and Moon, Hyeongi and Lee, Kyungyun and Chon, Ben Sangbae",
    title = "FALL-E: Gaudio Foley Synthesis System",
    institution = "Gaudio Lab, Inc., Seoul, South Kore",
    year = "2023",
    month = "June",
}
```

## License
[MIT License](LICENSE)
