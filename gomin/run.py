import argparse
import os
from typing import Union

import librosa
import numpy as np
import soundfile as sf
import torch

from . import config
from . import models
from .utils import AUDIO_EXTS, MEL_EXTS, device_type, preprocess_inout_files


def load_audio(
    audio_path: Union[str, bytes, os.PathLike],
    sample_rate: int,
    mono=True,
    fast_resample=False,
):
    """Load and resample audio file"""

    if fast_resample:
        audio, orig_sr = librosa.load(audio_path, sr=None, mono=mono)
        audio = librosa.resample(
            audio, orig_sr=orig_sr, target_sr=sample_rate, res_type="polyphase"
        )
    else:
        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=mono)

    audio = np.atleast_2d(audio)
    audio = torch.from_numpy(audio).float()
    return audio


@torch.no_grad()
def _analysis(
    model: torch.nn.Module, audio: Union[np.ndarray, torch.Tensor], device: torch.device
):
    """Convert waveform into melspectrogram."""
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()

    if audio.ndim < 2:
        audio = torch.atleast_2d(audio)

    if audio.device != device:
        audio = audio.to(device)

    melspec = model.prepare_melspectrogram(audio)
    return melspec


@torch.no_grad()
def _synthesis(model: torch.nn.Module, melspec: torch.Tensor, device: torch.device):
    """Convert melspectrogram into waveform."""
    if melspec.ndim < 3:
        melspec = torch.atleast_3d(melspec)

    if melspec.device != device:
        melspec = melspec.to(device)

    recon_audio = model(melspec)
    recon_audio = recon_audio.squeeze().cpu()
    return recon_audio


def analysis_synthesis(
    model: torch.nn.Module,
    in_file: Union[str, bytes, os.PathLike],
    out_file: Union[str, bytes, os.PathLike],
    device: torch.device,
    fast_resample=False,
):
    """Process and save files."""
    _, input_ext = os.path.splitext(in_file)
    _, output_ext = os.path.splitext(out_file)

    if input_ext in AUDIO_EXTS:
        audio = load_audio(in_file, model.sample_rate, fast_resample=fast_resample)
        melspec = _analysis(model, audio, device=device)
    elif input_ext in MEL_EXTS:
        if input_ext == ".npy":
            melspec = np.load(in_file)
            melspec = torch.from_numpy(melspec).float()
        else:
            melspec_dict = torch.load(in_file)
            melspec = melspec_dict["melspec"]
            assert melspec_dict["n_mels"] == model.n_mels, (
                f"Wrong `n_mels`. expected [{model.n_mels}], got"
                f" [{melspec_dict['n_mels']}]."
            )
            assert melspec_dict["sample_rate"] == model.sample_rate, (
                f"Wrong `sample_rate`. expected [{model.sample_rate}], got"
                f" [{melspec_dict['sample_rate']}]."
            )
            assert melspec_dict["win_length"] == model.win_length, (
                f"Wrong `win_length`. expected [{model.win_length}], got"
                f" [{melspec_dict['win_length']}]."
            )
            assert melspec_dict["hop_length"] == model.hop_length, (
                f"Wrong `hop_length`. expected [{model.hop_length}], got"
                f" [{melspec_dict['hop_length']}]."
            )
    else:
        print(f"Unsupported input file extension: {input_ext} for {in_file}.")

    if output_ext in MEL_EXTS:
        assert output_ext == ".pt", (
            "Only '.pt' file is supported for melspectrogram extraction. got"
            f" '{output_ext}'."
        )
        torch.save(
            {
                "melspec": melspec.cpu(),
                "n_mels": model.n_mels,
                "sample_rate": model.sample_rate,
                "win_length": model.win_length,
                "hop_length": model.hop_length,
            },
            out_file,
        )
    elif output_ext in AUDIO_EXTS:
        assert (
            output_ext == ".wav"
        ), "Only '.wav' file is supported for melspectrogram inversion. got"
        f" '{output_ext}'."
        recon_audio = _synthesis(model, melspec, device=device)
        sf.write(out_file, recon_audio, model.sample_rate, subtype="PCM_16")
    else:
        print(f"Unsupported output file extension: {output_ext} for {out_file}.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        choices=["diffusion", "gan"],
        help="Model type to run.",
    )
    parser.add_argument(
        "-p",
        "--model_path",
        type=str,
        default="checkpoints/",
        help="Directory path to model checkpoint.",
    )
    parser.add_argument(
        "-i",
        "--input_files",
        nargs="+",
        type=str,
        help=f"Path to input files. Audio files with {AUDIO_EXTS} extension and"
        f" melspectrogram files with {MEL_EXTS} extension are supported.",
    )
    parser.add_argument(
        "-o",
        "--output_files",
        nargs="+",
        type=str,
        default=["outputs/"],
        help="(Optional) Path to output files. Audio files with '.wav' extension and"
        " melspectrogram files with '.pt' extension are supported. If both input and"
        " output files are melspectrogram, error will be raised. (default: `outputs/`)",
    )
    parser.add_argument(
        "-d",
        "--device",
        nargs="+",
        type=device_type,
        default=["cpu"],
        help="Device to use. Currently multi-GPU inference is not supported. (default: "
        "'cpu')",
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=None,
        help="Number of bins in melspectrogram. If `args.model` is provided, this value"
        " will be ignored.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=None,
        help="Sample rate for audio sample and melspetrogram. If `args.model` is"
        " provided, this value will be ignored.",
    )
    parser.add_argument(
        "--win_length",
        type=int,
        default=None,
        help="Windoew length for melspetrogram. If `args.model` is provided, this value"
        " will be ignored",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=None,
        help="Hop length for melspectrogram. If `args.model` is provided, this value"
        " will be ignored.",
    )

    args = parser.parse_args()
    return args


def process(model, device, input_files, output_files):
    model.to(device)

    for in_file, out_file in zip(input_files, output_files):
        analysis_synthesis(
            model=model,
            in_file=in_file,
            out_file=out_file,
            device=device,
            fast_resample=len(input_files) > 1,
        )


def main():
    args = parse_args()
    args.input_files, args.output_files = preprocess_inout_files(
        args.input_files, args.output_files
    )

    if args.model == "gan":
        model = models.GomiGAN.from_pretrained(
            pretrained_model_path="checkpoints/gan_state_dict.pt",
            **config.GANConfig().__dict__,
        )
    elif args.model == "diffusion":
        model = models.DiffusionWrapper.from_pretrained(
            pretrained_model_path="checkpoints/diffusion_state_dict.pt",
            **config.DiffusionConfig().__dict__,
        )
    elif args.model is None:
        model = models.MelspecInversion(
            n_mels=args.n_mels,
            sample_rate=args.sample_rate,
            win_length=args.win_length,
            hop_length=args.hop_length,
        )

    process(model, args.device[0], args.input_files, args.output_files)


if __name__ == "__main__":
    main()
