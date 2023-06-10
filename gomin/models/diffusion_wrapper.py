import diffusers
import torch
from tqdm import tqdm

from .common import MelspecInversion
from .diffusion import GomiDiff


class DiffusionWrapper(MelspecInversion):
    def __init__(
        self,
        in_channels: int,
        residual_layers: int,
        residual_channels: int,
        dilation_cycle_length: int,
        num_diffusion_steps: int,
        **mel_config,
    ):
        super().__init__(n_mels=in_channels, **mel_config)
        self.model = GomiDiff(
            in_channels=in_channels,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
            num_diffusion_steps=num_diffusion_steps,
        )
        self.scheduler = diffusers.DDPMScheduler(
            beta_start=0.0001,
            beta_end=0.05,
            num_train_timesteps=self.model.num_diffusion_steps,
        )
        self.scheduler.set_timesteps(num_inference_steps=self.model.num_diffusion_steps)

    @torch.no_grad()
    def forward(self, spectrogram, return_whole_sequence=False):
        shape = (spectrogram.size(0), 1, self.hop_length * spectrogram.size(-1))

        x = torch.randn(*shape, device=spectrogram.device)

        if return_whole_sequence:
            output_sequence = [x.clone()]

        for t in tqdm(self.scheduler.timesteps, total=len(self.scheduler.timesteps)):
            timestep = torch.tensor([t], device=spectrogram.device).long()
            predicted_noise = self.model(x, timestep, spectrogram)

            scheduler_output = self.scheduler.step(
                predicted_noise, timestep=t, sample=x
            )
            x = scheduler_output["prev_sample"]

            if return_whole_sequence:
                output_sequence.insert(0, x.clone())

        if return_whole_sequence:
            return output_sequence
        return x
