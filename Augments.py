import torch
import torch_audiomentations

sample_rate = 16000


class Permute(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


augments_weak = torch_audiomentations.Compose([
    # [batch, time, channels] -> [batch, channels, time]
    Permute([0, 2, 1]),

    # Randomly invert waveform polarity
    torch_audiomentations.PolarityInversion(
        p=0.5,
        output_type="tensor",
        sample_rate=16000
    ),

    # Randomly modify recording volume
    torch_audiomentations.Gain(
        min_gain_in_db=-3,
        max_gain_in_db=3,
        p=0.5,
        output_type="tensor",
        sample_rate=16000
    ),

    # Add relatively mild background noise
    torch_audiomentations.AddColoredNoise(
        min_snr_in_db=3,
        max_snr_in_db=30,
        p=0.5,
        output_type="tensor",
        sample_rate=16000
    ),

    # [batch, channels, time] -> [batch, time, channels]
    Permute([0, 2, 1]),

], output_type="tensor")