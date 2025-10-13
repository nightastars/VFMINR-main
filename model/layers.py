import math

import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding layer.

    Positionally encode inputs by projecting them through sinusoidal functions at multiple frequencies.
    Frequencies are scaled logarithmically. The original input is also included in the output so that the
    absolute position information is not lost.

    Args:
    ----
    in_dim: int
        Input dimension.
    frequency_bands: int
        Number of frequencies to encode input into.

    """

    def __init__(self, in_dim, frequency_bands=6, include_input=True):
        super().__init__()
        self.in_dim = in_dim
        if include_input:
            self.out_dim = in_dim + (2 * frequency_bands * in_dim)
        else:
            self.out_dim = 2 * frequency_bands * in_dim
        self.frequency_bands = frequency_bands
        self.include_input = include_input

        freqs = 2.0 ** torch.linspace(0.0, frequency_bands - 1, frequency_bands, dtype=torch.float)
        self.freqs = torch.nn.Parameter(freqs, requires_grad=False)

    def forward(self, x):
        if self.include_input:
            encoding = [x]
        else:
            encoding = []

        for freq in self.freqs:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(x * freq))
        encoding = torch.cat(encoding, dim=-1)
        return encoding