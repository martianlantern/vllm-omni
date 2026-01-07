# Copyright (c) 2024 Xinsheng Wang (w.xinshawn@gmail.com)
# Adapted for vLLM-Omni integration
#
# Licensed under the Apache License, Version 2.0
# Adapted from https://github.com/descriptinc/descript-audio-codec

"""WaveGenerator module for SparkTTS BiCodec.

HiFi-GAN style generator that converts feature representations to audio waveforms.
"""

from typing import List

import torch.nn as nn

from .blocks import Snake1d, WNConv1d, ResidualUnit, WNConvTranspose1d, init_weights, DecoderBlock


class WaveGenerator(nn.Module):
    """HiFi-GAN style waveform generator."""

    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: List[int],
        kernel_sizes: List[int],
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, (kernel_size, stride) in enumerate(zip(kernel_sizes, rates)):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, kernel_size, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)
