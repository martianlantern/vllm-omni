# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
# Adapted for vLLM-Omni integration
#
# Licensed under the Apache License, Version 2.0
# See: https://huggingface.co/SparkAudio/Spark-TTS-0.5B

"""Decoder module for SparkTTS BiCodec.

Converts latent representations back to audio features via VocosBackbone
with upsampling.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from .blocks import VocosBackbone, SamplingBlock


class Decoder(nn.Module):
    """Decoder module with convnext and upsampling blocks."""

    def __init__(
        self,
        input_channels: int,
        vocos_dim: int,
        vocos_intermediate_dim: int,
        vocos_num_layers: int,
        out_channels: int,
        condition_dim: Optional[int] = None,
        sample_ratios: List[int] = [1, 1],
        use_tanh_at_final: bool = False,
    ):
        super().__init__()

        self.linear_pre = nn.Linear(input_channels, vocos_dim)
        modules = [
            nn.Sequential(
                SamplingBlock(
                    dim=vocos_dim,
                    groups=vocos_dim,
                    upsample_scale=ratio,
                ),
                VocosBackbone(
                    input_channels=vocos_dim,
                    dim=vocos_dim,
                    intermediate_dim=vocos_intermediate_dim,
                    num_layers=2,
                    condition_dim=None,
                ),
            )
            for ratio in sample_ratios
        ]

        self.downsample = nn.Sequential(*modules)

        self.vocos_backbone = VocosBackbone(
            input_channels=vocos_dim,
            dim=vocos_dim,
            intermediate_dim=vocos_intermediate_dim,
            num_layers=vocos_num_layers,
            condition_dim=condition_dim,
        )
        self.linear = nn.Linear(vocos_dim, out_channels)
        self.use_tanh_at_final = use_tanh_at_final

    def forward(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_channels, length)
            c: Optional condition tensor

        Returns:
            x: (batch_size, out_channels, length * upsample_ratio)
        """
        x = self.linear_pre(x.transpose(1, 2))
        x = self.downsample(x).transpose(1, 2)
        x = self.vocos_backbone(x, condition=c)
        x = self.linear(x).transpose(1, 2)
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        return x
