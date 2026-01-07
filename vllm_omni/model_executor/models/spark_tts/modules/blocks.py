# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
# Adapted for vLLM-Omni integration
#
# Licensed under the Apache License, Version 2.0
# See: https://huggingface.co/SparkAudio/Spark-TTS-0.5B

"""Common building blocks for SparkTTS neural network modules.

This file consolidates the necessary blocks from flashtts/modules/blocks/
for use in vLLM-Omni's SparkTTS integration.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm


# =============================================================================
# Basic Layers
# =============================================================================


def WNConv1d(*args, **kwargs):
    """Weight-normalized 1D convolution."""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    """Weight-normalized 1D transposed convolution."""
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


@torch.jit.script
def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Snake activation function - brings 1.4x speedup when scripted."""
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """Snake activation for 1D signals."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)


class ResidualUnit(nn.Module):
    """Residual unit with dilated convolutions."""

    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


def init_weights(m: nn.Module) -> None:
    """Initialize weights for Conv1d layers."""
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# =============================================================================
# Sampling Blocks
# =============================================================================


class SamplingBlock(nn.Module):
    """Sampling block for upsampling or downsampling."""

    def __init__(
        self,
        dim: int,
        groups: int = 1,
        upsample_scale: int = 1,
        downsample_scale: int = 1,
    ) -> None:
        super().__init__()
        self.upsample_scale = upsample_scale
        self.downsample_scale = downsample_scale

        if self.upsample_scale > 1:
            self.de_conv_upsampler = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(
                    dim,
                    dim,
                    kernel_size=upsample_scale * 2,
                    stride=upsample_scale,
                    padding=upsample_scale // 2 + upsample_scale % 2,
                    output_padding=upsample_scale % 2,
                    groups=groups,
                ),
            )

        if self.downsample_scale > 1:
            self.conv_downsampler = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv1d(
                    dim,
                    dim,
                    kernel_size=2 * downsample_scale,
                    stride=downsample_scale,
                    padding=downsample_scale // 2 + downsample_scale % 2,
                    groups=groups,
                ),
            )

    @staticmethod
    def repeat_upsampler(x: torch.Tensor, upsample_scale: int) -> torch.Tensor:
        return x.repeat_interleave(upsample_scale, dim=2)

    @staticmethod
    def skip_downsampler(x: torch.Tensor, downsample_scale: int) -> torch.Tensor:
        return F.avg_pool1d(x, kernel_size=downsample_scale, stride=downsample_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        if self.upsample_scale > 1:
            repeat_res = self.repeat_upsampler(x, self.upsample_scale)
            deconv_res = self.de_conv_upsampler(x)
            upmerge_res = repeat_res + deconv_res
        else:
            upmerge_res = x
            repeat_res = x

        if self.downsample_scale > 1:
            conv_res = self.conv_downsampler(upmerge_res)
            skip2_res = self.skip_downsampler(upmerge_res, self.downsample_scale)
            skip1_res = self.skip_downsampler(repeat_res, self.downsample_scale)
        else:
            conv_res = upmerge_res
            skip2_res = upmerge_res
            skip1_res = repeat_res

        return conv_res + skip1_res + skip2_res


# =============================================================================
# ConvNeXt / Vocos Blocks
# =============================================================================


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization with learnable embeddings."""

    def __init__(self, condition_dim: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Linear(condition_dim, embedding_dim)
        self.shift = nn.Linear(condition_dim, embedding_dim)
        nn.init.ones_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding)
        shift = self.shift(cond_embedding)
        x = F.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted for 1D audio signal."""

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        condition_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.adanorm = condition_dim is not None
        if condition_dim:
            self.norm = AdaLayerNorm(condition_dim, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + x


class VocosBackbone(nn.Module):
    """Vocos backbone built with ConvNeXt blocks."""

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        condition_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = condition_dim is not None
        if condition_dim:
            self.norm = AdaLayerNorm(condition_dim, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    condition_dim=condition_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self, x: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embed(x)
        if self.adanorm:
            assert condition is not None
            x = self.norm(x.transpose(1, 2), condition)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, condition)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


# =============================================================================
# HiFi-GAN Decoder Block
# =============================================================================


class DecoderBlock(nn.Module):
    """Decoder block with upsampling for waveform generation."""

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        kernel_size: int = 2,
        stride: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
