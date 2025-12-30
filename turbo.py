import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pyloudnorm as ln
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta
import torchaudio.compliance.kaldi as Kaldi
from einops import pack, rearrange, repeat
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from scipy import signal
from scipy.signal import get_window
from torch.distributions.uniform import Uniform
from torch.nn import Conv1d, ConvTranspose1d, Parameter
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2Model,
    LlamaConfig,
    LlamaModel,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

# Constants
S3GEN_SR = 24000
S3GEN_SIL = 4299


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


CFM_PARAMS = AttrDict(
    {
        "sigma_min": 1e-06,
        "solver": "euler",
        "t_scheduler": "cosine",
        "training_cfg_rate": 0.2,
        "inference_cfg_rate": 0.7,
        "reg_loss_type": "l1",
    }
)

LLAMA_520M_CONFIG_DICT = dict(
    vocab_size=8,
    max_position_embeddings=131072,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=30,
    num_attention_heads=16,
    attn_implementation="sdpa",
    head_dim=64,
    tie_word_embeddings=False,
    hidden_act="silu",
    attention_bias=False,
    attention_dropout=0.0,
    initializer_range=0.02,
    mlp_bias=False,
    model_type="llama",
    num_key_value_heads=16,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling=dict(
        factor=8.0, high_freq_factor=4.0, low_freq_factor=1.0, original_max_position_embeddings=8192, rope_type="llama3"
    ),
    rope_theta=500000.0,
    torch_dtype="bfloat16",
    use_cache=True,
)

GPT2_MEDIUM_CONFIG = {
    "activation_function": "gelu_new",
    "architectures": ["GPT2LMHeadModel"],
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 50256,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": 8196,
    "n_embd": 1024,
    "hidden_size": 1024,
    "n_head": 16,
    "n_layer": 24,
    "n_positions": 8196,
    "n_special": 0,
    "predict_special_tokens": True,
    "resid_pdrop": 0.1,
    "summary_activation": None,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": True,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "task_specific_params": {"text-generation": {"do_sample": True, "max_length": 50}},
    "vocab_size": 50276,
}

LLAMA_CONFIGS = {
    "Llama_520M": LLAMA_520M_CONFIG_DICT,
    "GPT2_medium": GPT2_MEDIUM_CONFIG,
}


class T3Config:
    def __init__(self, text_tokens_dict_size=704):
        self.start_text_token = 255
        self.stop_text_token = 0
        self.text_tokens_dict_size = text_tokens_dict_size
        self.max_text_tokens = 2048

        self.start_speech_token = 6561
        self.stop_speech_token = 6562
        self.speech_tokens_dict_size = 8194
        self.max_speech_tokens = 4096

        self.llama_config_name = "Llama_520M"
        self.input_pos_emb = "learned"
        self.speech_cond_prompt_len = 150

        self.encoder_type = "voice_encoder"
        self.speaker_embed_size = 256
        self.use_perceiver_resampler = True
        self.emotion_adv = True

    @property
    def n_channels(self):
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]

    @property
    def is_multilingual(self):
        return self.text_tokens_dict_size == 2454

    @classmethod
    def english_only(cls):
        return cls(text_tokens_dict_size=704)

    @classmethod
    def multilingual(cls):
        return cls(text_tokens_dict_size=2454)


class VoiceEncConfig:
    num_mels = 40
    sample_rate = 16000
    speaker_embed_size = 256
    ve_hidden_size = 256
    flatten_lstm_params = False
    n_fft = 400
    hop_size = 160
    win_size = 400
    fmax = 8000
    fmin = 0
    preemphasis = 0.0
    mel_power = 2.0
    mel_type = "amp"
    normalized_mels = False
    ve_partial_frames = 160
    ve_final_relu = True
    stft_magnitude_min = 1e-4


# Melspectrogram Utilities
@lru_cache
def mel_basis(hp):
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin, fmax=hp.fmax)


def preemphasis(wav, hp):
    assert hp.preemphasis != 0
    wav = signal.lfilter([1, -hp.preemphasis], [1], wav)
    wav = np.clip(wav, -1, 1)
    return wav


def melspectrogram(wav, hp, pad=True):
    if hp.preemphasis > 0:
        wav = preemphasis(wav, hp)
        assert np.abs(wav).max() - 1 < 1e-07

    spec_complex = librosa.stft(
        wav,
        n_fft=hp.n_fft,
        hop_length=hp.hop_size,
        win_length=hp.win_size,
        center=pad,
        pad_mode="reflect",
    )
    spec_magnitudes = np.abs(spec_complex)
    if hp.mel_power != 1.0:
        spec_magnitudes **= hp.mel_power

    mel = np.dot(mel_basis(hp), spec_magnitudes)
    if hp.mel_type == "db":
        mel = 20 * np.log10(np.maximum(hp.stft_magnitude_min, mel))

    if hp.normalized_mels:
        min_level_db = 20 * np.log10(hp.stft_magnitude_min)
        mel = (mel - min_level_db) / (-min_level_db + 15)
        mel = mel.astype(np.float32)

    assert not pad or mel.shape[1] == 1 + len(wav) // hp.hop_size
    return mel


def melspectrogram_torch(wav, hp):
    # expect wav in (1, T) or (T,)
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    if wav.ndim == 2:
        wav = wav.squeeze(0)
    return torch.from_numpy(melspectrogram(wav, hp))


# Modules


class Swish(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)
        return x


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000, reverse: bool = False):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, offset: int | torch.Tensor = 0) -> tuple[torch.Tensor, torch.Tensor]:
        self.pe = self.pe.to(x.device)
        pos_emb = self.position_encoding(offset, x.size(1), False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int | torch.Tensor, size: int, apply_dropout: bool = True) -> torch.Tensor:
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset : offset + size]
        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset : offset + size]
        else:
            assert torch.max(offset) + size <= self.max_len
            index = offset.unsqueeze(1) + torch.arange(0, size).to(offset.device)
            flag = index > 0
            index = index * flag
            pos_emb = F.embedding(index, self.pe[0])

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


class RelPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self, x: torch.Tensor, offset: int | torch.Tensor = 0) -> tuple[torch.Tensor, torch.Tensor]:
        self.pe = self.pe.to(x.device)
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.size(1), False)
        return self.dropout(x), self.dropout(pos_emb)


class WhisperPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 1500):
        super().__init__(d_model, dropout_rate, max_len)
        self.xscale = 1.0
        log_timescale_increment = np.log(10000) / (d_model // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(d_model // 2))
        scaled_time = torch.arange(max_len)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        delattr(self, "pe")
        self.register_buffer("pe", pe.unsqueeze(0))


class LearnablePositionalEncoding(PositionalEncoding):
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 448):
        super().__init__(d_model, dropout_rate, max_len)
        self.pe = torch.nn.Parameter(torch.empty(1, max_len, d_model))
        self.xscale = 1.0


# Conformer Modules


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_attention(
        self, value: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        n_batch = value.size(0)
        if mask.size(2) > 0:
            mask = mask.unsqueeze(1).eq(0)
            mask = mask[:, :, :, : scores.size(-1)]
            scores = scores.masked_fill(mask, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self.forward_qkv(query, key, value)
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        super().__init__(n_head, n_feat, dropout_rate, key_bias)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u.to(q.device)).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v.to(q.device)).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class PositionwiseFeedForward(torch.nn.Module):
    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class ConvolutionModule(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: nn.Module = nn.ReLU(),
        norm: str = "batch_norm",
        causal: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias)
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size, stride=1, padding=padding, groups=channels, bias=bias
        )
        assert norm in ["batch_norm", "layer_norm"]
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:
            x.masked_fill_(~mask_pad, 0.0)

        if self.lorder > 0:
            if cache.size(2) == 0:
                x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
            else:
                x = torch.cat((cache, x), dim=2)
            new_cache = x[:, :, -self.lorder :]
        else:
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        if mask_pad.size(2) > 0:
            x.masked_fill_(~mask_pad, 0.0)
        return x.transpose(1, 2), new_cache


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-12)
        self.norm2 = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb=pos_emb, cache=att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        fake_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        return x, mask, new_att_cache, fake_cnn_cache


class ConformerEncoderLayer(nn.Module):
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: nn.Module | None = None,
        feed_forward_macaron: nn.Module | None = None,
        conv_module: nn.Module | None = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-12)
            self.norm_final = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm_conv(x)

        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache


class BaseSubsampling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: int | torch.Tensor, size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-5),
            torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, offset: int | torch.Tensor = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class Conv2dSubsampling4(BaseSubsampling):
    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 4
        self.right_context = 6

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, offset: int | torch.Tensor = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2]


from diffusers.models.attention import (  # noqa: E402
    GEGLU,
    GELU,
    AdaLayerNorm,
    AdaLayerNormZero,
    ApproximateGELU,
)
from diffusers.models.attention_processor import Attention  # noqa: E402
from diffusers.models.lora import LoRACompatibleLinear  # noqa: E402
from diffusers.utils.torch_utils import maybe_allow_in_graph  # noqa: E402


def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    pos_idx = torch.arange(size, device=device)
    block_value = (torch.div(pos_idx, chunk_size, rounding_mode="trunc") + 1) * chunk_size
    ret = pos_idx.unsqueeze(0) < block_value.unsqueeze(1)
    return ret


def add_optional_chunk_mask(
    xs: torch.Tensor,
    masks: torch.Tensor,
    use_dynamic_chunk: bool,
    use_dynamic_left_chunk: bool,
    decoding_chunk_size: int,
    static_chunk_size: int,
    num_decoding_left_chunks: int,
    enable_full_context: bool = True,
):
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            chunk_size = torch.randint(1, max_len, (1,)).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks, (1,)).item()
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size, num_left_chunks, xs.device)
        chunk_masks = chunk_masks.unsqueeze(0)
        chunk_masks = masks & chunk_masks
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size, num_left_chunks, xs.device)
        chunk_masks = chunk_masks.unsqueeze(0)
        chunk_masks = masks & chunk_masks
    else:
        chunk_masks = masks
    if (chunk_masks.sum(dim=-1) == 0).sum().item() != 0:
        chunk_masks[chunk_masks.sum(dim=-1) == 0] = True
    return chunk_masks


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    lengths = lengths.long()
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


# IntMeanFlow Utils


def get_intmeanflow_time_mixer(dims):
    layer = nn.Linear(dims * 2, dims, bias=False)
    with torch.no_grad():
        target_weight = torch.zeros(dims, 2 * dims)
        target_weight[:, 0:dims] = torch.eye(dims)
        layer.weight.data = target_weight
    return layer


# Matcha Transformers & Decoder Components


class SnakeBeta(nn.Module):
    def __init__(self, in_features, out_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.in_features = out_features if isinstance(out_features, list) else [out_features]
        self.proj = LoRACompatibleLinear(in_features, out_features)
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(self.in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(self.in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        x = self.proj(x)
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta
        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)
        return x


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)
        elif activation_fn == "snakebeta":
            act_fn = SnakeBeta(dim, inner_dim)

        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(LoRACompatibleLinear(inner_dim, dim_out))
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: int | None = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.norm2 = None
            self.attn2 = None

        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        timestep: torch.LongTensor | None = None,
        cross_attention_kwargs: dict[str, Any] = None,
        class_labels: torch.LongTensor | None = None,
    ):
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=encoder_attention_mask if self.only_cross_attention else attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm3(hidden_states)
        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states
        return hidden_states


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block1D(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: str | None = None,
        cond_proj_dim=None,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None
        self.act = nn.SiLU() if act_fn == "silu" else nn.ReLU()  # Simplified
        self.linear_2 = nn.Linear(time_embed_dim, out_dim or time_embed_dim)
        self.post_act = None

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class Upsample1D_Decoder(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=True, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        if self.use_conv_transpose:
            return self.conv(inputs)
        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            outputs = self.conv(outputs)
        return outputs


# Upsample Conformer Encoder


class Upsample1D_Encoder(nn.Module):
    def __init__(self, channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv = nn.Conv1d(self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        outputs = F.interpolate(inputs, scale_factor=float(self.stride), mode="nearest")
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride


class PreLookaheadLayer(nn.Module):
    def __init__(self, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=pre_lookahead_len + 1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.transpose(1, 2).contiguous()
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode="constant", value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        outputs = F.pad(outputs, (2, 0), mode="constant", value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()
        outputs = outputs + inputs
        return outputs


class UpsampleConformerEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        input_layer: str = "linear",
        pos_enc_layer_type: str = "rel_pos_espnet",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = False,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = False,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self._output_size = output_size
        self.global_cmvn = global_cmvn

        # Simplified subsample classes mapping (Hardcoded for Turbo)
        if input_layer == "linear":
            self.embed = LinearNoSubsampling(
                input_size, output_size, dropout_rate, RelPositionalEncoding(output_size, positional_dropout_rate)
            )
        else:
            self.embed = Conv2dSubsampling4(
                input_size, output_size, dropout_rate, RelPositionalEncoding(output_size, positional_dropout_rate)
            )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing
        activation = Swish()

        # self-attention module definition
        # Using RelPositionMultiHeadedAttention as default
        self_attn_class = RelPositionMultiHeadedAttention

        encoder_selfattn_layer_args = (attention_heads, output_size, attention_dropout_rate, key_bias)
        positionwise_layer_args = (output_size, linear_units, dropout_rate, activation)
        convolution_layer_args = (output_size, cnn_module_kernel, activation, cnn_module_norm, causal)

        self.pre_lookahead_layer = PreLookaheadLayer(channels=512, pre_lookahead_len=3)
        self.encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    self_attn_class(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args) if macaron_style else None,
                    ConvolutionModule(*convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(num_blocks)
            ]
        )
        self.up_layer = Upsample1D_Encoder(channels=512, out_channels=512, stride=2)

        # up_embed uses same classes
        if input_layer == "linear":
            self.up_embed = LinearNoSubsampling(
                input_size, output_size, dropout_rate, RelPositionalEncoding(output_size, positional_dropout_rate)
            )
        else:
            self.up_embed = Conv2dSubsampling4(
                input_size, output_size, dropout_rate, RelPositionalEncoding(output_size, positional_dropout_rate)
            )

        self.up_encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    self_attn_class(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args) if macaron_style else None,
                    ConvolutionModule(*convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(4)
            ]
        )

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            self.static_chunk_size,
            num_decoding_left_chunks,
        )
        xs = self.pre_lookahead_layer(xs)
        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)

        xs = xs.transpose(1, 2).contiguous()
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(1, 2).contiguous()
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        xs, pos_emb, masks = self.up_embed(xs, masks)
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            self.static_chunk_size * self.up_layer.stride,
            num_decoding_left_chunks,
        )
        xs = self.forward_up_layers(xs, chunk_masks, pos_emb, mask_pad)

        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_layers(
        self, xs: torch.Tensor, chunk_masks: torch.Tensor, pos_emb: torch.Tensor, mask_pad: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    def forward_up_layers(
        self, xs: torch.Tensor, chunk_masks: torch.Tensor, pos_emb: torch.Tensor, mask_pad: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.up_encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs


# Conditional Decoder


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    mask = mask.to(dtype)
    mask = (1.0 - mask) * -1.0e10
    return mask


class Transpose(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor):
        return torch.transpose(x, self.dim0, self.dim1)


class CausalBlock1D(Block1D):
    def __init__(self, dim: int, dim_out: int):
        super().__init__(dim, dim_out)
        self.block = torch.nn.Sequential(
            CausalConv1d(dim, dim_out, 3),
            Transpose(1, 2),
            nn.LayerNorm(dim_out),
            Transpose(1, 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        output = self.block(x * mask)
        return output * mask


class CausalResnetBlock1D(ResnetBlock1D):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super().__init__(dim, dim_out, time_emb_dim, groups)
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.causal_padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, self.causal_padding)
        x = super().forward(x)
        return x


class ConditionalDecoder(nn.Module):
    def __init__(
        self,
        in_channels=320,
        out_channels=80,
        causal=True,
        channels=[256],
        dropout=0.0,
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=12,
        num_heads=8,
        act_fn="gelu",
        meanflow=False,
    ):
        super().__init__()
        channels = tuple(channels)
        self.meanflow = meanflow
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(in_channels=in_channels, time_embed_dim=time_embed_dim, act_fn="silu")

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.static_chunk_size = 0

        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = (
                CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
                if self.causal
                else ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            )
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel)
                if not is_last
                else CausalConv1d(output_channel, output_channel, 3)
                if self.causal
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        for _ in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]
            resnet = (
                CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
                if self.causal
                else ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            )
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i] * 2
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2
            resnet = (
                CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
                if self.causal
                else ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            )
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D_Decoder(output_channel, use_conv_transpose=True)
                if not is_last
                else CausalConv1d(output_channel, output_channel, 3)
                if self.causal
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        self.final_block = (
            CausalBlock1D(channels[-1], channels[-1]) if self.causal else Block1D(channels[-1], channels[-1])
        )
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)
        self.initialize_weights()
        self.time_embed_mixer = None
        if self.meanflow:
            self.time_embed_mixer = get_intmeanflow_time_mixer(time_embed_dim)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, x, mask, mu, t, spks=None, cond=None, r=None):
        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)
        if self.meanflow:
            r = self.time_embeddings(r).to(t.dtype)
            r = self.time_mlp(r)
            concat_embed = torch.cat([t, r], dim=1)
            t = self.time_embed_mixer(concat_embed)

        x = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_down.bool(), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = rearrange(x, "b t c -> b c t").contiguous()
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_mid.bool(), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = rearrange(x, "b t c -> b c t").contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, : skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_up.bool(), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = rearrange(x, "b t c -> b c t").contiguous()
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask


# Conditional CFM


def cast_all(*args, dtype):
    return [a if (not a.dtype.is_floating_point) or a.dtype == dtype else a.to(dtype) for a in args]


class ConditionalCFM(nn.Module):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__()
        self.n_feats = in_channels
        self.cfm_params = cfm_params
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.estimator = estimator
        self.sigma_min = cfm_params.sigma_min

    @torch.inference_mode()
    def forward(
        self,
        mu,
        mask,
        n_timesteps,
        temperature=1.0,
        spks=None,
        cond=None,
        prompt_len=0,
        flow_cache=torch.zeros(1, 80, 0, 2),
    ):
        # placeholder for inheritance
        pass

    def solve_euler(self, x, t_span, mu, mask, spks, cond, meanflow=False):
        in_dtype = x.dtype
        x, t_span, mu, mask, spks, cond = cast_all(x, t_span, mu, mask, spks, cond, dtype=self.estimator.dtype)
        B, T = mu.size(0), x.size(2)

        # Allocate buffers
        x_in = torch.zeros([2 * B, 80, T], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2 * B, 1, T], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2 * B, 80, T], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([2 * B], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2 * B, 80], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2 * B, 80, T], device=x.device, dtype=x.dtype)
        r_in = torch.zeros([2 * B], device=x.device, dtype=x.dtype)

        for t, r in zip(t_span[:-1], t_span[1:]):
            t = t.unsqueeze(dim=0)
            r = r.unsqueeze(dim=0)

            x_in[:B] = x_in[B:] = x
            mask_in[:B] = mask_in[B:] = mask
            mu_in[:B] = mu
            t_in[:B] = t_in[B:] = t
            spks_in[:B] = spks
            cond_in[:B] = cond
            r_in[:B] = r_in[B:] = r

            dxdt = self.estimator.forward(
                x=x_in,
                mask=mask_in,
                mu=mu_in,
                t=t_in,
                spks=spks_in,
                cond=cond_in,
                r=r_in if meanflow else None,
            )
            dxdt, cfg_dxdt = torch.split(dxdt, [B, B], dim=0)
            dxdt = (1.0 + self.inference_cfg_rate) * dxdt - self.inference_cfg_rate * cfg_dxdt
            dt = r - t
            x = x + dt * dxdt

        return x.to(in_dtype)


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels=240, cfm_params=CFM_PARAMS, n_spks=1, spk_emb_dim=80, estimator=None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        self.rand_noise = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, noised_mels=None, meanflow=False):
        z = torch.randn_like(mu)
        if noised_mels is not None:
            prompt_len = mu.size(2) - noised_mels.size(2)
            z[..., prompt_len:] = noised_mels

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if (not meanflow) and (self.t_scheduler == "cosine"):
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        if meanflow:
            # Basic euler
            in_dtype = z.dtype
            x, t_span, mu, mask, spks, cond = cast_all(z, t_span, mu, mask, spks, cond, dtype=self.estimator.dtype)
            for t, r in tqdm(zip(t_span[..., :-1], t_span[..., 1:]), total=t_span.shape[-1] - 1):
                t, r = t[None], r[None]
                dxdt = self.estimator.forward(x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)
                dt = r - t
                x = x + dt * dxdt
            return x.to(in_dtype), None

        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, meanflow=meanflow), None


# Causal Masked Diff With Xvec


def _repeat_batch_dim(tnsr, batch_size, ndim):
    if tnsr is not None:
        while tnsr.ndim < ndim:
            tnsr = tnsr[None]
        if batch_size > 1 and tnsr.size(0) == 1:
            tnsr = tnsr.repeat(batch_size, *([1] * (ndim - 1)))
    return tnsr


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 6561,
        input_frame_rate: int = 25,
        only_mask_loss: bool = True,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        encoder: torch.nn.Module = None,
        decoder: torch.nn.Module = None,
        decoder_conf: dict = None,
        mel_feat_conf: dict = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        finalize,
        n_timesteps=10,
        noised_mels=None,
        meanflow=False,
    ):
        B = token.size(0)
        embedding = torch.atleast_2d(embedding)
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        prompt_token = _repeat_batch_dim(prompt_token, B, ndim=2)
        prompt_token_len = _repeat_batch_dim(prompt_token_len, B, ndim=1)
        prompt_feat = _repeat_batch_dim(prompt_feat, B, ndim=3)
        prompt_feat_len = _repeat_batch_dim(prompt_feat_len, B, ndim=1)
        embedding = _repeat_batch_dim(embedding, B, ndim=2)

        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)

        if (token >= self.vocab_size).any():
            print(f"{token.max()}>{self.vocab_size}\n out-of-range special tokens found in flow")
        token = self.input_embedding(token.long()) * mask

        h, h_masks = self.encoder(token, token_len)
        if finalize is False:
            h = h[:, : -self.pre_lookahead_len * self.token_mel_ratio]

        h_lengths = h_masks.sum(dim=-1).squeeze(dim=-1)
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        conds = torch.zeros([B, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(h_lengths)).unsqueeze(1).to(h)
        if mask.shape[0] != B:
            mask = mask.repeat(B, 1, 1)

        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask,
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
            noised_mels=noised_mels,
            meanflow=meanflow,
        )
        feat = feat[:, :, mel_len1:]
        return feat, None


# -----------------------------------------------------------------------------
# Chunk 3: Speaker Encoders, HiFiGAN, S3Tokenizer
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# CAMPPlus (TDNN Speaker Encoder)
# -----------------------------------------------------------------------------


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]
    return pad


def extract_feature(audio):
    features = []
    feature_times = []
    feature_lengths = []
    for au in audio:
        feature = Kaldi.fbank(au.unsqueeze(0), num_mel_bins=80)
        feature = feature - feature.mean(dim=0, keepdim=True)
        features.append(feature)
        feature_times.append(au.shape[0])
        feature_lengths.append(feature.shape[0])
    features_padded = pad_list(features, pad_value=0)
    return features_padded, feature_lengths, feature_times


class BasicResBlock_CAMP(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=(stride, 1), bias=False),
                torch.nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCM(torch.nn.Module):
    def __init__(self, block=BasicResBlock_CAMP, num_blocks=[2, 2], m_channels=32, feat_dim=80):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = torch.nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.conv2 = torch.nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))
        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


def get_nonlinear(config_str, channels):
    nonlinear = torch.nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", torch.nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", torch.nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", torch.nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", torch.nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f"Unexpected module ({name}).")
    return nonlinear


class StatsPool(torch.nn.Module):
    def forward(self, x):
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=True)
        return torch.cat([mean, std], dim=-1)


class TDNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
    ):
        super().__init__()
        if padding < 0:
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        return self.nonlinear(self.linear(x))


class CAMLayer(torch.nn.Module):
    def __init__(self, bn_channels, out_channels, kernel_size, stride, padding, dilation, bias, reduction=2):
        super().__init__()
        self.linear_local = torch.nn.Conv1d(
            bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.linear1 = torch.nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = (
            x.mean(-1, keepdim=True)
            + F.avg_pool1d(x, kernel_size=100, stride=100, ceil_mode=True)
            .unsqueeze(-1)
            .expand(*x.shape[:-1], 100)
            .reshape(*x.shape[:-1], -1)[..., : x.shape[-1]]
        )
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m


class CAMDenseTDNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = torch.nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = torch.utils.checkpoint.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        return self.cam_layer(self.nonlinear2(x))


class CAMDenseTDNNBlock(torch.nn.ModuleList):
    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super().__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module(f"tdnnd{i + 1}", layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, config_str="batchnorm-relu"):
        super().__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.linear(self.nonlinear(x))


class DenseLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, config_str="batchnorm-relu"):
        super().__init__()
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        return self.nonlinear(x)


class CAMPPlus(torch.nn.Module):
    def __init__(
        self,
        feat_dim=80,
        embedding_size=192,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
        output_level="segment",
        **kwargs,
    ):
        super().__init__()
        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.output_level = output_level
        self.xvector = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(channels, init_channels, 5, stride=2, dilation=1, padding=-1, config_str=config_str),
                    )
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module(f"block{i + 1}", block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                f"transit{i + 1}", TransitLayer(channels, channels // 2, bias=False, config_str=config_str)
            )
            channels //= 2
        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))
        if self.output_level == "segment":
            self.xvector.add_module("stats", StatsPool())
            self.xvector.add_module("dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_"))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)
        x = self.xvector(x)
        if self.output_level == "frame":
            x = x.transpose(1, 2)
        return x

    def inference(self, audio_list):
        speech, speech_lengths, speech_times = extract_feature(audio_list)
        results = self.forward(speech.to(next(self.parameters()).device).to(torch.float32))
        return results


# -----------------------------------------------------------------------------
# VoiceEncoder (LSTM Speaker Encoder)
# -----------------------------------------------------------------------------


def pack_sequences(arrays, seq_len: int = None, pad_value=0):
    if seq_len is None:
        seq_len = max(len(array) for array in arrays)
    if isinstance(arrays[0], list):
        arrays = [np.array(array) for array in arrays]
    device = arrays[0].device if isinstance(arrays[0], torch.Tensor) else None
    tensors = arrays if device else [torch.as_tensor(array) for array in arrays]
    packed_shape = (len(tensors), seq_len, *tensors[0].shape[1:])
    packed_tensor = torch.full(packed_shape, pad_value, dtype=tensors[0].dtype, device=device)
    for i, tensor in enumerate(tensors):
        packed_tensor[i, : tensor.size(0)] = tensor
    return packed_tensor


def get_num_wins(n_frames, step, min_coverage, hp):
    win_size = hp.ve_partial_frames
    n_wins, remainder = divmod(max(n_frames - win_size + step, 0), step)
    if n_wins == 0 or (remainder + (win_size - step)) / win_size >= min_coverage:
        n_wins += 1
    target_n = win_size + step * (n_wins - 1)
    return n_wins, target_n


def get_frame_step(overlap, rate, hp):
    if rate is None:
        frame_step = int(np.round(hp.ve_partial_frames * (1 - overlap)))
    else:
        frame_step = int(np.round((hp.sample_rate / rate) / hp.ve_partial_frames))
    return frame_step


class VoiceEncoder(nn.Module):
    def __init__(self, hp=VoiceEncConfig()):
        super().__init__()
        self.hp = hp
        self.lstm = nn.LSTM(self.hp.num_mels, self.hp.ve_hidden_size, num_layers=3, batch_first=True)
        if hp.flatten_lstm_params:
            self.lstm.flatten_parameters()
        self.proj = nn.Linear(self.hp.ve_hidden_size, self.hp.speaker_embed_size)
        self.similarity_weight = nn.Parameter(torch.tensor([10.0]), requires_grad=True)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]), requires_grad=True)

    def forward(self, mels: torch.FloatTensor):
        _, (hidden, _) = self.lstm(mels)
        raw_embeds = self.proj(hidden[-1])
        if self.hp.ve_final_relu:
            raw_embeds = F.relu(raw_embeds)
        return raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

    def inference(
        self, mels: torch.Tensor, mel_lens, overlap=0.5, rate: float = None, min_coverage=0.8, batch_size=None
    ):
        mel_lens = mel_lens.tolist() if torch.is_tensor(mel_lens) else mel_lens
        frame_step = get_frame_step(overlap, rate, self.hp)
        n_partials, target_lens = zip(
            *(get_num_wins(mel_len, frame_step, min_coverage, self.hp) for mel_len in mel_lens)
        )
        len_diff = max(target_lens) - mels.size(1)
        if len_diff > 0:
            pad = torch.full((mels.size(0), len_diff, self.hp.num_mels), 0, dtype=torch.float32)
            mels = torch.cat((mels, pad.to(mels.device)), dim=1)
        partials = [
            mel[i * frame_step : i * frame_step + self.hp.ve_partial_frames]
            for mel, n in zip(mels, n_partials)
            for i in range(n)
        ]
        partials = torch.stack(partials)
        n_chunks = int(np.ceil(len(partials) / (batch_size or len(partials))))
        partial_embeds = torch.cat([self(batch) for batch in partials.chunk(n_chunks)], dim=0).cpu()
        slices = np.concatenate(([0], np.cumsum(n_partials)))
        raw_embeds = [torch.mean(partial_embeds[start:end], dim=0) for start, end in zip(slices[:-1], slices[1:])]
        raw_embeds = torch.stack(raw_embeds)
        return raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

    def embeds_from_mels(self, mels, mel_lens=None, as_spk=False, batch_size=32, **kwargs):
        if isinstance(mels, list):
            mels = [np.asarray(mel) for mel in mels]
            mel_lens = [mel.shape[0] for mel in mels]
            mels = pack_sequences(mels)
        with torch.inference_mode():
            utt_embeds = self.inference(
                mels.to(next(self.parameters()).device), mel_lens, batch_size=batch_size, **kwargs
            ).numpy()
        if as_spk:
            utt_embeds = np.mean(utt_embeds, axis=0)
            return utt_embeds / np.linalg.norm(utt_embeds, 2)
        return utt_embeds

    def embeds_from_wavs(self, wavs, sample_rate, as_spk=False, batch_size=32, trim_top_db=20, **kwargs):
        # Assuming melspectrogram and librosa are available
        # Simplified version
        if sample_rate != self.hp.sample_rate:
            wavs = [librosa.resample(wav, orig_sr=sample_rate, target_sr=self.hp.sample_rate) for wav in wavs]
        if trim_top_db:
            wavs = [librosa.effects.trim(wav, top_db=trim_top_db)[0] for wav in wavs]
        mels = [melspectrogram(w, self.hp).T for w in wavs]  # Using global melspectrogram
        return self.embeds_from_mels(mels, as_spk=as_spk, batch_size=batch_size, **kwargs)


# -----------------------------------------------------------------------------
# HiFiGAN (HiFTGenerator)
# ----------------------------------------------------------------------------


class SineGen(torch.nn.Module):
    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).type(torch.float32)

    @torch.no_grad()
    def forward(self, f0):
        F_mat = torch.zeros((f0.size(0), self.harmonic_num + 1, f0.size(-1))).to(f0.device)
        for i in range(self.harmonic_num + 1):
            F_mat[:, i : i + 1, :] = f0 * (i + 1) / self.sampling_rate
        theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
        u_dist = Uniform(low=-np.pi, high=np.pi)
        phase_vec = u_dist.sample(sample_shape=(f0.size(0), self.harmonic_num + 1, 1)).to(F_mat.device)
        phase_vec[:, 0, :] = 0
        sine_waves = self.sine_amp * torch.sin(theta_mat + phase_vec)
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    def __init__(
        self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshold=0
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x.transpose(1, 2))
            sine_wavs = sine_wavs.transpose(1, 2)
            uv = uv.transpose(1, 2)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class ResBlock(torch.nn.Module):
    def __init__(self, channels: int = 512, kernel_size: int = 3, dilations: list[int] = [1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for dilation in dilations:
            self.convs1.append(
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation,
                        padding=int((kernel_size * dilation - dilation) / 2),
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=int((kernel_size * 1 - 1) / 2))
                )
            )
        self.activations1 = nn.ModuleList([Snake(channels, alpha_logscale=False) for _ in range(len(self.convs1))])
        self.activations2 = nn.ModuleList([Snake(channels, alpha_logscale=False) for _ in range(len(self.convs2))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for idx in range(len(self.convs1)):
            remove_weight_norm(self.convs1[idx])
            remove_weight_norm(self.convs2[idx])


class HiFTGenerator(nn.Module):
    def __init__(
        self,
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=22050,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 8],
        upsample_kernel_sizes=[16, 16],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        f0_predictor=None,
    ):
        super().__init__()
        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate,
            np.prod(upsample_rates) * istft_params["hop_len"],
            nb_harmonics,
            nsf_alpha,
            nsf_sigma,
            nsf_voiced_threshold,
        )
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates) * istft_params["hop_len"])
        self.conv_pre = weight_norm(Conv1d(in_channels, base_channels, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        base_channels // (2**i), base_channels // (2 ** (i + 1)), k, u, padding=(k - u) // 2
                    )
                )
            )
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(
            zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)
        ):
            if u == 1:
                self.source_downs.append(Conv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), 1, 1))
            else:
                self.source_downs.append(
                    Conv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), u * 2, u, padding=(u // 2))
                )
            self.source_resblocks.append(ResBlock(base_channels // (2 ** (i + 1)), k, d))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2 ** (i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, istft_params["n_fft"] + 2, 7, 1, padding=3))
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft_window = torch.from_numpy(get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32))
        self.f0_predictor = f0_predictor

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        self.m_source.remove_weight_norm()
        for layer in self.source_downs:
            remove_weight_norm(layer)
        for layer in self.source_resblocks:
            layer.remove_weight_norm()

    def _stft(self, x):
        spec = torch.stft(
            x,
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(x.device),
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        return torch.istft(
            torch.complex(real, img),
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(magnitude.device),
        )

    def decode(self, x: torch.Tensor, s: torch.Tensor = torch.zeros(1, 1, 0)) -> torch.Tensor:
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, : self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1 :, :])
        x = self._istft(magnitude, phase)
        return torch.clamp(x, -self.audio_limit, self.audio_limit)

    @torch.inference_mode()
    def inference(self, speech_feat: torch.Tensor, cache_source: torch.Tensor = torch.zeros(1, 1, 0)) -> torch.Tensor:
        f0 = self.f0_predictor(speech_feat)
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        if cache_source.shape[2] != 0:
            s[:, :, : cache_source.shape[2]] = cache_source
        generated_speech = self.decode(x=speech_feat, s=s)
        return generated_speech, s


# -----------------------------------------------------------------------------
# S3Tokenizer
# -----------------------------------------------------------------------------

from s3tokenizer.model_v2 import ModelConfig, S3TokenizerV2  # noqa: E402
from s3tokenizer.utils import padding  # noqa: E402

S3_SR = 16_000
S3_HOP = 160
S3_TOKEN_HOP = 640
S3_TOKEN_RATE = 25
SPEECH_VOCAB_SIZE = 6561


class S3Tokenizer(S3TokenizerV2):
    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(self, name: str = "speech_tokenizer_v2_25hz", config: ModelConfig = ModelConfig()):
        super().__init__(name)
        self.n_fft = 400
        _mel_filters = librosa.filters.mel(sr=S3_SR, n_fft=self.n_fft, n_mels=config.n_mels)
        self.register_buffer("_mel_filters", torch.FloatTensor(_mel_filters))
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def pad(self, wavs, sr) -> list[torch.Tensor]:
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            n_tokens = (wav.shape[1] / sr) * S3_TOKEN_RATE
            n_tokens = np.ceil(n_tokens)
            intended_wav_len = int(n_tokens * (sr / S3_TOKEN_RATE))
            wav = F.pad(wav, (0, intended_wav_len - wav.shape[-1]), mode="constant", value=0)
            processed_wavs.append(wav)
        return processed_wavs

    def _prepare_audio(self, wavs):
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            processed_wavs.append(wav)
        return processed_wavs

    @torch.no_grad()
    def forward(
        self, wavs: torch.Tensor, accelerator=None, max_len: int = None
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        processed_wavs = self._prepare_audio(wavs)
        mels, mel_lens = [], []
        for wav in processed_wavs:
            wav = wav.to(self.device)
            mel = self.log_mel_spectrogram(wav)
            if max_len is not None:
                mel = mel[..., : max_len * 4]
            mels.append(mel.squeeze(0))
        mels, mel_lens = padding(mels)
        tokenizer = self if accelerator is None else accelerator.unwrap_model(self)
        speech_tokens, speech_token_lens = tokenizer.quantize(mels, mel_lens.to(self.device))
        return speech_tokens.long().detach(), speech_token_lens.long().detach()

    def log_mel_spectrogram(self, audio: torch.Tensor, padding: int = 0):
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        stft = torch.stft(audio, self.n_fft, S3_HOP, window=self.window.to(self.device), return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self._mel_filters.to(self.device) @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec


# -----------------------------------------------------------------------------
# Chunk 4: S3Gen, T3, ChatterboxTurboTTS
# -----------------------------------------------------------------------------

# import perth

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# ConvRNNF0Predictor
# -----------------------------------------------------------------------------


class ConvRNNF0Predictor(nn.Module):
    def __init__(self, num_class: int = 1, in_channels: int = 80, cond_channels: int = 512):
        super().__init__()
        self.num_class = num_class
        try:
            from torch.nn.utils.parametrizations import weight_norm
        except ImportError:
            from torch.nn.utils import weight_norm

        self.condnet = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.condnet(x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


# -----------------------------------------------------------------------------
# S3Gen (S3Token2Wav / S3Token2Mel Logic)
# -----------------------------------------------------------------------------


@lru_cache(100)
def get_resampler(src_sr, dst_sr, device):
    return ta.transforms.Resample(src_sr, dst_sr).to(device)


class S3Token2Mel(torch.nn.Module):
    def __init__(self, meanflow=False):
        super().__init__()
        self.tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
        self.mel_extractor = melspectrogram  # from chunk 1
        self.speaker_encoder = CAMPPlus(memory_efficient=False)
        self.meanflow = meanflow

        encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer="linear",
            pos_enc_layer_type="rel_pos_espnet",
            selfattention_layer_type="rel_selfattn",
            input_size=512,
            use_cnn_module=False,
            macaron_style=False,
        )

        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn="gelu",
            meanflow=self.meanflow,
        )
        cfm_params = CFM_PARAMS  # from chunk 1
        decoder = CausalConditionalCFM(spk_emb_dim=80, cfm_params=cfm_params, estimator=estimator)
        self.flow = CausalMaskedDiffWithXvec(encoder=encoder, decoder=decoder)

    @property
    def device(self):
        return next(self.tokenizer.parameters()).device

    @property
    def dtype(self):
        return next(self.flow.parameters()).dtype

    def embed_ref(self, ref_wav, ref_sr, device="auto", ref_fade_out=True):
        device = self.device if device == "auto" else device
        if isinstance(ref_wav, np.ndarray):
            ref_wav = torch.from_numpy(ref_wav).float()
        if ref_wav.device != device:
            ref_wav = ref_wav.to(device)
        if len(ref_wav.shape) == 1:
            ref_wav = ref_wav.unsqueeze(0)

        ref_wav_24 = ref_wav
        if ref_sr != S3GEN_SR:
            ref_wav_24 = get_resampler(ref_sr, S3GEN_SR, device)(ref_wav)
        ref_wav_24 = ref_wav_24.to(device=device, dtype=self.dtype)
        ref_mels_24 = self.mel_extractor(ref_wav_24).transpose(1, 2).to(dtype=self.dtype)

        ref_wav_16 = ref_wav
        if ref_sr != 16000:
            ref_wav_16 = get_resampler(ref_sr, 16000, device)(ref_wav)
        ref_x_vector = self.speaker_encoder.inference(ref_wav_16.to(dtype=self.dtype))
        ref_speech_tokens, ref_speech_token_lens = self.tokenizer(ref_wav_16.float())

        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            ref_speech_tokens = ref_speech_tokens[:, : ref_mels_24.shape[1] // 2]
            ref_speech_token_lens[0] = ref_speech_tokens.shape[1]

        return dict(
            prompt_token=ref_speech_tokens.to(device),
            prompt_token_len=ref_speech_token_lens,
            prompt_feat=ref_mels_24,
            prompt_feat_len=None,
            embedding=ref_x_vector,
        )

    def forward(
        self,
        speech_tokens,
        ref_wav: torch.Tensor | None,
        ref_sr: int | None,
        ref_dict: dict | None = None,
        n_cfm_timesteps=None,
        finalize: bool = False,
        speech_token_lens=None,
        noised_mels=None,
    ):
        if ref_dict is None:
            ref_dict = self.embed_ref(ref_wav, ref_sr)
        else:
            for rk in list(ref_dict):
                if isinstance(ref_dict[rk], np.ndarray):
                    ref_dict[rk] = torch.from_numpy(ref_dict[rk])
                if torch.is_tensor(ref_dict[rk]):
                    ref_dict[rk] = ref_dict[rk].to(device=self.device, dtype=self.dtype)

        speech_tokens = torch.atleast_2d(speech_tokens)
        if speech_token_lens is None:
            speech_token_lens = torch.LongTensor([st.size(-1) for st in speech_tokens]).to(self.device)

        output_mels, _ = self.flow.inference(
            token=speech_tokens,
            token_len=speech_token_lens,
            finalize=finalize,
            noised_mels=noised_mels,
            n_timesteps=n_cfm_timesteps,
            meanflow=self.meanflow,
            **ref_dict,
        )

        return output_mels


class S3Gen(S3Token2Mel):
    def __init__(self, meanflow=False):
        super().__init__(meanflow)
        f0_predictor = ConvRNNF0Predictor()
        self.mel2wav = HiFTGenerator(
            sampling_rate=S3GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )
        n_trim = S3GEN_SR // 50
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False)

    @torch.inference_mode()
    def flow_inference(
        self,
        speech_tokens,
        ref_wav=None,
        ref_sr=None,
        ref_dict=None,
        n_cfm_timesteps=None,
        finalize=False,
        speech_token_lens=None,
    ):
        n_cfm_timesteps = n_cfm_timesteps or (2 if self.meanflow else 10)
        noise = None
        if self.meanflow:
            noise = torch.randn(1, 80, speech_tokens.size(-1) * 2, dtype=self.dtype, device=self.device)

        return super().forward(
            speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=finalize,
            noised_mels=noise,
        )

    @torch.inference_mode()
    def hift_inference(self, speech_feat, cache_source: torch.Tensor = None):
        if cache_source is None:
            cache_source = torch.zeros(1, 1, 0).to(device=self.device, dtype=self.dtype)
        return self.mel2wav.inference(speech_feat=speech_feat, cache_source=cache_source)

    @torch.inference_mode()
    def inference(
        self, speech_tokens, ref_wav=None, ref_sr=None, ref_dict=None, n_cfm_timesteps=None, speech_token_lens=None
    ):
        with torch.profiler.record_function("Flow Inference"):
            output_mels = self.flow_inference(
                speech_tokens,
                speech_token_lens=speech_token_lens,
                ref_wav=ref_wav,
                ref_sr=ref_sr,
                ref_dict=ref_dict,
                n_cfm_timesteps=n_cfm_timesteps,
                finalize=True,
            )

        output_mels = output_mels.to(dtype=self.dtype)

        with torch.profiler.record_function("HIFT Inference"):
            output_wavs, output_sources = self.hift_inference(output_mels, None)

        output_wavs[:, : len(self.trim_fade)] *= self.trim_fade

        return output_wavs, output_sources

    @torch.inference_mode()
    def flow_inference_chunk(self, speech_tokens, ref_dict, context_tokens=None, n_cfm_timesteps=None, is_final=False):
        """
        Process a chunk of tokens with optional context overlap.

        Args:
            speech_tokens: Current chunk tokens (1, chunk_size)
            ref_dict: Reference embeddings dictionary
            context_tokens: Overlap tokens from previous chunk for continuity
            n_cfm_timesteps: Number of CFM solver steps
            is_final: Whether this is the final chunk (unused, always finalizes)

        Returns:
            output_mels: Mel spectrogram for this chunk (1, 80, mel_len)
        """
        n_cfm_timesteps = n_cfm_timesteps or (2 if self.meanflow else 10)

        if context_tokens is not None and context_tokens.numel() > 0:
            # Prepend context tokens for encoder continuity
            full_tokens = torch.cat([context_tokens, speech_tokens], dim=-1)
            context_len = context_tokens.size(-1)
        else:
            full_tokens = speech_tokens
            context_len = 0

        # Use flow_inference which handles noise generation correctly
        # Always finalize=True because we handle continuity via token overlap,
        # not mel frame trimming
        output_mels = self.flow_inference(
            full_tokens,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=True,  # Always finalize - we don't need lookahead trimming
            speech_token_lens=None,
        )

        if context_len > 0:
            # Trim context mel frames from output (2 mels per token)
            context_mel_len = context_len * 2
            output_mels = output_mels[:, :, context_mel_len:]

        return output_mels

    @torch.inference_mode()
    def hift_inference_stream(self, speech_feat, cache_source=None, prev_audio_tail=None, crossfade_len=480):
        """
        Vocoder with overlap-add crossfade support.

        Args:
            speech_feat: Mel spectrogram (1, 80, mel_len)
            cache_source: Source cache from previous chunk
            prev_audio_tail: Last crossfade_len samples from previous chunk
            crossfade_len: Number of samples to crossfade (default 480 = 20ms)

        Returns:
            audio: Generated audio tensor
            source: Source features for next chunk's cache
            audio_tail: Last crossfade_len samples for next chunk
        """
        if cache_source is None:
            cache_source = torch.zeros(1, 1, 0).to(device=self.device, dtype=self.dtype)

        audio, source = self.mel2wav.inference(speech_feat=speech_feat, cache_source=cache_source)
        audio_tail = None

        # # Clone to allow in-place operations (inference tensors are read-only)
        # audio = audio.clone()

        # # Handle both 1D (samples,) and 2D (batch, samples) audio
        # is_2d = audio.dim() == 2
        # if is_2d:
        #     # Work with last dimension for 2D tensors
        #     audio_len = audio.size(-1)
        # else:
        #     audio_len = audio.size(0)

        # if prev_audio_tail is not None and prev_audio_tail.numel() > 0:
        #     # Create crossfade windows
        #     fade_out = torch.linspace(1, 0, crossfade_len, device=audio.device, dtype=audio.dtype)
        #     fade_in = torch.linspace(0, 1, crossfade_len, device=audio.device, dtype=audio.dtype)

        #     if is_2d:
        #         # Apply crossfade for 2D tensor
        #         prev_faded = prev_audio_tail * fade_out
        #         current_faded = audio[:, :crossfade_len] * fade_in
        #         audio[:, :crossfade_len] = prev_faded + current_faded
        #     else:
        #         # Apply crossfade for 1D tensor
        #         prev_faded = prev_audio_tail * fade_out
        #         current_faded = audio[:crossfade_len] * fade_in
        #         audio[:crossfade_len] = prev_faded + current_faded

        # # Save tail for next chunk
        # if is_2d:
        #     audio_tail = audio[:, -crossfade_len:].clone() if audio_len >= crossfade_len else audio.clone()
        # else:
        #     audio_tail = audio[-crossfade_len:].clone() if audio_len >= crossfade_len else audio.clone()

        return audio, source, audio_tail


class TurboStreamer:
    """
    Orchestrates streaming TTS by buffering tokens and processing in chunks.

    Manages the token buffer, context overlap, and audio crossfading to produce
    a continuous audio stream from incrementally generated tokens.
    """

    def __init__(
        self,
        s3gen: S3Gen,
        ref_dict: dict,
        chunk_size: int = 10,
        overlap_tokens: int = 2,
        n_cfm_steps: int = 2,
        crossfade_samples: int = 480,
    ):
        """
        Args:
            s3gen: S3Gen model instance
            ref_dict: Reference embeddings for voice cloning
            chunk_size: Number of tokens per processing chunk
            overlap_tokens: Tokens to keep as context between chunks
            n_cfm_steps: CFM solver steps per chunk
            crossfade_samples: Audio samples to crossfade between chunks
        """
        self.s3gen = s3gen
        self.ref_dict = ref_dict
        self.chunk_size = chunk_size
        self.overlap_tokens = overlap_tokens
        self.n_cfm_steps = n_cfm_steps
        self.crossfade_samples = crossfade_samples

        # State
        self.token_buffer = []
        self.full_buffer = []
        self.context_tokens = None
        self.prev_audio_tail = None
        self.source_cache = None
        self.is_first_chunk = True

    def reset(self):
        """Reset state for new generation."""
        self.token_buffer = []
        self.full_buffer = []
        self.context_tokens = None
        self.prev_audio_tail = None
        self.source_cache = None
        self.is_first_chunk = True

    def add_token(self, token: torch.Tensor):
        """
        Add a token to the buffer.

        Args:
            token: Single token tensor of shape (1, 1)

        Returns:
            audio_chunk: Audio if buffer is full, else None
        """
        self.token_buffer.append(token)
        self.full_buffer.append(token)

        if len(self.token_buffer) >= self.chunk_size:
            return self._process_chunk(is_final=False)
        return None

    def finalize(self):
        """Process remaining tokens in buffer."""
        if self.token_buffer:
            dev = self.token_buffer[0].device
            tiff = len(self.token_buffer) % self.chunk_size
            if tiff > 0:
                self.token_buffer += [torch.tensor([[S3GEN_SIL]]).to(dev)] * (self.chunk_size - tiff)
            return self._process_chunk(is_final=True)
        return None

    @torch.inference_mode()
    def _process_chunk(self, is_final: bool = False):
        """Process buffered tokens into audio."""
        if not self.token_buffer:
            return None

        # Stack buffered tokens
        chunk_tokens = torch.cat(self.token_buffer, dim=1)

        # Filter out special tokens
        valid_mask = chunk_tokens < 6561
        chunk_tokens = chunk_tokens[valid_mask].unsqueeze(0)

        if chunk_tokens.numel() == 0:
            self.token_buffer = []
            return None

        # Flow inference with context
        output_mels = self.s3gen.flow_inference_chunk(
            speech_tokens=chunk_tokens,
            ref_dict=self.ref_dict,
            context_tokens=self.context_tokens,
            n_cfm_timesteps=self.n_cfm_steps,
            is_final=is_final,
        )

        output_mels = output_mels.to(dtype=self.s3gen.dtype)

        # Vocoder with crossfade
        audio, self.source_cache, self.prev_audio_tail = self.s3gen.hift_inference_stream(
            speech_feat=output_mels,
            cache_source=self.source_cache,
            prev_audio_tail=self.prev_audio_tail if not self.is_first_chunk else None,
            crossfade_len=self.crossfade_samples,
        )

        # Apply fade-in on first chunk
        # if self.is_first_chunk:
        #     n_trim = S3GEN_SR // 50
        #     trim_fade = torch.zeros(2 * n_trim, device=audio.device, dtype=audio.dtype)
        #     trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim, device=audio.device)) + 1) / 2
        #     # Audio is 1D (after HiFT returns squeezed output)
        #     if audio.dim() == 1 and audio.size(0) >= len(trim_fade):
        #         audio[:len(trim_fade)] = audio[:len(trim_fade)] * trim_fade
        #     elif audio.dim() == 2 and audio.size(-1) >= len(trim_fade):
        #         audio[:, :len(trim_fade)] = audio[:len(trim_fade)] * trim_fade
        #     self.is_first_chunk = False

        # Update context for next chunk
        if self.overlap_tokens > 0:
            all_tokens = torch.cat(self.full_buffer, dim=1)
            self.context_tokens = all_tokens[:, -self.overlap_tokens :]

        # Clear buffer
        self.token_buffer = []

        return audio.squeeze(0).detach()

    def process_token_stream(self, token_generator):
        """
        Process a stream of tokens into audio chunks.

        Args:
            token_generator: Iterator yielding torch.Tensor tokens

        Yields:
            torch.Tensor: Audio chunks
        """
        self.reset()

        for token in token_generator:
            audio_chunk = self.add_token(token)
            if audio_chunk is not None:
                yield audio_chunk

        # Finalize remaining tokens
        final_chunk = self.finalize()
        if final_chunk is not None:
            yield final_chunk


# -----------------------------------------------------------------------------
# T3 Components
# -----------------------------------------------------------------------------


@dataclass
class T3Cond:
    speaker_emb: torch.Tensor
    clap_emb: torch.Tensor | None = None
    cond_prompt_speech_tokens: torch.Tensor | None = None
    cond_prompt_speech_emb: torch.Tensor | None = None
    emotion_adv: torch.Tensor | None = 0.5

    def to(self, *, device=None, dtype=None):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                is_fp = type(v.view(-1)[0].item()) is not int
                setattr(self, k, v.to(device=device, dtype=dtype if is_fp else None))
        return self


class T3CondEnc(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        if hp.encoder_type == "voice_encoder":
            self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)
        else:
            raise NotImplementedError(str(hp.encoder_type))
        self.emotion_adv_fc = None
        if hp.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False)

    def forward(self, cond: T3Cond):
        cond_spkr = self.spkr_enc(cond.speaker_emb.view(-1, self.hp.speaker_embed_size))[:, None]
        empty = torch.zeros_like(cond_spkr[:, :0])
        cond_clap = empty
        cond_prompt_speech_emb = cond.cond_prompt_speech_emb if cond.cond_prompt_speech_emb is not None else empty
        cond_emotion_adv = empty
        if self.hp.emotion_adv:
            cond_emotion_adv = self.emotion_adv_fc(cond.emotion_adv.view(-1, 1, 1))
        return torch.cat((cond_spkr, cond_clap, cond_prompt_speech_emb, cond_emotion_adv), dim=1)


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        return self.emb(torch.arange(0, x.shape[1], device=x.device))

    def get_fixed_embedding(self, idx):
        device = self.emb.weight.device
        idx = idx.to(device) if torch.is_tensor(idx) else torch.tensor(idx, device=device)
        return self.emb(torch.atleast_2d(idx))


class T3(nn.Module):
    def __init__(self, hp=None):
        if hp is None:
            hp = T3Config.english_only()
        super().__init__()
        self.hp = hp
        config_dict = LLAMA_CONFIGS[hp.llama_config_name]
        self.is_gpt = config_dict.get("model_type") == "gpt2"
        if self.is_gpt:
            self.cfg = GPT2Config(**config_dict)
            self.tfmr = GPT2Model(self.cfg)
        else:
            self.cfg = LlamaConfig(**config_dict)
            self.tfmr = LlamaModel(self.cfg)
        self.dim = self.cfg.hidden_size
        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)
        self.text_pos_emb = None
        self.speech_pos_emb = None
        if hp.input_pos_emb == "learned":
            self.text_pos_emb = LearnedPositionEmbeddings(hp.max_text_tokens + 2, self.dim)
            self.speech_pos_emb = LearnedPositionEmbeddings(hp.max_speech_tokens + 4, self.dim)
        self.text_head = nn.Linear(self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=self.is_gpt)

    @property
    def device(self):
        return self.speech_head.weight.device

    def prepare_conditioning(self, t3_cond: T3Cond):
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens)
            if not self.is_gpt:
                t3_cond.cond_prompt_speech_emb += self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)

    def prepare_input_embeds(self, *, t3_cond, text_tokens, speech_tokens, cfg_weight=0.0):
        cond_emb = self.prepare_conditioning(t3_cond)
        text_emb = self.text_emb(text_tokens)
        if cfg_weight > 0.0 and not self.is_gpt:
            text_emb[1].zero_()
        speech_emb = self.speech_emb(speech_tokens)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        if cond_emb.size(0) != text_emb.size(0):
            cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)
        embeds = torch.stack([torch.cat((ce, te, se)) for ce, te, se in zip(cond_emb, text_emb, speech_emb)])
        return embeds, cond_emb.size(1)

    @torch.inference_mode()
    def inference_turbo(
        self, t3_cond, text_tokens, temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2, max_gen_len=1000
    ):
        logits_processors = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            logits_processors.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            logits_processors.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            logits_processors.append(TopPLogitsWarper(top_p))
        if repetition_penalty != 1.0:
            logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

        speech_start_token = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = self.prepare_input_embeds(
            t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=speech_start_token, cfg_weight=0.0
        )
        generated_speech_tokens = []
        llm_outputs = self.tfmr(inputs_embeds=embeds, use_cache=True)
        past_key_values = llm_outputs.past_key_values
        speech_hidden = llm_outputs[0][:, -1:]
        speech_logits = self.speech_head(speech_hidden)
        processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
        probs = F.softmax(processed_logits, dim=-1)
        next_speech_token = torch.multinomial(probs, num_samples=1)
        generated_speech_tokens.append(next_speech_token)
        current_speech_token = next_speech_token

        for _ in tqdm(range(max_gen_len)):
            current_speech_embed = self.speech_emb(current_speech_token)
            llm_outputs = self.tfmr(inputs_embeds=current_speech_embed, past_key_values=past_key_values, use_cache=True)
            past_key_values = llm_outputs.past_key_values
            speech_logits = self.speech_head(llm_outputs[0])
            input_ids = torch.cat(generated_speech_tokens, dim=1)
            processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])
            if torch.all(processed_logits == -float("inf")):
                break
            probs = F.softmax(processed_logits, dim=-1)
            next_speech_token = torch.multinomial(probs, num_samples=1)
            generated_speech_tokens.append(next_speech_token)
            current_speech_token = next_speech_token
            if torch.all(next_speech_token == self.hp.stop_speech_token):
                break
        all_tokens = torch.cat(generated_speech_tokens, dim=1)
        if all_tokens.size(1) > 0 and all_tokens[0, -1] == self.hp.stop_speech_token:
            all_tokens = all_tokens[:, :-1]
        return all_tokens

    @torch.inference_mode()
    def inference_stream(
        self, t3_cond, text_tokens, temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2, max_gen_len=1000
    ):
        """
        Streaming token generator - yields speech tokens one at a time.

        Yields:
            torch.Tensor: Single speech token tensor of shape (1, 1)
        """
        logits_processors = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            logits_processors.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            logits_processors.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            logits_processors.append(TopPLogitsWarper(top_p))
        if repetition_penalty != 1.0:
            logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

        speech_start_token = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = self.prepare_input_embeds(
            t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=speech_start_token, cfg_weight=0.0
        )

        generated_speech_tokens = []
        llm_outputs = self.tfmr(inputs_embeds=embeds, use_cache=True)
        past_key_values = llm_outputs.past_key_values
        speech_hidden = llm_outputs[0][:, -1:]
        speech_logits = self.speech_head(speech_hidden)
        processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
        probs = F.softmax(processed_logits, dim=-1)
        next_speech_token = torch.multinomial(probs, num_samples=1)
        generated_speech_tokens.append(next_speech_token)
        current_speech_token = next_speech_token

        # Yield first token
        yield next_speech_token

        for _ in range(max_gen_len):
            current_speech_embed = self.speech_emb(current_speech_token)
            llm_outputs = self.tfmr(inputs_embeds=current_speech_embed, past_key_values=past_key_values, use_cache=True)
            past_key_values = llm_outputs.past_key_values
            speech_logits = self.speech_head(llm_outputs[0])
            input_ids = torch.cat(generated_speech_tokens, dim=1)
            processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])

            if torch.all(processed_logits == -float("inf")):
                break

            probs = F.softmax(processed_logits, dim=-1)
            next_speech_token = torch.multinomial(probs, num_samples=1)

            # Check for stop token before yielding
            if torch.all(next_speech_token == self.hp.stop_speech_token):
                break

            generated_speech_tokens.append(next_speech_token)
            current_speech_token = next_speech_token

            yield next_speech_token


# -----------------------------------------------------------------------------
# ChatterboxTurboTTS
# -----------------------------------------------------------------------------


@dataclass
class Conditionals:
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs["t3"]), kwargs["gen"])


class ChatterboxTurboTTS:
    ENC_COND_LEN = 15 * 16000
    DEC_COND_LEN = 10 * S3GEN_SR
    REPO_ID = "ResembleAI/chatterbox-turbo"

    def __init__(self, t3: T3, s3gen: S3Gen, ve: VoiceEncoder, tokenizer, device: str, conds: Conditionals = None):
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        # self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> "ChatterboxTurboTTS":
        ckpt_dir = Path(ckpt_dir)
        map_location = torch.device("cpu") if device in ["cpu", "mps"] else None

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        hp = T3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False

        t3 = T3(hp)
        t3_state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        del t3.tfmr.wte
        t3.to(device).eval()

        s3gen = S3Gen(meanflow=True)
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen_meanflow.safetensors"), strict=True)
        s3gen.to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        conds = None
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)
        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> "ChatterboxTurboTTS":
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        local_path = snapshot_download(
            repo_id=cls.REPO_ID, token=False, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )
        return cls.from_local(local_path, device)

    def norm_loudness(self, wav, sr, target_lufs=-27):
        try:
            meter = ln.Meter(sr)
            loudness = meter.integrated_loudness(wav)
            gain_db = target_lufs - loudness
            gain_linear = 10.0 ** (gain_db / 20.0)
            if math.isfinite(gain_linear) and gain_linear > 0.0:
                wav = wav * gain_linear
        except Exception as e:
            print(f"Warning: Error in norm_loudness, skipping: {e}")
        return wav

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5, norm_loudness=True):
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        assert len(s3gen_ref_wav) / _sr > 5.0, "Audio prompt must be longer than 5 seconds!"
        if norm_loudness:
            s3gen_ref_wav = self.norm_loudness(s3gen_ref_wav, _sr)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=16000)
        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[: self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=16000))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)
        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.00,
        top_p=0.95,
        audio_prompt_path=None,
        exaggeration=0.0,
        cfg_weight=0.0,
        temperature=0.8,
        top_k=1000,
        norm_loudness=True,
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration, norm_loudness=norm_loudness)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Punctuation normalization (inlined)
        if len(text) == 0:
            text = "You need to add some text for me to talk."
        if text[0].islower():
            text = text[0].upper() + text[1:]
        text = " ".join(text.split())
        for old, new in [
            ("…", ", "),
            (":", ","),
            ("—", "-"),
            ("–", "-"),
            (" ,", ","),
            ("“", '"'),
            ("”", '"'),
            ("‘", "'"),
            ("’", "'"),
        ]:
            text = text.replace(old, new)
        text = text.rstrip(" ")
        if not any(text.endswith(p) for p in {".", "!", "?", "-", ","}):
            text += "."

        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)

        with torch.profiler.record_function("T3 Inference"):
            speech_tokens = self.t3.inference_turbo(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        speech_tokens = speech_tokens[speech_tokens < 6561].to(self.device)
        silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(self.device)
        speech_tokens = torch.cat([speech_tokens, silence])

        with torch.profiler.record_function("S3 Gen Inference"):
            wav, _ = self.s3gen.inference(speech_tokens=speech_tokens, ref_dict=self.conds.gen, n_cfm_timesteps=2)

        wav = wav.squeeze(0).detach().cpu().numpy()

        return wav

        # with torch.profiler.record_function("Apply Watermark"):
        #     watermarked_wav = self.watermarker.apply_watermark(
        #         wav,
        #         sample_rate=self.sr
        #     )

        # return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def generate_stream(
        self,
        text,
        chunk_size: int = 10,
        overlap_tokens: int = 2,
        repetition_penalty: float = 1.2,
        top_p: float = 0.95,
        audio_prompt_path: str = None,
        exaggeration: float = 0.0,
        temperature: float = 0.8,
        top_k: int = 1000,
        norm_loudness: bool = True,
    ):
        """
        Streaming TTS generation - yields audio chunks as they're generated.

        Args:
            text: Text to synthesize
            chunk_size: Number of tokens per audio chunk (default 10 = ~400ms)
            overlap_tokens: Context tokens between chunks (default 2)
            repetition_penalty: Repetition penalty for token generation
            top_p: Top-p sampling parameter
            audio_prompt_path: Optional path to voice reference audio
            exaggeration: Voice exaggeration factor
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            norm_loudness: Whether to normalize audio loudness

        Yields:
            numpy.ndarray: Audio chunks at 24kHz sample rate
        """
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration, norm_loudness=norm_loudness)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Punctuation normalization (same as generate)
        if len(text) == 0:
            text = "You need to add some text for me to talk."
        if text[0].islower():
            text = text[0].upper() + text[1:]
        text = " ".join(text.split())
        for old, new in [
            ("…", ", "),
            (":", ","),
            ("—", "-"),
            ("–", "-"),
            (" ,", ","),
            (""", "\""), (""", '"'),
            ("'", "'"),
            ("'", "'"),
        ]:
            text = text.replace(old, new)
        text = text.rstrip(" ")
        if not any(text.endswith(p) for p in {".", "!", "?", "-", ","}):
            text += "."

        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)

        # Create token generator
        token_gen = self.t3.inference_stream(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Create streaming orchestrator
        streamer = TurboStreamer(
            s3gen=self.s3gen,
            ref_dict=self.conds.gen,
            chunk_size=chunk_size,
            overlap_tokens=overlap_tokens,
            n_cfm_steps=2,
            crossfade_samples=480,
        )

        # Stream audio chunks
        for audio_chunk in streamer.process_token_stream(token_gen):
            yield audio_chunk.cpu().numpy()
