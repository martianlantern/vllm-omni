# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chatterbox Turbo T3 model for vllm-omni.

T3 is the autoregressive text-to-speech token generation stage.
It uses a GPT2-based transformer backbone to generate speech tokens from text.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2Config, GPT2Model
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.sample.sampler import Sampler

logger = init_logger(__name__)

# T3 Configuration Constants
LLAMA_CONFIGS = {
    "turbo_600m": {
        "hidden_size": 1024,
        "intermediate_size": 2618,
        "num_attention_heads": 8,
        "num_hidden_layers": 24,
        "num_key_value_heads": 8,
        "vocab_size": 152064,
        "model_type": "gpt2",
        "max_position_embeddings": 8192,
    },
}


@dataclass
class T3Config:
    """T3 Model Configuration."""

    n_channels: int = 1024
    text_tokens_dict_size: int = 152064
    speech_tokens_dict_size: int = 6561
    max_text_tokens: int = 2048
    max_speech_tokens: int = 3000
    start_speech_token: int = 6560
    stop_speech_token: int = 6562
    input_pos_emb: str = "learned"
    speaker_embed_size: int = 192
    encoder_type: str = "voice_encoder"
    llama_config_name: str = "turbo_600m"
    emotion_adv: bool = False

    @classmethod
    def english_only(cls):
        return cls()


@dataclass
class T3Cond:
    """T3 conditioning data structure."""

    speaker_emb: torch.Tensor
    clap_emb: torch.Tensor | None = None
    cond_prompt_speech_tokens: torch.Tensor | None = None
    cond_prompt_speech_emb: torch.Tensor | None = None
    emotion_adv: float = 0.5

    def to(self, *, device=None, dtype=None):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                is_fp = type(v.view(-1)[0].item()) is not int
                setattr(self, k, v.to(device=device, dtype=dtype if is_fp else None))
        return self


class LearnedPositionEmbeddings(nn.Module):
    """Learned position embeddings for T3."""

    def __init__(self, seq_len: int, model_dim: int, init: float = 0.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(torch.arange(0, x.shape[1], device=x.device))

    def get_fixed_embedding(self, idx):
        device = self.emb.weight.device
        idx = idx.to(device) if torch.is_tensor(idx) else torch.tensor(idx, device=device)
        return self.emb(torch.atleast_2d(idx))


class T3CondEnc(nn.Module):
    """T3 conditioning encoder."""

    def __init__(self, hp: T3Config):
        super().__init__()
        self.hp = hp
        if hp.encoder_type == "voice_encoder":
            self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)
        else:
            raise NotImplementedError(str(hp.encoder_type))
        self.emotion_adv_fc = None
        if hp.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False)

    def forward(self, cond: T3Cond) -> torch.Tensor:
        cond_spkr = self.spkr_enc(cond.speaker_emb.view(-1, self.hp.speaker_embed_size))[:, None]
        empty = torch.zeros_like(cond_spkr[:, :0])
        cond_clap = empty
        cond_prompt_speech_emb = cond.cond_prompt_speech_emb if cond.cond_prompt_speech_emb is not None else empty
        cond_emotion_adv = empty
        if self.hp.emotion_adv:
            emotion_tensor = torch.tensor([[cond.emotion_adv]], device=cond_spkr.device, dtype=cond_spkr.dtype)
            cond_emotion_adv = self.emotion_adv_fc(emotion_tensor.view(-1, 1, 1))
        return torch.cat((cond_spkr, cond_clap, cond_prompt_speech_emb, cond_emotion_adv), dim=1)


class ChatterboxTurboT3ForConditionalGeneration(nn.Module):
    """
    T3 text-to-speech token generator for vllm-omni.

    This is an autoregressive model that generates speech tokens from text.
    Uses GPT2 backbone with custom embeddings for text and speech tokens.

    Forward pass returns logits for next-token prediction.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.hp = T3Config.english_only()

        # Initialize GPT2 backbone
        config_dict = LLAMA_CONFIGS[self.hp.llama_config_name]
        self.is_gpt = config_dict.get("model_type") == "gpt2"
        self.cfg = GPT2Config(**config_dict)
        self.tfmr = GPT2Model(self.cfg)
        self.dim = self.cfg.hidden_size

        # Conditioning encoder
        self.cond_enc = T3CondEnc(self.hp)

        # Token embeddings
        self.text_emb = nn.Embedding(self.hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(self.hp.speech_tokens_dict_size, self.dim)

        # Position embeddings
        self.text_pos_emb = None
        self.speech_pos_emb = None
        if self.hp.input_pos_emb == "learned":
            self.text_pos_emb = LearnedPositionEmbeddings(self.hp.max_text_tokens + 2, self.dim)
            self.speech_pos_emb = LearnedPositionEmbeddings(self.hp.max_speech_tokens + 4, self.dim)

        # Output heads
        self.text_head = nn.Linear(self.cfg.hidden_size, self.hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.cfg.hidden_size, self.hp.speech_tokens_dict_size, bias=self.is_gpt)

        # Tokenizer for text processing
        self._tokenizer = None

        # Caching for inference
        self._past_key_values = None
        self._generated_tokens = []

    @property
    def device(self) -> torch.device:
        return self.speech_head.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.speech_head.weight.dtype

    @cached_property
    def sampler(self):
        return Sampler()

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            model_path = self.vllm_config.model_config.model
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        return self._tokenizer

    def prepare_conditioning(self, t3_cond: T3Cond) -> torch.Tensor:
        """Prepare conditioning embeddings from T3Cond."""
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens)
            if not self.is_gpt:
                t3_cond.cond_prompt_speech_emb += self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.Tensor,
        speech_tokens: torch.Tensor,
        cfg_weight: float = 0.0,
    ) -> tuple[torch.Tensor, int]:
        """Prepare input embeddings for the transformer."""
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

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input token IDs as speech tokens."""
        return self.speech_emb(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        kv_caches: list | None = None,
        attn_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for T3 model.

        During prefill: Processes full sequence (conditioning + text + initial speech token)
        During decode: Processes single speech token with cached KV

        Returns hidden states for speech head.
        """
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.speech_emb(input_ids)

        # Use internal caching for now (TODO: integrate with vLLM paged attention)
        use_cache = kv_caches is not None or self._past_key_values is not None

        outputs = self.tfmr(
            inputs_embeds=inputs_embeds,
            past_key_values=self._past_key_values if use_cache else None,
            use_cache=use_cache,
        )

        if use_cache:
            self._past_key_values = outputs.past_key_values

        return outputs.last_hidden_state

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute speech logits from hidden states."""
        return self.speech_head(hidden_states)

    def sample(self, logits: torch.Tensor, sampling_metadata) -> torch.Tensor:
        """Sample next token from logits."""
        return self.sampler(logits, sampling_metadata)

    def reset_cache(self):
        """Reset KV cache for new generation."""
        self._past_key_values = None
        self._generated_tokens = []

    @torch.inference_mode()
    def inference_step(
        self,
        t3_cond: T3Cond,
        text_tokens: torch.Tensor,
        speech_tokens: torch.Tensor | None = None,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """
        Single inference step for T3.

        Args:
            t3_cond: Conditioning embeddings
            text_tokens: Tokenized text input
            speech_tokens: Previously generated speech tokens (None for first step)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty

        Returns:
            Next speech token
        """
        from transformers import (
            LogitsProcessorList,
            RepetitionPenaltyLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        # Build logits processors
        logits_processors = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            logits_processors.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            logits_processors.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            logits_processors.append(TopPLogitsWarper(top_p))
        if repetition_penalty != 1.0:
            logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

        if speech_tokens is None:
            # First step: process full prompt
            speech_start_token = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
            embeds, _ = self.prepare_input_embeds(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                speech_tokens=speech_start_token,
                cfg_weight=0.0,
            )
            hidden_states = self.forward(inputs_embeds=embeds)
            self._generated_tokens = [speech_start_token]
        else:
            # Subsequent steps: process single token with cache
            current_embed = self.speech_emb(speech_tokens[:, -1:])
            hidden_states = self.forward(inputs_embeds=current_embed)

        # Get logits for last position
        speech_logits = self.speech_head(hidden_states[:, -1:])

        # Apply logits processors
        input_ids = torch.cat(self._generated_tokens, dim=1) if self._generated_tokens else speech_tokens
        processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])

        # Sample next token
        probs = F.softmax(processed_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        self._generated_tokens.append(next_token)

        return next_token

    @torch.inference_mode()
    def inference_stream(
        self,
        t3_cond: T3Cond,
        text_tokens: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_gen_len: int = 1000,
    ):
        """
        Streaming token generator - yields speech tokens one at a time.

        Yields:
            torch.Tensor: Single speech token tensor of shape (1, 1)
        """
        self.reset_cache()

        # First token
        next_token = self.inference_step(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=None,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        yield next_token

        # Subsequent tokens
        for _ in range(max_gen_len):
            next_token = self.inference_step(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                speech_tokens=torch.cat(self._generated_tokens, dim=1),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            # Check for stop token
            if torch.all(next_token == self.hp.stop_speech_token):
                break

            yield next_token

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load model weights from checkpoint."""
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # Map weight names if needed
            if name.startswith("t3."):
                name = name[3:]  # Remove 't3.' prefix

            if name in params_dict:
                param = params_dict[name]
                if param.shape == loaded_weight.shape:
                    param.data.copy_(loaded_weight)
                else:
                    logger.warning(f"Shape mismatch for {name}: expected {param.shape}, got {loaded_weight.shape}")
            else:
                logger.debug(f"Skipping unknown weight: {name}")
