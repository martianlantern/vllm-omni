# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chatterbox Turbo TTS configuration class."""

from transformers import PretrainedConfig


class ChatterboxTurboConfig(PretrainedConfig):
    """Configuration class for Chatterbox Turbo TTS model.

    This configuration class is used to instantiate a Chatterbox Turbo model
    according to the specified arguments, defining the model architecture.
    """

    model_type = "chatterbox_turbo"

    def __init__(
        self,
        # T3 (text-to-speech token generator) configuration
        hidden_size: int = 1024,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 24,
        intermediate_size: int = 2618,
        # Updated to match actual model weights
        vocab_size: int = 50276,
        speech_vocab_size: int = 6563,
        max_position_embeddings: int = 8196,
        # S3Gen (flow matching + vocoder) configuration
        sample_rate: int = 24000,
        mel_channels: int = 80,
        cfm_channels: int = 512,
        meanflow: bool = True,
        # Speaker encoder (updated to match weights)
        speaker_embed_dim: int = 256,
        # Stage configuration (set by vllm-omni)
        model_stage: str | None = None,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.speech_vocab_size = speech_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.sample_rate = sample_rate
        self.mel_channels = mel_channels
        self.cfm_channels = cfm_channels
        self.meanflow = meanflow
        self.speaker_embed_dim = speaker_embed_dim
        self.model_stage = model_stage

        super().__init__(**kwargs)


# Register the configuration with AutoConfig
def register_chatterbox_turbo_config():
    """Register ChatterboxTurboConfig with transformers AutoConfig."""
    from transformers import AutoConfig

    AutoConfig.register("chatterbox_turbo", ChatterboxTurboConfig)
