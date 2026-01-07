# Copyright (c) 2025 SparkAudio
# Adapted for vLLM-Omni integration
#
# Licensed under the Apache License, Version 2.0
# See: https://huggingface.co/SparkAudio/Spark-TTS-0.5B

"""SparkTTS main model for vLLM-Omni.

This module provides the main entry point for SparkTTS integration,
supporting three stages:
- audio_tokenizer: wav2vec2 + BiCodec encoder for voice cloning
- speech_llm: Qwen2-0.5B for text-to-semantic token generation
- bicodec: BiCodec decoder for semantic tokens to audio
"""

from __future__ import annotations

import torch
import torch.nn as nn
from functools import cached_property
from typing import Optional

from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


class SparkTTSForConditionalGeneration(nn.Module, CustomProcessMixin):
    """SparkTTS main model class for vLLM-Omni integration.

    Routes to appropriate sub-model based on `model_stage`:
    - "audio_tokenizer": SparkTTSAudioTokenizerForGeneration
    - "speech_llm": SparkTTSSpeechLLMForGeneration
    - "bicodec": SparkTTSBiCodecForGeneration
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.has_preprocess = False
        self.have_multimodal_outputs = True

        config = vllm_config.model_config.hf_config
        self.model_stage = vllm_config.model_config.model_stage

        # Get model path for loading components
        self.model_path = vllm_config.model_config.model

        logger.info(f"Initializing SparkTTS with stage: {self.model_stage}")

        if self.model_stage == "audio_tokenizer":
            # Stage 0: wav2vec2 + BiCodec encoder for voice cloning
            from vllm_omni.model_executor.models.spark_tts.spark_tts_audio_tokenizer import (
                SparkTTSAudioTokenizerForGeneration,
            )

            self.audio_tokenizer = SparkTTSAudioTokenizerForGeneration(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "audio_tokenizer")
            )
            self.model = self.audio_tokenizer

        elif self.model_stage == "speech_llm":
            # Stage 1: Qwen2-0.5B based speech LLM
            from vllm_omni.model_executor.models.spark_tts.spark_tts_speech_llm import (
                SparkTTSSpeechLLMForGeneration,
            )

            self.speech_llm = SparkTTSSpeechLLMForGeneration(
                vllm_config=vllm_config, prefix=prefix
            )
            self.model = self.speech_llm
            # Enable preprocessing for prompt construction
            self.has_preprocess = True
            self.set_custom_preprocess(self.speech_llm_preprocess)

        elif self.model_stage == "bicodec":
            # Stage 2: BiCodec decoder
            from vllm_omni.model_executor.models.spark_tts.spark_tts_bicodec import (
                SparkTTSBiCodecForGeneration,
            )

            self.bicodec = SparkTTSBiCodecForGeneration(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "bicodec")
            )
            self.model = self.bicodec

        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage}. "
                "Must be 'audio_tokenizer', 'speech_llm', or 'bicodec'"
            )

        # Set up intermediate tensors passthrough
        self.make_empty_intermediate_tensors = getattr(
            self.model, "make_empty_intermediate_tensors", lambda: None
        )

    @cached_property
    def sampler(self) -> Sampler:
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return Sampler()

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        """Get the device of a module."""
        try:
            return next(module.parameters()).device
        except StopIteration:
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        """Embed input IDs using the appropriate model."""
        if self.model_stage == "bicodec":
            # BiCodec doesn't use embeddings in the traditional sense
            return torch.zeros_like(input_ids).reshape(-1, 1).repeat(
                1, self.vllm_config.model_config.get_hidden_size()
            )
        return self.model.embed_input_ids(
            input_ids=input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        sampling_metadata: Optional[SamplingMetadata] = None,
        logits_index: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        additional_information: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        """Forward pass routing to appropriate stage."""
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            sampling_metadata=sampling_metadata,
            logits_index=logits_index,
            sampler=sampler,
            additional_information=additional_information,
            **kwargs,
        )

    def speech_llm_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        **info_dict,
    ):
        """Preprocess for speech LLM stage.

        Constructs the prompt with global tokens from audio tokenizer
        (for voice cloning) or with control parameters (for controllable TTS).
        """
        return self.model.preprocess(input_ids, input_embeds, **info_dict)

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput
    ) -> Optional[torch.Tensor]:
        """Compute logits from hidden states."""
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        return self.model.compute_logits(hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """Sample from logits."""
        return self.model.sample(logits, sampling_metadata)

    def load_weights(self, weights):
        """Load weights for the model."""
        return self.model.load_weights(weights)
