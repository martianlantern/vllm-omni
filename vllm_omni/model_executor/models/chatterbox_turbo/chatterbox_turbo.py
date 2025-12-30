# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chatterbox Turbo TTS model for vllm-omni.

This module implements the Chatterbox Turbo TTS model as a two-stage pipeline:
- Stage 0 (T3): Autoregressive text-to-speech token generation
- Stage 1 (S3Gen): Non-autoregressive flow matching + HiFT vocoder

Reference: https://huggingface.co/ResembleAI/chatterbox-turbo
"""

from functools import cached_property
from pathlib import Path

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# Constants from the original Chatterbox Turbo model
S3GEN_SR = 24000
S3GEN_SIL = 4299
STOP_SPEECH_TOKEN = 6562


class ChatterboxTurboForConditionalGeneration(nn.Module, CustomProcessMixin):
    """
    Main Chatterbox Turbo TTS model for vllm-omni.

    This is a dispatcher class that initializes the appropriate stage model
    based on the `model_stage` in vllm_config:
    - model_stage='t3': Text → Speech tokens (Autoregressive)
    - model_stage='s3gen': Speech tokens → Audio (Non-Autoregressive)
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model_stage = getattr(self.config, "model_stage", None)

        if self.model_stage is None:
            # Try to get from engine args
            self.model_stage = getattr(vllm_config.model_config, "model_stage", "t3")

        logger.info(f"Initializing ChatterboxTurbo for stage: {self.model_stage}")

        if self.model_stage == "t3":
            from vllm_omni.model_executor.models.chatterbox_turbo.chatterbox_turbo_t3 import (
                ChatterboxTurboT3ForConditionalGeneration,
            )

            self.t3 = ChatterboxTurboT3ForConditionalGeneration(
                vllm_config=vllm_config,
                prefix=prefix,
            )
            self.model = self.t3
            self.s3gen = None
            self.has_preprocess = False
            self.have_multimodal_outputs = False

        elif self.model_stage == "s3gen":
            from vllm_omni.model_executor.models.chatterbox_turbo.chatterbox_turbo_s3gen import (
                ChatterboxTurboS3GenForConditionalGeneration,
            )

            self.s3gen = ChatterboxTurboS3GenForConditionalGeneration(
                vllm_config=vllm_config,
                prefix=prefix,
            )
            self.model = self.s3gen
            self.t3 = None
            self.has_preprocess = True
            self.have_multimodal_outputs = True
            # Register custom preprocess for stage input transformation
            self.set_custom_preprocess(self.s3gen_preprocess)

        else:
            raise ValueError(f"Invalid model_stage: {self.model_stage}. Must be 't3' or 's3gen'.")

        # Default voice conditionals (loaded on demand)
        self._default_ref_dict: dict | None = None

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        """Get the device of a module."""
        try:
            return next(module.parameters()).device
        except StopIteration:
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    @cached_property
    def sampler(self):
        """Get the sampler for this model stage."""
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        from vllm.v1.sample.sampler import Sampler

        return Sampler()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict | None = None,
        **kwargs,
    ) -> torch.Tensor | OmniOutput:
        """
        Forward pass dispatcher.

        For T3 stage: text → speech tokens
        For S3Gen stage: speech tokens → audio
        """
        if self.model_stage == "t3":
            return self._forward_t3(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        elif self.model_stage == "s3gen":
            return self._forward_s3gen(
                additional_information=additional_information,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown model_stage: {self.model_stage}")

    def _forward_t3(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for T3 (text-to-speech token) stage.

        Returns raw hidden states tensor (not OmniOutput) since have_multimodal_outputs=False.
        """
        hidden_states = self.t3(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return hidden_states

    def _forward_s3gen(
        self,
        additional_information: dict | None = None,
        **kwargs,
    ) -> OmniOutput:
        """Forward pass for S3Gen (speech token to audio) stage."""
        # Handle warmup/dummy run - return dummy audio output
        if additional_information is None:
            # Warmup with no inputs - return dummy audio
            dummy_audio = torch.zeros(
                24000,  # 1 second of silence at 24kHz
                device=self._module_device(self.model),
                dtype=torch.float16,
            )
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": dummy_audio},
            )

        speech_tokens = additional_information.get("speech_tokens")
        ref_dict = additional_information.get("ref_dict")

        if speech_tokens is None:
            # No speech tokens - return silence
            dummy_audio = torch.zeros(
                24000,
                device=self._module_device(self.model),
                dtype=torch.float16,
            )
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": dummy_audio},
            )

        # Use default voice if no ref_dict provided
        if ref_dict is None:
            ref_dict = self._get_default_ref_dict()

        # Generate audio from speech tokens
        audio_tensor = self.s3gen(
            speech_tokens=speech_tokens,
            ref_dict=ref_dict,
            **kwargs,
        )

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"audio": audio_tensor},
        )

    def s3gen_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        **info_dict,
    ):
        """
        Custom preprocess for S3Gen stage.

        Extracts speech tokens and ref_dict from preceding stage outputs.
        """
        additional_info = info_dict.get("additional_information", {})
        return input_ids, input_embeds, {"additional_information": additional_info}

    def _get_default_ref_dict(self) -> dict:
        """Load default voice conditionals from model checkpoint."""
        if self._default_ref_dict is not None:
            return self._default_ref_dict

        model_path = Path(self.vllm_config.model_config.model)
        conds_path = model_path / "conds.pt"

        if conds_path.exists():
            logger.info(f"Loading default voice conditionals from {conds_path}")
            self._default_ref_dict = torch.load(
                conds_path,
                map_location=self._module_device(self.model),
                weights_only=True,
            )
        else:
            logger.warning(f"No default voice conditionals found at {conds_path}. Voice reference audio is required.")
            self._default_ref_dict = {}

        return self._default_ref_dict

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
        """Compute logits from hidden states."""
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        if hidden_states is None:
            return None

        return self.model.compute_logits(hidden_states)

    def sample(self, logits: torch.Tensor, sampling_metadata) -> torch.Tensor | None:
        """Sample tokens from logits."""
        return self.model.sample(logits, sampling_metadata)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        """Embed input token IDs."""
        if self.model_stage == "s3gen":
            # S3Gen doesn't use text embeddings
            device = self._module_device(self.model)
            hidden_size = getattr(self.config, "hidden_size", 512)
            return torch.zeros(
                input_ids.shape[0],
                hidden_size,
                dtype=torch.float16,
                device=device,
            )
        return self.model.embed_input_ids(input_ids)

    def load_weights(self, weights: dict):
        """Load model weights."""
        return self.model.load_weights(weights)

    def prepare_conditionals(
        self,
        audio_path: str | None = None,
        audio_tensor: torch.Tensor | None = None,
        sample_rate: int = 16000,
    ) -> dict:
        """
        Prepare voice conditioning from reference audio.

        Args:
            audio_path: Path to reference audio file
            audio_tensor: Pre-loaded audio tensor
            sample_rate: Sample rate of audio_tensor

        Returns:
            ref_dict for S3Gen inference
        """
        if audio_path is None and audio_tensor is None:
            return self._get_default_ref_dict()

        if self.s3gen is None:
            raise ValueError("prepare_conditionals requires s3gen stage to be loaded")

        if audio_path is not None:
            import librosa

            audio_tensor, sample_rate = librosa.load(audio_path, sr=16000)
            audio_tensor = torch.from_numpy(audio_tensor)

        return self.s3gen.embed_ref(audio_tensor, sample_rate)
