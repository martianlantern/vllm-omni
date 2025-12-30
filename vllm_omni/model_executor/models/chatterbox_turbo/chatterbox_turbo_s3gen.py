# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chatterbox Turbo S3Gen model for vllm-omni.

S3Gen is the non-autoregressive speech synthesis stage that converts
speech tokens to audio using:
- Flow matching (CFM) for mel spectrogram generation
- HiFT vocoder for waveform synthesis

This module wraps the S3Gen components from turbo.py and integrates
them with vllm-omni's generation infrastructure.
"""

import sys
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# Constants
S3GEN_SR = 24000
S3GEN_SIL = 4299


class ChatterboxTurboS3GenForConditionalGeneration(nn.Module):
    """
    S3Gen speech tokens to audio generator for vllm-omni.

    This is a non-autoregressive model that processes speech tokens
    and generates audio waveforms. The pipeline is:
    1. Flow matching: tokens → mel spectrogram
    2. HiFT vocoder: mel → audio waveform

    Optimizations applied:
    - CFM step caching for buffer reuse
    - Half precision inference
    - Speculative CFM with reduced steps (meanflow mode)
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix

        # Load S3Gen model from turbo.py
        # We import dynamically to avoid circular imports and ensure
        # the model directory is in the path
        self._s3gen = None
        self._meanflow = True  # Use faster meanflow mode by default

        # CFM step cache for buffer reuse
        self._cfm_cache: dict = {}

        # Streaming state
        self._source_cache = None
        self._context_tokens = None

    @property
    def device(self) -> torch.device:
        if self._s3gen is not None:
            return self._s3gen.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def dtype(self) -> torch.dtype:
        if self._s3gen is not None:
            return self._s3gen.dtype
        return torch.float16

    def _lazy_load_s3gen(self):
        """Lazily load S3Gen model on first use."""
        if self._s3gen is not None:
            return

        # Try to import from turbo.py in the workspace
        turbo_path = Path("/Users/darshan.makwana/Desktop/Projects/vllm-omni")
        if turbo_path.exists() and str(turbo_path) not in sys.path:
            sys.path.insert(0, str(turbo_path))

        try:
            from turbo import S3Gen

            logger.info("Loading S3Gen model with meanflow enabled")
            self._s3gen = S3Gen(meanflow=self._meanflow)
            self._s3gen.eval()

        except ImportError as e:
            logger.error(f"Failed to import S3Gen from turbo.py: {e}")
            raise

    def forward(
        self,
        speech_tokens: torch.Tensor,
        ref_dict: dict | None = None,
        n_cfm_timesteps: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate audio from speech tokens.

        Args:
            speech_tokens: Speech token tensor (batch, seq_len)
            ref_dict: Voice reference embeddings dictionary
            n_cfm_timesteps: Number of CFM solver steps (default: 2 for meanflow)

        Returns:
            Audio tensor (batch, samples)
        """
        self._lazy_load_s3gen()

        if ref_dict is None:
            raise ValueError("ref_dict is required for S3Gen inference")

        # Default CFM steps for meanflow mode
        if n_cfm_timesteps is None:
            n_cfm_timesteps = 2 if self._meanflow else 10

        # Move inputs to device
        speech_tokens = speech_tokens.to(device=self.device)

        # Filter out special tokens
        if speech_tokens.dim() == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
        valid_mask = speech_tokens < 6561
        filtered_tokens = []
        for i in range(speech_tokens.size(0)):
            tokens = speech_tokens[i][valid_mask[i]]
            filtered_tokens.append(tokens)
        speech_tokens = torch.nn.utils.rnn.pad_sequence(filtered_tokens, batch_first=True, padding_value=S3GEN_SIL)

        # Append silence tokens at the end
        silence = torch.full(
            (speech_tokens.size(0), 3),
            S3GEN_SIL,
            device=speech_tokens.device,
            dtype=speech_tokens.dtype,
        )
        speech_tokens = torch.cat([speech_tokens, silence], dim=1)

        # Run inference
        output_wavs, _ = self._s3gen.inference(
            speech_tokens,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
        )

        return output_wavs.squeeze()

    def flow_inference(
        self,
        speech_tokens: torch.Tensor,
        ref_dict: dict,
        n_cfm_timesteps: int | None = None,
        finalize: bool = True,
    ) -> torch.Tensor:
        """
        Run flow matching inference only (without vocoder).

        Returns mel spectrogram.
        """
        self._lazy_load_s3gen()
        return self._s3gen.flow_inference(
            speech_tokens=speech_tokens,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=finalize,
        )

    def flow_inference_chunk(
        self,
        speech_tokens: torch.Tensor,
        ref_dict: dict,
        context_tokens: torch.Tensor | None = None,
        n_cfm_timesteps: int | None = None,
        is_final: bool = False,
    ) -> torch.Tensor:
        """
        Process a chunk of tokens with context overlap.

        For streaming, this prepends context tokens for encoder continuity.
        """
        self._lazy_load_s3gen()
        return self._s3gen.flow_inference_chunk(
            speech_tokens=speech_tokens,
            ref_dict=ref_dict,
            context_tokens=context_tokens,
            n_cfm_timesteps=n_cfm_timesteps,
            is_final=is_final,
        )

    def hift_inference(
        self,
        speech_feat: torch.Tensor,
        cache_source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run HiFT vocoder inference.

        Args:
            speech_feat: Mel spectrogram (batch, 80, mel_len)
            cache_source: Source cache from previous chunk

        Returns:
            audio: Audio waveform
            source: Source features for caching
        """
        self._lazy_load_s3gen()
        return self._s3gen.hift_inference(speech_feat, cache_source)

    def hift_inference_stream(
        self,
        speech_feat: torch.Tensor,
        cache_source: torch.Tensor | None = None,
        prev_audio_tail: torch.Tensor | None = None,
        crossfade_len: int = 480,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Streaming vocoder with crossfade support.
        """
        self._lazy_load_s3gen()
        return self._s3gen.hift_inference_stream(
            speech_feat=speech_feat,
            cache_source=cache_source,
            prev_audio_tail=prev_audio_tail,
            crossfade_len=crossfade_len,
        )

    def embed_ref(
        self,
        ref_wav: torch.Tensor,
        ref_sr: int,
        ref_fade_out: bool = True,
    ) -> dict:
        """
        Compute reference embeddings from audio.

        Args:
            ref_wav: Reference audio waveform
            ref_sr: Sample rate of reference audio

        Returns:
            ref_dict containing embeddings for conditioning
        """
        self._lazy_load_s3gen()
        return self._s3gen.embed_ref(
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_fade_out=ref_fade_out,
        )

    def reset_streaming_state(self):
        """Reset streaming state for new generation."""
        self._source_cache = None
        self._context_tokens = None
        self._cfm_cache.clear()

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """S3Gen doesn't use logits - returns None."""
        return None

    def sample(self, logits: torch.Tensor, sampling_metadata) -> torch.Tensor | None:
        """S3Gen doesn't sample - returns None."""
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load model weights from checkpoint."""
        self._lazy_load_s3gen()

        # Build weight mapping
        params_dict = dict(self._s3gen.named_parameters())
        loaded_count = 0

        for name, loaded_weight in weights:
            # Map weight names if needed
            if name.startswith("s3gen."):
                name = name[6:]  # Remove 's3gen.' prefix

            if name in params_dict:
                param = params_dict[name]
                if param.shape == loaded_weight.shape:
                    param.data.copy_(loaded_weight)
                    loaded_count += 1
                else:
                    logger.warning(f"Shape mismatch for {name}: expected {param.shape}, got {loaded_weight.shape}")
            else:
                logger.debug(f"Skipping unknown weight: {name}")

        logger.info(f"Loaded {loaded_count} weights for S3Gen")


class StreamingTTSEngine:
    """
    Unified streaming TTS engine that orchestrates T3 → S3Gen pipeline.

    Key optimizations:
    1. Overlapped token generation and mel synthesis
    2. Efficient buffer management
    3. Crossfade-free audio concatenation using source cache
    """

    def __init__(
        self,
        s3gen: ChatterboxTurboS3GenForConditionalGeneration,
        ref_dict: dict,
        chunk_size: int = 10,
        overlap_tokens: int = 2,
        n_cfm_steps: int = 2,
        crossfade_samples: int = 480,
    ):
        """
        Initialize streaming engine.

        Args:
            s3gen: S3Gen model instance
            ref_dict: Voice reference embeddings
            chunk_size: Tokens per audio chunk
            overlap_tokens: Context tokens for continuity
            n_cfm_steps: CFM solver steps (2 for meanflow)
            crossfade_samples: Audio crossfade length
        """
        self.s3gen = s3gen
        self.ref_dict = ref_dict
        self.chunk_size = chunk_size
        self.overlap_tokens = overlap_tokens
        self.n_cfm_steps = n_cfm_steps
        self.crossfade_samples = crossfade_samples

        # Streaming state
        self.token_buffer: list[torch.Tensor] = []
        self.full_buffer: list[torch.Tensor] = []
        self.context_tokens: torch.Tensor | None = None
        self.source_cache: torch.Tensor | None = None
        self.prev_audio_tail: torch.Tensor | None = None
        self.is_first_chunk = True

    def reset(self):
        """Reset state for new generation."""
        self.token_buffer = []
        self.full_buffer = []
        self.context_tokens = None
        self.source_cache = None
        self.prev_audio_tail = None
        self.is_first_chunk = True

    def add_token(self, token: torch.Tensor) -> torch.Tensor | None:
        """
        Add a token to the buffer.

        Args:
            token: Single token tensor (1, 1)

        Returns:
            Audio chunk if buffer is full, else None
        """
        self.token_buffer.append(token)
        self.full_buffer.append(token)

        if len(self.token_buffer) >= self.chunk_size:
            return self._process_chunk(is_final=False)
        return None

    def finalize(self) -> torch.Tensor | None:
        """Process remaining tokens in buffer."""
        if self.token_buffer:
            # Pad to chunk size with silence
            device = self.token_buffer[0].device
            remaining = len(self.token_buffer) % self.chunk_size
            if remaining > 0:
                padding_needed = self.chunk_size - remaining
                self.token_buffer.extend([torch.tensor([[S3GEN_SIL]], device=device)] * padding_needed)
            return self._process_chunk(is_final=True)
        return None

    @torch.inference_mode()
    def _process_chunk(self, is_final: bool = False) -> torch.Tensor | None:
        """Process buffered tokens into audio."""
        if not self.token_buffer:
            return None

        # Stack buffered tokens
        chunk_tokens = torch.cat(self.token_buffer, dim=1)

        # Filter special tokens
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

        # Vocoder
        audio, self.source_cache, self.prev_audio_tail = self.s3gen.hift_inference_stream(
            speech_feat=output_mels,
            cache_source=self.source_cache,
            prev_audio_tail=self.prev_audio_tail if not self.is_first_chunk else None,
            crossfade_len=self.crossfade_samples,
        )

        self.is_first_chunk = False

        # Update context for next chunk
        if self.overlap_tokens > 0 and self.full_buffer:
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
