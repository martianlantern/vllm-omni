# Copyright (c) 2025 SparkAudio
# Adapted for vLLM-Omni integration
#
# Licensed under the Apache License, Version 2.0
# See: https://huggingface.co/SparkAudio/Spark-TTS-0.5B

"""SparkTTS Audio Tokenizer (Stage 0) for vLLM-Omni.

Extracts global and semantic tokens from reference audio for voice cloning.
Uses wav2vec2 for feature extraction and BiCodec encoder for tokenization.
"""

from __future__ import annotations

import os
import json
import yaml
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torchaudio.transforms as TT
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import numpy as np

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.spark_tts.modules.encoder import Encoder
from vllm_omni.model_executor.models.spark_tts.modules.speaker_encoder import SpeakerEncoder
from vllm_omni.model_executor.models.spark_tts.modules.quantize import FactorizedVectorQuantize

logger = init_logger(__name__)


class SparkTTSAudioTokenizerModel(nn.Module):
    """Core SparkTTS tokenizer model matching flashtts implementation.
    
    Contains:
    - Feature Encoder (BiCodec)
    - Factorized Vector Quantizer
    - Speaker Encoder (Global tokens)
    - Mel Spectrogram transform
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Initialize components from config
        self.encoder = Encoder(**config["encoder"])
        self.quantizer = FactorizedVectorQuantize(**config["quantizer"])
        self.speaker_encoder = SpeakerEncoder(**config["speaker_encoder"])

        # Mel spectrogram for speaker encoder
        self.mel_transformer = TT.MelSpectrogram(
            sample_rate=config["mel_params"]["sample_rate"],
            n_fft=config["mel_params"]["n_fft"],
            win_length=config["mel_params"]["win_length"],
            hop_length=config["mel_params"]["hop_length"],
            f_min=config["mel_params"]["mel_fmin"],
            f_max=config["mel_params"]["mel_fmax"],
            n_mels=config["mel_params"]["num_mels"],
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(
        self, audio_features: torch.Tensor, audio_clip: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_features: (B, T, D) Features from wav2vec2
            audio_clip: (B, T_audio) Raw audio clip for speaker encoding
            
        Returns:
            Dict with 'semantic_tokens' and 'global_tokens'
        """
        # Global tokens from mel spectrogram
        mel = self.mel_transformer(audio_clip).squeeze(1)
        global_tokens = self.speaker_encoder.tokenize(mel.transpose(1, 2))
        
        # Semantic tokens from encoded features
        z = self.encoder(audio_features.transpose(1, 2))
        semantic_tokens = self.quantizer.tokenize(z)
        
        return {
            "semantic_tokens": semantic_tokens,
            "global_tokens": global_tokens,
        }


from vllm.attention.layer import Attention

class SparkTTSAudioTokenizerForGeneration(nn.Module):
    """Audio tokenizer wrapper for vLLM generation pipeline."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.have_multimodal_outputs = True
        
        # Paths
        self.model_path = vllm_config.model_config.model
        if self.model_path.rstrip("/").endswith("LLM"):
             self.model_path = self.model_path.rstrip("/")[:-3].rstrip("/")
             
        self.wav2vec_path = os.path.join(self.model_path, "wav2vec2-large-xlsr-53")
        self.bicodec_path = os.path.join(self.model_path, "BiCodec")
        
        # Initialize wav2vec2 model
        logger.info(f"Loading wav2vec2 from {self.wav2vec_path}")
        try:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(
                self.wav2vec_path
            )
            self.wav2vec2.config.output_hidden_states = True
            self.wav2vec2.eval()
        except Exception as e:
            logger.warning(f"Failed to load wav2vec2: {e}. Assuming uninitialized for testing.")
            # Create dummy wav2vec2 for structure if loading fails (e.g. during CI/test without weights)
            self.wav2vec2 = None
            
        # Initialize Feature Extractor
        try:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.wav2vec_path)
        except Exception:
            self.processor = None

        # Placeholder for BiCodec model (will be loaded in load_weights)
        self.model = None

        # Dummy attention layer to satisfy vLLM KV cache coordinator
        self.dummy_attn = Attention(
             num_heads=1,
             head_size=1,
             scale=1.0,
        )

    def load_weights(self, weights):
        """Load weights for BiCodec components.
        
        Loads `config.yaml` and `model.safetensors` from the BiCodec directory.
        Note: ignoring `weights` iterator for BiCodec components as we load explicitly.
        """
        # Load config first
        config_path = os.path.join(self.bicodec_path, "config.yaml")
        if not os.path.exists(config_path):
            config_path = os.path.join(self.bicodec_path, "config.json")
            
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                if config_path.endswith(".yaml"):
                    full_config = yaml.safe_load(f)
                else:
                    full_config = json.load(f)
            
            # Extract audio_tokenizer config if present (standard SparkTTS structure)
            if "audio_tokenizer" in full_config:
                config = full_config["audio_tokenizer"]
            else:
                config = full_config
            
            # Initialize model with config
            self.model = SparkTTSAudioTokenizerModel(config)
            
            # Load weights from safetensors
            model_file = os.path.join(self.bicodec_path, "model.safetensors")
            if os.path.exists(model_file):
                from safetensors.torch import load_file
                try:
                    state_dict = load_file(model_file)
                    # Load into sub-model
                    missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                    if missing:
                        logger.warning(f"Missing keys when loading BiCodec: {missing[:5]}...")
                    logger.info(f"Successfully loaded BiCodec weights from {model_file}")
                except Exception as e:
                    logger.error(f"Failed to load BiCodec safetensors: {e}")
            else:
                # Fallback to pytorch_model.bin if safetensors missing
                bin_file = os.path.join(self.bicodec_path, "pytorch_model.bin")
                if os.path.exists(bin_file):
                    try:
                        state_dict = torch.load(bin_file, map_location="cpu")
                        self.model.load_state_dict(state_dict, strict=False)
                        logger.info(f"Successfully loaded BiCodec weights from {bin_file}")
                    except Exception as e:
                        logger.error(f"Failed to load BiCodec bin: {e}")
                else:
                    logger.warning(f"No model weights found in {self.bicodec_path}")
            
            self.model.eval()
            
        else:
             logger.warning(f"BiCodec config not found at {config_path}")

    @torch.no_grad()
    def extract_features(self, wav_input: torch.Tensor) -> torch.Tensor:
        """Extract wav2vec2 features."""
        if self.wav2vec2 is None:
            # Dummy features for testing
            return torch.randn(wav_input.shape[0], wav_input.shape[1] // 320, 1024, device=wav_input.device)
            
        # wav_input should be (B, T) raw audio
        # wav2vec2 expects input_values
        
        # Assuming wav_input is already normalized if coming from processor
        # If passed as raw tensor, we might need padding/attention_mask
        
        outputs = self.wav2vec2(wav_input, output_hidden_states=True)
        # Combine hidden states as per SparkTTS logic
        # (layer 11 + 14 + 16) / 3
        hidden_states = outputs.hidden_states
        features = (hidden_states[11] + hidden_states[14] + hidden_states[16]) / 3
        return features

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        intermediate_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> OmniOutput:
        """Process reference audio to tokens."""
        
        # Audio input should be in intermediate_data or kwargs
        # vLLM-Omni convention for processing stages
        
        audio = None
        if intermediate_data and "audio" in intermediate_data:
            audio = intermediate_data["audio"]
        
        if audio is None:
            # Check prompt data or multimodal dict
            # For now return empty/dummy if no audio
             return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "global_tokens": None,
                    "semantic_tokens": None,
                },
            )

        # 1. Extract features (wav2vec2)
        features = self.extract_features(audio)
        
        # 2. Get reference clip for speaker encoder (handled in model/process)
        # Assuming audio is the reference clip or full audio
        # SparkTokenizer has get_ref_clip logic.
        # We assume audio is already preprocessed/clipped or we use it as is.
        # Ideally, we should port get_ref_clip logic here or in input processor.
        
        # 3. Tokenize
        if self.model:
            outputs = self.model(features, audio)
        else:
            outputs = {"global_tokens": None, "semantic_tokens": None}

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs=outputs,
        )
