# Copyright (c) 2025 SparkAudio
# Adapted for vLLM-Omni integration
#
# Licensed under the Apache License, Version 2.0
# See: https://huggingface.co/SparkAudio/Spark-TTS-0.5B

"""SparkTTS BiCodec Decoder (Stage 2) for vLLM-Omni.

Converts semantic tokens and global tokens to audio waveforms.
Non-autoregressive generation using BiCodec decoder architecture.
"""

from __future__ import annotations

import os
import json
import yaml
from typing import Optional, List, Any, Dict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm_omni.model_executor.models.output_templates import OmniOutput

from vllm_omni.model_executor.models.spark_tts.modules.decoder import Decoder
from vllm_omni.model_executor.models.spark_tts.modules.wave_generator import WaveGenerator
from vllm_omni.model_executor.models.spark_tts.modules.speaker_encoder import SpeakerEncoder
from vllm_omni.model_executor.models.spark_tts.modules.quantize import FactorizedVectorQuantize

logger = init_logger(__name__)


class SparkTTSBiCodecModel(nn.Module):
    """Core BiCodec decoder model matching flashtts implementation.
    
    Contains:
    - Quantizer (detokenizer)
    - Speaker Encoder (detokenizer)
    - Prenet (Decoder)
    - WaveGenerator (Decoder)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.quantizer = FactorizedVectorQuantize(**config["quantizer"])
        self.prenet = Decoder(**config["prenet"])
        self.decoder = WaveGenerator(**config["decoder"])
        self.speaker_encoder = SpeakerEncoder(**config["speaker_encoder"])

    def forward(
        self, semantic_tokens: torch.Tensor, global_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            semantic_tokens: (B, T) Semantic token IDs
            global_tokens: (B, D) Global token IDs
        
        Returns:
            wav_recon: (B, 1, T_audio) Reconstructed waveform
        """
        z_q = self.quantizer.detokenize(semantic_tokens)
        d_vector = self.speaker_encoder.detokenize(global_tokens)
        
        # Conditioned generation
        x = self.prenet(z_q, d_vector)
        x = x + d_vector.unsqueeze(-1)
        wav_recon = self.decoder(x)
        return wav_recon



class SparkTTSBiCodecForGeneration(nn.Module):
    """BiCodec decoder wrapper for vLLM generation pipeline."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.have_multimodal_outputs = True

        self.model_path = vllm_config.model_config.model
        if self.model_path.rstrip("/").endswith("LLM"):
            self.model_path = self.model_path.rstrip("/")[:-3].rstrip("/")

        self.bicodec_path = os.path.join(self.model_path, "BiCodec")
        
        self.model = None

    def load_weights(self, weights) -> set[str]:
        """Load weights for BiCodec decoder.
        
        - BiCodec: Loaded from safetensors in BiCodec directory
        
        Returns:
            Set of loaded parameter names (relative to this module).
        """
        loaded_params: set[str] = set()
        
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
            
            # Extract audio_tokenizer config if present (assuming same config structure)
            if "audio_tokenizer" in full_config:
                config = full_config["audio_tokenizer"]
            else:
                config = full_config
            
            # Initialize model with config
            self.model = SparkTTSBiCodecModel(config)
            
            # Load weights from safetensors
            model_file = os.path.join(self.bicodec_path, "model.safetensors")
            if os.path.exists(model_file):
                from safetensors.torch import load_file
                try:
                    state_dict = load_file(model_file)
                    missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                    if missing:
                        logger.warning(f"Missing keys when loading BiCodec: {missing[:5]}...")
                    logger.info(f"Successfully loaded BiCodec weights from {model_file}")
                    # Track loaded params with 'model.' prefix (UPDATE, don't overwrite)
                    loaded_params.update({f"model.{k}" for k in state_dict.keys() if k not in missing})
                except Exception as e:
                    logger.error(f"Failed to load BiCodec safetensors: {e}")
            else:
                 # Fallback to pytorch_model.bin
                bin_file = os.path.join(self.bicodec_path, "pytorch_model.bin")
                if os.path.exists(bin_file):
                    try:
                        state_dict = torch.load(bin_file, map_location="cpu")
                        self.model.load_state_dict(state_dict, strict=False)
                        logger.info(f"Successfully loaded BiCodec weights from {bin_file}")
                        loaded_params.update({f"model.{k}" for k in state_dict.keys()})
                    except Exception as e:
                        logger.error(f"Failed to load BiCodec bin: {e}")
                else:
                    logger.warning(f"No model weights found in {self.bicodec_path}")
            
            self.model.eval()
            
        else:
             logger.warning(f"BiCodec config not found at {config_path}")
        
        return loaded_params

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        additional_information: Optional[Any] = None,
        **kwargs,
    ) -> OmniOutput:
        """Generate audio from semantic and global tokens."""
        
        if self.model is None:
            # Placeholder or uninitialized
            logger.warning("BiCodec model not initialized!")
            return OmniOutput(text_hidden_states=None, multimodal_outputs=None)

        # Batch processing: extract tokens from additional_information
        # additional_information is expected to be a list of dicts (one per request)
        # or a single dict if batch=1?
        # vLLM usually collates lists?
        
        semantic_tokens_list = []
        global_tokens_list = []
        
        if isinstance(additional_information, list):
            items = additional_information
        elif isinstance(additional_information, dict):
             # Maybe vLLM passed a single dict? Or collated dict?
             # Assuming list for now if batching enabled
             items = [additional_information]
        else:
            items = []
        
        for item in items:
            if item is None:
                continue
            
            s_tokens = item.get("semantic_tokens")
            g_tokens = item.get("global_tokens")
            
            if s_tokens is not None and g_tokens is not None:
                semantic_tokens_list.append(s_tokens)
                global_tokens_list.append(g_tokens)
        
        if not semantic_tokens_list:
             return OmniOutput(text_hidden_states=None, multimodal_outputs=None)

        # Collate
        # semantic_tokens are varying length: Pad to max length
        semantic_tokens = pad_sequence(semantic_tokens_list, batch_first=True, padding_value=0).to(input_ids.device)
        global_tokens = torch.stack(global_tokens_list).to(input_ids.device)
        
        if global_tokens.dim() == 2 and global_tokens.shape[1] == 1:
            global_tokens = global_tokens.squeeze(1) # Ensure correct shape if unsqueezed

        # Generate audio
        with torch.no_grad():
            wav_recon = self.model(semantic_tokens, global_tokens)
        
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": wav_recon},
        )
