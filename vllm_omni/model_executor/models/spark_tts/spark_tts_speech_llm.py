# Copyright (c) 2025 SparkAudio
# Adapted for vLLM-Omni integration
#
# Licensed under the Apache License, Version 2.0
# See: https://huggingface.co/SparkAudio/Spark-TTS-0.5B

"""SparkTTS Speech LLM (Stage 1) for vLLM-Omni.

Generates semantic tokens from text using Qwen2-0.5B LLM.
Supports both controllable TTS (gender/pitch/speed) and voice cloning.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# Constants (from flashtts)
TASK_TOKEN_MAP = {
    "vc": "<|task_vc|>",
    "tts": "<|task_tts|>",
    "asr": "<|task_asr|>",
    "s2s": "<|task_s2s|>",
    "t2s": "<|task_t2s|>",
    "understand": "<|task_understand|>",
    "caption": "<|task_cap|>",
    "controllable_tts": "<|task_controllable_tts|>",
    "prompt_tts": "<|task_prompt_tts|>",
    "speech_edit": "<|task_edit|>",
}

LEVELS_MAP = {
    "very_low": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "very_high": 4,
}

GENDER_MAP = {
    "female": 0,
    "male": 1,
}


class SparkTTSSpeechLLMForGeneration(Qwen2ForCausalLM):
    """Speech LLM for semantic token generation in SparkTTS.

    Stage 1: Generates semantic tokens from text input using Qwen2-0.5B.
    Supports controllable TTS with gender/pitch/speed parameters
    and voice cloning with global tokens from audio tokenizer.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Initialize Qwen2 backbone
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        
        self.tokenizer = None
        self._init_tokenizer(vllm_config)

    def _init_tokenizer(self, vllm_config):
        try:
            model_path = vllm_config.model_config.model
            # Try loading keys from LLM subdir if available, else root
            tokenizer_path = f"{model_path}/LLM"
            import os
            if not os.path.isdir(tokenizer_path):
                tokenizer_path = model_path
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer in SpeechLLM: {e}")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        # vLLM v1 specific arguments that Qwen2ForCausalLM doesn't accept
        sampling_metadata: Optional[SamplingMetadata] = None,
        logits_index: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        additional_information: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for speech LLM.
        
        This method accepts vLLM v1 specific arguments (sampling_metadata, 
        logits_index, sampler, additional_information) and filters them out
        before calling the parent Qwen2ForCausalLM.forward() which doesn't
        expect these arguments.
        """
        # Filter out vLLM v1 specific kwargs that Qwen2ForCausalLM doesn't accept
        # The parent class expects: input_ids, positions, intermediate_tensors, inputs_embeds
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        **info_dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Preprocess input for speech LLM prompt construction.

        Arguments:
            input_ids: Input text tokens
            input_embeds: Input embeddings (usually None here)
            info_dict: Contains 'global_tokens', 'semantic_ref_tokens', 'text', 'control_params'
        """
        # If no tokenizer, return as is (shouldn't happen in production)
        if self.tokenizer is None:
            return input_ids, input_embeds, info_dict

        # Extract info
        global_tokens = info_dict.get("global_tokens")
        semantic_ref = info_dict.get("semantic_ref_tokens")
        text = info_dict.get("text")
        
        # If we have global tokens, it's Voice Cloning
        if global_tokens is not None:
            # Voice Cloning Mode
            # Structure: <|task_tts|><|start_content|>{text}<|end_content|><|start_global_token|>{global_tokens}<|end_global_token|>
            # If semantic_ref exists: ...<|start_semantic_token|>{semantic_ref}
            
            prompt_parts = [TASK_TOKEN_MAP["tts"], "<|start_content|>", text, "<|end_content|>"]
            
            # Global tokens
            prompt_parts.extend(["<|start_global_token|>", self._format_global_tokens(global_tokens), "<|end_global_token|>"])
            
            # Semantic reference (optional)
            if semantic_ref is not None:
                prompt_parts.extend(["<|start_semantic_token|>", self._format_semantic_tokens(semantic_ref)])
                
            full_prompt = "".join(prompt_parts)
            
        else:
            # Controllable TTS Mode (or default)
            # Structure: <|task_controllable_tts|><|start_content|>{text}<|end_content|><|start_style_label|>{style}<|end_style_label|>
            
            # Defaults
            gender = info_dict.get("gender", "female")
            pitch = info_dict.get("pitch", "moderate")
            speed = info_dict.get("speed", "moderate")
            
            gender_id = GENDER_MAP.get(gender, 0)
            pitch_id = LEVELS_MAP.get(pitch, 2)
            speed_id = LEVELS_MAP.get(speed, 2)
            
            style_tokens = f"<|gender_{gender_id}|><|pitch_label_{pitch_id}|><|speed_label_{speed_id}|>"
            
            full_prompt = "".join([
                TASK_TOKEN_MAP["controllable_tts"],
                "<|start_content|>",
                text if text else "",
                "<|end_content|>",
                "<|start_style_label|>",
                style_tokens,
                "<|end_style_label|>"
            ])

        # Tokenize new prompt
        new_inputs = self.tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
        new_ids = new_inputs.input_ids.to(input_ids.device)
        
        return new_ids, None, info_dict

    def _format_global_tokens(self, tokens: torch.Tensor) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
            if isinstance(tokens, list) and isinstance(tokens[0], list): # Handle batch dim if present
                tokens = tokens[0] 
        return "".join([f"<|bicodec_global_{i}|>" for i in tokens])

    def _format_semantic_tokens(self, tokens: torch.Tensor) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
            if isinstance(tokens, list) and isinstance(tokens[0], list):
                tokens = tokens[0]
        return "".join([f"<|bicodec_semantic_{i}|>" for i in tokens])
