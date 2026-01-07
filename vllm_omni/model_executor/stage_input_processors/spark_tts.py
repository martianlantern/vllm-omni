# Copyright (c) 2025 SparkAudio
# Adapted for vLLM-Omni integration
#
# Licensed under the Apache License, Version 2.0
# See: https://huggingface.co/SparkAudio/Spark-TTS-0.5B

"""Stage input processors for SparkTTS pipeline.

Handles data transformation between stages:
- tokenizer_to_speech_llm: Audio tokenizer (Stage 0) → Speech LLM (Stage 1)
- speech_llm_to_bicodec: Speech LLM (Stage 1) → BiCodec (Stage 2)
"""

from __future__ import annotations

import re
from typing import Optional

import torch

from vllm_omni.inputs.data import OmniTokensPrompt


def tokenizer_to_speech_llm(
    stage_list,
    engine_input_source,
    prompt=None,
    requires_multimodal_data: bool = False,
):
    """Transform audio tokenizer output to speech LLM input.

    Extracts global tokens from audio tokenizer and prepares
    prompt for speech LLM with voice cloning information.

    For controllable TTS (no reference audio), this processor
    passes through the prompt with control parameters.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    tokenizer_outputs = stage_list[source_stage_id].engine_outputs

    speech_llm_inputs = []
    if not isinstance(prompt, list):
        prompt = [prompt]

    for i, tokenizer_output in enumerate(tokenizer_outputs):
        # Get original prompt
        p = prompt[i] if i < len(prompt) else prompt[-1]

        # Extract global tokens from tokenizer output
        if hasattr(tokenizer_output, 'outputs') and tokenizer_output.outputs:
            multimodal_out = tokenizer_output.outputs[0].multimodal_output
            global_tokens = multimodal_out.get("global_tokens") if multimodal_out else None
            semantic_ref = multimodal_out.get("semantic_tokens") if multimodal_out else None
        else:
            global_tokens = None
            semantic_ref = None

        # Build additional information for speech LLM
        additional_information = {
            "global_tokens": global_tokens,
            "semantic_ref_tokens": semantic_ref,  # Reference semantic tokens for cloning
            "text": p.get("text", "") if isinstance(p, dict) else str(p),
        }

        # Pass through original prompt text with voice cloning info
        speech_llm_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=p.get("prompt_token_ids", [0]) if isinstance(p, dict) else [0],
                additional_information=additional_information,
                multi_modal_data=p.get("multi_modal_data") if isinstance(p, dict) and requires_multimodal_data else None,
                mm_processor_kwargs=None,
            )
        )

    return speech_llm_inputs


def speech_llm_to_bicodec(
    stage_list,
    engine_input_source,
    prompt=None,
    requires_multimodal_data: bool = False,
):
    """Transform speech LLM output to BiCodec input.

    Extracts semantic tokens from speech LLM generated text
    and combines with global tokens from audio tokenizer.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    # Get outputs from both stages
    tokenizer_stage_id = engine_input_source[0]  # Stage 0: audio tokenizer
    speech_llm_stage_id = engine_input_source[1] if len(engine_input_source) > 1 else 1  # Stage 1: speech LLM

    # Get global tokens from audio tokenizer (Stage 0)
    global_tokens_map = {}
    if tokenizer_stage_id < len(stage_list) and stage_list[tokenizer_stage_id].engine_outputs:
        for output in stage_list[tokenizer_stage_id].engine_outputs:
            if hasattr(output, 'outputs') and output.outputs:
                multimodal_out = output.outputs[0].multimodal_output
                if multimodal_out:
                    global_tokens_map[output.request_id] = multimodal_out.get("global_tokens")

    # Get semantic tokens from speech LLM (Stage 1)
    speech_llm_outputs = stage_list[speech_llm_stage_id].engine_outputs
    if speech_llm_outputs is None:
        raise RuntimeError(f"Stage {speech_llm_stage_id} has no outputs yet")

    bicodec_inputs = []
    for output in speech_llm_outputs:
        request_id = output.request_id

        # Extract generated text
        generated_text = ""
        if hasattr(output, 'outputs') and output.outputs:
            generated_text = output.outputs[0].text or ""

        # Parse semantic tokens from generated text
        semantic_tokens = extract_semantic_tokens(generated_text)

        # Get global tokens for this request
        global_tokens = global_tokens_map.get(request_id)
        if global_tokens is None:
            # Fallback: try to get from additional_information
            if hasattr(output, 'additional_information'):
                global_tokens = output.additional_information.get("global_tokens")

        bicodec_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],  # Dummy for non-AR model
                additional_information={
                    "semantic_tokens": semantic_tokens,
                    "global_tokens": global_tokens,
                },
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return bicodec_inputs


def extract_semantic_tokens(text: str) -> torch.Tensor:
    """Parse bicodec_semantic_N tokens from generated text.

    SparkTTS generates text like:
    "<|bicodec_semantic_0|><|bicodec_semantic_1|>...<|bicodec_semantic_N|>"

    This function extracts the token indices.
    """
    # Pattern to match bicodec_semantic_N tokens
    pattern = r"<\|bicodec_semantic_(\d+)\|>"
    matches = re.findall(pattern, text)

    if not matches:
        # Try alternative pattern without angle brackets
        pattern = r"bicodec_semantic_(\d+)"
        matches = re.findall(pattern, text)

    tokens = [int(t) for t in matches]
    return torch.tensor(tokens, dtype=torch.int32) if tokens else torch.tensor([], dtype=torch.int32)


def extract_global_tokens(text: str) -> Optional[torch.Tensor]:
    """Parse global token indices from generated text.

    SparkTTS may generate global tokens like:
    "<|bicodec_global_0|><|bicodec_global_1|>..."
    """
    pattern = r"<\|bicodec_global_(\d+)\|>"
    matches = re.findall(pattern, text)

    if not matches:
        return None

    tokens = [int(t) for t in matches]
    return torch.tensor(tokens, dtype=torch.int32)
