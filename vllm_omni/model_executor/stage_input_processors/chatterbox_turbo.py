# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for Chatterbox Turbo TTS.

This module handles data transformation between the T3 (token generation)
and S3Gen (audio synthesis) stages.
"""

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)

# Constants
S3GEN_SIL = 4299
STOP_SPEECH_TOKEN = 6562


def t3_to_s3gen(
    stage_list,
    engine_input_source,
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
):
    """
    Transform T3 stage outputs to S3Gen stage inputs.

    Extracts speech tokens from T3 output and prepares them for
    flow matching inference in S3Gen.

    Args:
        stage_list: List of stage objects containing engine outputs
        engine_input_source: List of source stage IDs
        prompt: Original prompt containing multi_modal_data with ref_dict
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for S3Gen stage input
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty for t3_to_s3gen")

    source_stage_id = engine_input_source[0]
    t3_outputs = stage_list[source_stage_id].engine_outputs

    if not t3_outputs:
        logger.warning("No T3 outputs found for S3Gen input")
        return []

    s3gen_inputs = []

    for t3_output in t3_outputs:
        if not t3_output.outputs:
            continue

        output = t3_output.outputs[0]
        token_ids = output.token_ids

        # Convert to tensor
        if isinstance(token_ids, list):
            speech_tokens = torch.tensor(token_ids, dtype=torch.long)
        else:
            speech_tokens = token_ids.clone()

        # Filter out special tokens (keep only valid speech tokens < 6561)
        valid_mask = speech_tokens < 6561
        speech_tokens = speech_tokens[valid_mask]

        # Remove stop token if present
        if speech_tokens.numel() > 0 and speech_tokens[-1] == STOP_SPEECH_TOKEN:
            speech_tokens = speech_tokens[:-1]

        # Append silence tokens for proper audio ending
        silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL], dtype=torch.long)
        speech_tokens = torch.cat([speech_tokens, silence])

        # Extract ref_dict from original prompt
        ref_dict = None
        if prompt is not None:
            if isinstance(prompt, dict):
                multi_modal_data = prompt.get("multi_modal_data", {})
                ref_dict = multi_modal_data.get("ref_dict")
            elif hasattr(prompt, "multi_modal_data"):
                ref_dict = getattr(prompt.multi_modal_data, "ref_dict", None)

        additional_information = {
            "speech_tokens": speech_tokens,
            "ref_dict": ref_dict,
        }

        s3gen_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],  # Dummy token - S3Gen doesn't use text IDs
                additional_information=additional_information,
            )
        )

        logger.debug(f"t3_to_s3gen: Processed {len(token_ids)} tokens → {speech_tokens.numel()} valid speech tokens")

    return s3gen_inputs
