from typing import Any

import torch
from vllm.multimodal import BaseProcessingInfo
from vllm.multimodal.inputs import MultiModalDataDict
from vllm.transformers_utils.tokenizer import AnyTokenizer


class VGGTProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> dict[str, int]:
        return {"image": 1}

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: dict[str, int]) -> dict[str, int]:
        return {"image": 0}


class VGGTMultiModalProcessor:
    def __init__(self, model_config: Any, tokenizer: AnyTokenizer, **kwargs: Any):
        self.model_config = model_config
        self.tokenizer = tokenizer

    def apply(
        self,
        prompt: str | list[int],
        multi_modal_data: MultiModalDataDict,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Process multimodal data for VGGT.
        Ref: vllm_omni/engine/input_processor.py

        Args:
            prompt: textual prompt (ignored for VGGT as it's vision-only/specialized)
            multi_modal_data: dict containing 'image'
        """
        mm_inputs = {}
        if "image" in multi_modal_data:
            image_data = multi_modal_data["image"]
            # Ensure it's a tensor [C, H, W] or [B, C, H, W]
            # If it comes from vLLM's image processor, it might be a list or different format
            # For now, assume generic tensor handling or rely on vLLM's standard image mapper if available.
            # But since we are writing a custom processor, we handle it raw.

            if isinstance(image_data, torch.Tensor):
                mm_inputs["images"] = image_data
            else:
                # If PIL or other, vLLM usually converts earlier or we handle it here.
                # Assuming vLLM standard flow passes tensors or we need to convert.
                # Only handling Tensor for now as per plan integration script will pass tensors.
                raise ValueError(f"VGGT processor expects 'image' as torch.Tensor, got {type(image_data)}")

        # No query_points here as per simplified plan

        return mm_inputs

    def _get_mm_fields(self) -> list[str]:
        return ["images"]


class VGGTDummyInputsBuilder:
    def __init__(self, model_config: Any):
        self.model_config = model_config

    def get_dummy_inputs(self, **kwargs: Any) -> dict[str, Any]:
        # Minimal dummy inputs for profiling/warmup
        return {"images": torch.zeros(1, 3, 518, 518)}
