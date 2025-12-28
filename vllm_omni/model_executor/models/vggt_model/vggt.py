from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.model_executor.models.vggt_model.vggt_processor import (
    VGGTDummyInputsBuilder,
    VGGTMultiModalProcessor,
)
from vllm_omni.model_executor.models.vggt_model.vggt_source.models.vggt import VGGT as VGGTImpl  # noqa: N811


@MULTIMODAL_REGISTRY.register_processor(
    VGGTMultiModalProcessor,
    dummy_inputs=VGGTDummyInputsBuilder,
)
class VGGT(nn.Module):
    """
    Wrapper for VGGT model to be compatible with vllm registration.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        # Extract arguments from config or use defaults matching VGGTImpl
        img_size = getattr(config, "img_size", 518)
        patch_size = getattr(config, "patch_size", 14)
        embed_dim = getattr(config, "embed_dim", 1024)
        enable_camera = getattr(config, "enable_camera", True)
        enable_point = getattr(config, "enable_point", True)
        enable_depth = getattr(config, "enable_depth", True)
        enable_track = getattr(config, "enable_track", True)

        self.model = VGGTImpl(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_camera=enable_camera,
            enable_point=enable_point,
            enable_depth=enable_depth,
            enable_track=enable_track,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attn_metadata: Any = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # VGGT is a vision model, input_ids/positions are dummy from vLLM's perspective
        # The actual inputs come via kwargs from the processor (e.g. 'images')

        images = kwargs.get("images")
        if images is None:
            # Fallback or error if images are missing
            # In vLLM, if input_ids are present, maybe we are doing something else?
            # But for VGGT, we expect images.
            if kwargs.get("dummy_run", False):  # hypothetical
                return {}
            # Check if 'multi_modal_kwargs' wrapper exists (depending on vLLM version) or direct kwargs
            pass

        # query_points might be passed if we re-enable it, for now None
        query_points = kwargs.get("query_points", None)

        predictions = self.model(images, query_points=query_points)
        return predictions

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]):
        # Allow loading weights if vllm's loader calls this
        # But usually vllm loads weights into parameters by name match
        # Since we wrap self.model, parameters are prefixed with 'model.'
        # We might need to adjust keys or rely on vllm's smart loading
        # To support automatic loading, we can map vllm's flattened keys to self.model
        params_dict = dict(self.model.named_parameters())
        buffers_dict = dict(self.model.named_buffers())

        for name, loaded_weight in weights:
            # vLLM loader might pass keys like "model.patch_embed.proj.weight"
            # Our self.model has "patch_embed.proj.weight"

            # If the loaded name starts with "model.", strip it to match VGGTImpl
            if name.startswith("model."):
                sub_name = name[6:]
            else:
                sub_name = name

            if sub_name in params_dict:
                param = params_dict[sub_name]
                if param.shape != loaded_weight.shape:
                    # Handle shape mismatch or error
                    pass
                else:
                    param.data.copy_(loaded_weight)
            elif sub_name in buffers_dict:
                buf = buffers_dict[sub_name]
                buf.data.copy_(loaded_weight)
