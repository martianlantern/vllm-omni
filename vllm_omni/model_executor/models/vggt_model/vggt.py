import torch
import torch.nn as nn
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.vggt_model.vggt_source.models.vggt import VGGT as VGGTImpl  # noqa: N811


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

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, **kwargs):
        return self.model(images, query_points=query_points)

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]):
        # Allow loading weights if vllm's loader calls this
        # But usually vllm loads weights into parameters by name match
        # Since we wrap self.model, parameters are prefixed with 'model.'
        # We might need to adjust keys or rely on vllm's smart loading
        pass
