import os
import sys

import torch

# Add project root to path
sys.path.append(os.getcwd())

from vllm.config import ModelConfig, VllmConfig

from vllm_omni.model_executor.models.vggt_model.vggt import VGGT


# Mock Config mimicking VGGT-1B parameters
class MockHFConfig:
    def __init__(self):
        self.img_size = 518
        self.patch_size = 14
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 4.0
        self.num_register_tokens = 4
        self.enable_camera = True
        self.enable_point = True
        self.enable_depth = True
        self.enable_track = True


class MockModelConfig(ModelConfig):
    def __init__(self):
        self.hf_config = MockHFConfig()


class MockVllmConfig(VllmConfig):
    def __init__(self):
        self.model_config = MockModelConfig()


def test_vggt_gpu():
    print("Testing VGGT on GPU...")
    if not torch.cuda.is_available():
        print("CUDA not available, but creating model on CPU to verify instantiation...")
        device = "cpu"
    else:
        device = "cuda"

    vllm_config = MockVllmConfig()

    print(f"Initializing model on {device}...")
    # Initialize model
    model = VGGT(vllm_config=vllm_config, prefix="")
    model.to(device)
    # Cast to half precision if on GPU for realistic memory usage
    if device == "cuda":
        model.half()

    print("Model initialized.")

    # Test forward pass
    print("Running forward pass...")
    img_size = vllm_config.model_config.hf_config.img_size
    # Benchmarking batch size 1, sequence length 1 (1 frame)
    # Shape: [B, S, 3, H, W]
    img = torch.randn(1, 1, 3, img_size, img_size, device=device)
    if device == "cuda":
        img = img.half()

    with torch.no_grad():
        out = model(img)

    print("Forward pass successful.")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"Output '{k}': shape={v.shape}, dtype={v.dtype}, device={v.device}")
        elif isinstance(v, list):
            print(f"Output '{k}': List of length {len(v)}")

    expected_keys = ["pose_enc", "depth", "world_points"]
    missing = set(expected_keys) - set(out.keys())
    if not missing:
        print("All required outputs present.")
    else:
        print(f"Missing outputs: {missing}")


if __name__ == "__main__":
    test_vggt_gpu()
