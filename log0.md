# SparkTTS Integration Note

## Overview
SparkTTS is integrated into `vllm-omni` as a 3-stage pipeline (Audio Tokenizer → Speech LLM → BiCodec Decoder). This implementation is self-contained (no external `flashtts` dependency) and adapted for vLLM V1.

## Implementation Details

### 1. Dependencies & Porting
- **Ported Modules**: All necessary `flashtts` components (`modules/vq`, `modules/speaker`, `modules/fsq`) are vendored into `vllm_omni/model_executor/models/spark_tts/modules/`.
- **Einx Removal**: `einx` operations in `ResidualFSQ` were reimplemented using native PyTorch (`tensor.permute`, `F.embedding`) to remove the dependency.
- **Local Imports**: All imports were updated to reference the local `vllm_omni` package structure.

### 2. vLLM V1 Compatibility Fixes

#### A. KV Cache for Non-Attention Models (Stage 0 & 2)
**Issue**: vLLM's `HybridKVCacheCoordinator` asserts that all models must have attention layers. Encoder-only models (`SparkTTSAudioTokenizerForGeneration`) and non-autoregressive decoders (`SparkTTSBiCodecForGeneration`) lack these.
**Fix**: Added a dummy attention layer wrapped in `nn.ModuleList` to satisfy the coordinator and layer naming parsers.

```python
# In SparkTTSAudioTokenizerForGeneration & SparkTTSBiCodecForGeneration
from vllm.attention.layer import Attention

# Use ModuleList to ensure layer name (dummy_layers.0) parses correctly as index 0
self.dummy_layers = nn.ModuleList([
    Attention(num_heads=1, head_size=1, scale=1.0)
])
```

#### B. Weight Loading Prefix Mismatch (All Stages)
**Issue**: The wrapper class `SparkTTSForConditionalGeneration` stores sub-models under attributes like `self.speech_llm`, `self.audio_tokenizer`, `self.bicodec`. vLLM's loader checks `named_parameters()` on the outer model, which includes these prefixes (e.g., `speech_llm.model.layers...`). However, sub-models return loaded weight names relative to themselves (e.g., `model.layers...`).
**Fix**: The `load_weights` method now adds the appropriate prefix to the returned loaded weights set.

```python
# In spark_tts.py
def load_weights(self, weights) -> set[str]:
    from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights
    
    loaded = self.model.load_weights(weights)
    
    prefix_map = {
        "audio_tokenizer": "audio_tokenizer",
        "speech_llm": "speech_llm",
        "bicodec": "bicodec",
    }
    prefix = prefix_map.get(self.model_stage, "")
    
    if prefix:
        return add_prefix_to_loaded_weights(loaded, prefix)
    return loaded
```

## Running the Model
```bash
uv run vllm serve /root/voice_agent/services/tts-spark/Spark-TTS-0.5B \
  --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/spark_tts.yaml \
  --port 8091 \
  --host 0.0.0.0
```

