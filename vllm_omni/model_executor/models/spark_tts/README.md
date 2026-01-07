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

#### B. Weight Loading Prefix Mismatch (Stage 1)
**Issue**: The standard `SparkTTS` wrapper added a `speech_llm` prefix to the Qwen2-based Speech LLM. However, vLLM's `AutoWeightsLoader` expects keys to match the checkpoint (standard `model.layers...`), causing initialization failures.
**Fix**: Removed the `speech_llm` prefix in `spark_tts.py` for Stage 1 initialization.

```python
# In spark_tts.py
self.speech_llm = SparkTTSSpeechLLMForGeneration(
    vllm_config=vllm_config, prefix=prefix  # Was: maybe_prefix(prefix, "speech_llm")
)
```

## Running the Model
```bash
uv run vllm serve /root/voice_agent/services/tts-spark/Spark-TTS-0.5B \
  --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/spark_tts.yaml \
  --port 8091 \
  --host 0.0.0.0
```
