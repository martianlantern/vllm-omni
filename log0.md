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
**Fix**: Added a dummy attention layer in `nn.ModuleList`. Named `layers` (not `dummy_layers`) to satisfy vLLM's `extract_layer_index` which expects `layers.N` pattern.

```python
# In SparkTTSAudioTokenizerForGeneration & SparkTTSBiCodecForGeneration
from vllm.attention.layer import Attention

# Named 'layers' to match vLLM's expected pattern for layer index extraction
self.layers = nn.ModuleList([
    Attention(num_heads=1, head_size=1, scale=1.0)
])
```

#### C. Complete Weight Tracking (Stage 0 & 2)
**Issue**: vLLM's loader validates that ALL `named_parameters()` are covered by `load_weights()` return set. Stage 0's wav2vec2 (loaded via `from_pretrained`) and dummy attention weights weren't tracked.
**Fix**: `load_weights` now iterates over all model parameters and adds them to the returned set.

```python
# Track wav2vec2 weights (already loaded in __init__)
if self.wav2vec2 is not None:
    for name, _ in self.wav2vec2.named_parameters():
        loaded_params.add(f"wav2vec2.{name}")
    for name, _ in self.wav2vec2.named_buffers():
        loaded_params.add(f"wav2vec2.{name}")

# Track dummy attention layer weights
for name, _ in self.layers.named_parameters():
    loaded_params.add(f"layers.{name}")
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

#### D. Forward Method Signature for Speech LLM (Stage 1)
**Issue**: vLLM-Omni's `GPUARModelRunner` calls `model.forward()` with vLLM v1 specific arguments (`sampling_metadata`, `logits_index`, `sampler`, `additional_information`). `SparkTTSSpeechLLMForGeneration` extends `Qwen2ForCausalLM` directly, inheriting its `forward()` which doesn't accept these arguments.
**Fix**: Override `forward()` in `SparkTTSSpeechLLMForGeneration` to accept and filter out the vLLM v1 arguments.

```python
# In SparkTTSSpeechLLMForGeneration
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
    # Filter out vLLM v1 specific kwargs
    return super().forward(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
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

