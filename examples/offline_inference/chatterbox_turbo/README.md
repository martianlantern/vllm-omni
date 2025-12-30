# Chatterbox Turbo TTS Example

This example demonstrates how to use the Chatterbox Turbo TTS model
for text-to-speech synthesis via vllm-omni.

## Requirements

- GPU with at least 40GB VRAM (tested on A100 80GB)
- Python 3.10+

## Usage

### Basic Usage (Default Voice)

```bash
python examples/offline_inference/chatterbox_turbo/end2end.py \
    --text "Hello, this is a test of the Chatterbox Turbo TTS system."
```

### Custom Voice (Voice Cloning)

```bash
python examples/offline_inference/chatterbox_turbo/end2end.py \
    --text "Hello, this is a test." \
    --ref-audio /path/to/reference.wav
```

### Full Options

```bash
python examples/offline_inference/chatterbox_turbo/end2end.py \
    --model ResembleAI/chatterbox-turbo \
    --text "Your text here" \
    --output output.wav \
    --ref-audio /path/to/reference.wav \
    --temperature 0.8 \
    --top-p 0.95 \
    --top-k 1000 \
    --max-tokens 1000 \
    --repetition-penalty 1.2
```

## Architecture

The model runs as a two-stage pipeline:

1. **T3 Stage** (Autoregressive): Text → Speech tokens
   - GPT2-based transformer
   - Uses 45% of GPU memory

2. **S3Gen Stage** (Non-Autoregressive): Speech tokens → Audio
   - Flow matching for mel generation
   - HiFT vocoder for waveform synthesis
   - Uses 45% of GPU memory

## Streaming Support

For streaming audio generation, use the `generate_stream` API:

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(model="ResembleAI/chatterbox-turbo")

for audio_chunk in omni.generate_stream(prompt):
    # Process audio chunk (~400ms each)
    play_audio(audio_chunk)
```
