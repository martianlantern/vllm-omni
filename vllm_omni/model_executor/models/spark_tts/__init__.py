# SparkTTS model implementations for vLLM-Omni
#
# SparkTTS is a 3-stage TTS pipeline:
# - Stage 0: Audio Tokenizer (wav2vec2 + BiCodec encoder)
# - Stage 1: Speech LLM (Qwen2-0.5B)
# - Stage 2: BiCodec Decoder

from vllm_omni.model_executor.models.spark_tts.spark_tts import (
    SparkTTSForConditionalGeneration,
)

__all__ = [
    "SparkTTSForConditionalGeneration",
]

