# SparkTTS Modules - Audio processing components
# Adapted from flashtts/modules for vLLM-Omni integration

from vllm_omni.model_executor.models.spark_tts.modules.encoder import Encoder
from vllm_omni.model_executor.models.spark_tts.modules.decoder import Decoder
from vllm_omni.model_executor.models.spark_tts.modules.wave_generator import WaveGenerator
from vllm_omni.model_executor.models.spark_tts.modules.speaker_encoder import SpeakerEncoder
from vllm_omni.model_executor.models.spark_tts.modules.quantize import FactorizedVectorQuantize

__all__ = [
    "Encoder",
    "Decoder",
    "WaveGenerator",
    "SpeakerEncoder",
    "FactorizedVectorQuantize",
]
