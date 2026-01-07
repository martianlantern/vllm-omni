# Copyright (c) 2025 SparkAudio
# Adapted for vLLM-Omni integration
#
# Licensed under the Apache License, Version 2.0
# See: https://huggingface.co/SparkAudio/Spark-TTS-0.5B

"""SpeakerEncoder module for SparkTTS BiCodec.

Extracts speaker embeddings (x-vector and d-vector) from mel spectrograms.
Uses ECAPA-TDNN for speaker encoding and perceiver resampling for token generation.

Note: This module re-exports from flashtts for convenience.
The flashtts package must be available in the Python path.
"""

import sys
import os

# Add tts-spark parent dir to path if not already there
_tts_spark_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "tts-spark"
)
if os.path.isdir(_tts_spark_path) and _tts_spark_path not in sys.path:
    sys.path.insert(0, _tts_spark_path)

# Re-export from flashtts
try:
    from flashtts.modules.speaker.speaker_encoder import SpeakerEncoder
except ImportError:
    # Fallback: provide a placeholder that will error with helpful message
    class SpeakerEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "flashtts not found. Please ensure the tts-spark/flashtts package "
                "is installed or available in the Python path."
            )

__all__ = ["SpeakerEncoder"]
