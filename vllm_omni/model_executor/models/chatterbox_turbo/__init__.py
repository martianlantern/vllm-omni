# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chatterbox Turbo TTS model for vllm-omni."""

# Register custom configuration with transformers AutoConfig
from vllm_omni.model_executor.models.chatterbox_turbo.configuration_chatterbox_turbo import (
    ChatterboxTurboConfig,
    register_chatterbox_turbo_config,
)

register_chatterbox_turbo_config()

__all__ = ["ChatterboxTurboConfig"]
