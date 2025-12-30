#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end offline inference example for Chatterbox Turbo TTS.

This script demonstrates how to use the Chatterbox Turbo TTS model
for text-to-speech synthesis via the vllm-omni framework.

Usage:
    python examples/offline_inference/chatterbox_turbo/end2end.py

Requirements:
    - GPU with at least 40GB VRAM (tested on A100 80GB)
    - Reference audio for voice cloning (optional, uses default voice if not provided)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402

from vllm_omni.entrypoints.omni import Omni  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate speech from text using Chatterbox Turbo TTS")
    parser.add_argument(
        "--model",
        type=str,
        default="ResembleAI/chatterbox-turbo",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Oh, that's hilarious! Um anyway, do we have a new model in store?",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio for voice cloning (optional)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for T3 stage",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling for T3 stage",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Top-k sampling for T3 stage",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum speech tokens to generate",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,  # Disabled: T3 uses text input but speech output vocabs
        help="Repetition penalty for T3 stage (disabled by default)",
    )

    args = parser.parse_args()

    print(f"Loading model: {args.model}")

    # Initialize Omni with Chatterbox Turbo
    omni_llm = Omni(model=args.model)

    # Prepare voice reference if provided
    ref_dict = None
    if args.ref_audio:
        print(f"Loading reference audio: {args.ref_audio}")
        # Load and prepare conditionals from reference audio
        ref_dict = omni_llm.prepare_conditionals(audio_path=args.ref_audio)
    else:
        print("Using default voice (no reference audio provided)")

    # Prepare text prompt with multimodal data
    # Only include ref_dict if we have actual reference audio
    # vLLM multimodal validation rejects None values
    if ref_dict is not None:
        prompt = {
            "prompt": args.text,
            "multi_modal_data": {
                "ref_dict": ref_dict,
            },
        }
    else:
        # No reference audio - model should use internal default voice
        prompt = {
            "prompt": args.text,
        }

    # Sampling params for T3 stage (AR token generation)
    t3_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    # Sampling params for S3Gen stage (non-AR audio generation)
    s3gen_params = SamplingParams(max_tokens=1)

    print(f"Generating speech for: '{args.text}'")

    # Generate!
    outputs = omni_llm.generate([prompt], [t3_params, s3gen_params])

    # Extract audio from final stage output
    final_output = outputs[-1]
    if hasattr(final_output, "request_output"):
        audio_tensor = final_output.request_output[0].multimodal_output.get("audio")
    else:
        # Fallback for different output formats
        audio_tensor = final_output.outputs[0].multimodal_output.get("audio")

    if audio_tensor is not None:
        # Convert to numpy and save
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.cpu().numpy()
        else:
            audio_np = audio_tensor

        sample_rate = 24000  # S3Gen sample rate
        sf.write(args.output, audio_np, samplerate=sample_rate)
        print(f"Audio saved to: {args.output}")
        print(f"Duration: {len(audio_np) / sample_rate:.2f} seconds")
    else:
        print("Error: No audio output generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
