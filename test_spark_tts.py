#!/usr/bin/env python3
"""Test script for SparkTTS vLLM-Omni server.

Usage:
    python test_spark_tts.py [--text "Your text here"] [--output output.wav]
"""

import argparse
import httpx
import json
import base64
import io
import struct


def create_wav_header(audio_data: bytes, sample_rate: int = 16000) -> bytes:
    """Create WAV header for raw audio data."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(audio_data)
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,  # Subchunk1Size
        1,   # AudioFormat (PCM)
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header + audio_data


def test_tts(
    text: str = "Hello, this is a test of the SparkTTS text to speech system.",
    base_url: str = "http://localhost:8091",
    output_file: str = "output.wav",
):
    """Test TTS endpoint."""
    print(f"Testing SparkTTS at {base_url}")
    print(f"Input text: {text}")
    
    # Try the OpenAI-compatible TTS endpoint
    endpoint = f"{base_url}/v1/audio/speech"
    
    payload = {
        "model": "/root/Spark-TTS-0.5B",
        "input": text,
        "voice": "alloy",  # Default voice
        "response_format": "wav",  # Required field
    }
    
    print(f"\nSending request to {endpoint}...")
    
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                print(f"Content-Type: {content_type}")
                
                # Check if we got audio back
                if "audio" in content_type or len(response.content) > 1000:
                    # Assume raw audio or wav
                    audio_data = response.content
                    
                    # Check if it's already a WAV file
                    if audio_data[:4] != b'RIFF':
                        # Wrap in WAV header
                        audio_data = create_wav_header(audio_data)
                    
                    with open(output_file, "wb") as f:
                        f.write(audio_data)
                    print(f"\n✓ Audio saved to {output_file}")
                    print(f"  Size: {len(audio_data)} bytes")
                else:
                    # Try to parse as JSON
                    try:
                        result = response.json()
                        print(f"Response: {json.dumps(result, indent=2)}")
                        
                        # Check for base64 audio in response
                        if "audio" in result:
                            audio_b64 = result["audio"]
                            audio_data = base64.b64decode(audio_b64)
                            with open(output_file, "wb") as f:
                                f.write(audio_data)
                            print(f"\n✓ Audio (from JSON) saved to {output_file}")
                    except json.JSONDecodeError:
                        print(f"Raw response: {response.text[:500]}...")
            else:
                print(f"Error: {response.text}")
                
    except httpx.ConnectError:
        print(f"ERROR: Could not connect to {base_url}")
        print("Make sure the vLLM server is running.")
    except Exception as e:
        print(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test SparkTTS vLLM-Omni server")
    parser.add_argument(
        "--text", "-t",
        default="Hello, this is a test of the SparkTTS text to speech system.",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--url", "-u",
        default="http://localhost:8091",
        help="Server URL (default: http://localhost:8091)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output.wav",
        help="Output file (default: output.wav)"
    )
    
    args = parser.parse_args()
    test_tts(text=args.text, base_url=args.url, output_file=args.output)


if __name__ == "__main__":
    main()
