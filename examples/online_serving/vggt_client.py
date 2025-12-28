import base64
import os

from openai import OpenAI
from vllm.utils.argparse_utils import FlexibleArgumentParser

# Default API configuration
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local file to base64 format."""
    with open(file_path, "rb") as f:
        content = f.read()
        result = base64.b64encode(content).decode("utf-8")
    return result


def get_image_url_from_path(image_path: str | None) -> str:
    if not image_path:
        # Dummy placeholder
        return "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"

    if image_path.startswith(("http://", "https://")):
        return image_path

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image_path_lower = image_path.lower()
    if image_path_lower.endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    elif image_path_lower.endswith(".png"):
        mime_type = "image/png"
    else:
        mime_type = "image/jpeg"

    image_base64 = encode_base64_content_from_file(image_path)
    return f"data:{mime_type};base64,{image_base64}"


def run_vggt_inference(args):
    model_name = args.model
    image_url = get_image_url_from_path(args.image_path)

    # VGGT is vision-only, but we send a dummy prompt if required by API schema
    # Or just the image content
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
                {"type": "text", "text": "Process this image."},
            ],
        }
    ]

    print(f"Sending request to {openai_api_base} for model {model_name}...")
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            # VGGT is not generating tokens, so max_tokens might be irrelevant or small
            max_tokens=10,
            temperature=0.0,
        )

        # Check output
        print("Response received.")
        for choice in completion.choices:
            print(f"Message Content: {choice.message.content}")

            # vLLM-Omni might support custom fields?
            # Inspect object properties broadly
            if hasattr(choice.message, "audio"):
                print(f"Has Audio: {choice.message.audio}")

    except Exception as e:
        print(f"Error during inference: {e}")


def parse_args():
    parser = FlexibleArgumentParser(description="VGGT Online Inference Client")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="facebook/VGGT-1B", help="Model name served by vLLM")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_vggt_inference(args)
