"""
Utility functions for model loading, configuration, and cloud storage.

This module provides:
  - Configuration loading and validation
  - Model loading and initialization
  - LoRA weight handling and conversion
  - Cloud storage integration (S3/R2)
  - General utility functions
"""

from typing import Dict, Any, Optional
from pathlib import Path
import tempfile

from config_models import TrainingConfig, SampleConfig


# ============================================================================
# Configuration Utilities
# ============================================================================

def load_config_from_yaml(yaml_path: Path) -> TrainingConfig:
    """
    Load and parse training configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Validated TrainingConfig object

    Raises:
        FileNotFoundError: If YAML file not found
        ValueError: If configuration is invalid
    """
    import yaml

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path) as f:
        raw_config = yaml.safe_load(f)

    # Parse and validate with Pydantic
    return TrainingConfig.from_dict(raw_config)


def load_process_config(yaml_path: Path, process_index: int = 0) -> Dict[str, Any]:
    """
    Load a single training process configuration from YAML.

    This is the format used by the original inference.py:
    ```python
    config = load_config(path)  # Returns dict
    config["model"]["name_or_path"]  # Access nested values
    ```

    Args:
        yaml_path: Path to YAML configuration file
        process_index: Which process to load (default: 0)

    Returns:
        Dictionary representation of the process config
        (for backward compatibility with original code)
    """
    config = load_config_from_yaml(yaml_path)
    process = config.get_process(process_index)

    # Convert Pydantic model to dict for backward compatibility
    return process.dict()


# ============================================================================
# Model Loading Utilities
# ============================================================================

def load_model(base_model: str, to_cuda: bool = True):
    """
    Load the base diffusion model.

    This function loads the WAN 2.1 (or other) video generation model
    with its VAE and prepares it for inference.

    Args:
        base_model: HuggingFace model identifier
                   (e.g., "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        to_cuda: Whether to move model to GPU (CUDA)

    Returns:
        Initialized diffusers.WanPipeline ready for inference

    Example:
        ```python
        pipe = load_model("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        output = pipe(prompt="...", num_frames=40)
        ```
    """
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline

    # Load VAE first (handles image encoding)
    vae = AutoencoderKLWan.from_pretrained(
        base_model, subfolder="vae", torch_dtype=torch.float32
    )

    # Load main pipeline with VAE
    pipe = WanPipeline.from_pretrained(
        base_model,
        vae=vae,
        torch_dtype=torch.bfloat16,  # Use lower precision for memory efficiency
    )

    if to_cuda:
        pipe.to("cuda")

    return pipe


# ============================================================================
# LoRA Weight Handling
# ============================================================================

def load_and_convert(safetensors_file: Path) -> Dict[str, Any]:
    """
    Load LoRA weights from safetensors file and convert to diffusers format.

    The ai-toolkit outputs weights in its native format. This function
    converts them to the diffusers library format that WanPipeline expects.

    Args:
        safetensors_file: Path to safetensors LoRA weights file

    Returns:
        Dictionary of converted weight tensors

    Note:
        Conversion mappings:
        - diffusion_model. → transformer.
        - self_attn → attn1
        - cross_attn → attn2
        - q,k,v → to_q, to_k, to_v
        - o → to_out.0
        - ffn.0 → ffn.net.0.proj
        - ffn.2 → ffn.net.2
    """
    from safetensors import safe_open

    f = safe_open(safetensors_file, framework="pt", device=0)
    return convert_to_diffusers({key: f.get_tensor(key) for key in f.keys()})


def save_weights(state_dict: Dict[str, Any], output_path: Path) -> None:
    """
    Save weight dictionary to safetensors format.

    Args:
        state_dict: Dictionary of weight tensors
        output_path: Path to save safetensors file
    """
    from safetensors.torch import save_file

    save_file(state_dict, str(output_path))


def prep_lora_weights(finetune_id: str, models_dir: Path = Path("/root/models")) -> Path:
    """
    Prepare LoRA weights for inference.

    This function:
    1. Loads weights from ai-toolkit format
    2. Converts to diffusers format
    3. Saves to temporary location for pipeline loading

    Args:
        finetune_id: ID of the fine-tuned model
        models_dir: Root directory containing model files

    Returns:
        Path to converted weights file

    Example:
        ```python
        weights_path = prep_lora_weights("abc123def456")
        pipe.load_lora_weights(str(weights_path))
        ```
    """
    input_weights = models_dir / f"{finetune_id}/{finetune_id}.safetensors"
    output_weights = Path("diffusers_lora.safetensors")

    if not input_weights.exists():
        raise FileNotFoundError(f"LoRA weights not found: {input_weights}")

    converted = load_and_convert(input_weights)
    save_weights(converted, output_weights)

    return output_weights


def convert_to_diffusers(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert ai-toolkit LoRA weights to diffusers format.

    The ai-toolkit (ostris/ai-toolkit) trains models using a different
    naming scheme than the diffusers library. This function performs
    the necessary key name transformations.

    Conversions applied:
    - diffusion_model. → transformer.
    - self_attn → attn1 (self-attention)
    - cross_attn → attn2 (cross-attention to text embeddings)
    - attention components: q, k, v → to_q, to_k, to_v
    - attention output: o → to_out.0
    - feedforward: ffn.0 → ffn.net.0.proj
    - feedforward: ffn.2 → ffn.net.2

    Args:
        state_dict: Dictionary of weight tensors from ai-toolkit

    Returns:
        Dictionary with converted keys compatible with diffusers

    Example:
        ```python
        # Original key: diffusion_model.transformer.attn1.to_q.lora_A
        # Converted to: transformer.attn1.to_q.lora_A
        ```
    """
    new_state_dict = {}

    for key in state_dict:
        new_key = key

        # Base model name change
        if key.startswith("diffusion_model."):
            new_key = new_key.replace("diffusion_model.", "transformer.")

        # Attention blocks conversion
        if "self_attn" in new_key:
            new_key = new_key.replace("self_attn", "attn1")
        elif "cross_attn" in new_key:
            new_key = new_key.replace("cross_attn", "attn2")

        # Attention components conversion (q, k, v, o)
        parts = new_key.split(".")
        for i, part in enumerate(parts):
            if part in ["q", "k", "v"]:
                parts[i] = f"to_{part}"
            elif part == "o":
                parts[i] = "to_out.0"
        new_key = ".".join(parts)

        # Feedforward network conversion
        if "ffn.0" in new_key:
            new_key = new_key.replace("ffn.0", "ffn.net.0.proj")
        elif "ffn.2" in new_key:
            new_key = new_key.replace("ffn.2", "ffn.net.2")

        new_state_dict[new_key] = state_dict[key]

    return new_state_dict


# ============================================================================
# String Utilities
# ============================================================================

def slugify(text: str, max_length: int = 100) -> str:
    """
    Convert text to URL-safe slug format.

    Converts:
    - Whitespace → hyphens
    - Non-alphanumeric characters → hyphens
    - Multiple consecutive hyphens → single hyphen
    - Strips leading/trailing hyphens

    Args:
        text: Input text to slugify
        max_length: Maximum length of slug (truncates before conversion)

    Returns:
        URL-safe slug string

    Example:
        ```python
        slugify("Hello World!!") → "hello-world"
        slugify("[trigger] in the mountains") → "trigger-in-the-mountains"
        ```
    """
    # Truncate to max length first
    truncated = text[:max_length]

    # Replace non-alphanumeric with hyphens
    slug = "-".join(
        c if c.isalnum() else "-" for c in truncated.split(" ")
    ).strip("-")

    # Clean up multiple consecutive hyphens
    while "--" in slug:
        slug = slug.replace("--", "-")

    return slug.lower()


# ============================================================================
# Cloud Storage Integration (S3/Cloudflare R2)
# ============================================================================

class CloudStorageHandler:
    """
    Handler for cloud storage operations (S3/Cloudflare R2).

    Cloudflare R2 is S3-compatible, so the same boto3 interface works.
    """

    def __init__(
        self,
        bucket_name: str,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        """
        Initialize cloud storage handler.

        For Cloudflare R2, set:
          - endpoint_url: https://<account-id>.r2.cloudflarestorage.com
          - access_key: R2 API token access key
          - secret_key: R2 API token secret key

        For AWS S3, use standard credentials and region.

        Args:
            bucket_name: S3/R2 bucket name
            region: AWS region (for S3)
            endpoint_url: Custom endpoint URL (for R2)
            access_key: Access key ID
            secret_key: Secret access key

        Environment Variables (recommended):
            AWS_ACCESS_KEY_ID: Access key (or use parameter)
            AWS_SECRET_ACCESS_KEY: Secret key (or use parameter)
            R2_ENDPOINT_URL: R2 endpoint URL
            R2_BUCKET_NAME: Bucket name
        """
        import boto3

        self.bucket_name = bucket_name
        self.region = region or "auto"  # R2 uses "auto"

        # Build S3 client kwargs
        s3_kwargs = {}

        if endpoint_url:
            s3_kwargs["endpoint_url"] = endpoint_url

        if access_key:
            s3_kwargs["aws_access_key_id"] = access_key

        if secret_key:
            s3_kwargs["aws_secret_access_key"] = secret_key

        if region:
            s3_kwargs["region_name"] = region

        self.s3_client = boto3.client("s3", **s3_kwargs)

    def upload_file(
        self,
        file_path: Path,
        s3_key: str,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload a file to cloud storage.

        Args:
            file_path: Local file path
            s3_key: Key (path) in S3/R2 bucket
            content_type: MIME type (e.g., "video/mp4")

        Returns:
            Full S3 URL (s3://bucket/key)

        Example:
            ```python
            storage = CloudStorageHandler("my-bucket")
            url = storage.upload_file(
                Path("/tmp/video.mp4"),
                "videos/final.mp4",
                "video/mp4"
            )
            # Returns: s3://my-bucket/videos/final.mp4
            ```
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        put_kwargs = {}
        if content_type:
            put_kwargs["ContentType"] = content_type

        self.s3_client.upload_file(str(file_path), self.bucket_name, s3_key, ExtraArgs=put_kwargs)

        return f"s3://{self.bucket_name}/{s3_key}"

    def upload_bytes(
        self,
        data: bytes,
        s3_key: str,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload bytes to cloud storage.

        Args:
            data: File bytes
            s3_key: Key (path) in S3/R2 bucket
            content_type: MIME type

        Returns:
            Full S3 URL

        Example:
            ```python
            storage = CloudStorageHandler("my-bucket")
            url = storage.upload_bytes(
                video_bytes,
                "videos/final.mp4",
                "video/mp4"
            )
            ```
        """
        put_kwargs = {}
        if content_type:
            put_kwargs["ContentType"] = content_type

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=data,
            **put_kwargs,
        )

        return f"s3://{self.bucket_name}/{s3_key}"

    def download_file(self, s3_key: str, file_path: Path) -> Path:
        """
        Download a file from cloud storage.

        Args:
            s3_key: Key in bucket
            file_path: Local destination path

        Returns:
            Path to downloaded file

        Example:
            ```python
            storage = CloudStorageHandler("my-bucket")
            path = storage.download_file("videos/input.mp3", Path("/tmp/audio.mp3"))
            ```
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3_client.download_file(self.bucket_name, s3_key, str(file_path))
        return file_path

    def download_bytes(self, s3_key: str) -> bytes:
        """
        Download a file from cloud storage as bytes.

        Args:
            s3_key: Key in bucket

        Returns:
            File bytes

        Example:
            ```python
            storage = CloudStorageHandler("my-bucket")
            video_bytes = storage.download_bytes("videos/output.mp4")
            ```
        """
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
        return response["Body"].read()

    def get_signed_url(
        self,
        s3_key: str,
        expiration_seconds: int = 3600,
    ) -> str:
        """
        Generate a signed URL for direct access to a file.

        Useful for sharing video links with users without requiring
        AWS credentials.

        Args:
            s3_key: Key in bucket
            expiration_seconds: URL validity duration (default: 1 hour)

        Returns:
            Signed URL string

        Example:
            ```python
            storage = CloudStorageHandler("my-bucket")
            url = storage.get_signed_url("videos/final.mp4", expiration_seconds=86400)
            # Share this URL with user
            ```
        """
        return self.s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": s3_key},
            ExpiresIn=expiration_seconds,
        )

    def list_objects(self, prefix: str = "") -> list:
        """
        List objects in the bucket.

        Args:
            prefix: Filter by key prefix

        Returns:
            List of object metadata dictionaries

        Example:
            ```python
            storage = CloudStorageHandler("my-bucket")
            videos = storage.list_objects("videos/")
            ```
        """
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name, Prefix=prefix
        )

        return response.get("Contents", [])


def create_cloudflare_r2_handler(bucket_name: str) -> CloudStorageHandler:
    """
    Factory function to create a Cloudflare R2 storage handler.

    Expects these environment variables to be set:
      - R2_ACCOUNT_ID: Cloudflare account ID
      - R2_ACCESS_KEY: R2 API token access key
      - R2_SECRET_KEY: R2 API token secret key

    Args:
        bucket_name: R2 bucket name

    Returns:
        Configured CloudStorageHandler for R2

    Example:
        ```python
        # Set environment variables:
        # export R2_ACCOUNT_ID=abc123
        # export R2_ACCESS_KEY=xxxxx
        # export R2_SECRET_KEY=yyyyy

        storage = create_cloudflare_r2_handler("my-videos")
        url = storage.upload_bytes(video_data, "final.mp4", "video/mp4")
        ```
    """
    import os

    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY")
    secret_key = os.getenv("R2_SECRET_KEY")

    if not all([account_id, access_key, secret_key]):
        raise ValueError(
            "Missing Cloudflare R2 credentials. "
            "Set R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY environment variables."
        )

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    return CloudStorageHandler(
        bucket_name=bucket_name,
        endpoint_url=endpoint_url,
        access_key=access_key,
        secret_key=secret_key,
    )


# ============================================================================
# Export for convenience
# ============================================================================

__all__ = [
    # Configuration
    "load_config_from_yaml",
    "load_process_config",
    # Model loading
    "load_model",
    # LoRA utilities
    "load_and_convert",
    "save_weights",
    "prep_lora_weights",
    "convert_to_diffusers",
    # String utilities
    "slugify",
    # Cloud storage
    "CloudStorageHandler",
    "create_cloudflare_r2_handler",
]
