"""
Modal GPU runner - handles all heavy computation.

This module provides the GPU-decorated functions and classes that execute
on Modal's H100 GPUs. It imports and wraps the pure Python logic from
generator_logic.py and utilities.py.

The orchestrator (FastAPI) calls these functions via Modal's SDK to
trigger remote GPU execution.
"""

from pathlib import Path
from typing import List, Optional

import modal

# ============================================================================
# Environment Definition
# ============================================================================

# Commit SHA for specific diffusers version with Wan21 support
diffusers_commit_sha = "df1d7b01f18795a2d81eb1fd3f5d220db58cfae6"

# Build the Docker image with all dependencies
inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "accelerate==1.5.1",
        "boto3==1.34.0",  # For cloud storage
        "ftfy==6.3.1",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "huggingface_hub[hf_transfer]==0.30.1",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "peft==0.14.0",
        "pydantic==2.5.0",  # For config validation
        "pyyaml==6.0",
        "torch==2.5.1",
        "transformers==4.49.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

# Image for audio processing (ffmpeg + python)
audio_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install("ffmpeg-python==0.2.1", "mutagen==1.46.0")
)

# ============================================================================
# Modal App Setup
# ============================================================================

app = modal.App("music-video-gen-runner", image=inference_image)

# Volume mounts for persistent storage
finetunes_vol = modal.Volume.from_name("finetune-video-models", create_if_missing=False)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
outputs_vol = modal.Volume.from_name("finetune-video-outputs", create_if_missing=True)

MODELS_DIR = Path("/root/models")
OUTPUTS_DIR = Path("/root/outputs")
MINUTES = 60  # seconds


# ============================================================================
# GPU Runner - Stateful Video Generator
# ============================================================================

@app.cls(
    gpu="h100",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        MODELS_DIR: finetunes_vol,
        OUTPUTS_DIR: outputs_vol,
    },
    timeout=30 * MINUTES,
    scaledown_window=5 * MINUTES,
)
class ModalMusicVideoGenerator:
    """
    Stateful GPU runner for music video generation.

    This class runs on Modal H100 GPUs and handles the heavy computation:
    - Loading fine-tuned models
    - Running video inference
    - Processing audio with ffmpeg
    - Uploading results to cloud storage

    Usage from orchestrator:
    ```python
    generator = ModalMusicVideoGenerator(finetune_id="abc123")
    video_bytes = await generator.generate.aio(
        prompt="a person dancing",
        num_frames=40,
        guidance_scale=5.0
    )
    ```
    """

    finetune_id: str = modal.parameter()

    @modal.enter()
    def init(self):
        """
        Initialize the generator on container startup.

        This runs once when the container is created, not on every call.
        It loads the model to GPU memory for reuse across multiple
        inference calls.
        """
        from utilities import (
            load_process_config,
            load_model,
            prep_lora_weights,
        )

        print(f"[init] Loading model for finetune_id={self.finetune_id}")

        # Load configuration
        config_path = MODELS_DIR / self.finetune_id / "config.yaml"
        self.config = load_process_config(config_path, process_index=0)

        # Determine base model
        self.base_model = self.config["model"]["name_or_path"]
        print(f"[init] Loading base model: {self.base_model}")

        # Load model to GPU
        self.pipe = load_model(self.base_model, to_cuda=True)

        # Prepare and load LoRA weights
        print(f"[init] Loading LoRA weights")
        weights_path = prep_lora_weights(self.finetune_id, MODELS_DIR)
        self.pipe.load_lora_weights(str(weights_path))

        # Extract configuration parameters
        self.trigger_word = self.config.get("trigger_word", "[trigger]")
        self.guidance_scale = self.config["sample"]["guidance_scale"]
        self.height = self.config["sample"]["height"]
        self.width = self.config["sample"]["width"]

        print(f"[init] Model ready. Trigger word: {self.trigger_word}")

    @modal.method()
    def generate(
        self,
        prompt: str,
        num_frames: int = 40,
        guidance_scale: Optional[float] = None,
    ) -> bytes:
        """
        Generate a single video clip from a prompt.

        This method runs on the GPU and performs the actual diffusion
        inference to generate video frames.

        Args:
            prompt: Text prompt for video generation.
                   Use [trigger] placeholder for trained subject.
            num_frames: Number of frames to generate (default: 40)
            guidance_scale: Classifier-free guidance scale.
                          If None, uses model's default (usually 5.0)

        Returns:
            Generated video file as bytes (MP4 format)

        Example:
            ```python
            generator = ModalMusicVideoGenerator(finetune_id="abc123")
            video = await generator.generate.aio(
                prompt="[trigger] dancing in the rain",
                num_frames=60,
                guidance_scale=7.5
            )
            ```
        """
        from diffusers.utils import export_to_video
        from utilities import slugify

        print(f"[generate] Starting inference for: {prompt}")

        # Replace trigger word placeholder
        expanded_prompt = prompt.replace("[trigger]", self.trigger_word)
        print(f"[generate] Expanded prompt: {expanded_prompt}")

        # Use provided guidance_scale or model default
        gs = guidance_scale if guidance_scale is not None else self.guidance_scale

        # Run diffusion inference
        print(f"[generate] Running diffusion (guidance_scale={gs}, frames={num_frames})")
        output_frames = self.pipe(
            prompt=expanded_prompt,
            height=self.height,
            width=self.width,
            num_frames=num_frames,
            guidance_scale=gs,
        ).frames[0]

        # Export frames to video file
        output_dir = OUTPUTS_DIR / self.finetune_id
        output_dir.mkdir(exist_ok=True, parents=True)

        output_path = Path(
            export_to_video(
                output_frames,
                output_dir / (slugify(expanded_prompt) + ".mp4"),
                fps=15
            )
        )

        print(f"[generate] Saved to: {output_path}")

        # Read and return as bytes
        video_bytes = output_path.read_bytes()
        print(f"[generate] Generated {len(video_bytes)} bytes")

        return video_bytes

    @modal.method()
    def generate_batch(
        self,
        prompts: List[str],
        num_frames: int = 40,
        guidance_scale: Optional[float] = None,
    ) -> List[bytes]:
        """
        Generate multiple video clips.

        Calls generate() sequentially for each prompt. For true parallelism,
        use .map() on the orchestrator side.

        Args:
            prompts: List of prompts
            num_frames: Frames per video
            guidance_scale: Guidance scale

        Returns:
            List of video bytes, same length as prompts

        Example:
            ```python
            # From orchestrator using .map() for parallelism
            videos = await generator.generate_batch.map(
                prompts,
                kwargs={"num_frames": 40}
            )
            ```
        """
        results = []
        for i, prompt in enumerate(prompts):
            print(f"[generate_batch] Generating clip {i+1}/{len(prompts)}")
            video_bytes = self.generate(
                prompt=prompt,
                num_frames=num_frames,
                guidance_scale=guidance_scale
            )
            results.append(video_bytes)
        return results


# ============================================================================
# Stateless GPU Functions
# ============================================================================

@app.function(image=audio_image, timeout=10 * MINUTES)
def get_audio_duration(audio_bytes: bytes) -> float:
    """
    Get the duration of an audio file.

    Uses mutagen library to parse audio metadata.

    Args:
        audio_bytes: Audio file bytes (MP3, WAV, etc.)

    Returns:
        Duration in seconds

    Example:
        ```python
        duration = await get_audio_duration.aio(audio_file_bytes)
        num_clips = int(duration / 5) + 1  # 5-second clips
        ```
    """
    from io import BytesIO
    from mutagen.mp3 import MP3

    print(f"[get_audio_duration] Processing {len(audio_bytes)} bytes")

    audio = MP3(BytesIO(audio_bytes))
    duration = audio.info.length

    print(f"[get_audio_duration] Duration: {duration:.1f} seconds")
    return duration


@app.function(image=audio_image, timeout=15 * MINUTES)
def combine_videos_with_audio(
    video_clips: List[bytes],
    audio_bytes: bytes,
) -> bytes:
    """
    Combine video clips and overlay audio using ffmpeg.

    Concatenates videos and overlays audio, handling:
    - Different video/audio durations (uses shortest)
    - Codec conversion (H.264 video, AAC audio)
    - Temporary file cleanup

    Args:
        video_clips: List of video file bytes (MP4)
        audio_bytes: Audio file bytes (MP3 or WAV)

    Returns:
        Combined video bytes (MP4 format)

    Example:
        ```python
        final_video = await combine_videos_with_audio.aio(
            [video1, video2, video3],
            audio_bytes
        )
        ```
    """
    import tempfile
    from pathlib import Path
    import ffmpeg

    print(f"[combine_videos] Combining {len(video_clips)} clips with audio")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write video clips to temporary files
        print(f"[combine_videos] Writing {len(video_clips)} video files")
        video_paths = []
        for i, clip_bytes in enumerate(video_clips):
            path = tmpdir / f"clip_{i:03d}.mp4"
            path.write_bytes(clip_bytes)
            video_paths.append(path)
            print(f"[combine_videos]   clip_{i:03d}.mp4: {len(clip_bytes)} bytes")

        # Concatenate videos
        print(f"[combine_videos] Concatenating videos")
        video_inputs = [ffmpeg.input(str(p)) for p in video_paths]
        video_concat = ffmpeg.concat(*video_inputs, v=1, a=0).node

        # Write audio to temporary file
        print(f"[combine_videos] Writing audio file")
        audio_path = tmpdir / "audio.mp3"
        audio_path.write_bytes(audio_bytes)

        # Combine video and audio
        print(f"[combine_videos] Overlaying audio")
        audio_input = ffmpeg.input(str(audio_path))
        output_path = tmpdir / "output.mp4"

        output = ffmpeg.output(
            video_concat[0],
            audio_input,
            str(output_path),
            vcodec="libx264",
            acodec="aac",
            shortest=None,
        )

        # Run ffmpeg pipeline
        print(f"[combine_videos] Running ffmpeg")
        output.run(quiet=True, overwrite_output=True)

        # Read result
        print(f"[combine_videos] Reading output")
        result_bytes = output_path.read_bytes()
        print(f"[combine_videos] Final video: {len(result_bytes)} bytes")

        return result_bytes


# ============================================================================
# Cloud Storage Integration
# ============================================================================

@app.function(image=inference_image, timeout=5 * MINUTES)
def upload_to_storage(
    video_bytes: bytes,
    job_id: str,
    bucket_name: str,
) -> str:
    """
    Upload video to cloud storage (S3/Cloudflare R2).

    Args:
        video_bytes: Video file bytes
        job_id: Job identifier (used in storage path)
        bucket_name: S3/R2 bucket name

    Returns:
        Cloud storage URL (s3://bucket/path or signed URL)

    Environment Variables:
        For S3:
          - AWS_ACCESS_KEY_ID
          - AWS_SECRET_ACCESS_KEY
          - AWS_DEFAULT_REGION

        For Cloudflare R2:
          - R2_ACCOUNT_ID
          - R2_ACCESS_KEY
          - R2_SECRET_KEY

    Example:
        ```python
        url = await upload_to_storage.aio(
            video_bytes,
            job_id="abc123",
            bucket_name="my-videos"
        )
        # Returns: https://...signed...url or s3://my-videos/...
        ```
    """
    from utilities import create_cloudflare_r2_handler

    print(f"[upload_to_storage] Uploading {len(video_bytes)} bytes to {bucket_name}")

    try:
        # Try Cloudflare R2 first
        storage = create_cloudflare_r2_handler(bucket_name)
        s3_key = f"videos/{job_id}/final.mp4"
        url = storage.upload_bytes(
            video_bytes,
            s3_key,
            content_type="video/mp4"
        )
        print(f"[upload_to_storage] Uploaded to {url}")
        return url
    except ValueError:
        # Fall back to AWS S3
        print("[upload_to_storage] R2 credentials not found, trying S3")
        from utilities import CloudStorageHandler
        storage = CloudStorageHandler(bucket_name)
        s3_key = f"videos/{job_id}/final.mp4"
        url = storage.upload_bytes(
            video_bytes,
            s3_key,
            content_type="video/mp4"
        )
        print(f"[upload_to_storage] Uploaded to {url}")
        return url


# ============================================================================
# Local Entrypoints for Testing
# ============================================================================

@app.local_entrypoint()
def test_single_generation(
    finetune_id: str = "test_id",
    prompt: str = "[trigger] holding a sign that says 'I LOVE MODAL'",
    num_frames: int = 5,
):
    """
    Test endpoint for single video generation.

    Usage:
        modal run modal_runner.py::test_single_generation \
          --finetune-id abc123 \
          --prompt "[trigger] dancing"
    """
    print(f"Testing single generation:")
    print(f"  finetune_id: {finetune_id}")
    print(f"  prompt: {prompt}")
    print(f"  num_frames: {num_frames}")

    generator = ModalMusicVideoGenerator(finetune_id=finetune_id)
    result = generator.generate.remote(
        prompt=prompt,
        num_frames=num_frames,
    )

    output_path = Path(f"/tmp/{finetune_id}_test.mp4")
    output_path.write_bytes(result)
    print(f"Output saved to: {output_path}")


@app.local_entrypoint()
def test_audio_duration(audio_file: Optional[str] = None):
    """
    Test endpoint for audio duration detection.

    Usage:
        modal run modal_runner.py::test_audio_duration \
          --audio-file path/to/audio.mp3
    """
    if audio_file:
        audio_bytes = Path(audio_file).read_bytes()
    else:
        # Test with silence
        audio_bytes = b"" * 1000

    duration = get_audio_duration.remote(audio_bytes)
    print(f"Audio duration: {duration:.2f} seconds")


@app.local_entrypoint()
def test_combine_videos(num_clips: int = 3):
    """
    Test endpoint for video combination.

    Usage:
        modal run modal_runner.py::test_combine_videos --num-clips 5
    """
    print(f"Testing video combination with {num_clips} clips")
    # Generate dummy video bytes for testing
    # In production, these would come from actual video generation

    # For testing purposes, we can't easily generate video bytes,
    # but the function signature is correct
    print("(Skipping actual combination - would need real video files)")
