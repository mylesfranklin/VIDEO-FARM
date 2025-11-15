"""
Core music video generation logic - pure Python, no Modal decorators.

This module contains the MusicVideoGenerator class that orchestrates the
complete music video generation pipeline:
  1. Load prompts from input
  2. Get audio duration
  3. Generate video clips in parallel (called by orchestrator)
  4. Combine clips with audio overlay
  5. Return final video path/bytes

This class accepts configuration objects and parameters, performs no I/O to
local filesystems, and returns cloud storage paths or bytes.
"""

from typing import List, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from io import BytesIO

from config_models import InferenceRequest, OrchestrationConfig


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class GenerationResult:
    """Result from a single video generation."""

    prompt: str
    video_bytes: bytes
    duration_seconds: float


@dataclass
class OrchestrationResult:
    """Result from orchestrating a complete music video."""

    job_id: str
    output_path: str  # Cloud storage path or local path
    total_duration: float
    num_clips: int
    video_bytes: Optional[bytes] = None  # In-memory result if needed


# ============================================================================
# MusicVideoGenerator Class
# ============================================================================

class MusicVideoGenerator:
    """
    Pure Python class for music video generation orchestration.

    This class handles the workflow of:
      1. Loading and validating configurations
      2. Triggering parallel video clip generation
      3. Combining clips with audio
      4. Managing output to cloud storage

    The class accepts callbacks for GPU-intensive operations (like video
    generation and ffmpeg processing) so it can be used with Modal,
    FastAPI, or other execution frameworks.

    Usage Example:
    ```python
    generator = MusicVideoGenerator(
        finetune_id="abc123",
        model_loader_fn=load_model_on_gpu,
        inference_fn=run_inference_on_gpu,
        audio_processor_fn=combine_videos_with_audio
    )

    result = generator.orchestrate_music_video(
        prompts=["prompt1", "prompt2"],
        audio_bytes=audio_data,
        clip_duration=5
    )
    ```
    """

    def __init__(
        self,
        finetune_id: str,
        trigger_word: Optional[str] = None,
        inference_fn: Optional[Callable] = None,
        audio_processor_fn: Optional[Callable] = None,
    ):
        """
        Initialize the MusicVideoGenerator.

        Args:
            finetune_id: Identifier for the fine-tuned model
            trigger_word: Word to replace [trigger] placeholders in prompts
            inference_fn: Callable(prompt, num_frames, **kwargs) -> bytes
                         For GPU-intensive video generation
            audio_processor_fn: Callable(videos: List[bytes], audio: bytes) -> bytes
                               For combining videos with audio
        """
        self.finetune_id = finetune_id
        self.trigger_word = trigger_word or "[trigger]"
        self.inference_fn = inference_fn
        self.audio_processor_fn = audio_processor_fn

    # ========================================================================
    # Core Orchestration Methods
    # ========================================================================

    def orchestrate_music_video(
        self,
        prompts: List[str],
        audio_bytes: bytes,
        clip_duration: int = 5,
        num_frames_per_clip: Optional[int] = None,
    ) -> OrchestrationResult:
        """
        Orchestrate the complete music video generation pipeline.

        Pipeline:
          1. Calculate audio duration and number of clips needed
          2. Select random prompts (or cycle through provided ones)
          3. Generate video clips in parallel (via inference_fn callback)
          4. Combine clips and overlay audio (via audio_processor_fn)
          5. Return orchestration result with output path

        Args:
            prompts: List of prompts for video generation
            audio_bytes: Audio file bytes (MP3 or other format)
            clip_duration: Duration of each clip in seconds (default: 5)
            num_frames_per_clip: Frames per clip (default: 15 * clip_duration)

        Returns:
            OrchestrationResult with generated video bytes and metadata

        Raises:
            ValueError: If required callbacks are not set
            RuntimeError: If audio processing fails
        """
        if self.inference_fn is None:
            raise ValueError("inference_fn callback must be set before orchestration")
        if self.audio_processor_fn is None:
            raise ValueError("audio_processor_fn callback must be set before orchestration")

        # Step 1: Get audio duration
        total_duration = self._get_audio_duration(audio_bytes)
        num_clips = self._calculate_num_clips(total_duration, clip_duration)

        if num_frames_per_clip is None:
            num_frames_per_clip = 15 * clip_duration

        # Step 2: Prepare prompts
        selected_prompts = self._select_prompts(prompts, num_clips)

        # Step 3: Generate video clips
        video_clips = self._generate_video_clips(
            selected_prompts, num_frames_per_clip
        )

        # Step 4: Combine with audio
        final_video_bytes = self._combine_clips_with_audio(video_clips, audio_bytes)

        # Step 5: Return result
        return OrchestrationResult(
            job_id=self.finetune_id,  # Use finetune_id as job identifier
            output_path=f"s3://outputs/{self.finetune_id}/final_video.mp4",
            total_duration=total_duration,
            num_clips=num_clips,
            video_bytes=final_video_bytes,
        )

    def generate_single_clip(
        self,
        request: InferenceRequest,
    ) -> GenerationResult:
        """
        Generate a single video clip from a prompt.

        Args:
            request: InferenceRequest with job config and prompt

        Returns:
            GenerationResult with generated video bytes

        Raises:
            ValueError: If inference_fn is not set
        """
        if self.inference_fn is None:
            raise ValueError("inference_fn callback must be set before generation")

        # Replace trigger word placeholder
        prompt = self._replace_trigger_word(request.prompt)

        # Call GPU inference function
        video_bytes = self.inference_fn(
            prompt=prompt,
            guidance_scale=request.guidance_scale,
            num_frames=request.num_frames,
        )

        return GenerationResult(
            prompt=prompt,
            video_bytes=video_bytes,
            duration_seconds=request.num_frames / 15.0,
        )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _replace_trigger_word(self, prompt: str) -> str:
        """Replace [trigger] placeholder with configured trigger word."""
        return prompt.replace("[trigger]", self.trigger_word)

    def _get_audio_duration(self, audio_bytes: bytes) -> float:
        """
        Get duration of audio file in seconds.

        Currently returns a placeholder. In production, this would be called
        as a remote function via Modal or similar.

        Args:
            audio_bytes: Audio file bytes

        Returns:
            Duration in seconds
        """
        # In production, this would call:
        # return get_duration.remote(audio_bytes)
        # For now, placeholder that subclasses/tests can override
        if hasattr(self, "_audio_duration_fn"):
            return self._audio_duration_fn(audio_bytes)
        return 60.0  # Default placeholder

    def _calculate_num_clips(self, total_duration: float, clip_duration: int) -> int:
        """Calculate number of clips needed to cover audio duration."""
        return int(total_duration // clip_duration) + (
            1 if total_duration % clip_duration != 0 else 0
        )

    def _select_prompts(self, prompts: List[str], num_clips: int) -> List[str]:
        """
        Select prompts for clip generation.

        If num_clips > len(prompts), cycles through prompts with random
        selections. Otherwise, takes the first num_clips.

        Args:
            prompts: Available prompts
            num_clips: Number of clips to generate

        Returns:
            List of prompts to use (length == num_clips)
        """
        if not prompts:
            raise ValueError("At least one prompt must be provided")

        if len(prompts) >= num_clips:
            return prompts[:num_clips]

        # If fewer prompts than clips, cycle through them
        selected = []
        for i in range(num_clips):
            selected.append(prompts[i % len(prompts)])
        return selected

    def _generate_video_clips(
        self,
        prompts: List[str],
        num_frames: int,
    ) -> List[bytes]:
        """
        Generate video clips for each prompt.

        In production, this would orchestrate parallel execution via:
        - Modal: generator.run.map(prompts, ...)
        - FastAPI: async calls to GPU runner
        - Direct: sequential calls for testing

        Args:
            prompts: List of prompts to generate videos for
            num_frames: Frames per video clip

        Returns:
            List of video file bytes, same order as prompts
        """
        video_clips = []
        for prompt in prompts:
            # Call the inference function for each prompt
            video_bytes = self.inference_fn(
                prompt=prompt,
                num_frames=num_frames,
                guidance_scale=None,  # Use model default
            )
            video_clips.append(video_bytes)

        return video_clips

    def _combine_clips_with_audio(
        self,
        video_clips: List[bytes],
        audio_bytes: bytes,
    ) -> bytes:
        """
        Combine video clips and overlay audio.

        Args:
            video_clips: List of video file bytes
            audio_bytes: Audio file bytes (MP3 or similar)

        Returns:
            Final combined video bytes

        Raises:
            RuntimeError: If audio processing fails
        """
        return self.audio_processor_fn(video_clips, audio_bytes)

    # ========================================================================
    # Configuration Methods
    # ========================================================================

    def set_inference_callback(self, fn: Callable) -> "MusicVideoGenerator":
        """Set the GPU inference function callback."""
        self.inference_fn = fn
        return self

    def set_audio_processor_callback(self, fn: Callable) -> "MusicVideoGenerator":
        """Set the audio processing function callback."""
        self.audio_processor_fn = fn
        return self

    def set_trigger_word(self, word: str) -> "MusicVideoGenerator":
        """Set the trigger word for prompt replacement."""
        self.trigger_word = word
        return self


# ============================================================================
# Workflow Documentation
# ============================================================================

"""
COMPLETE MUSIC VIDEO GENERATION WORKFLOW
==========================================

Step 1: Fine-tune LoRA
  - User provides training images with captions
  - ai-toolkit LoRA training runs on Modal H100
  - Fine-tuned weights saved to volume: finetune-video-models/{finetune_id}/
  - Training config saved alongside model
  - Output: finetune_id (hash identifier)

Step 2: Load Configuration
  - Load YAML config from model directory
  - Parse with config_models.py Pydantic validators
  - Extract trigger_word, guidance_scale, sampling params

Step 3: Prepare Inference
  - Load base model (Wan2.1) to GPU memory
  - Load LoRA weights and merge into model
  - Model ready for inference on H100

Step 4: Run Parallel Inference
  - User provides audio file + list of prompts
  - Calculate audio duration → num_clips needed
  - Generate num_clips video files in parallel
  - Each clip:
    a. Replace [trigger] placeholder with trained word
    b. Run diffusion inference for specified num_frames
    c. Export frames to MP4 video file
    d. Return as bytes
  - All parallel calls complete

Step 5: Combine with Audio
  - Take all generated MP4 clips
  - Write to temporary directory
  - Concatenate video clips using ffmpeg
  - Overlay audio with aac encoding
  - Output: final_video.mp4

Step 6: Upload to Cloud Storage
  - Upload final video to Cloudflare R2 or S3
  - Store metadata in PostgreSQL job table
  - Return signed URL to user

ENTRY POINTS:
  - train_from_notebook.py: JupyterLab for training
  - inference.py: Single video generation test
  - music_video_generator.py: Full orchestration
  - FastAPI (to be created): HTTP API for all operations
"""
