"""
Pydantic data models for configuration parsing and validation.

This module provides type-safe configuration management for the music video
generation pipeline, replacing raw YAML/dict access with validated models.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, validator


# ============================================================================
# Dataset Configuration
# ============================================================================

class DatasetConfig(BaseModel):
    """Configuration for a training dataset."""

    folder_path: str = Field(..., description="Path to dataset folder")
    caption_ext: str = Field(default="txt", description="Caption file extension")
    resolution: List[int] = Field(..., description="[height, width] resolution")
    cache_latents_to_disk: bool = Field(default=True)
    caption_dropout_rate: float = Field(default=0.05)
    shuffle_tokens: bool = Field(default=False)

    class Config:
        extra = "allow"


# ============================================================================
# Model Configuration
# ============================================================================

class ModelConfig(BaseModel):
    """Configuration for the base model."""

    arch: str = Field(..., description="Model architecture (e.g., 'wan21')")
    name_or_path: str = Field(..., description="Model name or HuggingFace path")
    quantize_te: bool = Field(default=True, description="Quantize text encoder")

    class Config:
        extra = "allow"


# ============================================================================
# Network/LoRA Configuration
# ============================================================================

class NetworkConfig(BaseModel):
    """Configuration for LoRA adapter network."""

    type: Literal["lora"] = Field(default="lora")
    linear: int = Field(..., description="LoRA rank for linear layers")
    linear_alpha: int = Field(..., description="LoRA alpha scaling for linear layers")

    class Config:
        extra = "allow"


# ============================================================================
# Training Configuration
# ============================================================================

class OptimizerParams(BaseModel):
    """Optimizer hyperparameters."""

    weight_decay: float = Field(default=1e-4)

    class Config:
        extra = "allow"


class EmaConfig(BaseModel):
    """Exponential Moving Average configuration."""

    use_ema: bool = Field(default=True)
    ema_decay: float = Field(default=0.99)

    class Config:
        extra = "allow"


class TrainConfig(BaseModel):
    """Training hyperparameters."""

    steps: int = Field(..., description="Number of training steps")
    batch_size: int = Field(default=1)
    dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="bf16", description="Training precision (bf16, float16, float32)"
    )
    lr: float = Field(default=1e-4, description="Learning rate")
    optimizer: str = Field(default="adamw8bit", description="Optimizer type")
    optimizer_params: Optional[OptimizerParams] = None
    gradient_checkpointing: bool = Field(default=True)
    gradient_accumulation: int = Field(default=1)
    ema_config: Optional[EmaConfig] = None
    noise_scheduler: str = Field(default="flowmatch")
    timestep_type: str = Field(default="sigmoid")
    train_text_encoder: bool = Field(default=False)
    train_unet: bool = Field(default=True)

    class Config:
        extra = "allow"

    @validator("dtype", pre=True)
    def normalize_dtype(cls, v):
        """Normalize dtype aliases."""
        if v == "bf16":
            return "bfloat16"
        return v


# ============================================================================
# Save Configuration
# ============================================================================

class SaveConfig(BaseModel):
    """Configuration for model checkpointing and saving."""

    dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="float16", description="Saved model precision"
    )
    save_every: int = Field(default=100, description="Save checkpoint every N steps")
    max_step_saves_to_keep: int = Field(default=40, description="Max checkpoints to keep")
    push_to_hub: bool = Field(default=False, description="Push to HuggingFace Hub")

    class Config:
        extra = "allow"


# ============================================================================
# Sampling Configuration
# ============================================================================

class SampleConfig(BaseModel):
    """Configuration for inference sampling."""

    prompts: List[str] = Field(default=[], description="Sample prompts for inference")
    guidance_scale: float = Field(default=5.0, description="Classifier-free guidance scale")
    num_frames: int = Field(default=40, description="Number of frames to generate")
    height: int = Field(..., description="Video height in pixels")
    width: int = Field(..., description="Video width in pixels")
    fps: int = Field(default=15, description="Frames per second")
    sampler: str = Field(default="flowmatch", description="Sampling method")
    sample_steps: int = Field(default=30, description="Number of sampling steps")
    sample_every: int = Field(default=250, description="Sample every N training steps")
    seed: int = Field(default=42)
    walk_seed: bool = Field(default=True, description="Increment seed for each sample")
    neg: str = Field(default="", description="Negative prompt")

    class Config:
        extra = "allow"


# ============================================================================
# Training Process Configuration
# ============================================================================

class TrainingProcessConfig(BaseModel):
    """Configuration for a single training process."""

    type: str = Field(default="sd_trainer", description="Trainer type")
    device: str = Field(default="cuda:0", description="Device to train on")
    trigger_word: str = Field(..., description="Trigger word used in prompts")
    training_folder: str = Field(..., description="Path to save training outputs")

    # Sub-configurations
    datasets: List[DatasetConfig] = Field(..., description="Training datasets")
    model: ModelConfig = Field(..., description="Base model configuration")
    network: NetworkConfig = Field(..., description="LoRA network configuration")
    train: TrainConfig = Field(..., description="Training hyperparameters")
    sample: SampleConfig = Field(..., description="Sampling configuration")
    save: SaveConfig = Field(..., description="Save configuration")

    class Config:
        extra = "allow"


# ============================================================================
# Top-Level Configuration
# ============================================================================

class ConfigMetadata(BaseModel):
    """Metadata about the configuration."""

    name: str = Field(..., description="Configuration name")
    version: str = Field(default="1.0", description="Configuration version")

    class Config:
        extra = "allow"


class TrainingConfig(BaseModel):
    """Root-level configuration model."""

    config: ConfigMetadata = Field(..., description="Configuration metadata")
    process: List[TrainingProcessConfig] = Field(
        ..., description="List of training processes"
    )
    job: str = Field(default="extension", description="Job type")

    class Config:
        extra = "allow"

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        """Parse from dictionary (e.g., loaded YAML)."""
        return cls(**data)

    def get_process(self, index: int = 0) -> TrainingProcessConfig:
        """Get a specific process configuration."""
        return self.process[index]


# ============================================================================
# Inference Request Configuration
# ============================================================================

class InferenceRequest(BaseModel):
    """Configuration for an inference request."""

    job_id: str = Field(..., description="Unique job identifier")
    finetune_id: str = Field(..., description="Fine-tuned model identifier")
    prompt: str = Field(..., description="Generation prompt")
    guidance_scale: Optional[float] = Field(None, description="Override guidance scale")
    num_frames: int = Field(default=40, description="Number of frames to generate")
    image_urls: List[str] = Field(default=[], description="Reference image URLs")

    class Config:
        extra = "allow"


# ============================================================================
# Orchestration Configuration
# ============================================================================

class OrchestrationConfig(BaseModel):
    """Configuration for video orchestration (combining clips with audio)."""

    job_id: str = Field(..., description="Unique job identifier")
    finetune_id: str = Field(..., description="Fine-tuned model identifier")
    prompts: List[str] = Field(..., description="List of prompts for clips")
    audio_url: str = Field(..., description="URL to audio file in cloud storage")
    clip_duration: int = Field(default=5, description="Duration of each clip in seconds")
    output_bucket: str = Field(..., description="Cloud storage bucket for output")
    output_key: str = Field(..., description="Cloud storage key for final video")

    class Config:
        extra = "allow"
