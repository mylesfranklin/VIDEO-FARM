# Music Video Generation - Refactoring to Production Architecture

## Overview

This document describes the refactoring of the **music-video-gen** proof-of-concept into a robust, production-ready, service-oriented architecture.

### Target Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    FastAPI Orchestrator                       │
│             (HTTP API, Job Management, DB)                    │
└─────────────────────────────┬──────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
        ┌───────────▼──────────┐  ┌────▼────────────────┐
        │   Modal GPU Runner    │  │ PostgreSQL Database │
        │ (H100 GPU Inference)  │  │  (Job Metadata)     │
        │                       │  │                     │
        │ • Video Generation    │  │ • Job Status        │
        │ • Audio Processing    │  │ • Results Paths     │
        │ • Model Loading       │  │ • Timestamps        │
        └───────────┬───────────┘  └────────────────────┘
                    │
        ┌───────────▼──────────────┐
        │ Cloudflare R2 / S3       │
        │   Object Storage         │
        │                          │
        │ • Models & Weights       │
        │ • Generated Videos       │
        │ • Audio Files            │
        └──────────────────────────┘
```

---

## Refactored Modules

### 1. `config_models.py` - Configuration Validation

**Purpose:** Replace raw YAML/dict access with type-safe Pydantic models.

**Key Classes:**
- `TrainingConfig` - Root configuration
- `TrainingProcessConfig` - Single training process
- `ModelConfig`, `NetworkConfig`, `TrainConfig`, `SaveConfig`, `SampleConfig` - Sub-configurations
- `InferenceRequest` - API request model
- `OrchestrationConfig` - Video orchestration parameters

**Benefits:**
- Type safety and IDE autocomplete
- Automatic validation and error messages
- Easy serialization/deserialization
- Backward compatible with YAML

**Usage:**
```python
from config_models import TrainingConfig

# Load from YAML
config = TrainingConfig.from_dict(yaml.safe_load(f))

# Access with type hints
base_model = config.process[0].model.name_or_path
trigger = config.process[0].trigger_word

# Or get dict for backward compatibility
process_dict = config.get_process(0).dict()
```

---

### 2. `generator_logic.py` - Pure Python Core Logic

**Purpose:** Extract video generation orchestration logic with no Modal/Framework decorators.

**Key Class: `MusicVideoGenerator`**

This is the **orchestration** layer - pure Python that:
- Manages the workflow of video generation
- Accepts callbacks for GPU-intensive operations
- Can be used with Modal, FastAPI, or any other framework

**Workflow Steps:**
```
1. Load Configuration
   └─ Parse YAML, extract trigger_word, model params

2. Calculate Audio Duration
   └─ Determine how many clips needed

3. Generate Video Clips (Parallel)
   └─ For each prompt, call inference_fn callback
   └─ Collect video bytes from GPU runner

4. Combine with Audio
   └─ Call audio_processor_fn callback
   └─ Apply ffmpeg video concatenation + audio overlay

5. Upload to Cloud Storage
   └─ Save final video to S3/R2
   └─ Return signed URL
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `orchestrate_music_video()` | Full pipeline coordination |
| `generate_single_clip()` | Single video generation |
| `_generate_video_clips()` | Parallel clip generation |
| `_combine_clips_with_audio()` | Audio overlay |
| `_get_audio_duration()` | Audio metadata |

**Callback Pattern:**

The class accepts callbacks for operations that can't run locally:

```python
# Create generator
generator = MusicVideoGenerator(
    finetune_id="abc123",
    trigger_word="[trigger]",
    inference_fn=gpu_inference_function,      # Callback: GPU inference
    audio_processor_fn=audio_combination_fn,  # Callback: ffmpeg processing
)

# Run orchestration
result = generator.orchestrate_music_video(
    prompts=["prompt1", "prompt2"],
    audio_bytes=audio_data,
    clip_duration=5
)
```

**Advantages:**
- Pure Python, testable in isolation
- Decoupled from Modal, FastAPI, or execution framework
- Clear separation of orchestration vs. computation
- Easy to mock/stub for testing

---

### 3. `utilities.py` - Helper Functions & Cloud Integration

**Purpose:** Centralize all utility functions and provide cloud storage abstraction.

**Sections:**

#### Configuration Loading
```python
load_config_from_yaml(yaml_path)       # Returns Pydantic TrainingConfig
load_process_config(yaml_path)         # Returns dict for backward compat
```

#### Model Loading
```python
load_model(base_model, to_cuda=True)   # Load WAN2.1 to GPU
```

#### LoRA Weight Handling
```python
load_and_convert(safetensors_file)     # Load + convert ai-toolkit → diffusers format
prep_lora_weights(finetune_id)         # Prepare weights for inference
convert_to_diffusers(state_dict)       # Handle all weight key mappings
```

#### String Utilities
```python
slugify(text, max_length=100)          # Convert to URL-safe slugs
```

#### Cloud Storage - `CloudStorageHandler`

Abstraction layer for S3/Cloudflare R2:

```python
# Initialize
storage = CloudStorageHandler(
    bucket_name="my-videos",
    endpoint_url="https://account.r2.cloudflarestorage.com",
    access_key="...",
    secret_key="..."
)

# Upload
url = storage.upload_bytes(video_data, "videos/final.mp4", "video/mp4")

# Download
video = storage.download_bytes("videos/final.mp4")

# Signed URLs
signed_url = storage.get_signed_url("videos/final.mp4", expiration_seconds=3600)

# List objects
videos = storage.list_objects("videos/")
```

**Factory Function for Cloudflare R2:**

```python
# Environment variables:
# R2_ACCOUNT_ID=abc123
# R2_ACCESS_KEY=xxxxx
# R2_SECRET_KEY=yyyyy

storage = create_cloudflare_r2_handler("my-bucket")
```

---

### 4. `modal_runner.py` - GPU Execution Layer

**Purpose:** Modal-decorated wrapper that handles GPU computation.

**Key Components:**

#### A. Container Image Definition

```python
inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "torch==2.5.1",
        "diffusers@git+https://...",
        "accelerate==1.5.1",
        "peft==0.14.0",  # LoRA
        "boto3==1.34.0",  # Cloud storage
        "pydantic==2.5.0",  # Config validation
        # ... other deps
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
```

**Reproducible Environment:**
- All dependencies pinned to specific versions
- Explicit apt packages listed
- Environment variables configured
- No Dockerfile needed - purely programmatic

#### B. Stateful GPU Runner

```python
@app.cls(
    gpu="h100",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    timeout=30 * MINUTES,
)
class ModalMusicVideoGenerator:
    @modal.enter()
    def init(self):
        # Load model to GPU once
        self.pipe = load_model(...)
        self.pipe.load_lora_weights(...)

    @modal.method()
    def generate(self, prompt, num_frames) -> bytes:
        # Run inference, return video bytes
```

**Lifecycle:**
1. Container created with H100 GPU
2. `@modal.enter()` loads model to GPU memory
3. Multiple `.generate()` calls reuse loaded model
4. Container scales down after 5 minutes of inactivity

#### C. Stateless Functions

For operations that don't need model state:

```python
@app.function(image=audio_image)
def get_audio_duration(audio_bytes: bytes) -> float:
    # Fast operation, returns float

@app.function(image=audio_image)
def combine_videos_with_audio(videos, audio) -> bytes:
    # ffmpeg processing, returns video bytes

@app.function(image=inference_image)
def upload_to_storage(video_bytes, job_id, bucket) -> str:
    # Upload and return signed URL
```

#### D. Testing Entrypoints

```python
@app.local_entrypoint()
def test_single_generation(...):
    # Test video generation locally

@app.local_entrypoint()
def test_audio_duration(...):
    # Test audio processing

# Usage:
# modal run modal_runner.py::test_single_generation --finetune-id abc123
```

---

## Integration Pattern: How Components Work Together

### Scenario: FastAPI calls video generation

```
1. User POST to /api/videos/generate
   ├─ Body: InferenceRequest (Pydantic model)
   │  ├─ job_id
   │  ├─ finetune_id
   │  ├─ prompt
   │  └─ num_frames
   │
2. FastAPI Endpoint Handler
   ├─ Save job to PostgreSQL (status: PENDING)
   ├─ Create MusicVideoGenerator instance
   ├─ Set callbacks to modal_runner functions
   │
3. MusicVideoGenerator.generate_single_clip()
   ├─ Replace trigger word
   ├─ Call inference_fn callback (GPU)
   │  └─ ModalMusicVideoGenerator.generate.aio()
   │     └─ Runs on H100, returns video bytes
   │
4. Update Database
   ├─ Save video to S3/R2
   ├─ Store signed URL in PostgreSQL
   ├─ Update status: COMPLETED
   │
5. Return to User
   ├─ HTTP 200 with video URL
```

### Scenario: FastAPI orchestrates full music video

```
1. User POST to /api/music-videos/create
   ├─ Body: OrchestrationConfig (Pydantic)
   │  ├─ job_id
   │  ├─ finetune_id
   │  ├─ prompts: ["prompt1", "prompt2", ...]
   │  ├─ audio_url: "s3://bucket/audio.mp3"
   │  └─ clip_duration: 5
   │
2. FastAPI Orchestration Handler
   ├─ Download audio from cloud storage
   ├─ Create MusicVideoGenerator
   ├─ Set modal_runner callbacks
   │
3. generator.orchestrate_music_video()
   ├─ Get audio duration → 60 seconds
   ├─ Calculate clips → 12 clips needed
   ├─ Parallel generation via Modal
   │  └─ generator.generate.map(prompts)
   │     └─ Each prompt runs on separate GPU container
   ├─ Combine clips with audio
   │  └─ combine_videos_with_audio.remote()
   │     └─ ffmpeg concatenation + audio overlay
   ├─ Upload final video
   │  └─ upload_to_storage.remote()
   │
4. Return Orchestration Result
   ├─ Output path
   ├─ Total duration
   ├─ Number of clips
   └─ Video bytes (or signed URL)
```

---

## File Structure Comparison

### Before (POC)
```
VIDEO-FARM/
├── inference.py (195 lines, Modal mixed)
├── music_video_generator.py (113 lines, Modal mixed)
├── train_from_notebook.py (82 lines, Modal mixed)
├── data/
├── config/
│   └── train_lora_wan21_1b.yaml
└── notebooks/
```

**Issues:**
- Modal decorators mixed with business logic
- Hard to test without Modal
- Configuration as raw YAML dicts
- No abstraction for cloud storage
- Code duplicated across files

### After (Production)
```
VIDEO-FARM/
├── config_models.py (200+ lines)
│   └─ Pydantic config validation
│
├── generator_logic.py (300+ lines)
│   └─ Pure Python orchestration, no decorators
│
├── utilities.py (400+ lines)
│   ├─ Configuration loaders
│   ├─ Model utilities
│   ├─ LoRA weight handling
│   ├─ Cloud storage abstraction
│   └─ String helpers
│
├── modal_runner.py (400+ lines)
│   ├─ Modal image definitions
│   ├─ GPU runner class (@app.cls)
│   ├─ Stateless functions (@app.function)
│   └─ Local test entrypoints
│
├── fastapi_orchestrator.py (to be created)
│   ├─ HTTP API endpoints
│   ├─ Job management
│   └─ Database integration
│
├── config/
│   └── train_lora_wan21_1b.yaml
│
├── data/
├── notebooks/
└── REFACTORING_GUIDE.md (this file)
```

**Improvements:**
- ✅ Separation of concerns (orchestration vs. computation vs. config)
- ✅ Pure Python logic testable without Modal
- ✅ Type-safe configuration
- ✅ Cloud storage abstraction
- ✅ Production-ready error handling
- ✅ Clear extension points (callbacks)

---

## Migration Guide: Updating Existing Code

### Old Code (inference.py)

```python
# Load config as dict
config = load_config(MODELS_DIR / finetune_id / "config.yaml")
trigger = config["trigger_word"]
```

### New Code (generator_logic.py + utilities.py)

```python
from utilities import load_process_config
from config_models import TrainingConfig

# Option 1: Use Pydantic models
config = load_config_from_yaml(Path("..."))
trigger = config.process[0].trigger_word

# Option 2: Backward compatible dict
config_dict = load_process_config(Path("..."))
trigger = config_dict["trigger_word"]
```

---

## Deployment Considerations

### Local Testing

```bash
# Test generator logic without Modal
python -c "
from generator_logic import MusicVideoGenerator

gen = MusicVideoGenerator('test')
# Mock callbacks...
result = gen.orchestrate_music_video(...)
"
```

### Modal Testing

```bash
# Test GPU functions
modal run modal_runner.py::test_single_generation \
  --finetune-id abc123 \
  --prompt '[trigger] dancing'

modal run modal_runner.py::test_combine_videos \
  --num-clips 5
```

### FastAPI Deployment

```bash
# Start orchestrator (to be implemented)
uvicorn fastapi_orchestrator:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health

# Trigger generation
curl -X POST http://localhost:8000/api/videos/generate \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "job-123",
    "finetune_id": "abc123",
    "prompt": "[trigger] dancing",
    "num_frames": 40
  }'
```

---

## Next Steps: FastAPI Orchestrator

The `fastapi_orchestrator.py` module (to be created) should include:

### Endpoints

```python
# Health & Status
GET  /health                          # Service health
GET  /jobs/{job_id}                   # Check job status

# Video Generation
POST /api/videos/generate             # Single video from prompt
GET  /api/videos/{job_id}             # Get video result

# Music Video Orchestration
POST /api/music-videos/create         # Full orchestration
POST /api/music-videos/{job_id}/clips # Batch clip generation

# Model Management
GET  /models                          # List available fine-tuned models
POST /models/{finetune_id}/validate   # Check model status
```

### Database Integration

```python
# PostgreSQL schema
jobs
├── job_id (PK)
├── finetune_id (FK)
├── status (PENDING, RUNNING, COMPLETED, FAILED)
├── request_params (JSON)
├── result_path (S3/R2 URL)
├── signed_url (temporary access)
├── created_at
└── completed_at

models
├── finetune_id (PK)
├── base_model
├── trigger_word
├── config_yaml
├── storage_path
└── created_at
```

### Async Task Queue

For long-running operations (orchestration), use:
- **Option 1:** Modal's async (.aio) functions
- **Option 2:** Celery + Redis for distributed tasks
- **Option 3:** Background threads with worker pool

### Error Handling & Logging

```python
try:
    result = await generator.orchestrate_music_video(...)
except ValueError as e:
    logger.error(f"Validation error: {e}")
    db.update_job(job_id, status="FAILED", error=str(e))
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    db.update_job(job_id, status="FAILED", error="Internal server error")
```

---

## Summary

This refactoring transforms the POC into production-ready code by:

1. **Separating Concerns**
   - `config_models.py` - Validation & type safety
   - `generator_logic.py` - Business logic (testable)
   - `utilities.py` - Reusable helpers
   - `modal_runner.py` - GPU execution (replaceable)

2. **Enabling Testability**
   - Pure Python classes without framework decorators
   - Mock-friendly callback pattern
   - Clear data flow

3. **Adding Cloud Integration**
   - S3/Cloudflare R2 abstraction
   - Signed URL generation
   - Scalable blob storage

4. **Improving Configuration**
   - Type-safe Pydantic models
   - YAML validation
   - Easy extension

5. **Preparing for Production**
   - Reproducible container images
   - Stateful GPU runners for efficiency
   - Clear API contracts
   - Extensible for FastAPI, databases, logging

The architecture is now ready for HTTP API layer, database integration, and horizontal scaling.
