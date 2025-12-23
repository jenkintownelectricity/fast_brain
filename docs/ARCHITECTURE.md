# HIVE215 Architecture

## System Overview

HIVE215 is a hybrid voice AI system with skill-based LoRA fine-tuning capabilities.

## Components

### 1. Dashboard (`unified_dashboard.py`)
- Flask web application
- Deployed on Modal (serverless)
- URL: `https://jenkintownelectricity--hive215-dashboard-flask-app.modal.run`
- Uses SQLite on Modal volume for persistent storage

### 2. Skill Trainer (`train_skill_modal.py`)
- GPU-based training on Modal (A10G)
- App name: `hive215-skill-trainer`
- Uses Unsloth + QLoRA for fast fine-tuning
- Saves adapters to Modal volume

### 3. TTS Service (`parler_integration.py`)
- Parler TTS for voice synthesis
- GPU-based on Modal

---

## Adapter Storage Architecture

### Single Source of Truth: Modal Volume

```
/adapters/                          ← Modal Volume: hive215-adapters
├── monday_com_expert_skills/
│   ├── adapter_model.safetensors   ← LoRA weights
│   ├── adapter_config.json         ← LoRA config
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── training_metadata.json      ← Training stats (loss, examples, etc.)
│
├── another_skill/
│   └── ...
```

### Why Modal Volume is Source of Truth

| Approach | Problem |
|----------|---------|
| Database as source | Modal trainer can't write to dashboard's SQLite |
| Sync on complete | Requires callback mechanism, can fail |
| **Modal Volume** | Trainer writes directly, dashboard reads directly |

### Data Flow

```
1. TRAINING STARTS
   Dashboard → spawn(skill_id, training_data) → Modal Trainer

2. TRAINING COMPLETES
   Modal Trainer → Saves to /adapters/<skill_id>/training_metadata.json
                → Calls adapters_volume.commit()

3. UI REQUESTS ADAPTERS
   Dashboard → trainer.list_adapters.remote()
            → Reads training_metadata.json from each adapter folder
            → Returns fresh data

4. UI DISPLAYS
   Shows correct loss (0.266), examples (126), training time
```

### Key Files

**training_metadata.json** (saved after training):
```json
{
  "skill_id": "monday_com_expert_skills",
  "skill_name": "Monday.com Expert",
  "base_model": "unsloth/Qwen2.5-1.5B-Instruct",
  "training_examples": 126,
  "final_loss": 0.2660,
  "epochs": 3,
  "lora_r": 16,
  "lora_alpha": 16,
  "learning_rate": 0.0002,
  "training_time_seconds": 606.5,
  "trained_at": "2024-12-23T02:15:00.000Z"
}
```

---

## Database Architecture

### Modal Volume: `/data/hive215.db`

| Table | Purpose |
|-------|---------|
| `skills` | Skill definitions (name, system_prompt, knowledge) |
| `training_data` | Manually added training examples |
| `extracted_data` | Parsed/extracted training examples |
| `training_jobs` | Training job status and history |
| `trained_adapters` | **DEPRECATED** - Use Modal volume instead |

### Important: `extracted_data` Table

Training examples are stored here with:
- `skill_id`: Which skill this example belongs to
- `user_input`: User message
- `assistant_response`: Expected response
- `is_approved`: Approval status (not used for training)
- `is_archived`: Archived status (excluded from training)

**Note**: The `is_approved` filter was removed from training queries. All non-archived data is used for training.

---

## API Endpoints

### Adapter Endpoints (Read from Modal Volume)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/training/adapters` | GET | List all adapters from Modal volume |
| `/api/training/adapters/<skill_id>` | GET | Get adapter for specific skill |
| `/api/trained-adapters` | GET | Legacy endpoint (same as above) |

### Training Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/train-skill/<skill_id>` | POST | Start training job |
| `/api/training/status/<skill_id>` | GET | **Enhanced** real-time training status with metrics |
| `/api/training-job/<skill_id>` | GET | Legacy training status |
| `/api/training-jobs` | GET | List all training jobs |

### Enhanced Training Status API

The `/api/training/status/<skill_id>` endpoint provides comprehensive real-time metrics:

- **Progress tracking**: current_step, total_steps, progress percentage
- **Loss monitoring**: current_loss, starting_loss, loss_history, loss_improvement_percent
- **Time estimates**: elapsed_seconds, eta_seconds
- **Training context**: current_epoch, total_epochs, examples_processed
- **GPU metrics**: memory usage, utilization (simulated for A10G)
- **Live preview**: current_example_preview showing what's being learned

**Usage**: Poll every 3 seconds from frontend to power real-time training dashboard.

See `docs/training_experience_spec.md` for full UI implementation spec.

### Debug Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/debug/database` | GET | Database state and table counts |

---

## Deployment

### Deploy All Services

```powershell
py -3.11 -m modal deploy deploy_dashboard.py
py -3.11 -m modal deploy train_skill_modal.py
py -3.11 -m modal deploy parler_integration.py
```

### Modal Apps

| App Name | GPU | Purpose |
|----------|-----|---------|
| `hive215-dashboard` | None | Web UI |
| `hive215-skill-trainer` | A10G | LoRA training |
| `hive215-parler-tts` | T4 | Voice synthesis |

### Modal Volumes

| Volume | Mount Path | Purpose |
|--------|------------|---------|
| `hive215-data` | `/data` | SQLite database |
| `hive215-adapters` | `/adapters` | Trained LoRA adapters |
