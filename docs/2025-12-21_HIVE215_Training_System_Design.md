# HIVE215 / Fast Brain Training System
## Comprehensive Design Document

**Date:** December 21, 2025  
**Version:** 1.0  
**Status:** Research Complete, Ready for Implementation

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Market Research & Competitor Analysis](#market-research--competitor-analysis)
3. [Customer Pain Points & Loves](#customer-pain-points--loves)
4. [Best Practices](#best-practices)
5. [Technical Architecture](#technical-architecture)
6. [UI/UX Design Specification](#uiux-design-specification)
7. [Error Handling & Common Problems](#error-handling--common-problems)
8. [Logging & Monitoring](#logging--monitoring)
9. [Implementation Plan](#implementation-plan)

---

## Executive Summary

### Vision
Build a **one-click training system** integrated into the Fast Brain Unified Dashboard that enables users to fine-tune AI skills directly from the UI, with real-time progress streaming, intelligent error handling, and seamless adapter management.

### Key Differentiators
| Feature | Competitors | HIVE215 |
|---------|-------------|---------|
| Training Time | 30-60 min | ~10 min (Unsloth 2x faster) |
| Cost per Train | $2-10 | ~$0.50-2.00 |
| No-Code UI | Some | ✅ Full |
| Real-time Progress | Rare | ✅ Live streaming |
| Integrated Testing | Manual | ✅ One-click chat test |
| Error Recovery | Basic | ✅ Smart auto-recovery |

---

## Market Research & Competitor Analysis

### Top Competitor Platforms

#### 1. H2O LLM Studio
**Strengths:**
- No-code GUI designed for LLMs
- Visual training progress charts
- Chat with model after training
- One-click export to HuggingFace
- Neptune/W&B integration

**Weaknesses:**
- Requires self-hosting
- Limited to single dataset per run
- No built-in GPU provisioning

**Key Features to Copy:**
- Charts tab: train/validation loss, metrics, learning rate visualization
- Train Data Insights tab: verify input data before training
- Inline model chat testing

#### 2. LLaMA Factory
**Strengths:**
- Web UI + CLI dual interface
- 100+ model support
- Zero-code workflows
- Beginner-friendly

**Weaknesses:**
- Difficult to add custom datasets
- Limited logging integrations

**Key Features to Copy:**
- Model selection dropdown with previews
- Parameter presets (Easy/Medium/Advanced)
- Progress bar with ETA

#### 3. Together AI / Fireworks AI
**Strengths:**
- Managed GPU infrastructure
- LoRA fine-tuning as service
- OpenAI-compatible APIs
- Multi-LoRA serving

**Weaknesses:**
- Higher cost ($0.52/hr for hosting)
- Limited customization
- Vendor lock-in

**Key Features to Copy:**
- Transparent pricing calculator
- One-click deployment to inference
- Version comparison tools

#### 4. Axolotl
**Strengths:**
- Most flexible/powerful
- Multi-GPU support (FSDP, DeepSpeed)
- Latest techniques (GaLore, Ring FlashAttention)
- YAML-based configs

**Weaknesses:**
- Steep learning curve
- No GUI
- Complex setup

**Key Features to Copy:**
- Configuration presets
- W&B/MLflow integration
- Checkpoint management

#### 5. Unsloth (Our Current Choice)
**Strengths:**
- 2-5x faster training
- 80% less VRAM
- Works on consumer GPUs
- Free open-source

**Weaknesses:**
- Single-GPU only (OSS)
- Less model variety than Axolotl

**Why We Chose Unsloth:**
- Perfect for Modal's A10G GPUs
- Cost-efficient for voice AI use case
- Fast iteration cycles

---

## Customer Pain Points & Loves

### 😡 What Customers HATE (Must Avoid)

| Pain Point | Example | Our Solution |
|------------|---------|--------------|
| **Silent Failures** | Training crashes with no useful error | Clear error messages with suggested fixes |
| **VRAM OOM** | "CUDA out of memory" with no guidance | Auto-detect and suggest batch size reduction |
| **No Progress Visibility** | No idea if training is working | Real-time loss curves and ETA |
| **Template Mismatches** | Model performs worse after training | Validate chat template before training |
| **Slow Iteration** | Hours to test one change | ~10 min training cycles |
| **Data Quality Mystery** | Don't know if data is the problem | Pre-training data validation report |
| **Can't Reproduce** | "It worked yesterday" syndrome | Full config + data versioning |
| **Complex Setup** | 50 lines of config before first run | One-click with smart defaults |

### 😍 What Customers LOVE (Must Have)

| Feature | Why It Matters | Implementation |
|---------|----------------|----------------|
| **One-Click Training** | Reduces barrier to entry | "Train" button with smart defaults |
| **Live Progress** | Builds confidence | WebSocket streaming of loss/steps |
| **Fast Iteration** | Enables experimentation | Unsloth 2x speedup |
| **Clear Errors** | Actionable guidance | Curated error → fix mapping |
| **Reasonable Defaults** | Works out of the box | Pre-tuned hyperparameters |
| **Chat Testing** | Instant feedback | Test chat after training |
| **Cost Visibility** | Budget planning | Show estimated cost before start |
| **Export Anywhere** | Flexibility | HF Hub, GGUF, Modal volume |

---

## Best Practices

### Training Hyperparameters (Unsloth Recommended)

```yaml
# Optimal defaults for QLoRA fine-tuning
base_model: "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
method: "QLoRA"
quantization: "4-bit NF4"

# LoRA Configuration
lora_r: 16              # Rank (16 typical, 32-64 for complex)
lora_alpha: 16          # Same as r, or r * 2
lora_dropout: 0         # Unsloth recommends 0
target_modules:         # All attention + MLP for best results
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Training Configuration
learning_rate: 2e-4     # Start here
lr_scheduler: "cosine"  # Or "linear"
batch_size: 2           # Per device
gradient_accumulation: 4 # Effective batch = 8
max_steps: null         # Let epochs control
num_epochs: 3           # For small datasets
warmup_ratio: 0.03      # 3% warmup

# Optimization
bf16: true              # If GPU supports
gradient_checkpointing: true
packing: true           # Sample packing for efficiency
```

### Data Quality Guidelines

| Metric | Minimum | Recommended | Excellent |
|--------|---------|-------------|-----------|
| Examples | 10 | 50-100 | 500+ |
| Avg Token Length | 50 | 200-500 | 1000+ |
| Topic Coverage | 3 topics | 10+ topics | 50+ topics |
| Response Quality | Basic | Detailed | Expert-level |

### Training Data Format

```json
// HIVE215 format (auto-converted)
{
  "skill_id": "molasses-master-expert",
  "system_prompt": "You are the Molasses Alchemist...",
  "conversations": [
    {
      "user": "What is blackstrap molasses?",
      "assistant": "Blackstrap molasses is the dark, viscous..."
    }
  ]
}

// Training format (internal)
{
  "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are the Molasses Alchemist...<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is blackstrap molasses?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nBlackstrap molasses is the dark, viscous...<|eot_id|>"
}
```

---

## Technical Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED DASHBOARD (Flask)                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Skills    │  │   Train     │  │   Adapters  │              │
│  │   Manager   │  │   Console   │  │   Gallery   │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          │                                       │
│  ┌───────────────────────▼───────────────────────────────────┐  │
│  │                   Training API Layer                       │  │
│  │  POST /api/train-skill/{id}     - Start training          │  │
│  │  GET  /api/training-status/{id} - Get progress            │  │
│  │  WS   /api/training-stream/{id} - Live updates            │  │
│  │  POST /api/test-adapter/{id}    - Test adapter            │  │
│  │  GET  /api/adapters             - List adapters           │  │
│  │  DELETE /api/adapters/{id}      - Delete adapter          │  │
│  └───────────────────────┬───────────────────────────────────┘  │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODAL (GPU Cloud)                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               SkillTrainer Class                         │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    │
│  │  │   train()   │  │ test_adapter│  │ list_adapters│     │    │
│  │  │   A10G GPU  │  │   A10G GPU  │  │    CPU      │     │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               Modal Volumes                              │    │
│  │  hive215-adapters/                                      │    │
│  │  ├── molasses-master-expert/                            │    │
│  │  │   ├── adapter_model.safetensors                      │    │
│  │  │   ├── adapter_config.json                            │    │
│  │  │   └── training_metadata.json                         │    │
│  │  └── tara-sales/                                         │    │
│  │      └── ...                                             │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SUPABASE (Database)                          │
├─────────────────────────────────────────────────────────────────┤
│  fb_skills              fb_training_jobs        fb_adapters     │
│  ├── id                 ├── id                  ├── id          │
│  ├── name               ├── skill_id            ├── skill_id    │
│  ├── system_prompt      ├── status              ├── version     │
│  ├── knowledge_items    ├── progress            ├── metrics     │
│  └── training_data      ├── config              ├── created_at  │
│                         ├── started_at          └── storage_path│
│                         ├── completed_at                        │
│                         └── error_message                       │
└─────────────────────────────────────────────────────────────────┘
```

### Database Schema Additions

```sql
-- Training Jobs Table
CREATE TABLE fb_training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    skill_id TEXT NOT NULL REFERENCES fb_skills(id),
    status TEXT NOT NULL DEFAULT 'pending',
    -- pending, queued, running, completed, failed, cancelled
    
    -- Configuration
    config JSONB NOT NULL DEFAULT '{}',
    -- {base_model, lora_r, learning_rate, epochs, batch_size}
    
    -- Progress Tracking
    progress REAL DEFAULT 0,  -- 0-100
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER,
    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER,
    current_loss REAL,
    
    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    estimated_completion TIMESTAMPTZ,
    
    -- Results
    final_loss REAL,
    training_time_seconds INTEGER,
    cost_usd REAL,
    error_message TEXT,
    error_details JSONB,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID REFERENCES auth.users(id)
);

-- Adapters Table
CREATE TABLE fb_adapters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    skill_id TEXT NOT NULL REFERENCES fb_skills(id),
    training_job_id UUID REFERENCES fb_training_jobs(id),
    
    -- Versioning
    version INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    
    -- Metrics
    final_loss REAL,
    training_examples INTEGER,
    training_time_seconds INTEGER,
    
    -- Storage
    storage_path TEXT NOT NULL,  -- Modal volume path
    adapter_size_bytes BIGINT,
    
    -- Config used
    base_model TEXT,
    lora_config JSONB,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    notes TEXT
);

-- Training Logs Table (for detailed progress)
CREATE TABLE fb_training_logs (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID REFERENCES fb_training_jobs(id),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    level TEXT,  -- info, warning, error
    message TEXT,
    data JSONB
);

-- Indexes
CREATE INDEX idx_training_jobs_skill ON fb_training_jobs(skill_id);
CREATE INDEX idx_training_jobs_status ON fb_training_jobs(status);
CREATE INDEX idx_adapters_skill ON fb_adapters(skill_id);
CREATE INDEX idx_training_logs_job ON fb_training_logs(job_id);
```

---

## UI/UX Design Specification

### Training Console Tab (New Dashboard Section)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 🧠 Skill Training                                           [? Help]    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Select Skill: [The Molasses Alchemist ▼]                               │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ 📊 Training Data Status                                            │ │
│  │                                                                    │ │
│  │  Examples: 12 ✓        Quality Score: Good ★★★☆☆                  │ │
│  │  Avg Tokens: 245       Coverage: 7 topics                          │ │
│  │                                                                    │ │
│  │  ⚠️ Recommendation: Add 38 more examples for optimal results       │ │
│  │                                                                    │ │
│  │  [+ Add Training Data]  [📄 View Examples]                         │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ ⚙️ Training Configuration                    [Simple ▼] [Advanced] │ │
│  │                                                                    │ │
│  │  Training Intensity:  [●───────────────] Light                     │ │
│  │                        Quick   Standard   Deep                     │ │
│  │                                                                    │ │
│  │  Estimated Time: ~8 minutes                                        │ │
│  │  Estimated Cost: ~$0.65                                            │ │
│  │                                                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    [🚀 Start Training]                            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Training Progress View (During Training)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 🧠 Training: The Molasses Alchemist                      [Cancel]       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Status: Training in Progress                                            │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  42%        │ │
│  │                                                                    │ │
│  │ Step 4/10  •  Epoch 2/5  •  ETA: 6 min 23 sec                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────┐ ┌─────────────────────────────────┐│
│  │ 📉 Training Loss               │ │ 📈 Learning Rate                ││
│  │                                 │ │                                 ││
│  │   3.5 ┤●                       │ │ 2e-4┤      ●●●●                 ││
│  │   3.0 ┤  ●                     │ │     │   ●●                      ││
│  │   2.5 ┤    ●                   │ │     │ ●●                        ││
│  │   2.0 ┤      ●●                │ │     │●                          ││
│  │   1.5 ┤          ●             │ │   0 ┼─────────────────          ││
│  │       └─────────────           │ │     └─────────────────          ││
│  │         1  2  3  4  step       │ │       1  2  3  4  step          ││
│  └─────────────────────────────────┘ └─────────────────────────────────┘│
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ 📋 Training Log                                          [Clear]  │ │
│  │                                                                    │ │
│  │ 12:34:15  ✓ Loaded 12 training examples                           │ │
│  │ 12:34:18  ✓ Model loaded: Llama-3.1-8B (4-bit)                    │ │
│  │ 12:34:22  ✓ LoRA configured (r=16, trainable: 41.9M params)       │ │
│  │ 12:34:25  → Training started...                                   │ │
│  │ 12:35:02  📊 Step 1/10: loss=3.368                                │ │
│  │ 12:35:38  📊 Step 2/10: loss=2.890                                │ │
│  │ 12:36:14  📊 Step 3/10: loss=2.456                                │ │
│  │ 12:36:50  📊 Step 4/10: loss=2.123  ← Current                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Training Complete View

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ✅ Training Complete: The Molasses Alchemist                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        🎉 SUCCESS!                                 │ │
│  │                                                                    │ │
│  │     Final Loss: 1.847    Training Time: 8 min 42 sec              │ │
│  │     Examples: 12         Cost: $0.58                               │ │
│  │                                                                    │ │
│  │  ┌──────────────────────────────────────────────────────────────┐ │ │
│  │  │ Improvement: Loss decreased 45% (3.37 → 1.85)                │ │ │
│  │  │ Quality: ★★★★☆ Good - Model learned skill personality        │ │ │
│  │  └──────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ 💬 Test Your Trained Skill                                        │ │
│  │                                                                    │ │
│  │ ┌────────────────────────────────────────────────────────────────┐│ │
│  │ │ You: What makes blackstrap molasses special?                   ││ │
│  │ └────────────────────────────────────────────────────────────────┘│ │
│  │                                                                    │ │
│  │ ┌────────────────────────────────────────────────────────────────┐│ │
│  │ │ 🧪 Molasses Alchemist:                                         ││ │
│  │ │ Ah, blackstrap molasses - the "third boil" treasure! Unlike    ││ │
│  │ │ light or dark molasses, blackstrap is the concentrated         ││ │
│  │ │ essence remaining after the third extraction from sugar cane.  ││ │
│  │ │ It's remarkably rich in minerals - iron, calcium, magnesium... ││ │
│  │ └────────────────────────────────────────────────────────────────┘│ │
│  │                                                                    │ │
│  │ [Send Another Test]                                                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ [🔄 Train Again]  [📤 Export to HuggingFace]  [🏠 Back to Skills] │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Adapters Gallery Tab

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 📦 Trained Adapters                               [↻ Refresh] [+ New]   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐│
│  │ 🧪 Molasses         │ │ 💼 Tara Sales       │ │ ⚡ Electrician      ││
│  │    Alchemist        │ │                      │ │    Service          ││
│  │                     │ │                      │ │                     ││
│  │ v3 • Active ✓       │ │ v1 • Active ✓       │ │ v2 • Inactive       ││
│  │                     │ │                      │ │                     ││
│  │ Loss: 1.85          │ │ Loss: 2.12          │ │ Loss: 1.94          ││
│  │ Examples: 12        │ │ Examples: 45        │ │ Examples: 28        ││
│  │ Trained: 2h ago     │ │ Trained: 3d ago     │ │ Trained: 1w ago     ││
│  │                     │ │                      │ │                     ││
│  │ [Test] [⋮ More]     │ │ [Test] [⋮ More]     │ │ [Activate] [⋮]      ││
│  └─────────────────────┘ └─────────────────────┘ └─────────────────────┘│
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ 📊 Training History                                               │ │
│  │                                                                    │ │
│  │  Date         Skill              Status    Loss    Time    Cost   │ │
│  │  ─────────────────────────────────────────────────────────────────│ │
│  │  Dec 21 12:34  Molasses Alchemist  ✓ Done   1.85    8m      $0.58 │ │
│  │  Dec 21 11:22  Molasses Alchemist  ✓ Done   2.67   10m      $0.72 │ │
│  │  Dec 20 15:45  Tara Sales          ✓ Done   2.12   15m      $1.05 │ │
│  │  Dec 19 09:30  Electrician Service ✗ Failed  -       -       $0.12 │ │
│  │                                                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Advanced Configuration Panel (Collapsed by Default)

```
┌────────────────────────────────────────────────────────────────────┐
│ ⚙️ Advanced Configuration                              [▼ Expand]  │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Base Model                                                         │
│  [Llama-3.1-8B (Recommended) ▼]                                    │
│  ├── Llama-3.1-8B (Recommended)                                    │
│  ├── Llama-3.2-3B (Faster, less capable)                           │
│  ├── Mistral-7B                                                     │
│  └── Qwen-2-7B                                                      │
│                                                                     │
│  LoRA Configuration                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │ Rank (r)        │  │ Alpha           │  │ Dropout         │     │
│  │ [16 ▼]          │  │ [16 ▼]          │  │ [0.0 ▼]         │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│                                                                     │
│  Training Parameters                                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │ Learning Rate   │  │ Epochs          │  │ Batch Size      │     │
│  │ [2e-4        ]  │  │ [3 ▼]           │  │ [2 ▼]           │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 💡 Preset Configurations                                    │   │
│  │  [Quick Test]  [Standard]  [Deep Training]  [Custom]        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## Error Handling & Common Problems

### Error Classification System

```python
ERROR_CATEGORIES = {
    "CUDA_OOM": {
        "patterns": [
            "CUDA out of memory",
            "OutOfMemoryError",
            "tried to allocate"
        ],
        "severity": "recoverable",
        "user_message": "GPU memory exceeded. Automatically reducing batch size.",
        "auto_fix": {
            "action": "reduce_batch_size",
            "retry": True
        },
        "suggestions": [
            "Reduce batch size to 1",
            "Enable gradient checkpointing",
            "Use smaller LoRA rank (8 instead of 16)",
            "Train on a smaller model variant"
        ]
    },
    
    "MODEL_LOAD_FAIL": {
        "patterns": [
            "Failed to load model",
            "model not found",
            "404"
        ],
        "severity": "fatal",
        "user_message": "Could not load the base model. Please check model availability.",
        "auto_fix": None,
        "suggestions": [
            "Verify model name is correct",
            "Check HuggingFace Hub status",
            "Try a different base model"
        ]
    },
    
    "DATA_FORMAT_ERROR": {
        "patterns": [
            "ValueError",
            "KeyError",
            "invalid format"
        ],
        "severity": "user_action_required",
        "user_message": "Training data format issue detected.",
        "auto_fix": None,
        "suggestions": [
            "Check that all examples have 'user' and 'assistant' fields",
            "Ensure no empty conversations",
            "Validate JSON/JSONL format"
        ]
    },
    
    "NETWORK_ERROR": {
        "patterns": [
            "ConnectionError",
            "TimeoutError",
            "network"
        ],
        "severity": "recoverable",
        "user_message": "Network connection issue. Retrying...",
        "auto_fix": {
            "action": "retry",
            "max_retries": 3,
            "backoff": "exponential"
        }
    },
    
    "GPU_UNAVAILABLE": {
        "patterns": [
            "No CUDA GPUs are available",
            "CUDA not available"
        ],
        "severity": "infrastructure",
        "user_message": "GPU currently unavailable. Please try again in a few minutes.",
        "auto_fix": {
            "action": "queue_and_retry",
            "delay_minutes": 5
        }
    }
}
```

### Auto-Recovery Logic

```python
class TrainingErrorHandler:
    """Smart error recovery for training jobs."""
    
    async def handle_error(self, error: Exception, job: TrainingJob) -> RecoveryAction:
        error_category = self.classify_error(error)
        
        if error_category == "CUDA_OOM":
            # Progressive batch size reduction
            current_batch = job.config["batch_size"]
            if current_batch > 1:
                job.config["batch_size"] = current_batch // 2
                return RecoveryAction(
                    action="retry",
                    message=f"Reduced batch size to {job.config['batch_size']}"
                )
            elif not job.config.get("gradient_checkpointing"):
                job.config["gradient_checkpointing"] = True
                return RecoveryAction(
                    action="retry",
                    message="Enabled gradient checkpointing"
                )
            else:
                return RecoveryAction(
                    action="fail",
                    message="Cannot reduce memory usage further",
                    suggestions=[
                        "Use a smaller model (3B instead of 8B)",
                        "Reduce LoRA rank",
                        "Reduce max sequence length"
                    ]
                )
        
        elif error_category == "NETWORK_ERROR":
            if job.retry_count < 3:
                delay = 2 ** job.retry_count  # Exponential backoff
                return RecoveryAction(
                    action="retry",
                    delay_seconds=delay,
                    message=f"Retrying in {delay}s..."
                )
        
        return RecoveryAction(action="fail", message=str(error))
```

### User-Friendly Error Display

```
┌────────────────────────────────────────────────────────────────────┐
│ ⚠️ Training Issue Detected                                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  What happened:                                                     │
│  GPU memory was exceeded during training.                           │
│                                                                     │
│  What we tried:                                                     │
│  ✓ Reduced batch size from 2 → 1                                   │
│  ✓ Enabled gradient checkpointing                                  │
│  ✗ Still not enough memory                                         │
│                                                                     │
│  Recommended actions:                                               │
│  • Use a smaller base model (Llama-3.2-3B)                         │
│  • Reduce LoRA rank from 16 to 8                                   │
│  • Train with fewer examples per batch                              │
│                                                                     │
│  [🔄 Retry with Smaller Model]  [⚙️ Adjust Settings]  [❌ Cancel]  │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## Logging & Monitoring

### What to Log

| Category | Data Points | Purpose |
|----------|-------------|---------|
| **Training Metrics** | loss, grad_norm, learning_rate, epoch, step | Performance tracking |
| **Resource Usage** | GPU memory, GPU utilization, CPU usage | Cost optimization |
| **Timing** | step duration, total time, ETA | User feedback |
| **Data Quality** | num_examples, avg_tokens, data_hash | Reproducibility |
| **Configuration** | full config snapshot | Reproducibility |
| **Errors** | error type, stack trace, recovery attempts | Debugging |
| **Model Info** | base_model, adapter_size, trainable_params | Documentation |

### Logging Format (Structured JSON)

```json
{
  "timestamp": "2024-12-21T12:34:56Z",
  "job_id": "train_abc123",
  "skill_id": "molasses-master-expert",
  "event": "training_step",
  "data": {
    "step": 4,
    "total_steps": 10,
    "epoch": 2,
    "loss": 2.123,
    "learning_rate": 0.0002,
    "grad_norm": 1.54,
    "gpu_memory_used_gb": 18.2,
    "gpu_memory_total_gb": 22.0,
    "step_time_seconds": 5.2,
    "samples_per_second": 1.54
  }
}
```

### Metrics Dashboard Integration

```python
# Training progress callback
class ProgressCallback(TrainerCallback):
    def __init__(self, websocket_manager, job_id):
        self.ws = websocket_manager
        self.job_id = job_id
        self.metrics_history = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        metrics = {
            "step": state.global_step,
            "epoch": state.epoch,
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "grad_norm": logs.get("grad_norm"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.metrics_history.append(metrics)
        
        # Stream to frontend
        await self.ws.broadcast(self.job_id, {
            "type": "training_progress",
            "data": metrics,
            "history": self.metrics_history
        })
        
        # Save to database
        await self.save_metrics(metrics)
```

### Cost Tracking

```python
COST_ESTIMATES = {
    "modal_a10g_per_hour": 1.10,  # $1.10/hr
    "modal_t4_per_hour": 0.40,    # $0.40/hr
    
    "typical_training_times": {
        "10_examples": 5,    # minutes
        "50_examples": 12,   # minutes
        "100_examples": 20,  # minutes
        "500_examples": 60,  # minutes
    }
}

def estimate_training_cost(num_examples: int, gpu_type: str = "a10g") -> dict:
    """Estimate training cost before starting."""
    base_time = COST_ESTIMATES["typical_training_times"].get(
        f"{num_examples}_examples",
        num_examples * 0.2  # Fallback: 0.2 min per example
    )
    
    hourly_rate = COST_ESTIMATES[f"modal_{gpu_type}_per_hour"]
    estimated_cost = (base_time / 60) * hourly_rate
    
    return {
        "estimated_minutes": base_time,
        "estimated_cost_usd": round(estimated_cost, 2),
        "gpu_type": gpu_type,
        "confidence": "high" if num_examples <= 100 else "medium"
    }
```

---

## Implementation Plan

### Phase 1: Core Training Pipeline (Week 1)

**Goal:** Get basic training working end-to-end

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Update Modal trainer with error handling | P0 | 4h | None |
| Add training jobs DB table | P0 | 2h | None |
| Create `/api/train-skill` endpoint | P0 | 3h | Modal trainer |
| Create `/api/training-status` endpoint | P0 | 2h | DB table |
| Basic progress polling in UI | P0 | 3h | API endpoints |
| Test adapter endpoint | P1 | 2h | Modal trainer |

### Phase 2: Real-Time Progress (Week 2)

**Goal:** Live streaming of training progress

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| WebSocket server setup | P0 | 4h | None |
| Training callback integration | P0 | 3h | WebSocket |
| Live loss chart component | P0 | 4h | WebSocket |
| Progress bar with ETA | P1 | 2h | Callbacks |
| Training log stream | P1 | 2h | WebSocket |

### Phase 3: Error Handling & Recovery (Week 3)

**Goal:** Robust error handling and auto-recovery

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Error classification system | P0 | 4h | None |
| Auto-recovery logic (OOM, network) | P0 | 4h | Classification |
| User-friendly error messages | P0 | 3h | Classification |
| Retry UI with options | P1 | 3h | Error handling |
| Error analytics logging | P2 | 2h | None |

### Phase 4: Adapter Management (Week 4)

**Goal:** Full adapter lifecycle management

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Adapters gallery UI | P0 | 4h | None |
| Adapter versioning | P0 | 3h | DB schema |
| Adapter activation/deactivation | P1 | 2h | DB schema |
| Export to HuggingFace | P2 | 4h | None |
| Adapter comparison view | P2 | 3h | Versioning |

### Phase 5: Polish & Advanced Features (Week 5)

**Goal:** Production-ready with advanced features

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Training presets (Quick/Standard/Deep) | P0 | 2h | None |
| Cost calculator | P0 | 2h | None |
| Data quality validator | P1 | 4h | None |
| Training history charts | P1 | 3h | DB data |
| Documentation & help tooltips | P1 | 3h | All features |

---

## API Specification

### Training Endpoints

```
POST /api/train-skill/{skill_id}
Request:
{
  "config": {
    "preset": "standard",  // or "quick", "deep", "custom"
    "epochs": 3,
    "learning_rate": 2e-4,
    "lora_r": 16,
    "batch_size": 2
  }
}
Response:
{
  "job_id": "job_abc123",
  "status": "queued",
  "estimated_cost": 0.65,
  "estimated_time_minutes": 8
}

GET /api/training-status/{job_id}
Response:
{
  "job_id": "job_abc123",
  "status": "running",
  "progress": 42,
  "current_step": 4,
  "total_steps": 10,
  "current_loss": 2.123,
  "eta_seconds": 383,
  "metrics_history": [...]
}

WS /api/training-stream/{job_id}
Messages:
{
  "type": "progress",
  "step": 4,
  "loss": 2.123,
  "eta_seconds": 383
}
{
  "type": "log",
  "level": "info",
  "message": "Step 4/10 complete"
}
{
  "type": "complete",
  "final_loss": 1.847,
  "adapter_path": "/adapters/molasses-master-expert"
}

POST /api/test-adapter/{skill_id}
Request:
{
  "prompt": "What is blackstrap molasses?"
}
Response:
{
  "response": "Ah, blackstrap molasses - the third boil treasure!...",
  "generation_time_ms": 1234
}

GET /api/adapters
Response:
{
  "adapters": [
    {
      "id": "adapter_123",
      "skill_id": "molasses-master-expert",
      "version": 3,
      "is_active": true,
      "final_loss": 1.847,
      "training_examples": 12,
      "created_at": "2024-12-21T12:34:56Z"
    }
  ]
}

DELETE /api/adapters/{adapter_id}
Response:
{
  "success": true,
  "message": "Adapter deleted"
}
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training Success Rate | >95% | Jobs completed / Jobs started |
| Avg Training Time | <15 min | For 50 examples |
| Avg Cost per Train | <$1.50 | Modal GPU costs |
| User Satisfaction | >4.5/5 | Post-training survey |
| Error Recovery Rate | >80% | Auto-recovered / Total errors |
| Time to First Train | <5 min | New user onboarding |

---

## Summary

This design document outlines a comprehensive, user-friendly training system that addresses the key pain points identified in market research:

1. **One-Click Training** - Smart defaults, no configuration required
2. **Live Progress** - Real-time loss curves and ETA
3. **Smart Error Recovery** - Auto-fix common issues like OOM
4. **Cost Transparency** - Show estimates before training starts
5. **Instant Testing** - Chat with trained model immediately
6. **Full Lifecycle Management** - Version, compare, and export adapters

The implementation leverages our existing Unsloth + Modal infrastructure while adding the polish and UX that competitors lack.
