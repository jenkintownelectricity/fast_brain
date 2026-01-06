# Fast Brain Golden Package - MANIFEST

## Package Overview

Core logic files for the HIVE215 Voice AI platform - a hybrid "System 1 + System 2" architecture inspired by Daniel Kahneman's *Thinking, Fast and Slow*.

**Total Files:** 17 Python files + 5 Markdown files
**Total Lines:** 26,839 lines of code
**Package Size:** 1.1 MB uncompressed, 244 KB compressed

---

## Architecture Overview

### System 1 vs System 2 Routing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SYSTEM 1 + SYSTEM 2 HYBRID                               │
├─────────────────────────────────────────────────────────────────────────────┤
│   User: "Can you analyze my bill and predict next month's cost?"            │
│            │                                                                 │
│            ▼                                                                 │
│   ┌─────────────────┐                                                        │
│   │  🚀 SYSTEM 1    │  Groq + Llama 3.3 70B                                  │
│   │  (Fast Brain)   │  ~80ms TTFB, handles 90% of calls                     │
│   └────────┬────────┘                                                        │
│            │                                                                 │
│     ┌──────┴──────┐                                                          │
│     │             │                                                          │
│     ▼             ▼                                                          │
│  [SIMPLE]      [COMPLEX]                                                     │
│     │             │                                                          │
│     ▼             ▼                                                          │
│  Answer       ┌─────────────────┐                                            │
│  Directly     │  FILLER PHRASE  │  "Let me pull that up for you..."         │
│  (~80ms)      └────────┬────────┘  (Plays while Claude thinks)              │
│                        │                                                     │
│                        ▼                                                     │
│               ┌─────────────────┐                                            │
│               │  🧠 SYSTEM 2    │  Claude 3.5 Sonnet                         │
│               │  (Deep Brain)   │  ~2s, complex reasoning                    │
│               └────────┬────────┘                                            │
│                        │                                                     │
│                        ▼                                                     │
│               [Real Answer]  ← Arrives just as filler finishes!             │
│                                                                              │
│   User Perception: ZERO LATENCY. Agent feels human.                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Golden Prompts Optimization

- **Token Budget:** <3,000 tokens per skill prompt
- **Prefill Latency:** ~14-50ms on Groq (vs ~200ms for 10k+ token prompts)
- **Voice-Native:** Contractions, short sentences, no markdown
- **Filler Triggers:** Embedded phrases that activate System 2 handoff

### Filler Phrase Strategy

When System 1 detects a complex query, it:
1. Immediately returns a natural filler phrase (~80ms)
2. Triggers System 2 (Claude) in parallel
3. TTS plays the filler while Claude thinks
4. Claude's answer arrives as filler ends

**Categories:**
- `analysis`: "Let me pull up your information..."
- `calculation`: "Let me run those numbers..."
- `research`: "Let me check on that..."
- `complex`: "Let me think through that carefully..."

### LoRA Training Pipeline

```
Training Data Collection → DPO Pair Generation → Unsloth/QLoRA Training → Modal Volume Export
         │                        │                        │                       │
   Supabase logs          Chosen vs Rejected        A10G GPU (~$0.50)         /adapters/<skill_id>/
```

---

## File Manifest

### Core Deployment

| File | Lines | Description |
|------|-------|-------------|
| `fast_brain/deploy_groq.py` | 1,519 | Main Fast Brain API - System 1 + System 2 hybrid inference engine |
| `deploy_dashboard.py` | 124 | Modal deployment wrapper for Flask dashboard |
| `unified_dashboard.py` | 17,626 | Full management UI - Flask + embedded HTML/JS/CSS |
| `parler_integration.py` | 637 | Parler TTS Modal deployment - emotion-aware voice synthesis |

#### fast_brain/deploy_groq.py

**Description:** The core inference API implementing the hybrid System 1 (Groq/Llama) + System 2 (Claude) architecture.

**Key Exports:**
- `FastBrainAPI` - Modal web endpoint class
- `get_filler_phrase(category)` - Returns latency-masking phrases
- `TOOLS` - Tool definitions for Groq to call Claude
- `FILLER_PHRASES` - Dict of filler phrases by category

**Endpoints:**
- `GET /health` - Health check
- `GET /v1/skills` - List all available skills
- `POST /v1/chat/completions` - Standard chat (text only)
- `POST /v1/chat/voice` - Voice-aware chat (Brain + Mouth output)
- `POST /v1/chat/hybrid` - Hybrid System 1 + System 2 routing

**Dependencies:** `modal`, `anthropic`, `groq`, `skills.py`, `config.py`

**Configuration:**
```python
FAST_MODEL = "llama-3.3-70b-versatile"  # System 1
DEEP_MODEL = "claude-sonnet-4-20250514"      # System 2
FAST_MAX_TOKENS = 1024
CONTAINER_IDLE_TIMEOUT = 300
KEEP_WARM = 1
```

#### unified_dashboard.py

**Description:** Complete web dashboard for skill management, training, and monitoring. Single-file Flask app with embedded HTML/JS.

**Key Exports:**
- `app` - Flask application
- `@app.route(...)` - All API endpoints

**Endpoints (partial list):**
- `GET /` - Dashboard home
- `GET /api/skills` - List skills
- `POST /api/skills` - Create skill
- `POST /api/training/start` - Start LoRA training
- `GET /api/training/status/<job_id>` - Training progress
- `GET /api/trained-adapters` - List trained adapters

**Dependencies:** `flask`, `modal`, `database.py`

---

### Brain Logic

| File | Lines | Description |
|------|-------|-------------|
| `fast_brain/skills.py` | 347 | Built-in skill definitions with System 1/2 routing hints |
| `fast_brain/config.py` | 221 | Configuration dataclasses for hybrid system |
| `fast_brain/client.py` | 424 | Python client library for Fast Brain API |
| `fast_brain/model.py` | 394 | BitNet model wrapper with streaming generation |
| `golden_prompts.py` | 398 | Voice-optimized skill prompts (<3k tokens) |

#### fast_brain/skills.py

**Description:** Skill definitions for different business verticals (electrician, plumber, lawyer, etc.)

**Key Exports:**
- `BUILT_IN_SKILLS` - Dict of all built-in skills
- `VOICE_RULES` - Shared voice interaction rules
- `VOICE_CONTEXTS` - TTS voice descriptions
- `get_skill(skill_id)` - Get skill by ID
- `list_skills()` - List all skill IDs
- `get_skill_info(skill_id)` - Get skill metadata
- `create_custom_skill(...)` - Create new skill

**Built-in Skills:**
- `general`, `receptionist`, `electrician`, `plumber`
- `lawyer`, `solar`, `hvac`, `medical`, `restaurant`

**Dependencies:** None (standalone)

#### fast_brain/config.py

**Description:** Dataclass configurations for System 1 and System 2.

**Key Exports:**
- `FastBrainConfig` - System 1 (Groq) settings
- `DeepBrainConfig` - System 2 (Claude) settings
- `HybridConfig` - Combined configuration
- Constants: `FAST_MODEL`, `DEEP_MODEL`, `MODAL_REGION`

**Dependencies:** None (standalone)

#### fast_brain/client.py

**Description:** HTTP client for connecting to Fast Brain from voice agents.

**Key Exports:**
- `FastBrainClient` - Async HTTP client
- `StreamingResponse` - Response chunk dataclass
- `stream_chat()` - Streaming chat generator
- `chat()` - Non-streaming chat

**Dependencies:** `httpx`, `config.py`

#### golden_prompts.py

**Description:** Claude-optimized prompts under 3k tokens for fast Groq prefill.

**Key Exports:**
- `SKILL_MANUALS` - Dict of all skill prompts
- `RECEPTIONIST_MANUAL` - Professional call handling
- `ELECTRICIAN_MANUAL` - Jenkintown Electricity
- `PLUMBER_MANUAL` - Plumbing service
- `SOLAR_MANUAL` - Solar sales
- `MOLASSES_MANUAL` - Molasses master expert

**Design Principles:**
- Stay under 3k tokens (~50-100ms prefill on Groq)
- Voice-native (contractions, short sentences)
- Include "WHEN STUMPED" triggers for System 2

**Dependencies:** None (standalone)

---

### Training System

| File | Lines | Description |
|------|-------|-------------|
| `train_skill_modal.py` | 649 | LoRA training with Unsloth + QLoRA on Modal |
| `training_collector.py` | 688 | Supabase metrics + training data export |
| `continuous_learner.py` | 427 | DPO training from production feedback |
| `database.py` | 1,601 | SQLite database with full CRUD operations |

#### train_skill_modal.py

**Description:** One-click LoRA fine-tuning using Unsloth for 2-5x faster training.

**Key Exports:**
- `SkillTrainer` - Modal class for training
- `train(skill_id, config)` - Main training function
- `list_adapters()` - List trained adapters from volume
- `DEFAULT_CONFIG` - Default training hyperparameters

**Training Config:**
```python
DEFAULT_CONFIG = {
    "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "lora_r": 16,
    "lora_alpha": 16,
    "epochs": 3,
    "batch_size": 2,
    "learning_rate": 2e-4,
}
```

**Dependencies:** `modal`, `unsloth`, `transformers`, `peft`, `trl`, `database.py`

#### training_collector.py

**Description:** Captures System 2 responses for fine-tuning System 1.

**Key Exports:**
- `MetricsCollector` - Main collector class
- `log_interaction()` - Log voice interaction
- `export_training_data()` - Export LoRA-compatible pairs
- `SCHEMA_SQL` - Supabase table definitions

**Database Schema:**
```sql
CREATE TABLE voice_interactions (
    session_id TEXT,
    skill_id TEXT,
    user_input TEXT,
    agent_response TEXT,
    system_used TEXT,  -- 'fast' or 'deep'
    fast_latency_ms INTEGER,
    deep_latency_ms INTEGER,
    user_feedback INTEGER,
    is_training_candidate BOOLEAN
);
```

**Dependencies:** `modal`, `supabase`

#### continuous_learner.py

**Description:** Automated DPO training from production corrections.

**Key Exports:**
- `FeedbackCollector` - Collect user feedback
- `DPODataGenerator` - Generate chosen/rejected pairs
- `ContinuousTrainer` - Orchestrate improvement cycle
- `run_improvement_cycle()` - Main entry point

**Pipeline:**
1. Collect feedback from logs
2. Generate DPO pairs (chosen vs rejected)
3. Train improved LoRA
4. Hot-swap into production

**Dependencies:** `modal`, `unsloth`, `trl`, `datasets`

#### database.py

**Description:** SQLite database for skills, prompts, training data, and configurations.

**Key Exports:**
- `init_db()` - Initialize schema
- `get_db()` - Context manager for connections
- CRUD functions for each table:
  - `create_skill()`, `get_skill()`, `update_skill()`, `delete_skill()`
  - `get_golden_prompt()`, `save_golden_prompt()`
  - `create_training_example()`, `get_training_examples()`
  - `get_api_key()`, `set_api_key()`

**Tables:**
- `skills` - Skill definitions with metrics
- `golden_prompts` - Custom prompt overrides
- `configurations` - Key-value settings
- `training_examples` - LoRA training data
- `api_keys` - Encrypted API key storage
- `call_logs` - Voice call history

**Dependencies:** `sqlite3` (stdlib)

---

### Voice Transfer Files

| File | Lines | Description |
|------|-------|-------------|
| `_TRANSFER_TO_HIVE215/skill_command_center.py` | 430 | LatencyMasker + SmartRouter for hybrid routing |
| `_TRANSFER_TO_HIVE215/turn_taking.py` | 575 | Context-aware turn detection + backchanneling |

#### skill_command_center.py

**Description:** The brain of the voice platform - routes queries and masks latency.

**Key Exports:**
- `LatencyMasker` - Generates filler sounds during LLM waits
- `SmartRouter` - Routes to fast vs smart model
- `SkillCommandCenter` - Main orchestrator
- `mask_latency()` - Async generator wrapper

**LatencyMasker Features:**
- `FILLER_SOUNDS`: "Hmm...", "Umm...", "Well..."
- `THINKING_PHRASES`: "Let me think...", "One moment..."
- `SKILL_FILLERS`: Domain-specific phrases
- `max_wait_before_filler`: 300ms default
- `filler_interval`: 2s between fillers

**Dependencies:** `asyncio`, `dataclasses`

#### turn_taking.py

**Description:** State-of-the-art conversational turn-taking.

**Key Exports:**
- `TurnState` - Enum (IDLE, USER_SPEAKING, USER_DONE, AGENT_SPEAKING, etc.)
- `InterruptionType` - Enum (BACKCHANNEL, CORRECTION, BARGE_IN)
- `TurnManager` - Main turn-taking state machine
- `TurnConfig` - Configuration dataclass
- `TurnContext` - Context for decisions

**Turn Detection Signals:**
1. Voice Activity Detection (VAD)
2. Semantic endpoint prediction
3. Prosodic analysis (pitch/energy)
4. Barge-in handling
5. Backchanneling ("uh-huh", "got it")

**Dependencies:** `dataclasses`, `collections`

---

### Worker

| File | Lines | Description |
|------|-------|-------------|
| `worker/voice_agent.py` | 567 | LiveKit voice agent with turn detection |
| `worker/test_fast_brain.py` | 212 | Test suite for Fast Brain client |

#### worker/voice_agent.py

**Description:** LiveKit voice agent worker with hybrid Fast Brain integration.

**Key Exports:**
- `prewarm()` - Container prewarming
- `entrypoint()` - Main agent entry
- `FastBrainLLM` - LLM adapter for LiveKit
- `HybridResponse` - Response with routing info

**Architecture:**
```
User → Deepgram STT → Turn Detector → Fast Brain (Groq)
                                            │
                                     ┌──────┴──────┐
                                     │             │
                                  [SIMPLE]     [COMPLEX]
                                     │             │
                                     ▼             ▼
                                  Answer      Filler + Claude
                                  (~80ms)        (~2s hidden)
```

**Environment Variables:**
- `FAST_BRAIN_URL` - Modal endpoint
- `LIVEKIT_URL` - LiveKit server
- `DEEPGRAM_API_KEY` - STT
- `CARTESIA_API_KEY` - TTS

**Dependencies:** `livekit`, `livekit.agents`, `livekit.plugins`, `httpx`

---

### Documentation

| File | Lines | Description |
|------|-------|-------------|
| `CLAUDE.md` | 258 | Claude Code instructions for this project |
| `README.md` | 656 | Project documentation and quickstart |
| `CHANGELOG.md` | 124 | Version history |
| `INDEX.md` | 89 | Package overview and structure |
| `docs/HIVE215_INTEGRATION_GUIDE.md` | 438 | Integration guide for voice platforms |

---

## Skills Database Schema

```sql
CREATE TABLE skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    skill_type TEXT DEFAULT 'custom',  -- 'builtin', 'custom', 'cloned'
    system_prompt TEXT,
    knowledge TEXT,      -- JSON array of knowledge items
    voice_config TEXT,   -- JSON: {voice_id, speed, pitch}
    is_builtin INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 1,
    created_at TEXT,
    updated_at TEXT,
    -- Metrics
    total_requests INTEGER DEFAULT 0,
    avg_latency_ms REAL DEFAULT 0,
    satisfaction_rate REAL DEFAULT 0
);

CREATE TABLE golden_prompts (
    skill_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    tokens_estimate INTEGER,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE training_examples (
    id INTEGER PRIMARY KEY,
    skill_id TEXT NOT NULL,
    instruction TEXT NOT NULL,
    input TEXT,
    output TEXT NOT NULL,
    source TEXT,          -- 'manual', 'captured', 'generated'
    quality_score REAL,
    created_at TEXT
);
```

---

## Quick Reference

### Deploy Commands
```bash
# Dashboard
modal deploy deploy_dashboard.py

# Fast Brain API
modal deploy fast_brain/deploy_groq.py

# TTS
modal deploy parler_integration.py

# Training
modal run train_skill_modal.py --skill-id <skill_id>
```

### Environment Variables
```bash
# Required
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...

# Optional
SUPABASE_URL=https://...
SUPABASE_KEY=eyJ...
LIVEKIT_URL=wss://...
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
```

### Cost Estimates
- **Groq API:** Free tier (100k tokens/day), then ~$0.05/1M tokens
- **Claude API:** ~$3/1M input, ~$15/1M output tokens
- **Modal Training:** ~$0.50-2.00 per run (A10G GPU, 10-30 min)
- **Modal Inference:** ~$0.005/request (warm container)

---

*Generated: 2026-01-06*
