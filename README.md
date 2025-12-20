# Fast Brain LPU

**Hybrid Voice AI Engine for HIVE215 - "Thinking, Fast and Slow"**

Fast Brain is a dual-system inference engine inspired by Daniel Kahneman's cognitive model. It combines ultra-fast Groq inference (~80ms) for routine questions with Claude's deep reasoning (~2s) for complex analysis - all while maintaining a natural conversational experience through intelligent filler phrases.

---

## Architecture Overview

```
                     SYSTEM 1 + SYSTEM 2 HYBRID

   User: "Can you analyze my bill and predict next month's cost?"
            |
            v
   +------------------+
   |  SYSTEM 1        |  Groq + Llama 3.3 70B
   |  (Fast Brain)    |  ~80ms, handles 90% of calls
   |                  |
   |  Decision:       |  "This needs complex reasoning..."
   +--------+---------+
            |
     +------+------+
     |             |
     v             v
  [SIMPLE]      [COMPLEX]
     |             |
     v             v
  Answer       +------------------+
  Directly     |  Output A:       |  "Let me pull that up for you..."
  (~80ms)      |  FILLER PHRASE   |  (Plays while Claude thinks)
               +--------+---------+
                        |
                        v
               +------------------+
               |  SYSTEM 2        |  Claude Sonnet 4.5
               |  (Deep Brain)    |  ~2s, complex reasoning
               +--------+---------+
                        |
                        v
               [Real Answer]  <-- Arrives just as filler finishes!

   User Perception: ZERO LATENCY. Agent feels human.
```

---

## Quick Start (Modal Deployment)

### Prerequisites
- Modal account (free tier works)
- Python 3.11 (Modal doesn't support 3.14 yet)
- Groq API key (free at console.groq.com)
- Anthropic API key

### 🚀 DEPLOY ALL 3 SERVICES (Copy & Paste)

**Always deploy all 3 together:**

```bash
# Windows (PowerShell)
modal deploy deploy_dashboard.py; modal deploy fast_brain/deploy_groq.py; modal deploy parler_integration.py

# Mac/Linux
modal deploy deploy_dashboard.py && modal deploy fast_brain/deploy_groq.py && modal deploy parler_integration.py
```

**Or run each separately:**
```bash
modal deploy deploy_dashboard.py        # Dashboard UI
modal deploy fast_brain/deploy_groq.py  # Fast Brain API
modal deploy parler_integration.py      # Parler TTS (GPU)
```

### Your 3 Deployed URLs
```
Dashboard:       https://[username]--hive215-dashboard-flask-app.modal.run
Fast Brain API:  https://[username]--fast-brain-lpu-fastapi-app.modal.run
Parler TTS:      https://[username]--hive215-parler-tts-parlerttsmodel-*.modal.run
```

### First-Time Setup

#### Windows Users (Python 3.11 + venv - Recommended)
```powershell
# Create virtual environment with Python 3.11
py -3.11 -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install modal fastapi pydantic

# Authenticate with Modal (opens browser)
modal token new

# Set up secrets in Modal dashboard or CLI
modal secret create groq-api-key GROQ_API_KEY=gsk_your_key
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-your_key

# 🚀 DEPLOY ALL 3 SERVICES
modal deploy deploy_dashboard.py; modal deploy fast_brain/deploy_groq.py; modal deploy parler_integration.py
```

**VS Code Users:** Press `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose `.\venv\Scripts\python.exe`

#### Mac/Linux Users
```bash
pip install modal
modal token new
modal secret create groq-api-key GROQ_API_KEY=gsk_your_key
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-your_key

# 🚀 DEPLOY ALL 3 SERVICES
modal deploy deploy_dashboard.py && modal deploy fast_brain/deploy_groq.py && modal deploy parler_integration.py
```

---

## What's Implemented

### Core Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Hybrid Architecture** | Live | System 1 (Groq) + System 2 (Claude) |
| **Filler Phrase Strategy** | Live | Hide Claude's latency with natural phrases |
| **Golden Prompts** | Live | Voice-optimized skill manuals (<3k tokens) |
| **Parler TTS** | Live | Expressive voice synthesis with emotions |
| **Unified Dashboard** | Live | Full management UI on Modal |
| **Skills System** | Live | Domain-specific prompts and behaviors |
| **Training Collector** | Live | Supabase metrics + LoRA data export |

### Performance Metrics

| System | Latency | Use Case | % of Calls |
|--------|---------|----------|------------|
| **System 1 (Groq)** | ~80ms | Simple questions, greetings, intake | ~90% |
| **System 2 (Claude)** | ~2000ms | Analysis, calculations, complex advice | ~10% |
| **Golden Prompts** | ~14ms prefill | Voice-optimized system prompts | Always |
| **Parler TTS** | ~2-5s | Expressive voice with emotion | Per request |

---

## Dashboard Features

Access at: `https://[username]--hive215-dashboard-flask-app.modal.run`

### Four Pillars Navigation

The dashboard is organized into 4 main sections for streamlined workflow:

| Pillar | Contents |
|--------|----------|
| **Dashboard** | Overview, status cards, Getting Started onboarding |
| **Skills** | Skills Manager, Golden Prompts, Train LoRA, Test Chat, Outgoing API |
| **Voice** | Voice Projects, Create Voice, Training Queue, Skill Training |
| **Settings** | API Keys (7 providers), Platform Connections, Stats |

### Dark/Light Mode

Toggle between Slack-inspired themes:
- **Dark Mode**: Default cyberpunk theme with neon accents
- **Light Mode**: Clean, professional Slack-inspired light theme
- Theme persists via localStorage (survives browser restarts)
- Toggle button (sun/moon icon) in top-right corner

### Dashboard Tab
- Real-time metrics and activity feed
- Getting Started onboarding for new users
- Status cards for skills, voices, and API connections

### Skills Tab
- **Skills Manager**: Create, edit, and manage AI agent skills
- **Golden Prompts**: View/edit voice-optimized system prompts
- **Train LoRA**: Generate training data for fine-tuning
- **Test Chat**: Test skills with real LLM inference
- **Outgoing API**: Configure external API integrations

### Voice Tab (Voice Lab)

Full voice cloning and synthesis:

| Feature | Description |
|---------|-------------|
| **Create Project** | New voice projects with provider selection |
| **Upload Samples** | Add audio samples for voice training |
| **Train Voice** | Clone voices with ElevenLabs, Cartesia, or free gTTS |
| **Test Synthesis** | Generate audio and playback in browser |
| **Link to Skill** | Connect trained voices to specific agents |

**Supported Providers:**
- **ElevenLabs**: Professional voice cloning (requires API key)
- **Cartesia**: High-quality synthesis (requires API key)
- **gTTS**: Free Google Text-to-Speech (no API key needed)
- **Parler TTS**: Expressive GPU-based synthesis

### Settings Tab

**API Keys (7 providers):**
| Category | Providers |
|----------|-----------|
| LLM | Groq, OpenAI, Anthropic |
| Voice | ElevenLabs, Cartesia, Deepgram, PlayHT |

**API Endpoints:**
Display and copy your deployed API URLs:
- **Fast Brain API**: `https://jenkintownelectricity--fast-brain-lpu-fastapi-app.modal.run`
- **Dashboard URL**: `https://jenkintownelectricity--hive215-dashboard-flask-app.modal.run`
- One-click copy buttons for each URL
- Health check with status indicators (green=online, orange=warning, red=offline)
- Quick reference for common API endpoints

**Platform Connections:**
- LiveKit, Twilio, Vapi, Retell AI, Daily.co

### Test Chat

**Voice Test (Optional):**
Test any voice with custom text:
- **Voice Provider dropdown**: ElevenLabs, Cartesia, Deepgram, OpenAI, Edge TTS, Parler
- **Voice ID dropdown**: Dynamically loads voices from selected provider
- **Text input**: Enter any text to hear it synthesized
- **Audio playback**: Instant playback with HTML5 audio player
- Similar styling to Golden Prompts optional section

### Outgoing API Connections

Connect your agents to external services:

- **Add connections**: Name, URL, authentication type
- **Auth types**: Bearer token, X-API-Key, Basic auth, None
- **Custom headers**: Add any HTTP headers
- **Webhook URL**: Receive callbacks from external services
- **Test connection**: Verify with real HTTP requests
- **Request tester**: Send GET/POST/PUT/DELETE with custom paths and bodies
- **Live status**: See connection health at a glance

---

## Golden Prompts

Voice-native skill manuals optimized for ultra-fast Groq inference:

| Skill | Tokens | Prefill | Use Case |
|-------|--------|---------|----------|
| `receptionist` | ~850 | ~14ms | Professional call handling |
| `electrician` | ~900 | ~15ms | Jenkintown Electricity |
| `plumber` | ~850 | ~14ms | Plumbing service intake |
| `lawyer` | ~950 | ~16ms | Legal intake specialist |
| `solar` | ~900 | ~15ms | Solar sales qualification |
| `tara-sales` | ~1200 | ~20ms | TheDashTool sales assistant |
| `general` | ~750 | ~12ms | General assistant |

**Usage:**
```python
from golden_prompts import get_skill_prompt

prompt = get_skill_prompt(
    "electrician",
    business_name="Jenkintown Electricity",
    agent_name="Sarah"
)
```

---

## Parler TTS Integration

Expressive voice synthesis with emotion control:

```python
# Via Modal function
import modal

ParlerTTS = modal.Cls.lookup("hive215-parler-tts", "ParlerTTSModel")
model = ParlerTTS()

audio_bytes, description = model.synthesize_with_emotion.remote(
    text="Hello! How can I help you today?",
    skill_id="receptionist",
    emotion="warm"
)
```

### Available Emotions
- `neutral`, `warm`, `excited`, `calm`
- `concerned`, `apologetic`, `confident`
- `empathetic`, `urgent`, `cheerful`

### Voice Personas (per skill)
- **Receptionist**: Sarah - professional warm female
- **Electrician**: Mike - friendly knowledgeable male
- **Plumber**: Tom - reassuring calm male
- **Lawyer**: Jennifer - professional articulate female
- **Solar**: Alex - enthusiastic energetic neutral
- **General**: Jordan - warm natural neutral

---

## Database Schema

The dashboard uses SQLite for persistent storage on a Modal volume (`/data/hive215.db`):

| Table | Purpose |
|-------|---------|
| `skills` | Custom skills/agents with prompts and voice configs |
| `golden_prompts` | Custom prompt overrides for built-in skills |
| `api_keys` | Encrypted API key storage (7 providers) |
| `platform_connections` | Voice platform configurations |
| `activity_log` | System activity tracking and audit log |
| `voice_projects` | Voice cloning projects and settings |
| `voice_samples` | Audio samples for voice training |
| `api_connections` | Outgoing API integrations |
| `training_data` | Collected examples for LoRA fine-tuning |
| `configurations` | Key-value system settings |

Data persists across Modal container restarts via the `hive215-data` volume.

---

## File Structure

```
fast_brain/
├── deploy_groq.py          # Main Fast Brain API (v3.0 Hybrid)
├── golden_prompts.py       # Voice-optimized skill manuals
├── parler_integration.py   # Parler TTS Modal deployment
├── training_collector.py   # Supabase metrics + LoRA export
├── skills.py               # Skills definitions
├── config.py               # Configuration dataclasses
├── client.py               # Python client library
└── __init__.py             # Package exports

deploy_dashboard.py         # Dashboard Modal deployment
unified_dashboard.py        # Full management UI (Flask + HTML/JS)
database.py                 # SQLite database with CRUD operations
golden_prompts.py           # Voice-optimized skill prompts

worker/
├── voice_agent.py          # LiveKit voice agent
├── requirements.txt        # Worker dependencies
└── test_fast_brain.py      # Test client

docs/
├── HIVE215_INTEGRATION_GUIDE.md  # Complete integration guide for voice agents
└── visuals/                      # Visual HTML documentation
    ├── index.html                # Documentation library index
    ├── dashboard-improvement-plan.html
    ├── 2024-12-18_fast-brain-hive215-architecture.html
    └── 2024-12-18_unified-dashboard-features.html
```

---

## API Reference

### Base URL
```
https://[your-username]--fast-brain-lpu-fastapi-app.modal.run
```

### Endpoints

#### Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "architecture": "System 1 + System 2 Hybrid",
  "system1": {"model": "llama-3.3-70b-versatile", "latency": "~80ms"},
  "system2": {"model": "claude-sonnet-4-5-20250929", "latency": "~2000ms"},
  "skills": ["general", "receptionist", "electrician", "plumber", "lawyer", "solar", "default", "tara-sales"]
}
```

#### Hybrid Chat (Recommended)
```bash
POST /v1/chat/hybrid
Content-Type: application/json

{
  "messages": [{"role": "user", "content": "Can you analyze my electricity bill?"}],
  "skill": "electrician",
  "user_context": {"business_name": "Jenkintown Electricity"}
}

Response (simple question):
{
  "content": "I'd be happy to help! What's your address?",
  "filler": null,
  "system_used": "fast",
  "fast_latency_ms": 85
}

Response (complex question):
{
  "content": "Based on your usage of 850 kWh at $0.12/kWh...",
  "filler": "Let me pull up your information and analyze that for you...",
  "system_used": "deep",
  "fast_latency_ms": 85,
  "deep_latency_ms": 1950
}
```

#### Voice Chat (with TTS hints)
```bash
POST /v1/chat/voice

Response:
{
  "text": "Based on your usage...",
  "voice": "A patient, clear female voice. Teacher-like.",
  "filler_text": "Let me look into that...",
  "system_used": "deep"
}
```

---

## Training Collector (Supabase)

Collect metrics and export training data for LoRA fine-tuning:

```python
from training_collector import TrainingCollector

collector = TrainingCollector()

# Log an interaction
collector.log_interaction(
    skill_id="electrician",
    user_message="My lights are flickering",
    response="That could be a loose connection...",
    system_used="fast",
    latency_ms=85
)

# Export for fine-tuning
training_data = collector.export_for_training(
    min_quality=4,  # Only highly-rated responses
    format="jsonl"
)
```

### Supabase Setup
```bash
py -3.11 -m modal secret create supabase-credentials \
    SUPABASE_URL=https://xxx.supabase.co \
    SUPABASE_KEY=your_service_key
```

---

## Cost Estimates (per minute of voice)

| Component | Cost/min | Notes |
|-----------|----------|-------|
| Deepgram STT | ~$0.006 | Nova-3 streaming |
| Parler TTS | ~$0.002 | GPU time (~3s/request) |
| Groq (System 1) | ~$0.0001 | Free tier: 14,400 req/day |
| Claude (System 2) | ~$0.003 | Only ~10% of calls |
| LiveKit Cloud | ~$0.004 | Ship plan |
| Modal (Dashboard) | FREE | Scales to zero |
| **Total** | ~$0.015/min | ~$0.90/hour |

---

## Troubleshooting

### "Modal doesn't support Python 3.14"
Use Python 3.11:
```powershell
py -3.11 -m modal deploy deploy_dashboard.py
```

### "No audio was received" (Edge TTS)
Edge TTS WebSockets may be blocked from Modal. The dashboard uses gTTS as fallback.

### Parler TTS timeout
First request takes 30-60s (GPU cold start). Try again after warming up.

### Claude model not found (404)
Update to latest model name in `deploy_groq.py`:
```python
model="claude-sonnet-4-5-20250929"
```

---

## Development

### Recent Updates (December 2024)

**New Features:**
- **Voice Lab**: Full voice cloning workflow with ElevenLabs, Cartesia, and free gTTS
- **Dynamic Voice Fetching**: Live voice lists from ElevenLabs, Cartesia, Deepgram, OpenAI APIs
- **Browse Voices Tab**: Search and filter voices from all providers with preview playback
- **TTS Audio Caching**: In-memory cache for instant playback of repeated phrases (<5ms vs 500-2000ms)
- **Voice Preview**: One-click preview of provider sample audio
- **Outgoing API Connections**: Connect agents to external REST APIs with auth support
- **Dark/Light Mode**: Slack-inspired light theme with localStorage persistence
- **Four Pillars Navigation**: Consolidated from 6 tabs to 4 main sections
- **SQLite Database**: Persistent storage on Modal volume for all data
- **7 API Key Providers**: Groq, OpenAI, Anthropic, ElevenLabs, Cartesia, Deepgram, PlayHT
- **tara-sales skill**: TheDashTool sales assistant for voice demos
- **API Endpoints Card**: Display/copy Fast Brain API and Dashboard URLs with health check
- **Voice Test in Test Chat**: Test any voice with custom text (optional section)

**Voice Providers Supported:**
| Provider | Type | Features |
|----------|------|----------|
| ElevenLabs | Premium | Voice cloning, 70+ languages, emotional range |
| Cartesia | Premium | Sub-100ms latency, 3-second voice cloning |
| Deepgram | Premium | Aura TTS, enterprise-grade reliability |
| OpenAI | Premium | 11 voices (alloy, echo, nova, etc.) |
| Parler TTS | Free (GPU) | Expressive synthesis, emotion control |
| Edge TTS | Free | Microsoft voices, multiple accents |
| Kokoro | Free | High-quality open source |
| gTTS | Free | Google TTS, reliable fallback |

**Performance Improvements:**
- TTS cache: 500 items, 1-hour TTL, LRU eviction
- Voice list cache: 5-minute client-side TTL
- Cache hit latency: <5ms (vs 500-2000ms for fresh synthesis)
- Common phrases pre-cached for instant response

**Improvements:**
- Removed all placeholder/test data for production readiness
- Better empty states with Getting Started onboarding
- Improved audio playback with proper error handling
- Real HTTP testing for API connections
- Activity logging for all operations

**Infrastructure:**
- Python 3.11 venv setup (recommended deployment workflow)
- Parler TTS integration with emotion controls
- gTTS fallback for reliable voice testing
- Training collector for LoRA fine-tuning data export

---

## License

MIT

---

## Contact

**HIVE215** - AI Phone Assistant Platform
**Jenkintown Electricity** - Primary deployment

Built with Groq, Claude, LiveKit, Modal, Parler-TTS, and Claude Code
