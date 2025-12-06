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

### Windows Users (Python 3.11)
```powershell
# Use py launcher to specify Python 3.11
py -3.11 -m pip install modal
py -3.11 -m modal token new

# Set up secrets
py -3.11 -m modal secret create groq-api-key GROQ_API_KEY=gsk_your_key
py -3.11 -m modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-your_key

# Deploy Fast Brain API
py -3.11 -m modal deploy fast_brain/deploy_groq.py

# Deploy Dashboard
py -3.11 -m modal deploy deploy_dashboard.py

# Deploy Parler TTS (GPU - optional)
py -3.11 -m modal deploy parler_integration.py
```

### Mac/Linux Users
```bash
pip install modal
modal token new
modal secret create groq-api-key GROQ_API_KEY=gsk_your_key
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-your_key
modal deploy fast_brain/deploy_groq.py
modal deploy deploy_dashboard.py
modal deploy parler_integration.py
```

### Your Deployed URLs
```
Fast Brain API:  https://[username]--fast-brain-lpu-fastapi-app.modal.run
Dashboard:       https://[username]--hive215-dashboard-flask-app.modal.run
Parler TTS:      https://[username]--hive215-parler-tts-parlerttsmodel-*.modal.run
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

### Command Center
- Real-time metrics and activity feed
- Voice configuration with emotion controls
- LLM comparison testing (Groq vs Claude)

### Skills Factory
- **Business Profile**: Create skill configurations
- **Golden Prompts**: View/edit voice-optimized prompts
- **Upload Documents**: Add training data
- **Train Skill**: Generate LoRA training scripts
- **Manage Skills**: Full skill library

### Voice Lab
- **Parler TTS**: Expressive voices with emotion (GPU)
- **Edge TTS**: Free Microsoft voices
- **gTTS**: Google Text-to-Speech fallback
- Voice testing with audio playback

### Platform Connections
- LiveKit, Twilio, Vonage integration status
- Vapi configuration

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
unified_dashboard.py        # Full management UI (259KB)

worker/
├── voice_agent.py          # LiveKit voice agent
├── requirements.txt        # Worker dependencies
└── test_fast_brain.py      # Test client
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
  "skills": ["general", "receptionist", "electrician", "plumber", "lawyer", "solar"]
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

### Current Branch
```
claude/add-modal-instructions-01BymuqovBFYVM933PhHLvVg
```

### Recent Updates
- Parler TTS integration with emotion controls
- Golden Prompts management UI
- gTTS fallback for voice testing
- Training collector for LoRA fine-tuning
- Unified dashboard on Modal

---

## License

MIT

---

## Contact

**HIVE215** - AI Phone Assistant Platform
**Jenkintown Electricity** - Primary deployment

Built with Groq, Claude, LiveKit, Modal, Parler-TTS, and Claude Code
