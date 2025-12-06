# Fast Brain LPU

**Hybrid Voice AI Engine for HIVE215 - "Thinking, Fast and Slow"**

Fast Brain is a dual-system inference engine inspired by Daniel Kahneman's cognitive model. It combines ultra-fast Groq inference (~80ms) for routine questions with Claude's deep reasoning (~2s) for complex analysis - all while maintaining a natural conversational experience through intelligent filler phrases.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SYSTEM 1 + SYSTEM 2 HYBRID                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   User: "Can you analyze my bill and predict next month's cost?"            │
│            │                                                                 │
│            ▼                                                                 │
│   ┌─────────────────┐                                                        │
│   │  🚀 SYSTEM 1    │  Groq + Llama 3.3 70B                                  │
│   │  (Fast Brain)   │  ~80ms, handles 90% of calls                          │
│   │                 │                                                        │
│   │  Decision:      │  "This needs complex reasoning..."                     │
│   └────────┬────────┘                                                        │
│            │                                                                 │
│     ┌──────┴──────┐                                                          │
│     │             │                                                          │
│     ▼             ▼                                                          │
│  [SIMPLE]      [COMPLEX]                                                     │
│     │             │                                                          │
│     ▼             ▼                                                          │
│  Answer       ┌─────────────────┐                                            │
│  Directly     │  Output A:      │  "Let me pull that up for you..."         │
│  (~80ms)      │  FILLER PHRASE  │  (Plays while Claude thinks)              │
│               └────────┬────────┘                                            │
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
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## What's Implemented

### Core Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Hybrid Architecture** | ✅ Live | System 1 (Groq) + System 2 (Claude) |
| **Filler Phrase Strategy** | ✅ Live | Hide Claude's latency with natural phrases |
| **LiveKit Turn Detector** | ✅ Live | Context-aware turn detection (FREE, local) |
| **Skills System** | ✅ Live | Domain-specific prompts and behaviors |
| **Voice-Optimized Output** | ✅ Live | Contractions, short sentences, no markdown |
| **Modal Deployment** | ✅ Live | Serverless, auto-scaling, always warm |
| **Logic-Based Skill Routing** | ✅ Live | Route by phone number (~0ms, not LLM) |

### Performance Metrics

| System | Latency | Use Case | % of Calls |
|--------|---------|----------|------------|
| **System 1 (Groq)** | ~80ms | Simple questions, greetings, intake | ~90% |
| **System 2 (Claude)** | ~2000ms | Analysis, calculations, complex advice | ~10% |
| **Turn Detector** | ~10-25ms | End-of-utterance detection | Every turn |
| **Filler Phrases** | ~1500ms spoken | Covers Claude's thinking time | When needed |

### Built-in Skills

| Skill ID | Name | Best For |
|----------|------|----------|
| `general` | General Assistant | Default helpful assistant |
| `receptionist` | Professional Receptionist | Call handling, message taking |
| `electrician` | Electrician Service Intake | Jenkintown Electricity |
| `plumber` | Plumber Service Intake | Emergency/routine plumbing |
| `lawyer` | Legal Intake Specialist | Confidential legal intake |
| `solar` | Solar Company Receptionist | Solar qualification |

---

## File Structure

```
fast_brain/
├── deploy_groq.py          # Main deployment (v3.0 Hybrid) ← USE THIS
├── skills.py               # Skills definitions
├── config.py               # Configuration dataclasses
├── client.py               # Python client library
├── __init__.py             # Package exports
├── deploy_bitnet.py        # BitNet attempt (archived)
├── deploy_simple.py        # Simple stub (testing)
├── deploy.py               # Original design (archived)
└── model.py                # BitNet model wrapper (archived)

worker/
├── voice_agent.py          # LiveKit voice agent with turn detector
├── requirements.txt        # Worker dependencies
├── __init__.py
└── test_fast_brain.py      # Test client

# Documentation
README.md                   # This file
HIVE215_INTEGRATION.md      # Integration guide
FASTBRAIN_MODAL_DEPLOY.md   # Deployment documentation
hive215-architecture-overview.html  # Visual architecture

# Utilities
unified_dashboard.py        # Management dashboard
skill_factory.py            # Skill creation utilities
skill_command_center.py     # Command center UI
turn_taking.py              # Turn detection experiments
```

---

## API Reference

### Base URL
```
https://[your-username]--fast-brain-lpu.modal.run
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
  "system2": {"model": "claude-3-5-sonnet-20241022", "latency": "~2000ms"},
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
  "fast_latency_ms": 85,
  "total_latency_ms": 85
}

Response (complex question):
{
  "content": "Based on your usage of 850 kWh at $0.12/kWh...",
  "filler": "Let me pull up your information and analyze that for you...",
  "system_used": "deep",
  "fast_latency_ms": 85,
  "deep_latency_ms": 1950,
  "total_latency_ms": 2035
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
  "filler_voice": "A thoughtful, measured female voice.",
  "system_used": "deep"
}
```

#### Standard Chat (System 1 only)
```bash
POST /v1/chat/completions
```

#### List Skills
```bash
GET /v1/skills
```

#### Get Greeting
```bash
GET /v1/greeting/{skill_id}
```

---

## Voice Agent (LiveKit)

The voice agent (`worker/voice_agent.py`) integrates with LiveKit Cloud:

### Pipeline
```
Phone Call → LiveKit Cloud → Deepgram STT → Turn Detector → Fast Brain
                │                                              │
                │ (noise cancellation FREE)             ┌──────┴──────┐
                │                                    [SIMPLE]     [COMPLEX]
                │                                       │             │
                └──────────────────────────────────  Answer     Filler + Claude
                                                     (~80ms)       (~2s hidden)
                                                        │             │
                                                        └──────┬──────┘
                                                               │
                                                        Cartesia TTS → Audio
```

### Key Components

| Component | Provider | Cost | Notes |
|-----------|----------|------|-------|
| **STT** | Deepgram Nova-3 | ~$0.006/min | YOUR API key |
| **TTS** | Cartesia Sonic | ~$0.0225/min | YOUR API key |
| **VAD** | Silero | FREE | Runs locally |
| **Turn Detector** | LiveKit Plugin | FREE | Runs locally, ~200MB |
| **Noise Cancellation** | LiveKit Cloud | FREE | Standard tier |

### Turn Detector

Uses `livekit-plugins-turn-detector` with `EOUModel()`:

```python
from livekit.plugins.turn_detector import EOUModel

turn_detector = EOUModel()

agent = VoicePipelineAgent(
    vad=vad,
    stt=stt,
    llm=llm_adapter,
    tts=tts,
    turn_detector=turn_detector,  # Context-aware!
    min_endpointing_delay=0.5,
    max_endpointing_delay=6.0,
)
```

**Benefits over basic VAD:**
- Understands linguistic context (vs. just silence detection)
- Knows when user is mid-sentence vs. done speaking
- Reduces interruptions and awkward pauses
- ~10-25ms inference, runs locally

---

## Deployment

### Prerequisites
- Modal account (free tier works)
- Groq API key (free at console.groq.com)
- Anthropic API key
- Python 3.11+

### Deploy to Modal

```bash
# 1. Install Modal CLI
pip install modal
modal token new

# 2. Set up secrets
modal secret create groq-api-key GROQ_API_KEY=gsk_your_key
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-your_key

# 3. Deploy
modal deploy fast_brain/deploy_groq.py

# 4. Get your URL
# https://[your-username]--fast-brain-lpu.modal.run
```

### Deploy Voice Agent

```bash
# Set environment variables
export FAST_BRAIN_URL=https://your-username--fast-brain-lpu.modal.run
export LIVEKIT_URL=wss://your-project.livekit.cloud
export LIVEKIT_API_KEY=your_key
export LIVEKIT_API_SECRET=your_secret
export DEEPGRAM_API_KEY=your_deepgram_key
export CARTESIA_API_KEY=your_cartesia_key

# Install dependencies
pip install -r worker/requirements.txt

# Run agent
python -m worker.voice_agent dev
```

### Cost Estimates (per minute of voice)

| Component | Cost/min | Notes |
|-----------|----------|-------|
| Deepgram STT | ~$0.006 | Nova-3 streaming |
| Cartesia TTS | ~$0.023 | Sonic English |
| Groq (System 1) | ~$0.0001 | Free tier: 14,400 req/day |
| Claude (System 2) | ~$0.003 | Only ~10% of calls |
| LiveKit Cloud | ~$0.004 | Ship plan: 5,000 min |
| **Total** | ~$0.036/min | ~$2.16/hour |

---

## Future Roadmap

### In Progress

- [ ] Streaming responses (SSE) for faster perceived latency
- [ ] Multi-turn conversation memory with context window management
- [ ] A/B testing framework for skill prompt optimization

### Planned

- [ ] RAG with vector embeddings for business-specific knowledge
- [ ] Real-time analytics dashboard
- [ ] Webhook notifications for call events
- [ ] Direct Supabase skill sync
- [ ] Custom skill creation API
- [ ] Multi-language support (turn detector has `MultilingualModel()`)

### Future Ideas

- [ ] Fine-tuned models per skill domain
- [ ] Voice cloning integration (ElevenLabs/PlayHT)
- [ ] Real-time skill switching mid-call based on topic
- [ ] Sentiment-based response adaptation
- [ ] Call recording and transcription storage
- [ ] CRM integrations (HubSpot, Salesforce)
- [ ] Warm transfer to human agents with context handoff

---

## Development

### Current Branch
```
claude/implement-new-vision-01FkwcnCFbicK9uG2QrB8FSf
```

### Recent Commits
```
e40cf25 Add LiveKit turn detector plugin for context-aware turn detection
b672881 Implement hybrid System 1 + System 2 architecture (v3.0)
```

### Local Testing

```bash
# Set API keys
export GROQ_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key

# Run local test
python fast_brain/deploy_groq.py
```

---

## Troubleshooting

### "503 Service Unavailable"
- Check Modal deployment: `modal app list`
- Verify secrets are set: `modal secret list`
- Check logs: `modal app logs fast-brain-lpu`

### Slow first response
- First request warms container (~3s)
- Subsequent requests: ~80ms (System 1) or ~2s (System 2)
- Set `keep_warm=1` in deploy for always-warm containers

### Turn detector not working
- Ensure `livekit-plugins-turn-detector` is installed
- Check it's passed to `VoicePipelineAgent`
- Model downloads on first use (~200MB)

### Claude not being triggered
- Check if skill's system prompt includes `ask_expert` tool instructions
- Verify `anthropic-api-key` secret exists in Modal
- Look for "System used: deep" in logs

---

## License

MIT

---

## Contact

**HIVE215** - AI Phone Assistant Platform
**Jenkintown Electricity** - Primary deployment

Built with Groq, Claude, LiveKit, Modal, and Claude Code
