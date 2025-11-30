# fast_brain

Virtual Chip - A serverless BitNet.cpp LPU (Language Processing Unit) on Modal.

## Overview

This project creates a "Virtual Chip" that:
- Compiles specialized 1-bit kernels (BitNet.cpp) in a serverless environment
- Downloads and serves the BitNet Llama3-8B model
- Exposes a high-speed inference endpoint
- Supports hot-swappable skill adapters (LoRA)

## Quick Start

### 1. Install Modal

```bash
pip install modal
```

### 2. Setup Token

```bash
modal setup
```

### 3. Deploy the Virtual Chip

```bash
modal deploy modal_lpu.py
```

## Training Expert Skills

Use `training_the_expert.py` to create specialized LoRA adapters:

```bash
# Prepare your dataset as JSONL:
# {"instruction": "How do I connect?", "input": "", "output": "Use ctx.connect()..."}

python training_the_expert.py
```

## Upload Skills to the LPU

After training, upload adapters to the Modal volume:

```bash
modal volume put lpu-skills adapters/livekit_architect /root/skills/livekit_architect.lora
```

## Integration Guide

### Connect from Your Application

```python
import modal

# Connect to the remote class
lpu = modal.Cls.lookup("bitnet-lpu-v1", "VirtualLPU")()

# Generate with base model
for chunk in lpu.chat.remote_gen("User: Hello! Assistant:"):
    print(chunk, end="", flush=True)

# Generate with a skill adapter
for chunk in lpu.chat.remote_gen(
    "User: Help me build a LiveKit agent! Assistant:",
    skill_adapter="livekit_architect.lora"
):
    print(chunk, end="", flush=True)

# List available skills
skills = lpu.list_skills.remote()
print(f"Available skills: {skills}")
```

### LiveKit Agent Integration

```python
import modal
from livekit.agents import AutoSubscribe, JobContext, llm

# Get the Virtual LPU
lpu = modal.Cls.lookup("bitnet-lpu-v1", "VirtualLPU")()

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Use the LPU for fast inference
    async def generate_response(prompt: str) -> str:
        response = ""
        for chunk in lpu.chat.remote_gen(prompt, skill_adapter="livekit_architect.lora"):
            response += chunk
        return response

    # Your agent logic here...
```

## Architecture

```
+-------------------+     +------------------+     +------------------+
|   Your App        | --> |   Modal Cloud    | --> |   Virtual LPU    |
|   (LiveKit Agent) |     |   (Serverless)   |     |   (BitNet.cpp)   |
+-------------------+     +------------------+     +------------------+
                                   |
                          +--------v--------+
                          |  Skills Volume  |
                          |  (LoRA Adapters)|
                          +-----------------+
```

## Configuration

- `min_containers=1`: One instance always ready (zero cold start)
- `min_containers=0`: Standard serverless (~2s startup)
- `timeout=600`: 10 minute max inference time

## Files

- `modal_lpu.py` - Virtual Chip deployment code
- `training_the_expert.py` - LoRA adapter training script
- `unified_dashboard.py` - Unified dashboard with voice integration
- `bitnet_lpu_roadmap.html` - Project roadmap
- `lora_swarm.html` - Multi-skill swarm documentation

---

## NEXT SESSION: Hook Up Real Voice APIs

### Current State
The unified dashboard (`unified_dashboard.py`) has a complete UI for voice selection and platform connections, but uses **mock/simulated responses**. The next session needs to connect real TTS APIs.

### Voice Providers to Integrate

| Provider | API Docs | Priority |
|----------|----------|----------|
| **Chatterbox** | Local/self-hosted TTS | HIGH - Default voice |
| **ElevenLabs** | https://elevenlabs.io/docs/api-reference | HIGH - Best quality |
| **Cartesia** | https://docs.cartesia.ai/ | HIGH - Low latency |
| **PlayHT** | https://docs.play.ht/reference | MEDIUM |
| **OpenVoice** | https://github.com/myshell-ai/OpenVoice | MEDIUM - Voice cloning |
| **Coqui/XTTS** | https://docs.coqui.ai/ | LOW - Open source |
| **Bark** | https://github.com/suno-ai/bark | LOW - Open source |

### Platform Connections to Implement

| Platform | Purpose | Config Required |
|----------|---------|-----------------|
| **LiveKit** | Real-time voice/video | `api_key`, `api_secret`, `url` |
| **Vapi** | Phone voice agents | `api_key` |
| **Twilio** | Cloud voice calls | `account_sid`, `auth_token` |
| **Vocode** | Conversational AI | `api_key` |
| **Retell** | Voice AI agents | `api_key` |
| **Pipecat** | Open-source framework | `server_url` |

### Files to Modify

1. **`unified_dashboard.py`** - Replace mock implementations:
   - `test_voice()` at line ~654 - Currently returns simulated response
   - `connect_platform()` at line ~690 - Currently simulates connection
   - Add real API calls with proper error handling

2. **Add new file `voice_providers.py`** - Create provider classes:
   ```python
   class VoiceProvider:
       def synthesize(self, text: str, voice_id: str) -> bytes:
           """Return audio bytes"""
           raise NotImplementedError

   class ElevenLabsProvider(VoiceProvider):
       def __init__(self, api_key: str):
           self.api_key = api_key

       def synthesize(self, text: str, voice_id: str) -> bytes:
           # Call ElevenLabs API
           pass
   ```

3. **Add new file `platform_connectors.py`** - Create platform classes:
   ```python
   class PlatformConnector:
       def connect(self, config: dict) -> bool:
           raise NotImplementedError

       def send_audio(self, audio: bytes) -> None:
           raise NotImplementedError

   class LiveKitConnector(PlatformConnector):
       # Implement LiveKit WebRTC connection
       pass
   ```

### API Keys Needed

Store in environment variables or use the dashboard's API Keys section:
- `ELEVENLABS_API_KEY`
- `CARTESIA_API_KEY`
- `PLAYHT_API_KEY` + `PLAYHT_USER_ID`
- `LIVEKIT_API_KEY` + `LIVEKIT_API_SECRET`
- `VAPI_API_KEY`
- `TWILIO_ACCOUNT_SID` + `TWILIO_AUTH_TOKEN`

### Quick Start for Next Session

```bash
# 1. Run the dashboard
python unified_dashboard.py

# 2. Open http://localhost:5000

# 3. Go to "API Keys" tab and add your keys

# 4. Go to "Voice" tab to test voices

# 5. Go to "Connections" tab to link platforms
```

### Implementation Order

1. **ElevenLabs** - Best quality, well-documented API
2. **Cartesia** - Fastest latency for real-time
3. **LiveKit** - Connect to real-time voice rooms
4. **Chatterbox** - Local TTS fallback

### Testing

After implementing, test with:
```bash
curl -X POST http://localhost:5000/api/voice/test \
  -H "Content-Type: application/json" \
  -d '{"voice_id": "elevenlabs_rachel", "text": "Hello world"}'
```

Should return actual audio data instead of mock response.
