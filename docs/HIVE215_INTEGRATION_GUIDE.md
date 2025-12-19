# HIVE215 Integration Guide for Fast Brain API

**Fast Brain LPU - Voice AI Inference Engine**

This guide explains how to integrate your HIVE215 voice application with the Fast Brain API to create ultra-responsive AI agents.

---

## Quick Start

### Base URL
```
https://[your-username]--fast-brain-lpu-fastapi-app.modal.run
```

### Test Connection
```bash
curl https://[your-username]--fast-brain-lpu-fastapi-app.modal.run/health
```

**Response:**
```json
{
  "status": "healthy",
  "architecture": "System 1 + System 2 Hybrid",
  "system1": {"model": "llama-3.3-70b-versatile", "latency": "~80ms"},
  "system2": {"model": "claude-3-5-sonnet", "latency": "~2000ms"},
  "skills": ["general", "receptionist", "electrician", "plumber", "lawyer", "solar", "tara-sales"]
}
```

---

## Architecture Overview

Fast Brain uses a **dual-system architecture** inspired by Daniel Kahneman's "Thinking, Fast and Slow":

| System | Model | Latency | Use Case |
|--------|-------|---------|----------|
| **System 1 (Fast Brain)** | Groq + Llama 3.3 70B | ~80ms | Simple questions, greetings, intake (90% of calls) |
| **System 2 (Deep Brain)** | Claude 3.5 Sonnet | ~2000ms | Complex analysis, calculations, reasoning (10% of calls) |

**Key Innovation:** When System 2 is needed, Fast Brain returns a **filler phrase** to play while Claude thinks. This creates a seamless, natural conversation flow.

---

## API Endpoints

### 1. Hybrid Chat (Recommended)
**`POST /v1/chat/hybrid`**

The primary endpoint for voice agents. Automatically routes between System 1 and System 2.

```bash
curl -X POST https://[your-url]/v1/chat/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What time do you open?"}],
    "skill": "electrician"
  }'
```

**Simple Question Response (System 1):**
```json
{
  "content": "We're open Monday through Friday, 8 AM to 6 PM, and Saturdays 9 AM to 2 PM.",
  "filler": null,
  "system_used": "fast",
  "fast_latency_ms": 85
}
```

**Complex Question Response (System 2):**
```json
{
  "content": "Based on your 850 kWh usage at $0.12/kWh, your monthly bill should be around $102...",
  "filler": "Let me pull up your information and analyze that for you, just a moment...",
  "system_used": "deep",
  "fast_latency_ms": 85,
  "deep_latency_ms": 1950
}
```

### 2. Voice Chat (with TTS hints)
**`POST /v1/chat/voice`**

Returns text + voice description for Parler TTS or similar expressive synthesis.

```bash
curl -X POST https://[your-url]/v1/chat/voice \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "I need help with a flickering light"}],
    "skill": "electrician"
  }'
```

**Response:**
```json
{
  "text": "That could be a loose connection or a failing switch. Let me ask a few questions to help diagnose the issue.",
  "voice": "A patient, clear male voice. Knowledgeable and reassuring.",
  "filler_text": "Let me look into that for you...",
  "filler_voice": "A warm, conversational voice. Friendly and patient.",
  "system_used": "deep"
}
```

### 3. Standard Chat (OpenAI-compatible)
**`POST /v1/chat/completions`**

OpenAI-compatible endpoint for direct integration with existing tools.

```bash
curl -X POST https://[your-url]/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "llama-3.3-70b-versatile",
    "skill": "receptionist"
  }'
```

### 4. Skills Management

**List all skills:**
```bash
GET /v1/skills
```

**Get skill details:**
```bash
GET /v1/skills/{skill_id}
```

**Create custom skill:**
```bash
POST /v1/skills
{
  "skill_id": "my_business",
  "name": "My Business Assistant",
  "description": "Handles inquiries for My Business",
  "system_prompt": "You are a helpful assistant for My Business...",
  "knowledge": ["Hours: 9-5 M-F", "Phone: 555-1234"]
}
```

**Get skill greeting:**
```bash
GET /v1/greeting/{skill_id}
```

### 5. Filler Phrases
**`GET /v1/fillers`**

Get all filler phrase categories for custom handling.

```json
{
  "categories": ["analysis", "calculation", "research", "complex", "default"],
  "phrases": {
    "analysis": ["Let me pull up your information..."],
    "calculation": ["Let me run those numbers..."]
  }
}
```

---

## Integration Patterns

### Pattern 1: Basic Voice Agent (LiveKit)

```python
from livekit import agents
import httpx

FAST_BRAIN_URL = "https://your-username--fast-brain-lpu-fastapi-app.modal.run"

async def handle_user_message(message: str, skill: str = "receptionist"):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{FAST_BRAIN_URL}/v1/chat/hybrid",
            json={
                "messages": [{"role": "user", "content": message}],
                "skill": skill
            }
        )
        data = response.json()

        # If filler exists, play it while waiting for main response
        if data.get("filler"):
            # Play filler audio first (synthesize with TTS)
            await play_filler(data["filler"])

        # Return main content
        return data["content"]
```

### Pattern 2: Filler Phrase Strategy

The key to a natural-feeling agent is using filler phrases correctly:

```python
async def process_with_filler(message: str, skill: str, tts_client):
    response = await fast_brain.hybrid_chat(message, skill)

    # Check if filler was returned (indicates System 2 was used)
    if response.filler:
        # Start TTS for filler immediately
        filler_audio = await tts_client.synthesize(
            text=response.filler,
            voice="warm_female"  # Consistent with agent persona
        )

        # Play filler (takes ~2-3 seconds, matches Claude's latency!)
        await play_audio(filler_audio)

    # Now play the main response
    main_audio = await tts_client.synthesize(
        text=response.content,
        voice="warm_female"
    )
    await play_audio(main_audio)
```

### Pattern 3: Skill-Based Routing

Route different callers to different skills based on context:

```python
SKILL_ROUTING = {
    "electrical": "electrician",
    "plumbing": "plumber",
    "legal": "lawyer",
    "solar": "solar",
    "sales": "tara-sales",
    "default": "receptionist"
}

async def route_call(caller_intent: str):
    skill = SKILL_ROUTING.get(caller_intent, SKILL_ROUTING["default"])

    # Get skill-specific greeting
    greeting = await fast_brain.get_greeting(skill)

    return {
        "skill": skill,
        "greeting": greeting["text"],
        "voice": greeting["voice"]
    }
```

### Pattern 4: Context Passing

Pass business context to personalize responses:

```python
response = await client.post(
    f"{FAST_BRAIN_URL}/v1/chat/hybrid",
    json={
        "messages": messages,
        "skill": "electrician",
        "user_context": {
            "business_name": "Jenkintown Electricity",
            "caller_name": "John Smith",
            "account_number": "12345",
            "location": "Philadelphia, PA"
        }
    }
)
```

---

## Dashboard Integration

### Managing Skills via Dashboard

1. **Access Dashboard**: `https://[username]--hive215-dashboard-flask-app.modal.run`
2. **Navigate to Skills tab**
3. **Create/Edit skills** with custom prompts and knowledge bases
4. **Test skills** in the Test Chat sub-tab

### API Keys Setup

Configure your API keys in **Settings > API Keys**:

| Provider | Required For |
|----------|--------------|
| Groq | System 1 (Fast Brain) |
| Anthropic | System 2 (Deep Brain) |
| ElevenLabs | Premium voice cloning |
| Cartesia | Ultra-low latency TTS |
| Deepgram | Speech-to-text |

### Voice Integration

The Dashboard's **Voice Lab** lets you:
- Browse voices from all providers
- Preview voice samples
- Create custom voice clones
- Link voices to specific skills

---

## Best Practices

### 1. Choose the Right Endpoint

| Use Case | Endpoint |
|----------|----------|
| Voice agents with TTS | `/v1/chat/voice` |
| General voice agents | `/v1/chat/hybrid` |
| Text-only apps | `/v1/chat/completions` |

### 2. Always Handle Fillers

When `filler` is not null, always play it! This is what makes the agent feel human.

```python
if response.get("filler"):
    await synthesize_and_play(response["filler"])
await synthesize_and_play(response["content"])
```

### 3. Use Appropriate Skills

Match skills to caller intent for best results:
- **receptionist**: General inquiries, call routing
- **electrician**: Technical electrical questions
- **plumber**: Plumbing emergencies and scheduling
- **lawyer**: Legal intake and appointment setting
- **solar**: Solar panel sales qualification
- **tara-sales**: Product demos and sales

### 4. Pass Context

Always include relevant context in `user_context`:
```python
{
    "business_name": "Your Business",
    "service_area": "Philadelphia metro",
    "hours": "Mon-Fri 9am-5pm"
}
```

### 5. Monitor Performance

Track these metrics:
- **System 1 vs System 2 ratio**: Should be ~90/10
- **Fast latency**: Should be <100ms
- **Deep latency**: Should be <2500ms
- **Cache hit rate**: Higher = faster responses

---

## Error Handling

```python
async def safe_chat(message: str, skill: str):
    try:
        response = await client.post(
            f"{FAST_BRAIN_URL}/v1/chat/hybrid",
            json={"messages": [{"role": "user", "content": message}], "skill": skill},
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()

    except httpx.TimeoutException:
        # Fallback response
        return {
            "content": "I apologize, I'm having a moment. Could you repeat that?",
            "filler": None,
            "system_used": "fallback"
        }

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            # Rate limited - back off
            await asyncio.sleep(1)
            return await safe_chat(message, skill)
        raise
```

---

## Sample Integration (Full Example)

```python
"""
HIVE215 Voice Agent with Fast Brain Integration
"""
import httpx
import asyncio
from typing import Optional

class FastBrainClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=15.0)

    async def health_check(self) -> dict:
        response = await self.client.get(f"{self.base_url}/health")
        return response.json()

    async def get_skills(self) -> list:
        response = await self.client.get(f"{self.base_url}/v1/skills")
        return response.json()["skills"]

    async def chat(
        self,
        message: str,
        skill: str = "general",
        context: Optional[dict] = None,
        history: Optional[list] = None
    ) -> dict:
        messages = history or []
        messages.append({"role": "user", "content": message})

        response = await self.client.post(
            f"{self.base_url}/v1/chat/hybrid",
            json={
                "messages": messages,
                "skill": skill,
                "user_context": context or {}
            }
        )
        return response.json()

    async def voice_chat(
        self,
        message: str,
        skill: str = "general",
        context: Optional[dict] = None
    ) -> dict:
        response = await self.client.post(
            f"{self.base_url}/v1/chat/voice",
            json={
                "messages": [{"role": "user", "content": message}],
                "skill": skill,
                "user_context": context or {}
            }
        )
        return response.json()

    async def get_greeting(self, skill: str) -> dict:
        response = await self.client.get(f"{self.base_url}/v1/greeting/{skill}")
        return response.json()

    async def close(self):
        await self.client.aclose()


# Usage Example
async def main():
    client = FastBrainClient(
        "https://your-username--fast-brain-lpu-fastapi-app.modal.run"
    )

    # Check health
    health = await client.health_check()
    print(f"Status: {health['status']}")
    print(f"Skills: {health['skills']}")

    # Get greeting for electrician skill
    greeting = await client.get_greeting("electrician")
    print(f"Greeting: {greeting['text']}")

    # Simple question (System 1)
    response = await client.chat(
        "What are your hours?",
        skill="electrician",
        context={"business_name": "Jenkintown Electricity"}
    )
    print(f"Response: {response['content']}")
    print(f"System used: {response['system_used']}")
    print(f"Latency: {response['fast_latency_ms']}ms")

    # Complex question (System 2)
    response = await client.chat(
        "Can you analyze my 850 kWh usage and predict next month's bill?",
        skill="electrician",
        context={"business_name": "Jenkintown Electricity"}
    )
    print(f"Filler: {response.get('filler')}")
    print(f"Response: {response['content']}")
    print(f"System used: {response['system_used']}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Deployment URLs

After running `modal deploy`, your services will be at:

| Service | URL Pattern |
|---------|-------------|
| Dashboard | `https://[username]--hive215-dashboard-flask-app.modal.run` |
| Fast Brain API | `https://[username]--fast-brain-lpu-fastapi-app.modal.run` |
| Parler TTS | `https://[username]--hive215-parler-tts-parlerttsmodel-*.modal.run` |

---

## Support

- **Documentation**: `/docs/visuals/` in the repository
- **Issues**: Check `TODO.md` for known issues
- **Logs**: Check `CLAUDE_LOG.md` for recent changes

---

*Built for HIVE215 - AI Phone Assistant Platform*
*Powered by Groq, Claude, LiveKit, and Modal*
