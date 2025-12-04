# Fast Brain + HIVE215 Integration Guide

**Instructions for the next Claude Code session working on HIVE215**

This document provides everything needed to integrate Fast Brain LPU into the HIVE215 voice assistant platform.

---

## Quick Summary

Fast Brain is a **Groq-powered inference endpoint** deployed on Modal that provides:
- **~80ms TTFB** (Time to First Byte)
- **200+ tokens/second** throughput
- **Built-in skills** for different business types
- **OpenAI-compatible API** for easy integration

**Deployed URL:** `https://[username]--fast-brain-lpu.modal.run`

---

## Integration Checklist

### Phase 1: Environment Setup

- [ ] **1.1** Add `FAST_BRAIN_URL` to Railway worker environment
  ```
  FAST_BRAIN_URL=https://[username]--fast-brain-lpu.modal.run
  ```

- [ ] **1.2** Verify Fast Brain is deployed and healthy
  ```bash
  curl https://[username]--fast-brain-lpu.modal.run/health
  ```
  Expected: `{"status": "healthy", "model_loaded": true, ...}`

- [ ] **1.3** Test a chat completion
  ```bash
  curl -X POST https://[username]--fast-brain-lpu.modal.run/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello!"}], "skill": "receptionist"}'
  ```

### Phase 2: Voice Agent Integration

- [ ] **2.1** Update `worker/voice_agent.py` to use Fast Brain as primary LLM
- [ ] **2.2** Implement fallback chain: Fast Brain → Groq → Anthropic
- [ ] **2.3** Add skill selection based on user profile
- [ ] **2.4** Test voice flow end-to-end

### Phase 3: Skills Sync

- [ ] **3.1** Create `/api/skills` endpoint in Hive215 backend
- [ ] **3.2** Sync skills from Fast Brain on user login
- [ ] **3.3** Allow users to select skills from Hive215 dashboard
- [ ] **3.4** Push custom skills to Fast Brain when created in Hive215

### Phase 4: Production

- [ ] **4.1** Add monitoring/logging for Fast Brain calls
- [ ] **4.2** Set up alerts for latency spikes
- [ ] **4.3** Document in Hive215 README
- [ ] **4.4** Update pricing/usage tracking

---

## Code Changes Required

### 1. Voice Agent LLM Integration

**File:** `worker/voice_agent.py`

Add Fast Brain as the primary LLM in the fallback chain:

```python
import os
import httpx
from typing import AsyncIterator

class FastBrainLLM:
    """Fast Brain LPU client for voice agent."""

    def __init__(self, url: str, skill: str = "receptionist"):
        self.url = url.rstrip('/')
        self.skill = skill
        self.client = httpx.AsyncClient(timeout=30.0)

    async def chat(
        self,
        messages: list[dict],
        user_profile: str = None,
        max_tokens: int = 256,
    ) -> str:
        """Send chat request to Fast Brain."""
        response = await self.client.post(
            f"{self.url}/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "skill": self.skill,
                "user_profile": user_profile,
            }
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        messages: list[dict],
        user_profile: str = None,
    ) -> AsyncIterator[str]:
        """Stream chat response (when implemented)."""
        # For now, return full response
        # TODO: Implement SSE streaming
        content = await self.chat(messages, user_profile)
        yield content


def get_llm():
    """Get LLM with fallback chain."""
    fast_brain_url = os.getenv("FAST_BRAIN_URL")
    groq_key = os.getenv("GROQ_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    # Primary: Fast Brain
    if fast_brain_url:
        try:
            return FastBrainLLM(url=fast_brain_url)
        except Exception as e:
            print(f"Fast Brain unavailable: {e}")

    # Fallback 1: Direct Groq
    if groq_key:
        from groq import Groq
        return Groq(api_key=groq_key)

    # Fallback 2: Anthropic
    if anthropic_key:
        from anthropic import Anthropic
        return Anthropic(api_key=anthropic_key)

    raise RuntimeError("No LLM configured!")
```

### 2. Backend Status Endpoint

**File:** `backend/main.py`

Add Fast Brain status to the admin status endpoint:

```python
@app.get("/admin/status")
async def get_admin_status():
    """Get status of all external services."""
    import httpx

    status = {
        "fast_brain": {"status": "unknown", "latency_ms": None},
        "groq": {"status": "unknown"},
        "deepgram": {"status": "unknown"},
        "cartesia": {"status": "unknown"},
    }

    # Check Fast Brain
    fast_brain_url = os.getenv("FAST_BRAIN_URL")
    if fast_brain_url:
        try:
            import time
            start = time.time()
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{fast_brain_url}/health")
                latency = (time.time() - start) * 1000
                if response.status_code == 200:
                    health = response.json()
                    status["fast_brain"] = {
                        "status": "online" if health.get("status") == "healthy" else "degraded",
                        "latency_ms": round(latency, 1),
                        "skills": health.get("skills_available", []),
                        "backend": health.get("backend", "unknown"),
                    }
        except Exception as e:
            status["fast_brain"] = {"status": "offline", "error": str(e)}
    else:
        status["fast_brain"] = {"status": "not_configured"}

    return status
```

### 3. Skills Sync Endpoint

**File:** `backend/main.py`

Add endpoint to fetch skills from Fast Brain:

```python
@app.get("/api/skills/fast-brain")
async def get_fast_brain_skills():
    """Fetch available skills from Fast Brain."""
    import httpx

    fast_brain_url = os.getenv("FAST_BRAIN_URL")
    if not fast_brain_url:
        return {"skills": [], "error": "Fast Brain not configured"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{fast_brain_url}/v1/skills")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"skills": [], "error": str(e)}


@app.post("/api/skills/fast-brain")
async def create_fast_brain_skill(skill: dict):
    """Create a custom skill in Fast Brain."""
    import httpx

    fast_brain_url = os.getenv("FAST_BRAIN_URL")
    if not fast_brain_url:
        return {"success": False, "error": "Fast Brain not configured"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{fast_brain_url}/v1/skills",
                json=skill
            )
            response.raise_for_status()
            return {"success": True, "skill": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### 4. Frontend Skills Display

**File:** `web/src/app/dashboard/developer/page.tsx`

Add Fast Brain skills section:

```tsx
// Add to the developer dashboard

const [fastBrainSkills, setFastBrainSkills] = useState([]);
const [fastBrainStatus, setFastBrainStatus] = useState('unknown');

useEffect(() => {
  // Fetch Fast Brain skills
  fetch('/api/skills/fast-brain')
    .then(res => res.json())
    .then(data => setFastBrainSkills(data.skills || []))
    .catch(console.error);

  // Fetch status
  fetch('/admin/status')
    .then(res => res.json())
    .then(data => setFastBrainStatus(data.fast_brain?.status || 'unknown'))
    .catch(console.error);
}, []);

// In the JSX:
<Card>
  <CardHeader>
    <CardTitle>Fast Brain LPU</CardTitle>
    <Badge variant={fastBrainStatus === 'online' ? 'success' : 'secondary'}>
      {fastBrainStatus}
    </Badge>
  </CardHeader>
  <CardContent>
    <h4>Available Skills</h4>
    <div className="grid grid-cols-2 gap-2">
      {fastBrainSkills.map(skill => (
        <div key={skill.id} className="p-2 border rounded">
          <strong>{skill.name}</strong>
          <p className="text-sm text-muted">{skill.description}</p>
        </div>
      ))}
    </div>
  </CardContent>
</Card>
```

---

## API Reference

### Fast Brain Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with skills list |
| `/v1/models` | GET | List available models |
| `/v1/skills` | GET | List all skills |
| `/v1/skills` | POST | Create custom skill |
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible) |

### Chat Completion Request

```json
{
  "messages": [
    {"role": "system", "content": "Optional override"},
    {"role": "user", "content": "User message"}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "skill": "receptionist",
  "user_profile": "Business name and details"
}
```

### Chat Completion Response

```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "fast-brain-groq",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Response text"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 30,
    "total_tokens": 80
  },
  "metrics": {
    "ttfb_ms": 79.7,
    "total_time_ms": 557.2,
    "tokens_per_sec": 213.2
  },
  "skill_used": "receptionist"
}
```

---

## Built-in Skills

| Skill ID | Name | Best For |
|----------|------|----------|
| `general` | General Assistant | Default, general queries |
| `receptionist` | Professional Receptionist | Phone answering, call screening |
| `electrician` | Electrician Assistant | Electrical service businesses |
| `plumber` | Plumber Assistant | Plumbing service businesses |
| `lawyer` | Legal Intake Assistant | Law firm intake calls |

### Using Skills

Pass the `skill` parameter in chat requests:

```python
response = await fast_brain.chat(
    messages=[{"role": "user", "content": "I need an electrician"}],
    skill="electrician",  # Uses electrician system prompt
    user_profile="Acme Electric - Philadelphia area"
)
```

---

## Environment Variables

### Required in Railway Worker

```bash
# Fast Brain (Primary LLM)
FAST_BRAIN_URL=https://[username]--fast-brain-lpu.modal.run

# Fallback LLMs
GROQ_API_KEY=your_groq_key
ANTHROPIC_API_KEY=your_anthropic_key

# Voice
DEEPGRAM_API_KEY=your_deepgram_key
CARTESIA_API_KEY=your_cartesia_key
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_key
LIVEKIT_API_SECRET=your_livekit_secret
```

---

## Testing

### 1. Test Fast Brain Directly

```bash
# Health check
curl https://[username]--fast-brain-lpu.modal.run/health

# Chat test
curl -X POST https://[username]--fast-brain-lpu.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, I need help"}],
    "skill": "receptionist"
  }'
```

### 2. Test from Hive215 Backend

```python
# In Python console
import httpx
import os

url = os.getenv("FAST_BRAIN_URL")
response = httpx.post(
    f"{url}/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "Test"}],
        "skill": "general"
    }
)
print(response.json())
```

### 3. Test Voice Flow

1. Start LiveKit worker with Fast Brain configured
2. Call Twilio number
3. Verify response latency is <500ms total

---

## Troubleshooting

### Fast Brain returns 503

**Cause:** Container is cold or Groq API key not set

**Fix:**
```bash
# Check Modal deployment
modal app list

# Check Groq secret
modal secret list
# Should show: groq-api-key

# Redeploy if needed
cd fast_brain_repo
modal deploy fast_brain/deploy_groq.py
```

### High latency (>500ms TTFB)

**Cause:** Cold start (first request warms container)

**Fix:**
- Send a warmup request on service start
- Set `container_idle_timeout=300` in Modal config (already done)

### Skills not loading

**Cause:** Network timeout or Fast Brain not reachable

**Fix:**
```python
# Add retry logic
async def get_skills_with_retry(url, retries=3):
    for i in range(retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{url}/v1/skills")
                return response.json()
        except:
            if i == retries - 1:
                raise
            await asyncio.sleep(1)
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              HIVE215 VOICE PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   [Caller] ──► [Twilio] ──► [LiveKit Worker] ──► [Fast Brain LPU] ──► [Response] │
│                                    │                    │                        │
│                                    │              ┌─────┴─────┐                  │
│                              [Deepgram STT]      │   Groq    │                  │
│                                    │             │ Llama 3.3 │                  │
│                              [Cartesia TTS]      │   70B     │                  │
│                                    │             └───────────┘                  │
│                                    ▼                    │                        │
│                           [Audio Response]    [Skills Database]                  │
│                                                         │                        │
│   ┌──────────────────────────────────────────────────────┘                      │
│   │                                                                              │
│   │  Skills: receptionist │ electrician │ plumber │ lawyer │ custom...         │
│   │                                                                              │
└───┴──────────────────────────────────────────────────────────────────────────────┘

Latency Breakdown:
- Deepgram STT: ~100ms
- Fast Brain LLM: ~80ms TTFB
- Cartesia TTS: ~40ms TTFB
- Total perceived: ~220ms (target <500ms ✅)
```

---

## Files in Fast Brain Repo

```
fast_brain/
├── deploy_groq.py      # Main deployment ← THIS IS WHAT'S DEPLOYED
├── deploy_bitnet.py    # BitNet attempt (didn't work)
├── deploy_simple.py    # Stub for testing
├── deploy.py           # Original design
└── ...

unified_dashboard.py    # Dashboard with skills management
README.md               # Main documentation
HIVE215_INTEGRATION.md  # This file
```

---

## Next Steps After Integration

1. **Add streaming support** - Currently returns full response, add SSE for token streaming
2. **Skill training UI** - Let users create skills from Hive215 dashboard
3. **Analytics** - Track which skills are used most, latency metrics
4. **A/B testing** - Test different skill prompts for better results
5. **RAG integration** - Upload documents to enhance skill knowledge

---

## Contact

**Fast Brain repo:** `jenkintownelectricity/fast_brain`
**Branch:** `claude/fast-brain-streaming-018wsUb2vztjfkc8EGVQ9t8j`

For questions, check the README.md in the Fast Brain repo or start a new Claude Code session with this context.
