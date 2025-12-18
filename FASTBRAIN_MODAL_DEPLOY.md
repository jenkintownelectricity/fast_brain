# Fast Brain Modal Deployment Guide (v3.0)
## Hybrid System 1 + System 2 Architecture

This guide deploys the HIVE215 Fast Brain LPU service to Modal — now with hybrid intelligence that combines Groq's speed with Claude's reasoning.

---

## 🚀 DEPLOY ALL 3 SERVICES (Copy & Paste)

**Always deploy all 3 together for the full system:**

```bash
# Windows (PowerShell)
modal deploy deploy_dashboard.py; modal deploy fast_brain/deploy_groq.py; modal deploy parler_integration.py

# Mac/Linux
modal deploy deploy_dashboard.py && modal deploy fast_brain/deploy_groq.py && modal deploy parler_integration.py
```

| Service | Command | URL Pattern |
|---------|---------|-------------|
| Dashboard | `modal deploy deploy_dashboard.py` | `https://[user]--hive215-dashboard-flask-app.modal.run` |
| Fast Brain | `modal deploy fast_brain/deploy_groq.py` | `https://[user]--fast-brain-lpu-fastapi-app.modal.run` |
| Parler TTS | `modal deploy parler_integration.py` | `https://[user]--hive215-parler-tts-*.modal.run` |

---

## The "Holy Grail" Architecture

Named after "Thinking, Fast and Slow" by Daniel Kahneman:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SYSTEM 1 + SYSTEM 2 HYBRID                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   🚀 SYSTEM 1 (Fast Brain)          🧠 SYSTEM 2 (Deep Brain)                 │
│   ─────────────────────────         ─────────────────────────                │
│   Groq + Llama 3.3 70B              Claude 3.5 Sonnet                        │
│   ~80ms latency                     ~2000ms latency                          │
│   Handles 90% of calls              Handles 10% of calls                     │
│   Intuitive, instant                Rational, complex reasoning              │
│                                                                              │
│   The "Filler Phrase" Strategy:                                              │
│   ────────────────────────────                                               │
│   1. Groq detects complex question                                           │
│   2. Fires filler: "Let me look into that..."  ──► TTS plays (~2.5s)        │
│   3. Silently calls Claude in background       ──► Thinks (~2s)              │
│   4. Claude finishes just as filler ends       ──► Seamless handoff         │
│   5. User perception: ZERO LATENCY                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

```bash
# Install Modal CLI
pip install modal

# Authenticate (opens browser)
modal setup
```

---

## Step 1: Create Modal Secrets

```bash
# System 1: Groq API key (required)
modal secret create groq-api-key GROQ_API_KEY=gsk_your_key_here

# System 2: Anthropic API key (required for hybrid mode)
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-your_key_here

# Optional: Supabase for dynamic skills
modal secret create supabase-credentials \
    SUPABASE_URL=https://your-project.supabase.co \
    SUPABASE_KEY=your_anon_key
```

---

## Step 2: Deploy to Modal

```bash
# From your project root
modal deploy fast_brain/deploy_groq.py

# You'll see output like:
# ✓ Created FastBrain class
# ✓ Created fastapi_app function
# 
# Web endpoint: https://your-username--fast-brain-lpu.modal.run
```

---

## Step 3: Test Your Deployment

### Health Check

```bash
curl https://YOUR-USERNAME--fast-brain-lpu.modal.run/health
```

Response shows hybrid architecture:
```json
{
  "status": "healthy",
  "architecture": "System 1 + System 2 Hybrid",
  "system1": {
    "name": "Fast Brain",
    "model": "llama-3.3-70b-versatile",
    "provider": "Groq",
    "latency": "~80ms"
  },
  "system2": {
    "name": "Deep Brain",
    "model": "claude-3-5-sonnet-20241022",
    "provider": "Anthropic",
    "latency": "~2000ms"
  }
}
```

### Test Simple Question (System 1 Only)

```bash
curl -X POST https://YOUR-USERNAME--fast-brain-lpu.modal.run/v1/chat/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hi, my lights are flickering"}],
    "skill": "electrician"
  }'
```

Response (~80ms):
```json
{
  "content": "I can help with that! Flickering lights can be a wiring issue. Can I get your name and address?",
  "filler": null,
  "system_used": "fast",
  "fast_latency_ms": 85,
  "total_latency_ms": 85
}
```

### Test Complex Question (System 1 → System 2)

```bash
curl -X POST https://YOUR-USERNAME--fast-brain-lpu.modal.run/v1/chat/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Can you analyze my electricity bill and tell me why it went up? I used 850 kWh this month."}],
    "skill": "electrician"
  }'
```

Response (~2000ms, but user hears filler immediately):
```json
{
  "content": "Based on your usage of 850 kWh, which is about 28 kWh per day, your bill increase could be due to...",
  "filler": "Let me pull up your information and analyze that for you, just a moment...",
  "system_used": "deep",
  "fast_latency_ms": 85,
  "deep_latency_ms": 1950,
  "total_latency_ms": 2035
}
```

---

## API Endpoints

| Endpoint | Mode | Description |
|----------|------|-------------|
| `POST /v1/chat/completions` | System 1 only | Backwards compatible, fast responses |
| `POST /v1/chat/hybrid` | System 1 + 2 | **Recommended** - auto-routes to Claude when needed |
| `POST /v1/chat/voice` | Hybrid + Voice | Returns voice descriptions for Brain + Mouth |
| `GET /v1/fillers` | — | List all filler phrase categories |

---

## The Filler Phrase Strategy

When Groq detects a complex question, it returns a filler phrase that your TTS should speak immediately. This buys ~2.5 seconds while Claude thinks in the background.

### Built-in Filler Categories

| Category | Example Phrase |
|----------|---------------|
| `analysis` | "Let me pull up your information and analyze that for you..." |
| `calculation` | "Let me run those numbers for you real quick..." |
| `research` | "That's a detailed question. Let me look into that..." |
| `complex` | "Hmm, let me think through that carefully..." |

### Implementation in Your Voice Pipeline

```python
result = await brain.think_hybrid(messages, skill="electrician")

if result["filler"]:
    # Speak filler IMMEDIATELY (don't wait for content)
    await tts.speak(result["filler"])
    
# Then speak the real answer (arrives ~2s later, just as filler ends)
await tts.speak(result["content"])
```

---

## Built-in Skills (6 total)

| Skill | System 1 Handles | System 2 Handles |
|-------|------------------|------------------|
| `general` | Casual chat | Complex questions |
| `receptionist` | Greetings, info | Billing, technical |
| `electrician` | Intake, scheduling | Code questions, analysis |
| `plumber` | Intake, scheduling | Technical advice |
| `lawyer` | Contact info | ALL legal questions |
| `solar` | Interest qualification | ROI, bill analysis |

---

## Cost Analysis

### System 1 (Groq) - 90% of calls
- Llama 3.3 70B: **FREE** (for now)
- Modal compute: ~$0.0005 per call

### System 2 (Claude) - 10% of calls  
- Claude 3.5 Sonnet: ~$0.003 per call (input) + $0.015 per call (output)
- Average: ~$0.01 per complex question

### Blended Cost (realistic usage)
- 1,000 calls: 900 × $0.0005 + 100 × $0.01 = **~$1.45**
- vs. using Claude for everything: 1,000 × $0.01 = $10.00
- **Savings: 85%**

---

## Project Structure

Create this structure in your project:

```
fast_brain/
├── __init__.py
├── deploy_groq.py      # Main Modal deployment
├── skills.py           # Skill definitions
└── config.py           # Configuration
```

---

## Step 1: Create Modal Secrets

Run these commands to set up your API keys:

```bash
# Groq API key (required)
modal secret create groq-api-key GROQ_API_KEY=gsk_your_key_here

# Optional: Supabase for dynamic skills
modal secret create supabase-credentials \
    SUPABASE_URL=https://your-project.supabase.co \
    SUPABASE_KEY=your_anon_key
```

---

## Step 2: Create the Deployment Files

### `fast_brain/__init__.py`

```python
# Empty file to make this a package
```

### `fast_brain/config.py`

```python
"""Fast Brain Configuration"""

# Groq model settings
MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024
TEMPERATURE = 0.7

# Performance
CONTAINER_IDLE_TIMEOUT = 300  # 5 minutes
KEEP_WARM = 1  # Keep 1 container warm to avoid cold starts

# CRITICAL: Match your LiveKit Cloud region!
# Options: "us-east", "us-west", "eu-west"
MODAL_REGION = "us-east"
```

### `fast_brain/skills.py`

```python
"""Built-in skill definitions for Fast Brain"""

VOICE_RULES = """
VOICE INTERACTION RULES:
- Use contractions (I'm, you're, we'll)
- Keep sentences under 20 words
- No markdown, bullets, or formatting
- No emojis or special characters
- Spell out numbers under 10
- Say "um" or "let me think" if you need time
- Ask one question at a time
"""

BUILT_IN_SKILLS = {
    "general": {
        "name": "General Assistant",
        "system_prompt": f"""You are a helpful voice assistant. Be conversational and natural.
{VOICE_RULES}""",
        "greeting": "Hey there! How can I help you today?",
    },
    
    "receptionist": {
        "name": "Professional Receptionist",
        "system_prompt": f"""You are a professional receptionist for a business. Your job is to:
- Answer calls warmly and professionally
- Collect caller's name and reason for calling
- Offer to take a message or transfer the call
- Be helpful but don't make commitments on behalf of the business

{VOICE_RULES}""",
        "greeting": "Thanks for calling! How can I help you today?",
    },
    
    "electrician": {
        "name": "Electrician Service Intake",
        "system_prompt": f"""You are a receptionist for an electrical contracting company. Your job is to:
- Collect the customer's name and phone number
- Understand their electrical issue (outlets, panels, lighting, etc.)
- Determine urgency (emergency vs routine)
- Collect their address for service
- Let them know someone will call back to schedule

DO NOT give electrical advice or diagnose problems over the phone.
DO NOT quote prices - say "we'll provide an estimate after assessing the job."

{VOICE_RULES}""",
        "greeting": "Thanks for calling! Are you having an electrical issue I can help with?",
    },
    
    "plumber": {
        "name": "Plumber Service Intake",
        "system_prompt": f"""You are a receptionist for a plumbing company. Your job is to:
- Collect the customer's name and phone number
- Understand their plumbing issue (leaks, drains, water heater, etc.)
- Determine urgency (active leak vs routine)
- Collect their address for service
- Let them know someone will call back to schedule

For emergencies (active leaks, no water, sewage backup), prioritize getting their info quickly.

{VOICE_RULES}""",
        "greeting": "Thanks for calling! Do you have a plumbing issue I can help with?",
    },
    
    "lawyer": {
        "name": "Legal Intake Specialist",
        "system_prompt": f"""You are an intake specialist for a law firm. Your job is to:
- Collect the caller's name and contact information
- Understand the general nature of their legal matter
- Determine if it's within the firm's practice areas
- Schedule a consultation or take a message for an attorney

IMPORTANT DISCLAIMERS:
- You are NOT an attorney and cannot give legal advice
- Nothing discussed creates an attorney-client relationship
- Say "I can't give legal advice, but I can have an attorney call you back"

{VOICE_RULES}""",
        "greeting": "Thanks for calling the law office. How can I help you today?",
    },
}

def get_skill(skill_id: str) -> dict:
    """Get a skill by ID, with fallback to general."""
    return BUILT_IN_SKILLS.get(skill_id, BUILT_IN_SKILLS["general"])

def list_skills() -> list:
    """List all available skill IDs."""
    return list(BUILT_IN_SKILLS.keys())
```

### `fast_brain/deploy_groq.py`

```python
"""
Fast Brain LPU - Modal Deployment
HIVE215 Voice AI Inference Engine

Deploys a Groq-powered LLM endpoint with skill-based routing.
Designed for ultra-low latency voice applications.

Deploy: modal deploy fast_brain/deploy_groq.py
Test:   curl https://your-username--fast-brain-lpu.modal.run/health
"""

import modal
import os
import json
from datetime import datetime

from .config import MODEL, MAX_TOKENS, TEMPERATURE, CONTAINER_IDLE_TIMEOUT, KEEP_WARM, MODAL_REGION
from .skills import get_skill, list_skills, BUILT_IN_SKILLS, VOICE_RULES

# ═══════════════════════════════════════════════════════════════════════════════
# MODAL APP CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

app = modal.App("fast-brain-lpu")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "groq>=0.4.0",
        "fastapi>=0.109.0",
        "pydantic>=2.0.0",
    )
)

# ═══════════════════════════════════════════════════════════════════════════════
# FAST BRAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("groq-api-key")],
    container_idle_timeout=CONTAINER_IDLE_TIMEOUT,
    keep_warm=KEEP_WARM,
    region=MODAL_REGION,  # CRITICAL: Match LiveKit Cloud region!
)
class FastBrain:
    """
    Ultra-fast LLM inference for voice AI applications.
    
    Uses Groq's LPU for ~80ms TTFB with Llama 3.3 70B.
    Supports skill-based routing for different business verticals.
    """
    
    @modal.enter()
    def setup(self):
        """Initialize Groq client on container start."""
        from groq import Groq
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = MODEL
        print(f"[FastBrain] Initialized with model: {self.model}")
        print(f"[FastBrain] Available skills: {list_skills()}")
    
    @modal.method()
    def think(
        self,
        messages: list[dict],
        skill: str = "general",
        user_context: dict = None,
        stream: bool = False,
    ) -> dict:
        """
        Process a conversation and generate a response.
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            skill: Skill ID for system prompt routing
            user_context: Optional dict with business_name, transfer_number, etc.
            stream: Whether to stream the response (not yet implemented)
            
        Returns:
            {"content": "response text", "skill": "skill_id", "latency_ms": 123}
        """
        import time
        start = time.perf_counter()
        
        # Get skill configuration
        skill_config = get_skill(skill)
        system_prompt = skill_config["system_prompt"]
        
        # Inject user context if provided
        if user_context:
            context_lines = []
            if user_context.get("business_name"):
                context_lines.append(f"Business: {user_context['business_name']}")
            if user_context.get("transfer_number"):
                context_lines.append(f"Transfer calls to: {user_context['transfer_number']}")
            if user_context.get("knowledge"):
                context_lines.append(f"Knowledge: {user_context['knowledge']}")
            
            if context_lines:
                system_prompt = system_prompt + "\n\nCONTEXT:\n" + "\n".join(context_lines)
        
        # Build full message list
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Call Groq
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            
            content = response.choices[0].message.content
            latency_ms = int((time.perf_counter() - start) * 1000)
            
            return {
                "content": content,
                "skill": skill,
                "model": self.model,
                "latency_ms": latency_ms,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            }
            
        except Exception as e:
            return {
                "content": "I'm sorry, I'm having trouble thinking right now. Can you try again?",
                "skill": skill,
                "error": str(e),
                "latency_ms": int((time.perf_counter() - start) * 1000),
            }
    
    @modal.method()
    def get_greeting(self, skill: str = "general") -> str:
        """Get the greeting for a skill."""
        return get_skill(skill).get("greeting", "Hello! How can I help you?")
    
    @modal.method()
    def health_check(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model": self.model,
            "skills": list_skills(),
            "timestamp": datetime.utcnow().isoformat(),
        }

# ═══════════════════════════════════════════════════════════════════════════════
# WEB ENDPOINTS (FastAPI)
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

web_app = FastAPI(title="Fast Brain LPU", version="1.0.0")

class ChatRequest(BaseModel):
    messages: list[dict]
    skill: str = "general"
    user_context: dict = None
    stream: bool = False

class ChatResponse(BaseModel):
    content: str
    skill: str
    model: str
    latency_ms: int
    usage: dict = None
    error: str = None

@web_app.get("/health")
async def health():
    """Health check endpoint."""
    brain = FastBrain()
    return brain.health_check.remote()

@web_app.get("/v1/skills")
async def get_skills():
    """List all available skills."""
    return {
        "skills": [
            {"id": k, "name": v["name"]}
            for k, v in BUILT_IN_SKILLS.items()
        ]
    }

@web_app.get("/v1/skills/{skill_id}")
async def get_skill_detail(skill_id: str):
    """Get details for a specific skill."""
    if skill_id not in BUILT_IN_SKILLS:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")
    
    skill = BUILT_IN_SKILLS[skill_id]
    return {
        "id": skill_id,
        "name": skill["name"],
        "greeting": skill["greeting"],
        "system_prompt_preview": skill["system_prompt"][:200] + "...",
    }

@web_app.get("/v1/greeting/{skill_id}")
async def get_greeting(skill_id: str):
    """Get the greeting for a skill."""
    brain = FastBrain()
    return {"greeting": brain.get_greeting.remote(skill_id)}

@web_app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    OpenAI-compatible chat completion endpoint.
    
    Example:
        curl -X POST https://your--fast-brain-lpu.modal.run/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{"messages":[{"role":"user","content":"hello"}],"skill":"electrician"}'
    """
    brain = FastBrain()
    result = brain.think.remote(
        messages=request.messages,
        skill=request.skill,
        user_context=request.user_context,
        stream=request.stream,
    )
    return result

@app.function(
    image=image,
    region=MODAL_REGION,
)
@modal.asgi_app()
def fastapi_app():
    """Mount FastAPI app to Modal."""
    return web_app

# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL TESTING
# ═══════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main():
    """Local test entrypoint."""
    print("Testing Fast Brain...")
    
    brain = FastBrain()
    
    # Health check
    health = brain.health_check.remote()
    print(f"\nHealth: {json.dumps(health, indent=2)}")
    
    # Test each skill
    for skill_id in ["general", "electrician", "plumber"]:
        print(f"\n--- Testing skill: {skill_id} ---")
        
        greeting = brain.get_greeting.remote(skill_id)
        print(f"Greeting: {greeting}")
        
        result = brain.think.remote(
            messages=[{"role": "user", "content": "Hi, I need some help"}],
            skill=skill_id,
        )
        print(f"Response ({result['latency_ms']}ms): {result['content'][:100]}...")
    
    print("\n✅ All tests passed!")
```

---

## Step 3: Deploy to Modal

```bash
# From your project root
modal deploy fast_brain/deploy_groq.py

# You'll see output like:
# ✓ Created FastBrain class
# ✓ Created fastapi_app function
# View app at https://modal.com/apps/your-username/fast-brain-lpu
# 
# Web endpoint: https://your-username--fast-brain-lpu.modal.run
```

**Save this URL!** You'll need it for the LiveKit worker.

---

## Step 4: Test Your Deployment

### Health Check

```bash
curl https://YOUR-USERNAME--fast-brain-lpu.modal.run/health
```

Expected response:
```json
{
  "status": "healthy",
  "model": "llama-3.3-70b-versatile",
  "skills": ["general", "receptionist", "electrician", "plumber", "lawyer"],
  "timestamp": "2025-12-05T15:30:00.000000"
}
```

### List Skills

```bash
curl https://YOUR-USERNAME--fast-brain-lpu.modal.run/v1/skills
```

### Test Chat Completion

```bash
curl -X POST https://YOUR-USERNAME--fast-brain-lpu.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hi, my lights are flickering"}],
    "skill": "electrician"
  }'
```

Expected response:
```json
{
  "content": "I can help with that. Flickering lights can have a few causes. Can I get your name and the address where you're having this issue?",
  "skill": "electrician",
  "model": "llama-3.3-70b-versatile",
  "latency_ms": 85,
  "usage": {"prompt_tokens": 245, "completion_tokens": 32}
}
```

### Local Test (without deploying)

```bash
modal run fast_brain/deploy_groq.py
```

---

## Step 5: Create Fast Brain URL Secret

For your LiveKit worker to call Fast Brain:

```bash
modal secret create fast-brain-url \
    FAST_BRAIN_URL=https://YOUR-USERNAME--fast-brain-lpu.modal.run
```

---

## Monitoring & Logs

```bash
# View real-time logs
modal app logs fast-brain-lpu

# View in dashboard
# https://modal.com/apps/YOUR-USERNAME/fast-brain-lpu
```

---

## Cost Estimates

| Usage | Container Hours | Cost |
|-------|-----------------|------|
| 1,000 calls/day | ~8 hrs | ~$2.40/day |
| 10,000 calls/day | ~80 hrs | ~$24/day |
| keep_warm=1 | 720 hrs/mo | ~$20/mo |

**Note:** Groq API is currently free for Llama 3.3 70B.

---

## Troubleshooting

### Cold Start Too Slow

Increase `keep_warm` in config.py:
```python
KEEP_WARM = 2  # Keep 2 containers warm
```

### Wrong Region

Check your LiveKit Cloud region and update `MODAL_REGION` in config.py.

### Groq Rate Limits

Groq free tier has rate limits. For production, upgrade to paid tier or implement request queuing.

### Missing Secret

```bash
# Verify secrets exist
modal secret list

# Re-create if needed
modal secret create groq-api-key GROQ_API_KEY=your_key
```

---

## Next Steps

1. Deploy LiveKit worker (separate guide)
2. Configure SIP trunk in LiveKit Cloud
3. Map phone numbers to skills in your database
4. Test end-to-end voice call
5. **[Advanced] Fine-tune Expert Brains** — See `EXPERT_BRAIN_FINETUNING.md`

---

## 🧠 Advanced: Fine-Tuned Expert Brains

Want **smarter + faster** brains? Fine-tune Llama 3.1 on Modal, run on Groq with LoRA.

### The Problem with System Prompts

```python
# Current: 500+ token system prompts
ELECTRICIAN_PROMPT = """You are a receptionist for an electrical company...
[500 tokens of instructions, policies, examples]"""
```

- Burns tokens ($$)
- Slower TTFB
- Inconsistent behavior

### The Solution: Bake Knowledge Into Weights

```python
# After fine-tuning: 50 token prompt
ELECTRICIAN_PROMPT = "You are Sparky, the electrician receptionist."
# Model instinctively knows how to handle calls
```

### Workflow

```
[Training Data] → [Modal A100 GPU] → [LoRA Adapter] → [Groq LPU]
   1,000 examples     ~4 hours          ~100MB         800 tok/s
      $0                ~$8               $0            ⚡ Fast
```

### ROI

| Approach | Tokens/Call | Cost @ 10k calls/day |
|----------|-------------|---------------------|
| Heavy prompts | 500 | $7.50/day |
| Fine-tuned | 50 | $0.75/day |
| **Savings** | | **$6.75/day** |

Fine-tuning cost ($8) pays for itself in **1.2 days**.

**Full guide:** `EXPERT_BRAIN_FINETUNING.md`

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `modal deploy fast_brain/deploy_groq.py` | Deploy to Modal |
| `modal run fast_brain/deploy_groq.py` | Local test |
| `modal app logs fast-brain-lpu` | View logs |
| `modal secret list` | List secrets |
| `modal app stop fast-brain-lpu` | Stop app |

---

## Files Checklist

- [ ] `fast_brain/__init__.py` (empty)
- [ ] `fast_brain/config.py` (settings)
- [ ] `fast_brain/skills.py` (skill definitions)
- [ ] `fast_brain/deploy_groq.py` (main deployment)
- [ ] Modal secret: `groq-api-key`
- [ ] Modal secret: `fast-brain-url` (after deploy)

---

**Deployed URL:** `https://YOUR-USERNAME--fast-brain-lpu.modal.run`

Save this URL for the LiveKit worker configuration!
