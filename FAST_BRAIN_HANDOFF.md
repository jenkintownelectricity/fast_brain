# Fast Brain Integration Guide for HIVE215

> **Handoff Document** | Last Updated: December 18, 2025
>
> This document explains how HIVE215 Voice Platform integrates with Fast Brain LPU.
> Use this as the CLAUDE.md reference for the Fast Brain repository.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HIVE215 VOICE PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Speech ──▶ Deepgram STT ──▶ FAST BRAIN ──▶ Cartesia TTS ──▶ Audio   │
│                     (~30ms)        (~40ms)         (~30ms)                  │
│                                       │                                     │
│                                       ▼                                     │
│                              ┌────────────────┐                            │
│                              │  Skill Router  │                            │
│                              │  (by skill_id) │                            │
│                              └────────────────┘                            │
│                                       │                                     │
│                    ┌──────────────────┼──────────────────┐                 │
│                    ▼                  ▼                  ▼                 │
│              ┌──────────┐      ┌──────────┐      ┌──────────┐             │
│              │ tara-    │      │ default  │      │ custom   │             │
│              │ sales    │      │          │      │ skills   │             │
│              └──────────┘      └──────────┘      └──────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Fast Brain's Role:** Receive user text, apply skill-specific system prompt, generate response via Groq LPU.

---

## 2. Required API Endpoints

HIVE215 expects these endpoints from Fast Brain:

### 2.1 Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "backend": "groq-llama-3.3-70b",
  "skills_available": ["default", "receptionist", "tara-sales"],
  "version": "1.0.0"
}
```

**Used by:** HIVE215 worker on startup to verify Fast Brain is available.

---

### 2.2 Chat Completions (OpenAI-Compatible)
```
POST /v1/chat/completions
```

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What is The Dash?"}
  ],
  "skill": "tara-sales",
  "max_tokens": 256,
  "temperature": 0.7,
  "user_profile": "optional business context"
}
```

**Response:**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The Dash is a service that connects all your business tools..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 45,
    "total_tokens": 95
  },
  "skill_used": "tara-sales",
  "metrics": {
    "ttfb_ms": 42,
    "tokens_per_sec": 450
  }
}
```

**Key Points:**
- `skill` parameter selects which system prompt to use
- Must be OpenAI-compatible for fallback to work
- `metrics` object is Fast Brain-specific (optional but helpful)

---

### 2.3 List Skills
```
GET /v1/skills
```

**Response:**
```json
{
  "skills": [
    {
      "id": "default",
      "name": "Default Assistant",
      "description": "General-purpose voice assistant",
      "version": "1.0"
    },
    {
      "id": "tara-sales",
      "name": "Tara's Sales Assistant",
      "description": "Sales agent for TheDashTool demos",
      "version": "1.0"
    }
  ]
}
```

---

### 2.4 Create Skill
```
POST /v1/skills
```

**Request:**
```json
{
  "skill_id": "tara-sales",
  "name": "Tara's Sales Assistant",
  "description": "Sales agent for TheDashTool demos",
  "system_prompt": "You are Tara, founder of The Dash...",
  "knowledge": [
    "TheDashTool.com is the website",
    "Tara Horn is the founder",
    "..."
  ]
}
```

**Response:**
```json
{
  "success": true,
  "skill_id": "tara-sales",
  "message": "Skill created successfully"
}
```

---

## 3. Skill Structure

A skill defines an agent's personality and knowledge. Structure:

```python
skill = {
    "skill_id": "unique-identifier",      # Used in API calls
    "name": "Human Readable Name",         # Display name
    "description": "What this skill does", # For documentation
    "system_prompt": """                   # The core personality
        You are [Name], [role/title].

        ## Your Personality
        - [trait 1]
        - [trait 2]

        ## Your Knowledge
        - [fact 1]
        - [fact 2]

        ## Response Style
        - Keep responses concise (2-3 sentences)
        - Be conversational, not robotic
        ...
    """,
    "knowledge": [                         # Quick facts for RAG/context
        "Company website: example.com",
        "Hours: 9-5 Mon-Fri",
        "..."
    ]
}
```

---

## 4. Pre-loaded Skills

### 4.1 Default Skill
```python
DEFAULT_SKILL = {
    "skill_id": "default",
    "name": "Default Assistant",
    "system_prompt": """You are a helpful, friendly voice assistant.

Guidelines:
- Keep responses concise and conversational (1-2 sentences when possible)
- Be natural and engaging, like talking to a friend
- Ask clarifying questions when needed
- If you don't know something, say so honestly
- Match the user's energy and communication style"""
}
```

### 4.2 Receptionist Skill
```python
RECEPTIONIST_SKILL = {
    "skill_id": "receptionist",
    "name": "Business Receptionist",
    "system_prompt": """You are a professional receptionist for a business.

Your role:
- Answer calls warmly and professionally
- Collect caller name and reason for calling
- Offer to take a message or transfer the call
- Provide basic business information (hours, location)

Keep responses brief and efficient. Always confirm you understood correctly."""
}
```

### 4.3 Tara Sales Skill (TheDashTool)
```python
TARA_SALES_SKILL = {
    "skill_id": "tara-sales",
    "name": "Tara's Sales Assistant",
    "description": "Sales assistant for TheDashTool demos",
    "system_prompt": """You are Tara, the founder of The Dash (TheDashTool.com). You're a workflow optimization expert who has helped nearly 200 companies improve their operational efficiency since 2015.

## Your Personality
- Warm, friendly, and genuinely curious about businesses
- Confident but not pushy - you ask questions and listen
- You speak conversationally, not like a salesperson reading a script
- You understand the pain of data chaos because you've seen it hundreds of times
- You're enthusiastic about helping businesses get clarity

## About The Dash
The Dash is a complete BI dashboard service (not just software) that:
- Connects ALL your business tools into one unified dashboard
- Provides AI-powered insights that anticipate what's next
- Is fully custom-built for each business - no one-size-fits-all
- Does all the technical work FOR the client

## Key Differentiators
1. "We do the work FOR you" - Unlike other tools, clients don't figure things out themselves
2. "Built to grow with you" - Ongoing support, dashboards evolve with the business
3. "We understand business, not just technology" - We speak their language
4. "No data science degree required" - Clarity without complexity

## The Process
1. Map Your Business - Learn goals, team, tools, what matters most
2. Connect Your Tools - CRM, accounting, project management, marketing, ticketing - everything
3. Design Your Dashboards - Custom metrics, uncover what's missing, visualize the gaps
4. AI Insights - Anticipate patterns, predict risks, highlight opportunities

## Pricing Approach
- Don't quote specific prices (pricing is custom based on complexity)
- If asked, say: "Pricing depends on how many tools you're connecting and the complexity of your dashboards. The best way to get a clear picture is to book a quick demo where we can learn about your specific situation."

## Your Goal
Help prospects understand how The Dash can give them clarity, and guide them to book a free demo.

## Demo Booking
When interested, say: "That's great! The easiest next step is to book a free demo. You can do that at thedashtool.com, or I can have someone from the team reach out to you directly. Which would you prefer?"

## Handling Objections
- "We already use [tool]": "That's actually perfect - The Dash connects TO those tools. We don't replace them, we unify them."
- "We don't have time": "That's exactly why we do the work for you. Our team handles all the setup."
- "We're too small": "We work with businesses of all sizes. Getting clarity early helps you scale smarter."
- "It sounds expensive": "I understand. That's why we do a free demo first - so you can see exactly what you'd get."

## Response Style
- Keep responses conversational and concise (2-3 sentences usually)
- Ask follow-up questions to understand their situation
- Use "you" and "your business" - make it personal
- Avoid jargon - speak plainly
- Sound like you're having a friendly conversation, not giving a presentation""",

    "knowledge": [
        "TheDashTool.com is the website. Email is info@thedashtool.com",
        "Tara Horn is the founder and workflow optimization expert since 2015",
        "The Dash has helped nearly 200 companies improve operational efficiency",
        "Industries served: Finance & Banking, Healthcare, Retail, Manufacturing, Professional Services",
        "The Dash integrates with: CRM systems, accounting software, project management tools, marketing platforms, ticketing systems",
        "Demo booking: Free demo available at thedashtool.com or by phone",
        "Operating hours: Mon-Fri 9:00AM - 5:00PM, Sat-Sun 10:00AM - 6:00PM",
        "YouTube channel: youtube.com/@thedashtool"
    ]
}
```

---

## 5. HIVE215 Brain Client

HIVE215 uses this client to communicate with Fast Brain:

```python
# backend/brain_client.py (in HIVE215 repo)

class FastBrainClient:
    def __init__(
        self,
        base_url: str,              # Fast Brain URL
        default_skill: str = "default",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        ...

    async def is_healthy(self) -> bool:
        """Check GET /health returns status: healthy"""

    async def think(self, user_input: str, skill: str = None) -> ThinkResponse:
        """Single-turn request to POST /v1/chat/completions"""

    async def chat(self, messages: list, skill: str = None) -> ThinkResponse:
        """Multi-turn request to POST /v1/chat/completions"""

    async def list_skills(self) -> list[Skill]:
        """GET /v1/skills"""

    async def create_skill(self, skill_id, name, description, system_prompt, knowledge) -> dict:
        """POST /v1/skills"""
```

---

## 6. Environment Variables

### Fast Brain (Modal) needs:
```bash
GROQ_API_KEY=gsk_xxxxx           # Groq API for LPU inference
```

### HIVE215 Worker sends to Fast Brain:
```bash
FAST_BRAIN_URL=https://your-app--fast-brain-lpu-xxx.modal.run
DEFAULT_SKILL=tara-sales         # Which skill to use by default
```

---

## 7. Fallback Chain

If Fast Brain is unavailable, HIVE215 falls back:

```
Fast Brain (primary) → Groq Direct (fallback 1) → Anthropic Claude (fallback 2)
```

Fast Brain should aim for:
- **TTFB < 50ms** (time to first byte)
- **99.9% uptime**
- **Graceful error responses** (never hang)

---

## 8. Modal Deployment

Fast Brain is deployed on Modal. Typical structure:

```
fast_brain/
├── modal_app.py          # Modal app definition
├── api.py                # FastAPI endpoints
├── skills/
│   ├── __init__.py
│   ├── default.py        # Default skill
│   ├── receptionist.py   # Receptionist skill
│   └── tara_sales.py     # Tara sales skill (TheDashTool)
├── llm.py                # Groq LPU integration
└── config.py             # Configuration
```

---

## 9. Testing Integration

To verify Fast Brain works with HIVE215:

```bash
# 1. Health check
curl https://your-fast-brain.modal.run/health

# 2. List skills
curl https://your-fast-brain.modal.run/v1/skills

# 3. Test chat completion
curl -X POST https://your-fast-brain.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, what is The Dash?"}],
    "skill": "tara-sales",
    "max_tokens": 100
  }'
```

---

## 10. Adding New Skills

To add a new skill:

1. **Create skill definition** (system_prompt + knowledge)
2. **Register via API** or add to preloaded skills
3. **Update HIVE215** `DEFAULT_SKILL` environment variable
4. **Test** with curl or via HIVE215 voice interface

---

## 11. Performance Requirements

| Metric | Target | Why |
|--------|--------|-----|
| TTFB | < 50ms | Voice needs instant response |
| Total Latency | < 200ms | Part of 300ms total pipeline budget |
| Tokens/sec | > 400 | Groq LPU capability |
| Uptime | 99.9% | Production voice service |

---

## 12. Quick Reference

**Fast Brain URL Format:**
```
https://[username]--fast-brain-lpu-[app-name].modal.run
```

**HIVE215 Repo:** `github.com/jenkintownelectricity/premier_voice_assistant`

**Key Files in HIVE215:**
- `backend/brain_client.py` - Fast Brain client
- `backend/livekit_agent.py` - Voice agent (uses Fast Brain)
- `skills/tara_sales.py` - Tara skill definition

**Documentation:**
- `docs/visuals/2025-12-18_voice_agent_guide.html` - User guide for creating agents
- `CLAUDE.md` - Project instructions for HIVE215

---

*Use this document as the CLAUDE.md reference for the Fast Brain repository.*
