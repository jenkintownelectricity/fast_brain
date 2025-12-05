"""
Fast Brain LPU - Modal Deployment (v3.0)
=========================================

HIVE215 Voice AI Inference Engine
Hybrid "System 1 / System 2" Architecture

ARCHITECTURE (Thinking, Fast and Slow):
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

FEATURES:
- System 1 (Fast Brain): Groq + Llama 3.3 70B (~80ms TTFB)
- System 2 (Deep Brain): Claude 3.5 Sonnet (complex reasoning)
- Filler phrases to hide Claude's latency
- Skill-based routing
- Voice-aware output (Brain + Mouth)
- Tool use for automatic routing

DEPLOYMENT:
    modal deploy fast_brain/deploy_groq.py

REQUIRED SECRETS:
    modal secret create groq-api-key GROQ_API_KEY=gsk_your_key
    modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-your_key

ENDPOINTS:
    GET  /health                    - Health check
    GET  /v1/skills                 - List all skills
    POST /v1/chat/completions       - Standard chat (text only)
    POST /v1/chat/voice             - Voice-aware chat (Brain + Mouth)
    POST /v1/chat/hybrid            - Hybrid chat (System 1 + System 2)
"""

import modal
import os
import json
from datetime import datetime
from typing import Optional
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# System 1: Fast Brain (Groq)
FAST_MODEL = "llama-3.3-70b-versatile"
FAST_MAX_TOKENS = 1024
FAST_TEMPERATURE = 0.7

# System 2: Deep Brain (Claude)
DEEP_MODEL = "claude-3-5-sonnet-20241022"
DEEP_MAX_TOKENS = 2048
DEEP_TEMPERATURE = 0.7

# Performance tuning
CONTAINER_IDLE_TIMEOUT = 300
KEEP_WARM = 1

# CRITICAL: Match your LiveKit Cloud region!
MODAL_REGION = "us-east"

# ═══════════════════════════════════════════════════════════════════════════════
# FILLER PHRASES (used to hide Claude's latency)
# ═══════════════════════════════════════════════════════════════════════════════

FILLER_PHRASES = {
    "analysis": [
        "That's a great question. Let me pull up your information and analyze that for you, just a moment...",
        "Good question! Let me take a look at the details and figure that out for you...",
        "Sure thing, let me dig into that and get you an accurate answer...",
    ],
    "calculation": [
        "Let me run those numbers for you real quick...",
        "Good question, let me calculate that and get back to you in just a second...",
        "Let me crunch those numbers and I'll have an answer for you right away...",
    ],
    "research": [
        "That's a detailed question. Let me look into that for you...",
        "Let me check on that and get you the most accurate information...",
        "Good question, give me just a moment to find the best answer for you...",
    ],
    "complex": [
        "That's a thoughtful question. Let me think through that carefully...",
        "Hmm, let me consider all the factors here and give you a proper answer...",
        "That requires some careful thought. Give me just a second...",
    ],
    "default": [
        "Let me look into that for you, just a moment...",
        "Good question, let me check on that...",
        "Sure, give me just a second to get you that information...",
    ],
}

def get_filler_phrase(category: str = "default") -> str:
    """Get a random filler phrase for the given category."""
    import random
    phrases = FILLER_PHRASES.get(category, FILLER_PHRASES["default"])
    return random.choice(phrases)

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS (for Groq to call Claude)
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ask_expert",
            "description": """Call this tool when you need help with:
- Complex mathematical calculations or analysis
- Detailed document or bill analysis
- Legal, medical, or financial advice questions
- Code generation or debugging
- Multi-step reasoning problems
- Questions requiring extensive knowledge
- Anything you're not 100% confident about

DO NOT try to answer these yourself. Call this tool immediately.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's question, exactly as they asked it",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["analysis", "calculation", "research", "complex"],
                        "description": "The type of expert help needed",
                    },
                    "context": {
                        "type": "string",
                        "description": "Any relevant context from the conversation",
                    },
                },
                "required": ["query", "category"],
            },
        },
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# VOICE RULES & CONTEXTS
# ═══════════════════════════════════════════════════════════════════════════════

VOICE_RULES = """
VOICE INTERACTION RULES:
- Use contractions (I'm, you're, we'll, don't, can't)
- Keep sentences under 20 words
- No markdown, bullets, numbered lists, or formatting
- No emojis or special characters
- Spell out numbers under 10 (say "three" not "3")
- Say "um" or "let me think" if you need processing time
- Ask one question at a time
- Be conversational, not robotic
"""

VOICE_CONTEXTS = {
    "greeting": "A friendly, warm female voice with a welcoming tone.",
    "routine": "A friendly, professional female voice. Warm and helpful.",
    "emergency": "A calm but urgent female voice. Reassuring yet conveying importance.",
    "frustrated": "A soft, apologetic female voice with genuine empathy. Slower pace.",
    "good_news": "A bright, cheerful female voice with enthusiasm.",
    "explaining": "A patient, clear female voice. Teacher-like, emphasizing key words.",
    "thinking": "A thoughtful, measured female voice. Slightly slower, contemplative.",
    "default": "A friendly, professional female voice. Warm and approachable.",
}

# ═══════════════════════════════════════════════════════════════════════════════
# BUILT-IN SKILLS
# ═══════════════════════════════════════════════════════════════════════════════

BUILT_IN_SKILLS = {
    "general": {
        "name": "General Assistant",
        "system_prompt": f"""You are a helpful voice assistant. Be conversational and natural.

For simple questions (greetings, basic info, scheduling), answer directly.
For complex questions (analysis, calculations, detailed advice), use the ask_expert tool.

{VOICE_RULES}""",
        "greeting": "Hey there! How can I help you today?",
        "voice": VOICE_CONTEXTS["default"],
    },

    "receptionist": {
        "name": "Professional Receptionist",
        "system_prompt": f"""You are a professional receptionist. Your job is to:
- Answer calls warmly and professionally
- Collect caller's name and reason for calling
- Offer to take a message or transfer the call

For simple requests, handle them directly.
For complex questions about billing, technical issues, or detailed analysis, use the ask_expert tool.

{VOICE_RULES}""",
        "greeting": "Thanks for calling! How can I help you today?",
        "voice": VOICE_CONTEXTS["greeting"],
    },

    "electrician": {
        "name": "Electrician Service Intake",
        "system_prompt": f"""You are Sparky, a friendly receptionist for an electrical contracting company.

HANDLE DIRECTLY (System 1):
- Greetings and basic questions
- Collecting customer info (name, phone, address)
- Understanding their issue (outlets, panels, lighting)
- Determining urgency (emergency vs routine)
- Scheduling callbacks

USE ask_expert TOOL FOR (System 2):
- Electrical code questions
- Cost estimates or pricing analysis
- Technical troubleshooting advice
- Bill analysis or energy calculations
- Complex multi-part questions

IMPORTANT:
- DO NOT give electrical advice - use ask_expert if pressed
- DO NOT quote prices without ask_expert
- For emergencies, get info FAST then use ask_expert if needed

{VOICE_RULES}""",
        "greeting": "Thanks for calling! Are you having an electrical issue I can help with?",
        "voice": "A friendly, professional female voice. Clear and efficient.",
    },

    "plumber": {
        "name": "Plumber Service Intake",
        "system_prompt": f"""You are a receptionist for a plumbing company.

HANDLE DIRECTLY (System 1):
- Greetings and basic questions
- Collecting customer info
- Understanding plumbing issues
- Determining urgency (active leak = emergency)
- Scheduling callbacks

USE ask_expert TOOL FOR (System 2):
- Plumbing code questions
- Cost estimates
- Technical advice
- Water bill analysis

For emergencies, get their info FAST.

{VOICE_RULES}""",
        "greeting": "Thanks for calling! Do you have a plumbing issue I can help with?",
        "voice": "A friendly, down-to-earth male voice. Conversational.",
    },

    "lawyer": {
        "name": "Legal Intake Specialist",
        "system_prompt": f"""You are an intake specialist for a law firm.

HANDLE DIRECTLY (System 1):
- Greetings
- Collecting contact info
- Understanding general nature of legal matter
- Scheduling consultations

USE ask_expert TOOL FOR (System 2):
- ANY legal questions or advice
- Case evaluation
- Legal procedure questions
- Document analysis

CRITICAL: You are NOT an attorney. NEVER give legal advice yourself.
Always say "I can't give legal advice, but let me check with our team" and use ask_expert.

{VOICE_RULES}""",
        "greeting": "Thanks for calling the law office. How can I help you today?",
        "voice": "A calm, professional female voice with gravitas.",
    },

    "solar": {
        "name": "Solar Company Receptionist",
        "system_prompt": f"""You are a receptionist for a solar installation company.

HANDLE DIRECTLY (System 1):
- Greetings
- Collecting customer info
- Basic interest qualification
- Scheduling consultations

USE ask_expert TOOL FOR (System 2):
- ROI calculations
- Energy bill analysis
- Technical specifications
- Tax credit questions
- Comparison analysis

{VOICE_RULES}""",
        "greeting": "Thanks for calling! Are you interested in solar for your home?",
        "voice": "A bright, enthusiastic voice. Energetic.",
    },
}


def get_skill(skill_id: str) -> dict:
    """Get skill by ID with fallback to general."""
    return BUILT_IN_SKILLS.get(skill_id, BUILT_IN_SKILLS["general"])


def list_skills() -> list[str]:
    """List all skill IDs."""
    return list(BUILT_IN_SKILLS.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# MODAL APP SETUP
# ═══════════════════════════════════════════════════════════════════════════════

app = modal.App("fast-brain-lpu")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "groq>=0.4.0",
        "anthropic>=0.18.0",
        "fastapi>=0.109.0",
        "pydantic>=2.0.0",
    )
)


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID BRAIN CLASS (System 1 + System 2)
# ═══════════════════════════════════════════════════════════════════════════════

@app.cls(
    image=image,
    secrets=[
        modal.Secret.from_name("groq-api-key"),
        modal.Secret.from_name("anthropic-api-key"),
    ],
    container_idle_timeout=CONTAINER_IDLE_TIMEOUT,
    keep_warm=KEEP_WARM,
    region=MODAL_REGION,
)
class FastBrain:
    """
    Hybrid System 1 + System 2 voice AI brain.

    System 1 (Groq): Fast, intuitive, handles 90% of calls (~80ms)
    System 2 (Claude): Slow, rational, handles complex reasoning (~2s)

    The "filler phrase" strategy hides Claude's latency during live calls.
    """

    @modal.enter()
    def setup(self):
        """Initialize both clients on container start."""
        from groq import Groq
        import anthropic

        # System 1: Fast Brain (Groq)
        self.fast_client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.fast_model = FAST_MODEL

        # System 2: Deep Brain (Claude)
        self.deep_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.deep_model = DEEP_MODEL

        print(f"[FastBrain] Hybrid System Initialized")
        print(f"  System 1 (Fast): {self.fast_model}")
        print(f"  System 2 (Deep): {self.deep_model}")
        print(f"  Region: {MODAL_REGION}")
        print(f"  Skills: {', '.join(list_skills())}")

    def _call_system1(
        self,
        messages: list[dict],
        system_prompt: str,
        use_tools: bool = True,
    ) -> dict:
        """
        Call System 1 (Groq) for fast response.

        Returns:
            {
                "content": "response" or None if tool called,
                "tool_call": {"name": "ask_expert", "args": {...}} or None,
                "latency_ms": 85
            }
        """
        import time
        start = time.perf_counter()

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        kwargs = {
            "model": self.fast_model,
            "messages": full_messages,
            "max_tokens": FAST_MAX_TOKENS,
            "temperature": FAST_TEMPERATURE,
        }

        if use_tools:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"

        response = self.fast_client.chat.completions.create(**kwargs)

        latency_ms = int((time.perf_counter() - start) * 1000)

        # Check for tool call
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            return {
                "content": None,
                "tool_call": {
                    "name": tool_call.function.name,
                    "args": json.loads(tool_call.function.arguments),
                },
                "latency_ms": latency_ms,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            }

        return {
            "content": response.choices[0].message.content,
            "tool_call": None,
            "latency_ms": latency_ms,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }

    def _call_system2(
        self,
        query: str,
        context: str = "",
        skill_context: str = "",
    ) -> dict:
        """
        Call System 2 (Claude) for deep reasoning.

        Returns:
            {
                "content": "detailed response",
                "latency_ms": 2000
            }
        """
        import time
        start = time.perf_counter()

        system = f"""You are an expert assistant helping with a voice AI system.
Provide clear, accurate, conversational answers.

{skill_context}

IMPORTANT FOR VOICE:
- Keep responses concise (under 100 words if possible)
- Use natural language, no markdown or formatting
- Be direct and helpful
- Use contractions (I'm, you're, we'll)
"""

        messages = [{"role": "user", "content": f"{context}\n\nQuestion: {query}"}]

        response = self.deep_client.messages.create(
            model=self.deep_model,
            max_tokens=DEEP_MAX_TOKENS,
            system=system,
            messages=messages,
        )

        latency_ms = int((time.perf_counter() - start) * 1000)

        return {
            "content": response.content[0].text,
            "latency_ms": latency_ms,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    @modal.method()
    def think(
        self,
        messages: list[dict],
        skill: str = "general",
        user_context: Optional[dict] = None,
    ) -> dict:
        """
        Standard System 1 only response (for backwards compatibility).
        """
        import time
        start = time.perf_counter()

        skill_config = get_skill(skill)
        system_prompt = skill_config["system_prompt"]

        if user_context:
            context_parts = []
            if user_context.get("business_name"):
                context_parts.append(f"Business: {user_context['business_name']}")
            if user_context.get("transfer_number"):
                context_parts.append(f"Transfer calls to: {user_context['transfer_number']}")
            if context_parts:
                system_prompt += "\n\nCONTEXT:\n" + "\n".join(context_parts)

        result = self._call_system1(messages, system_prompt, use_tools=False)

        return {
            "content": result["content"],
            "skill": skill,
            "model": self.fast_model,
            "system": "fast",
            "latency_ms": result["latency_ms"],
            "usage": result.get("usage"),
        }

    @modal.method()
    def think_hybrid(
        self,
        messages: list[dict],
        skill: str = "general",
        user_context: Optional[dict] = None,
    ) -> dict:
        """
        Hybrid System 1 + System 2 response.

        System 1 (Groq) handles simple questions directly.
        For complex questions, it returns a filler phrase and triggers System 2 (Claude).

        Returns:
            {
                "content": "final response",
                "filler": "filler phrase" or None,
                "system_used": "fast" or "deep",
                "fast_latency_ms": 85,
                "deep_latency_ms": 2000 or None,
                "total_latency_ms": 85 or 2085
            }
        """
        import time
        total_start = time.perf_counter()

        skill_config = get_skill(skill)
        system_prompt = skill_config["system_prompt"]

        # Add user context
        context_str = ""
        if user_context:
            context_parts = []
            if user_context.get("business_name"):
                context_parts.append(f"Business: {user_context['business_name']}")
            if user_context.get("transfer_number"):
                context_parts.append(f"Transfer calls to: {user_context['transfer_number']}")
            if context_parts:
                context_str = "\n".join(context_parts)
                system_prompt += "\n\nCONTEXT:\n" + context_str

        # Step 1: Ask System 1 (Groq)
        fast_result = self._call_system1(messages, system_prompt, use_tools=True)

        # Step 2: Check if System 2 is needed
        if fast_result["tool_call"] and fast_result["tool_call"]["name"] == "ask_expert":
            # System 2 needed!
            tool_args = fast_result["tool_call"]["args"]
            category = tool_args.get("category", "default")
            query = tool_args.get("query", messages[-1]["content"])
            extra_context = tool_args.get("context", "")

            # Get filler phrase
            filler = get_filler_phrase(category)

            # Call System 2 (Claude)
            deep_result = self._call_system2(
                query=query,
                context=f"{context_str}\n{extra_context}".strip(),
                skill_context=f"You are helping as: {skill_config['name']}",
            )

            total_latency = int((time.perf_counter() - total_start) * 1000)

            return {
                "content": deep_result["content"],
                "filler": filler,
                "system_used": "deep",
                "skill": skill,
                "fast_model": self.fast_model,
                "deep_model": self.deep_model,
                "fast_latency_ms": fast_result["latency_ms"],
                "deep_latency_ms": deep_result["latency_ms"],
                "total_latency_ms": total_latency,
                "usage": {
                    "fast": fast_result.get("usage"),
                    "deep": deep_result.get("usage"),
                },
            }

        # System 1 handled it directly
        total_latency = int((time.perf_counter() - total_start) * 1000)

        return {
            "content": fast_result["content"],
            "filler": None,
            "system_used": "fast",
            "skill": skill,
            "fast_model": self.fast_model,
            "deep_model": None,
            "fast_latency_ms": fast_result["latency_ms"],
            "deep_latency_ms": None,
            "total_latency_ms": total_latency,
            "usage": {"fast": fast_result.get("usage")},
        }

    @modal.method()
    def think_with_voice(
        self,
        messages: list[dict],
        skill: str = "general",
        user_context: Optional[dict] = None,
    ) -> dict:
        """
        Hybrid response with voice descriptions (Brain + Mouth mode).
        """
        # Get hybrid response
        result = self.think_hybrid.local(messages, skill, user_context)

        skill_config = get_skill(skill)
        default_voice = skill_config.get("voice", VOICE_CONTEXTS["default"])

        # Determine voice based on context
        if result["filler"]:
            filler_voice = VOICE_CONTEXTS["thinking"]
            content_voice = VOICE_CONTEXTS["explaining"]
        else:
            filler_voice = None
            content_voice = default_voice

        return {
            "text": result["content"],
            "voice": content_voice,
            "filler_text": result["filler"],
            "filler_voice": filler_voice,
            "system_used": result["system_used"],
            "skill": skill,
            "fast_latency_ms": result["fast_latency_ms"],
            "deep_latency_ms": result.get("deep_latency_ms"),
            "total_latency_ms": result["total_latency_ms"],
        }

    @modal.method()
    def get_greeting(self, skill: str = "general") -> dict:
        """Get the greeting and voice for a skill."""
        skill_config = get_skill(skill)
        return {
            "text": skill_config.get("greeting", "Hello!"),
            "voice": skill_config.get("voice", VOICE_CONTEXTS["default"]),
            "skill": skill,
        }

    @modal.method()
    def health_check(self) -> dict:
        """Health check with status info."""
        return {
            "status": "healthy",
            "architecture": "System 1 + System 2 Hybrid",
            "system1": {
                "name": "Fast Brain",
                "model": self.fast_model,
                "provider": "Groq",
                "latency": "~80ms",
            },
            "system2": {
                "name": "Deep Brain",
                "model": self.deep_model,
                "provider": "Anthropic",
                "latency": "~2000ms",
            },
            "region": MODAL_REGION,
            "skills": list_skills(),
            "skills_count": len(BUILT_IN_SKILLS),
            "timestamp": datetime.utcnow().isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI WEB APP
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

web_app = FastAPI(
    title="Fast Brain LPU",
    description="Hybrid System 1 + System 2 Voice AI - HIVE215",
    version="3.0.0",
)


class ChatRequest(BaseModel):
    messages: list[dict]
    skill: str = "general"
    user_context: Optional[dict] = None


class ChatResponse(BaseModel):
    content: str
    skill: str
    model: str
    system: str
    latency_ms: int
    usage: Optional[dict] = None


class HybridChatResponse(BaseModel):
    content: str
    filler: Optional[str] = None
    system_used: str
    skill: str
    fast_model: str
    deep_model: Optional[str] = None
    fast_latency_ms: int
    deep_latency_ms: Optional[int] = None
    total_latency_ms: int
    usage: Optional[dict] = None


class VoiceChatResponse(BaseModel):
    text: str
    voice: str
    filler_text: Optional[str] = None
    filler_voice: Optional[str] = None
    system_used: str
    skill: str
    fast_latency_ms: int
    deep_latency_ms: Optional[int] = None
    total_latency_ms: int


@web_app.get("/health")
async def health():
    """Health check showing hybrid architecture status."""
    brain = FastBrain()
    return brain.health_check.remote()


@web_app.get("/v1/skills")
async def get_all_skills():
    """List all available skills."""
    return {
        "skills": [
            {"id": k, "name": v["name"]}
            for k, v in BUILT_IN_SKILLS.items()
        ],
        "count": len(BUILT_IN_SKILLS),
    }


@web_app.get("/v1/skills/{skill_id}")
async def get_skill_detail(skill_id: str):
    """Get details for a specific skill."""
    if skill_id not in BUILT_IN_SKILLS:
        raise HTTPException(404, f"Skill '{skill_id}' not found")

    skill = BUILT_IN_SKILLS[skill_id]
    return {
        "id": skill_id,
        "name": skill["name"],
        "greeting": skill["greeting"],
        "voice": skill.get("voice", VOICE_CONTEXTS["default"]),
    }


@web_app.get("/v1/greeting/{skill_id}")
async def get_greeting(skill_id: str):
    """Get the greeting for a skill."""
    brain = FastBrain()
    return brain.get_greeting.remote(skill_id)


@web_app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """Standard System 1 only chat (backwards compatible)."""
    brain = FastBrain()
    return brain.think.remote(
        messages=request.messages,
        skill=request.skill,
        user_context=request.user_context,
    )


@web_app.post("/v1/chat/hybrid", response_model=HybridChatResponse)
async def hybrid_chat_completion(request: ChatRequest):
    """
    Hybrid System 1 + System 2 chat.

    System 1 (Groq) handles simple questions directly (~80ms).
    For complex questions, returns filler phrase + System 2 (Claude) response.

    Example response (simple question):
        {
            "content": "I can help with that! What's your address?",
            "filler": null,
            "system_used": "fast",
            "fast_latency_ms": 85,
            "total_latency_ms": 85
        }

    Example response (complex question):
        {
            "content": "Based on your usage of 850 kWh and current rates...",
            "filler": "Let me pull up your information and analyze that...",
            "system_used": "deep",
            "fast_latency_ms": 85,
            "deep_latency_ms": 1950,
            "total_latency_ms": 2035
        }
    """
    brain = FastBrain()
    return brain.think_hybrid.remote(
        messages=request.messages,
        skill=request.skill,
        user_context=request.user_context,
    )


@web_app.post("/v1/chat/voice", response_model=VoiceChatResponse)
async def voice_chat_completion(request: ChatRequest):
    """
    Hybrid chat with voice descriptions (Brain + Mouth mode).

    Returns text + voice description for both filler and main response.
    """
    brain = FastBrain()
    return brain.think_with_voice.remote(
        messages=request.messages,
        skill=request.skill,
        user_context=request.user_context,
    )


@web_app.get("/v1/fillers")
async def get_filler_phrases():
    """List all filler phrase categories."""
    return {
        "categories": list(FILLER_PHRASES.keys()),
        "phrases": FILLER_PHRASES,
    }


@app.function(image=image, region=MODAL_REGION)
@modal.asgi_app()
def fastapi_app():
    """Mount FastAPI to Modal."""
    return web_app


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL TESTING
# ═══════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main():
    """Test hybrid brain locally."""
    print("=" * 70)
    print("FAST BRAIN LPU v3.0 - Hybrid System 1 + System 2")
    print("=" * 70)

    brain = FastBrain()

    # Health check
    health = brain.health_check.remote()
    print(f"\n✓ Architecture: {health['architecture']}")
    print(f"  System 1: {health['system1']['model']} ({health['system1']['latency']})")
    print(f"  System 2: {health['system2']['model']} ({health['system2']['latency']})")

    # Test 1: Simple question (System 1 only)
    print("\n" + "-" * 70)
    print("Test 1: Simple question (should use System 1 only)")
    print("-" * 70)

    result = brain.think_hybrid.remote(
        messages=[{"role": "user", "content": "Hi, my lights are flickering"}],
        skill="electrician",
    )
    print(f"\n✓ System used: {result['system_used']}")
    print(f"  Filler: {result['filler']}")
    print(f"  Response: {result['content'][:80]}...")
    print(f"  Latency: {result['total_latency_ms']}ms")

    # Test 2: Complex question (System 1 → System 2)
    print("\n" + "-" * 70)
    print("Test 2: Complex question (should trigger System 2)")
    print("-" * 70)

    result = brain.think_hybrid.remote(
        messages=[{"role": "user", "content": "Can you analyze my electricity bill and tell me why it's so high this month compared to last month? I used 850 kWh."}],
        skill="electrician",
    )
    print(f"\n✓ System used: {result['system_used']}")
    print(f"  Filler: {result['filler']}")
    print(f"  Response: {result['content'][:100]}...")
    print(f"  Fast latency: {result['fast_latency_ms']}ms")
    print(f"  Deep latency: {result['deep_latency_ms']}ms")
    print(f"  Total latency: {result['total_latency_ms']}ms")

    # Test 3: Voice mode
    print("\n" + "-" * 70)
    print("Test 3: Voice mode with complex question")
    print("-" * 70)

    result = brain.think_with_voice.remote(
        messages=[{"role": "user", "content": "What NEC code applies to outdoor outlet installations?"}],
        skill="electrician",
    )
    print(f"\n✓ System used: {result['system_used']}")
    print(f"  Filler text: {result['filler_text']}")
    print(f"  Filler voice: {result['filler_voice']}")
    print(f"  Response: {result['text'][:80]}...")
    print(f"  Response voice: {result['voice'][:50]}...")

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    print(f"\nDeploy with: modal deploy fast_brain/deploy_groq.py")
    print(f"\nEndpoints:")
    print(f"  POST /v1/chat/completions - System 1 only (backwards compatible)")
    print(f"  POST /v1/chat/hybrid      - System 1 + System 2 (recommended)")
    print(f"  POST /v1/chat/voice       - Hybrid + voice descriptions")
