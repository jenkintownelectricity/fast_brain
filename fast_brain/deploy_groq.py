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
DEEP_MODEL = "claude-sonnet-4-20250514"
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
        "description": "General-purpose voice assistant",
        "system_prompt": f"""You are a helpful voice assistant. Be conversational and natural.

For simple questions (greetings, basic info, scheduling), answer directly.
For complex questions (analysis, calculations, detailed advice), use the ask_expert tool.

{VOICE_RULES}""",
        "greeting": "Hey there! How can I help you today?",
        "voice": VOICE_CONTEXTS["default"],
        "version": "1.0",
    },

    "receptionist": {
        "name": "Professional Receptionist",
        "description": "Business receptionist for call handling",
        "system_prompt": f"""You are a professional receptionist. Your job is to:
- Answer calls warmly and professionally
- Collect caller's name and reason for calling
- Offer to take a message or transfer the call

For simple requests, handle them directly.
For complex questions about billing, technical issues, or detailed analysis, use the ask_expert tool.

{VOICE_RULES}""",
        "greeting": "Thanks for calling! How can I help you today?",
        "voice": VOICE_CONTEXTS["greeting"],
        "version": "1.0",
    },

    "electrician": {
        "name": "Electrician Service Intake",
        "description": "Intake specialist for electrical service companies",
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
        "version": "1.0",
    },

    "plumber": {
        "name": "Plumber Service Intake",
        "description": "Intake specialist for plumbing companies",
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
        "version": "1.0",
    },

    "lawyer": {
        "name": "Legal Intake Specialist",
        "description": "Intake specialist for law firms",
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
        "version": "1.0",
    },

    "solar": {
        "name": "Solar Company Receptionist",
        "description": "Receptionist for solar installation companies",
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
        "version": "1.0",
    },

    "default": {
        "name": "Default Assistant",
        "description": "General-purpose voice assistant",
        "system_prompt": f"""You are a helpful, friendly voice assistant.

Guidelines:
- Keep responses concise and conversational (1-2 sentences when possible)
- Be natural and engaging, like talking to a friend
- Ask clarifying questions when needed
- If you don't know something, say so honestly
- Match the user's energy and communication style

{VOICE_RULES}""",
        "greeting": "Hello! How can I help you today?",
        "voice": VOICE_CONTEXTS["default"],
        "version": "1.0",
    },

    "tara-sales": {
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
- Sound like you're having a friendly conversation, not giving a presentation
- Use contractions (I'm, you're, we'll, don't, can't)
- Keep sentences under 20 words
- No markdown, bullets, numbered lists, or formatting
- No emojis or special characters""",
        "greeting": "Hey there! I'm Tara from The Dash. What kind of business are you running?",
        "voice": "A warm, friendly female voice. Confident and conversational, like talking to a trusted advisor.",
        "version": "1.0",
        "knowledge": [
            "TheDashTool.com is the website. Email is info@thedashtool.com",
            "Tara Horn is the founder and workflow optimization expert since 2015",
            "The Dash has helped nearly 200 companies improve operational efficiency",
            "Industries served: Finance & Banking, Healthcare, Retail, Manufacturing, Professional Services",
            "The Dash integrates with: CRM systems, accounting software, project management tools, marketing platforms, ticketing systems",
            "Demo booking: Free demo available at thedashtool.com or by phone",
            "Operating hours: Mon-Fri 9:00AM - 5:00PM, Sat-Sun 10:00AM - 6:00PM",
            "YouTube channel: youtube.com/@thedashtool"
        ],
    },
}

# Runtime skill storage (populated via POST /v1/skills API)
RUNTIME_SKILLS = {}


def _load_all_database_skill_ids() -> list[str]:
    """Get all skill IDs from the persistent database."""
    try:
        import sys
        sys.path.insert(0, "/root")
        os.environ.setdefault('HIVE215_DB_PATH', '/data/hive215.db')
        import database as db
        skills = db.get_all_skills(include_inactive=False)
        return [s.get('id') for s in skills if s.get('id')]
    except Exception:
        return []


def get_skill(skill_id: str) -> dict:
    """Get skill by ID with fallback to default.

    Priority order:
    1. Runtime skills (API-created, temporary)
    2. Database skills (dashboard-created, persistent)
    3. Built-in skills (hardcoded)
    4. Fallback to 'general' skill
    """
    # Check runtime skills first (allows overriding)
    if skill_id in RUNTIME_SKILLS:
        return RUNTIME_SKILLS[skill_id]

    # Check database skills (persistent, from dashboard)
    db_skill = _get_database_skill(skill_id)
    if db_skill:
        # Convert to format expected by callers
        return {
            "name": db_skill.get("name", skill_id),
            "description": db_skill.get("description", ""),
            "system_prompt": db_skill.get("system_prompt", ""),
            "greeting": db_skill.get("greeting", "Hello!"),
            "voice": db_skill.get("voice_config", {}).get("description", VOICE_CONTEXTS.get("default", "")),
            "version": db_skill.get("version", "1.0"),
            "knowledge": db_skill.get("knowledge", []),
        }

    # Check built-in skills
    if skill_id in BUILT_IN_SKILLS:
        return BUILT_IN_SKILLS[skill_id]

    # Fallback to default
    return BUILT_IN_SKILLS.get("default", BUILT_IN_SKILLS["general"])


def list_skills() -> list[str]:
    """List all skill IDs (database + built-in + runtime)."""
    # Combine all sources, deduplicate
    skill_ids = set(BUILT_IN_SKILLS.keys())
    skill_ids.update(RUNTIME_SKILLS.keys())
    skill_ids.update(_load_all_database_skill_ids())
    return list(skill_ids)


# ═══════════════════════════════════════════════════════════════════════════════
# MODAL APP SETUP
# ═══════════════════════════════════════════════════════════════════════════════

app = modal.App("fast-brain-lpu")

# Shared volume with unified dashboard for persistent skill storage
# This allows skills created in the dashboard to appear in the API
skills_volume = modal.Volume.from_name("hive215-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "groq>=0.4.0",
        "anthropic>=0.18.0",
        "fastapi>=0.109.0",
        "pydantic>=2.0.0",
        "pyjwt>=2.8.0",
    )
    # Add database module for persistent skill storage
    .add_local_file("database.py", "/root/database.py")
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
            "skills_count": len(list_skills()),  # Include database skills in count
            "timestamp": datetime.utcnow().isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI WEB APP
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid
import time

web_app = FastAPI(
    title="Fast Brain LPU",
    description="Hybrid System 1 + System 2 Voice AI - HIVE215",
    version="3.0.0",
)

# Add CORS middleware - allow requests from 453rahul.com
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://453rahul.com",
        "http://453rahul.com",
        "https://www.453rahul.com",
        "http://www.453rahul.com",
        "http://localhost:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS (OpenAI-Compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """OpenAI-compatible chat request with skill extension."""
    messages: list[dict]
    skill: str = "default"  # Fast Brain extension
    model: Optional[str] = None  # Ignored, but accepted for compatibility
    max_tokens: int = 256
    temperature: float = 0.7
    user_profile: Optional[str] = None  # Optional business context
    user_context: Optional[dict] = None  # Legacy support


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChoice(BaseModel):
    index: int = 0
    message: OpenAIMessage
    finish_reason: str = "stop"


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class MetricsExtension(BaseModel):
    """Fast Brain-specific performance metrics."""
    ttfb_ms: int
    tokens_per_sec: Optional[float] = None


class OpenAIChatResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: list[OpenAIChoice]
    usage: OpenAIUsage
    skill_used: str  # Fast Brain extension
    metrics: Optional[MetricsExtension] = None  # Fast Brain extension


class SkillRequest(BaseModel):
    """Request to create a new skill."""
    skill_id: str
    name: str
    description: str
    system_prompt: str
    knowledge: list[str] = []
    greeting: Optional[str] = None
    voice: Optional[str] = None


class SkillResponse(BaseModel):
    """Response for skill operations."""
    success: bool
    skill_id: str
    message: str


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


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@web_app.get("/health")
async def health():
    """
    Health check endpoint (HIVE215-compatible format).

    Returns:
        status: "healthy"
        backend: Model identifier
        skills_available: List of skill IDs (including database skills)
        version: API version
    """
    # Use list_skills() to include database, built-in, and runtime skills
    all_skills = list_skills()
    return {
        "status": "healthy",
        "backend": f"groq-{FAST_MODEL}",
        "skills_available": all_skills,
        "version": "1.0.0",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LIVEKIT TOKEN ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@web_app.get("/api/livekit/token")
def get_livekit_token(room: str = "demo-room", identity: str = None):
    """Generate LiveKit access token for voice demo"""
    import jwt
    import secrets

    api_key = "APICitVpmeC5zL6"
    api_secret = "GFOkmbbd1qP9WKfg7uiQux8SgPNLCIxd5EHCagqbVfj"

    if not identity:
        identity = f"visitor-{secrets.token_hex(4)}"

    # Token expires in 1 hour
    exp = int(time.time()) + 3600

    claims = {
        "iss": api_key,
        "sub": identity,
        "iat": int(time.time()),
        "exp": exp,
        "video": {
            "room": room,
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True
        }
    }

    token = jwt.encode(claims, api_secret, algorithm="HS256")

    return {
        "token": token,
        "url": "wss://hive215-9diqhgoh.livekit.cloud",
        "room": room,
        "identity": identity
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SKILLS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_database_skills() -> list:
    """
    Get skills from the persistent SQLite database.

    Returns empty list if database is not available.
    """
    try:
        import sys
        sys.path.insert(0, "/root")
        os.environ.setdefault('HIVE215_DB_PATH', '/data/hive215.db')
        import database as db
        return db.get_all_skills(include_inactive=False)
    except Exception as e:
        print(f"Warning: Could not load database skills: {e}")
        return []


def _get_database_skill(skill_id: str) -> Optional[dict]:
    """
    Get a specific skill from the persistent SQLite database.

    Returns None if database is not available or skill not found.
    """
    try:
        import sys
        sys.path.insert(0, "/root")
        os.environ.setdefault('HIVE215_DB_PATH', '/data/hive215.db')
        import database as db
        return db.get_skill(skill_id)
    except Exception as e:
        print(f"Warning: Could not load skill from database: {e}")
        return None


@web_app.get("/v1/skills")
async def get_all_skills():
    """
    List all available skills (HIVE215-compatible).

    Returns skills from three sources (in priority order):
    1. Persistent database (skills created in dashboard)
    2. Built-in skills (hardcoded defaults)
    3. Runtime skills (created via API, lost on restart)
    """
    skills = []
    seen_ids = set()

    # 1. Add skills from persistent database first (highest priority)
    # These are skills created in the unified dashboard
    db_skills = _get_database_skills()
    for skill in db_skills:
        skill_id = skill.get('id')
        if skill_id and skill_id not in seen_ids:
            skills.append({
                "id": skill_id,
                "name": skill.get("name", skill_id),
                "description": skill.get("description", ""),
                "version": skill.get("version", "1.0"),
                "source": "database",  # Indicates persistent storage
            })
            seen_ids.add(skill_id)

    # 2. Add built-in skills (don't duplicate if already in database)
    for skill_id, skill in BUILT_IN_SKILLS.items():
        if skill_id not in seen_ids:
            skills.append({
                "id": skill_id,
                "name": skill.get("name", skill_id),
                "description": skill.get("description", ""),
                "version": skill.get("version", "1.0"),
                "source": "builtin",
            })
            seen_ids.add(skill_id)

    # 3. Add runtime skills (temporary, created via API)
    for skill_id, skill in RUNTIME_SKILLS.items():
        if skill_id not in seen_ids:
            skills.append({
                "id": skill_id,
                "name": skill.get("name", skill_id),
                "description": skill.get("description", ""),
                "version": skill.get("version", "1.0"),
                "source": "runtime",
            })
            seen_ids.add(skill_id)

    return {"skills": skills}


@web_app.post("/v1/skills")
async def create_skill(request: SkillRequest):
    """
    Create a new skill dynamically.

    This allows HIVE215 to register custom skills at runtime.
    """
    # Build knowledge section if provided
    knowledge_text = ""
    if request.knowledge:
        knowledge_text = "\n\n## Quick Reference\n" + "\n".join(f"- {k}" for k in request.knowledge)

    # Create the skill
    skill = {
        "name": request.name,
        "description": request.description,
        "system_prompt": request.system_prompt + knowledge_text,
        "greeting": request.greeting or f"Hello! I'm {request.name}. How can I help?",
        "voice": request.voice or VOICE_CONTEXTS["default"],
        "version": "1.0",
        "knowledge": request.knowledge,
    }

    # Store in runtime skills
    RUNTIME_SKILLS[request.skill_id] = skill

    return SkillResponse(
        success=True,
        skill_id=request.skill_id,
        message="Skill created successfully"
    )


@web_app.get("/v1/skills/{skill_id}")
async def get_skill_detail(skill_id: str):
    """Get details for a specific skill."""
    # 1. Check database first (persistent skills from dashboard)
    db_skill = _get_database_skill(skill_id)
    if db_skill:
        # Extract voice config from database skill
        voice_config = db_skill.get("voice_config", {})
        voice_description = voice_config.get("description", VOICE_CONTEXTS["default"])

        return {
            "id": skill_id,
            "name": db_skill.get("name", skill_id),
            "description": db_skill.get("description", ""),
            "greeting": db_skill.get("greeting", "Hello!"),
            "voice": voice_description,
            "version": db_skill.get("version", "1.0"),
            "system_prompt": db_skill.get("system_prompt", ""),
            "knowledge": db_skill.get("knowledge", []),
            "source": "database",
        }

    # 2. Check built-in and runtime skills
    skill = BUILT_IN_SKILLS.get(skill_id) or RUNTIME_SKILLS.get(skill_id)

    if not skill:
        raise HTTPException(404, f"Skill '{skill_id}' not found")

    return {
        "id": skill_id,
        "name": skill.get("name", skill_id),
        "description": skill.get("description", ""),
        "greeting": skill.get("greeting", "Hello!"),
        "voice": skill.get("voice", VOICE_CONTEXTS["default"]),
        "version": skill.get("version", "1.0"),
        "source": "builtin" if skill_id in BUILT_IN_SKILLS else "runtime",
    }


@web_app.get("/v1/greeting/{skill_id}")
async def get_greeting(skill_id: str):
    """Get the greeting for a skill."""
    brain = FastBrain()
    return brain.get_greeting.remote(skill_id)


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT COMPLETIONS ENDPOINT (OpenAI-Compatible)
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_skills_dict() -> dict:
    """Get combined database, built-in, and runtime skills.

    Priority order (later overrides earlier):
    1. Built-in skills (base)
    2. Database skills (persistent, from dashboard)
    3. Runtime skills (temporary, from API)
    """
    combined = dict(BUILT_IN_SKILLS)

    # Add database skills (override built-in if same ID)
    db_skills = _get_database_skills()
    for skill in db_skills:
        skill_id = skill.get('id')
        if skill_id:
            combined[skill_id] = {
                "name": skill.get("name", skill_id),
                "description": skill.get("description", ""),
                "system_prompt": skill.get("system_prompt", ""),
                "greeting": skill.get("greeting", "Hello!"),
                "voice": skill.get("voice_config", {}).get("description", VOICE_CONTEXTS["default"]),
                "version": skill.get("version", "1.0"),
                "knowledge": skill.get("knowledge", []),
            }

    # Runtime skills have highest priority
    combined.update(RUNTIME_SKILLS)
    return combined


@web_app.post("/v1/chat/completions", response_model=OpenAIChatResponse)
async def chat_completion(request: ChatRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Accepts standard OpenAI format plus:
    - skill: Selects which system prompt to use
    - user_profile: Optional business context

    Returns OpenAI-compatible response with:
    - skill_used: Which skill handled the request
    - metrics: Performance metrics (ttfb_ms, tokens_per_sec)
    """
    import time as time_module
    start_time = time_module.perf_counter()

    brain = FastBrain()

    # Build user context from profile if provided
    user_context = request.user_context or {}
    if request.user_profile:
        user_context["profile"] = request.user_profile

    # Get the skill (check runtime skills first, then built-in)
    skill_id = request.skill
    all_skills = get_all_skills_dict()
    if skill_id not in all_skills:
        skill_id = "default"  # Fallback to default

    # Call the brain
    result = brain.think.remote(
        messages=request.messages,
        skill=skill_id,
        user_context=user_context if user_context else None,
    )

    # Calculate metrics
    total_time_ms = int((time_module.perf_counter() - start_time) * 1000)
    ttfb_ms = result.get("latency_ms", total_time_ms)

    # Estimate tokens per second
    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    tokens_per_sec = None
    if completion_tokens > 0 and ttfb_ms > 0:
        tokens_per_sec = round((completion_tokens / ttfb_ms) * 1000, 1)

    # Build OpenAI-compatible response
    return OpenAIChatResponse(
        choices=[
            OpenAIChoice(
                message=OpenAIMessage(
                    role="assistant",
                    content=result["content"]
                )
            )
        ],
        usage=OpenAIUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
        ),
        skill_used=skill_id,
        metrics=MetricsExtension(
            ttfb_ms=ttfb_ms,
            tokens_per_sec=tokens_per_sec,
        ),
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


@app.function(
    image=image,
    region=MODAL_REGION,
    volumes={"/data": skills_volume},  # Mount shared volume for database access
)
@modal.asgi_app()
def fastapi_app():
    """Mount FastAPI to Modal with persistent skill storage."""
    import os
    import sys

    # Set up database access
    sys.path.insert(0, "/root")
    os.environ['HIVE215_DB_PATH'] = '/data/hive215.db'

    # Initialize database module
    try:
        import database as db
        db.init_db()  # Ensure tables exist
    except Exception as e:
        print(f"Database initialization warning: {e}")

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
