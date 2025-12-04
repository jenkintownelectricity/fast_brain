"""
Fast Brain - Groq-Powered LPU with Skills Layer

Uses Groq's ultra-fast inference (40ms TTFB, 800 tok/s) with
domain-specific skills to create expert AI agents.

Architecture:
    Skills Database → System Prompt Builder → Groq LLM → Fast Response

Usage:
    modal deploy fast_brain/deploy_groq.py

Environment Variables:
    GROQ_API_KEY - Your Groq API key (get free at console.groq.com)
"""

import modal
import time
import json
import os
from typing import Optional, List, Dict

# Lightweight image - no ML dependencies needed
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "uvicorn",
        "pydantic>=2.0",
        "httpx",
    )
)

app = modal.App("fast-brain-lpu")

# =============================================================================
# Skills Database (In-Memory for now, can connect to Supabase later)
# =============================================================================

SKILLS_DATABASE: Dict[str, dict] = {
    "receptionist": {
        "name": "Professional Receptionist",
        "description": "Expert phone answering and call handling",
        "system_prompt": """You are a professional AI receptionist. Respond in 1-2 short sentences maximum. Be warm, helpful, and conversational—never robotic.

Core behaviors:
- Answer calls naturally: "Hi, this is [Business], how can I help you?"
- Gather: caller name, reason for call, callback number
- For appointments: confirm date/time preferences
- For emergencies: immediately escalate with "Let me connect you right away"
- If unsure: "Let me check on that and get right back to you"

Never say you're an AI unless directly asked. Speak like a friendly, competent human receptionist.""",
        "knowledge": [],
    },
    "electrician": {
        "name": "Electrician Assistant",
        "description": "Expert in electrical services and scheduling",
        "system_prompt": """You are an AI assistant for an electrical contracting business. Be professional, safety-conscious, and helpful.

Core knowledge:
- Services: panel upgrades, rewiring, EV chargers, generators, lighting, outlets
- Always emphasize safety for electrical emergencies
- Collect: address, service needed, timeline, access info
- For emergencies (sparks, burning smell, no power): treat as urgent

Respond concisely in 1-2 sentences. Be warm but professional.""",
        "knowledge": [
            "Panel upgrades typically cost $1,500-$3,000",
            "EV charger installation: $500-$2,000 depending on distance from panel",
            "Emergency service available 24/7 with $150 service call fee",
            "Free estimates for jobs over $500",
        ],
    },
    "plumber": {
        "name": "Plumber Assistant",
        "description": "Expert in plumbing services",
        "system_prompt": """You are an AI assistant for a plumbing company. Be helpful and knowledgeable about plumbing services.

Core knowledge:
- Services: drain cleaning, water heaters, leak repair, pipe replacement, fixtures
- For water emergencies (flooding, burst pipes): treat as urgent
- Collect: address, issue description, when it started

Respond concisely in 1-2 sentences.""",
        "knowledge": [
            "Drain cleaning: $150-$300",
            "Water heater replacement: $1,000-$3,000",
            "Emergency service available with priority scheduling",
        ],
    },
    "lawyer": {
        "name": "Legal Intake Assistant",
        "description": "Professional legal intake and scheduling",
        "system_prompt": """You are a legal intake assistant. Be professional, confidential, and thorough.

Core behaviors:
- Gather: name, contact info, brief description of legal matter
- DO NOT provide legal advice
- Schedule consultations, don't diagnose cases
- Emphasize confidentiality

Respond professionally in 1-2 sentences.""",
        "knowledge": [
            "Initial consultations are typically free or low-cost",
            "All communications are confidential",
        ],
    },
    "general": {
        "name": "General Assistant",
        "description": "Helpful general-purpose assistant",
        "system_prompt": """You are a helpful AI assistant. Be friendly, concise, and helpful.
Respond in 1-2 sentences unless more detail is needed.""",
        "knowledge": [],
    },
}

# =============================================================================
# Groq Client
# =============================================================================

class GroqClient:
    """Fast inference via Groq API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "llama-3.3-70b-versatile"  # Fast & capable

    async def chat(
        self,
        messages: List[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> tuple[str, dict]:
        """Send chat request to Groq."""
        import httpx

        start_time = time.perf_counter()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()

        end_time = time.perf_counter()
        total_ms = (end_time - start_time) * 1000

        # Extract response
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})

        stats = {
            "ttfb_ms": total_ms,  # Groq is so fast, TTFB ≈ total time
            "total_time_ms": round(total_ms, 2),
            "tokens": usage.get("completion_tokens", 0),
            "tokens_per_sec": round(
                usage.get("completion_tokens", 0) / (total_ms / 1000), 2
            ) if total_ms > 0 else 0,
            "model": self.model,
        }

        return content, stats


# =============================================================================
# Skills Manager
# =============================================================================

class SkillsManager:
    """Manages skills and builds system prompts."""

    def __init__(self):
        self.skills = SKILLS_DATABASE.copy()
        self.custom_skills: Dict[str, dict] = {}

    def get_skill(self, skill_id: str) -> Optional[dict]:
        """Get a skill by ID."""
        return self.skills.get(skill_id) or self.custom_skills.get(skill_id)

    def list_skills(self) -> List[str]:
        """List all available skills."""
        return list(self.skills.keys()) + list(self.custom_skills.keys())

    def add_custom_skill(
        self,
        skill_id: str,
        name: str,
        description: str,
        system_prompt: str,
        knowledge: List[str] = None,
    ):
        """Add a custom skill."""
        self.custom_skills[skill_id] = {
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "knowledge": knowledge or [],
        }

    def build_system_prompt(
        self,
        skill_id: str,
        user_profile: str = None,
        additional_context: str = None,
    ) -> str:
        """Build a complete system prompt with skill + user profile."""
        skill = self.get_skill(skill_id)
        if not skill:
            skill = self.skills["general"]

        parts = [skill["system_prompt"]]

        # Add knowledge base
        if skill.get("knowledge"):
            parts.append("\nKey Information:")
            for item in skill["knowledge"]:
                parts.append(f"- {item}")

        # Add user profile
        if user_profile:
            parts.append(f"\nBusiness Profile:\n{user_profile}")

        # Add additional context
        if additional_context:
            parts.append(f"\nAdditional Context:\n{additional_context}")

        return "\n".join(parts)


# =============================================================================
# FastAPI Application
# =============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

web_app = FastAPI(
    title="Fast Brain LPU",
    description="Groq-powered inference with domain-specific skills",
    version="2.0.0",
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False
    skill: str = "general"
    user_profile: Optional[str] = None


class SkillRequest(BaseModel):
    skill_id: str
    name: str
    description: str
    system_prompt: str
    knowledge: List[str] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    skills_available: List[str]
    version: str
    backend: str


# Global instances
_groq_client: Optional[GroqClient] = None
_skills_manager: Optional[SkillsManager] = None


def get_groq_client() -> GroqClient:
    global _groq_client
    if not _groq_client:
        raise HTTPException(status_code=503, detail="Groq client not initialized")
    return _groq_client


def get_skills_manager() -> SkillsManager:
    global _skills_manager
    if not _skills_manager:
        _skills_manager = SkillsManager()
    return _skills_manager


@web_app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global _groq_client, _skills_manager

    skills = get_skills_manager()

    return HealthResponse(
        status="healthy" if _groq_client else "degraded",
        model_loaded=_groq_client is not None,
        skills_available=skills.list_skills(),
        version="2.0.0",
        backend="groq-llama-3.3-70b",
    )


@web_app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "fast-brain-groq",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "fast-brain",
                "backend": "groq-llama-3.3-70b",
            }
        ]
    }


@web_app.get("/v1/skills")
async def list_skills():
    """List all available skills."""
    skills = get_skills_manager()
    return {
        "skills": [
            {
                "id": skill_id,
                "name": skill["name"],
                "description": skill["description"],
            }
            for skill_id, skill in {**skills.skills, **skills.custom_skills}.items()
        ]
    }


@web_app.post("/v1/skills")
async def create_skill(request: SkillRequest):
    """Create a custom skill."""
    skills = get_skills_manager()
    skills.add_custom_skill(
        skill_id=request.skill_id,
        name=request.name,
        description=request.description,
        system_prompt=request.system_prompt,
        knowledge=request.knowledge,
    )
    return {"success": True, "skill_id": request.skill_id}


@web_app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatRequest):
    """Create a chat completion with skill-based system prompt."""
    groq = get_groq_client()
    skills = get_skills_manager()

    # Build system prompt from skill
    system_prompt = skills.build_system_prompt(
        skill_id=request.skill,
        user_profile=request.user_profile,
    )

    # Build messages with system prompt
    messages = [{"role": "system", "content": system_prompt}]
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    # Get response from Groq
    content, stats = await groq.chat(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "fast-brain-groq",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": sum(len(m["content"].split()) for m in messages),
            "completion_tokens": stats.get("tokens", 0),
            "total_tokens": sum(len(m["content"].split()) for m in messages) + stats.get("tokens", 0),
        },
        "metrics": stats,
        "skill_used": request.skill,
    }


# =============================================================================
# Modal Deployment
# =============================================================================

@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("groq-api-key")],
    cpu=1,
    memory=512,
    timeout=300,
    container_idle_timeout=300,
)
class FastBrainLPU:
    """Fast Brain with Groq backend and Skills Layer."""

    @modal.enter()
    def initialize(self):
        """Initialize on container startup."""
        global _groq_client, _skills_manager

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("WARNING: GROQ_API_KEY not set!")
        else:
            _groq_client = GroqClient(api_key)
            print(f"Groq client initialized with model: {_groq_client.model}")

        _skills_manager = SkillsManager()
        print(f"Skills loaded: {_skills_manager.list_skills()}")
        print("Fast Brain LPU ready!")

    @modal.asgi_app()
    def serve(self):
        """Serve the FastAPI application."""
        return web_app


# =============================================================================
# Local Testing
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    import asyncio

    # For local testing
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        _groq_client = GroqClient(api_key)
    _skills_manager = SkillsManager()

    uvicorn.run(web_app, host="0.0.0.0", port=8000)
