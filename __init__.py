"""
Fast Brain - Hybrid System 1 + System 2 Voice AI
=================================================

HIVE215 Voice AI Inference Engine using Kahneman's "Thinking, Fast and Slow" architecture:

- System 1 (Fast Brain): Groq + Llama 3.3 70B (~80ms latency) - handles 90% of calls
- System 2 (Deep Brain): Claude 3.5 Sonnet (~2s latency) - complex reasoning

Features:
- Filler phrase strategy to hide Claude's latency
- Skill-based routing for different business verticals
- Voice-aware output (Brain + Mouth mode)
- Tool-based automatic routing between systems

Usage:
    # Deploy to Modal
    modal deploy fast_brain/deploy_groq.py

    # Use the client
    from fast_brain.client import FastBrainClient

    client = FastBrainClient(os.getenv("FAST_BRAIN_URL"))
    async for chunk in client.stream_chat("Hello!"):
        print(chunk.text, end="")
"""

__version__ = "3.0.0"

# Export main classes for convenience
from .client import FastBrainClient, FastBrainWebSocket, StreamingResponse
from .config import (
    FastBrainConfig,
    DeepBrainConfig,
    HybridConfig,
    ServerConfig,
    VoiceAgentConfig,
    get_config,
    get_hybrid_config,
    get_server_config,
    get_voice_config,
    estimate_monthly_cost,
    FAST_MODEL,
    DEEP_MODEL,
    MODAL_REGION,
)
from .skills import (
    BUILT_IN_SKILLS,
    VOICE_RULES,
    VOICE_CONTEXTS,
    get_skill,
    list_skills,
    get_skill_info,
    get_voice_context,
    create_custom_skill,
)
from .model import BitNetModel, FastBrainLLM, GenerationResult, TokenChunk

__all__ = [
    # Version
    "__version__",
    # Client
    "FastBrainClient",
    "FastBrainWebSocket",
    "StreamingResponse",
    # Config
    "FastBrainConfig",
    "DeepBrainConfig",
    "HybridConfig",
    "ServerConfig",
    "VoiceAgentConfig",
    "get_config",
    "get_hybrid_config",
    "get_server_config",
    "get_voice_config",
    "estimate_monthly_cost",
    "FAST_MODEL",
    "DEEP_MODEL",
    "MODAL_REGION",
    # Skills
    "BUILT_IN_SKILLS",
    "VOICE_RULES",
    "VOICE_CONTEXTS",
    "get_skill",
    "list_skills",
    "get_skill_info",
    "get_voice_context",
    "create_custom_skill",
    # Model (legacy)
    "BitNetModel",
    "FastBrainLLM",
    "GenerationResult",
    "TokenChunk",
]
