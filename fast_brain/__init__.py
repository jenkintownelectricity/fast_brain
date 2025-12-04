"""
Fast Brain - BitNet LPU for Premier Voice Assistant

Primary LLM in the voice agent fallback chain:
Fast Brain -> Groq -> Anthropic Claude

Performance targets:
- TTFB: <50ms
- Throughput: >500 tokens/second
- Cold start: <5s

Usage:
    # Deploy to Modal
    modal deploy fast_brain/deploy.py

    # Use the client
    from fast_brain.client import FastBrainClient

    client = FastBrainClient(os.getenv("FAST_BRAIN_URL"))
    async for chunk in client.stream_chat("Hello!"):
        print(chunk.text, end="")
"""

__version__ = "0.1.0"

# Export main classes for convenience
from .client import FastBrainClient, FastBrainWebSocket, StreamingResponse
from .config import ModelConfig, ServerConfig, VoiceAgentConfig, get_config
from .model import BitNetModel, FastBrainLLM, GenerationResult, TokenChunk

__all__ = [
    # Client
    "FastBrainClient",
    "FastBrainWebSocket",
    "StreamingResponse",
    # Config
    "ModelConfig",
    "ServerConfig",
    "VoiceAgentConfig",
    "get_config",
    # Model
    "BitNetModel",
    "FastBrainLLM",
    "GenerationResult",
    "TokenChunk",
]
