"""
Fast Brain Configuration (v3.0)
================================

Configuration for the Fast Brain Hybrid System 1 + System 2 architecture.

System 1 (Fast Brain): Groq + Llama 3.3 70B - ~80ms latency
System 2 (Deep Brain): Claude 3.5 Sonnet - ~2000ms latency

Architecture based on Daniel Kahneman's "Thinking, Fast and Slow":
- System 1: Fast, intuitive, handles 90% of calls
- System 2: Slow, rational, handles complex reasoning
"""

import os
from dataclasses import dataclass, field
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM 1: FAST BRAIN (Groq)
# ═══════════════════════════════════════════════════════════════════════════════

# Groq model settings
FAST_MODEL = "llama-3.3-70b-versatile"
FAST_MAX_TOKENS = 1024
FAST_TEMPERATURE = 0.7

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM 2: DEEP BRAIN (Claude)
# ═══════════════════════════════════════════════════════════════════════════════

# Anthropic model settings
DEEP_MODEL = "claude-3-5-sonnet-20241022"
DEEP_MAX_TOKENS = 2048
DEEP_TEMPERATURE = 0.7

# ═══════════════════════════════════════════════════════════════════════════════
# MODAL DEPLOYMENT
# ═══════════════════════════════════════════════════════════════════════════════

# Performance tuning
CONTAINER_IDLE_TIMEOUT = 300  # 5 minutes
KEEP_WARM = 1  # Keep 1 container warm to avoid cold starts

# CRITICAL: Match your LiveKit Cloud region for lowest latency!
# Options: "us-east", "us-west", "eu-west"
MODAL_REGION = "us-east"

# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASS CONFIGS (for compatibility with existing code)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FastBrainConfig:
    """System 1 (Groq) configuration"""
    model: str = FAST_MODEL
    max_tokens: int = FAST_MAX_TOKENS
    temperature: float = FAST_TEMPERATURE
    provider: str = "groq"


@dataclass
class DeepBrainConfig:
    """System 2 (Claude) configuration"""
    model: str = DEEP_MODEL
    max_tokens: int = DEEP_MAX_TOKENS
    temperature: float = DEEP_TEMPERATURE
    provider: str = "anthropic"


@dataclass
class HybridConfig:
    """Combined hybrid system configuration"""
    fast_brain: FastBrainConfig = field(default_factory=FastBrainConfig)
    deep_brain: DeepBrainConfig = field(default_factory=DeepBrainConfig)
    region: str = MODAL_REGION
    container_idle_timeout: int = CONTAINER_IDLE_TIMEOUT
    keep_warm: int = KEEP_WARM

    # Filler phrase configuration
    filler_enabled: bool = True
    filler_min_duration_ms: int = 2000  # Minimum filler duration in ms

    # Voice configuration
    voice_rules_enabled: bool = True


@dataclass
class ServerConfig:
    """Server and deployment configuration"""
    # Modal settings
    app_name: str = "fast-brain-lpu"
    min_containers: int = 1  # Keep warm for low latency
    max_containers: int = 10
    container_idle_timeout: int = CONTAINER_IDLE_TIMEOUT
    request_timeout: int = 120

    # API settings
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list = field(default_factory=lambda: ["*"])

    # Performance targets
    target_fast_latency_ms: float = 100.0  # System 1 target
    target_deep_latency_ms: float = 2500.0  # System 2 target
    target_cold_start_s: float = 5.0


@dataclass
class VoiceAgentConfig:
    """Configuration for voice agent integration"""
    # System prompts for different scenarios
    default_system_prompt: str = """You are a helpful voice assistant.
Keep responses concise and conversational, suitable for spoken delivery.
Respond naturally as if speaking to someone on the phone."""

    phone_answering_prompt: str = """You are answering a phone call as a professional receptionist.
Be friendly, helpful, and efficient. Keep responses brief and to the point.
If you don't understand something, politely ask for clarification."""

    customer_service_prompt: str = """You are a customer service representative.
Be empathetic, patient, and solution-oriented.
Acknowledge the customer's concerns and provide clear next steps."""

    # Response settings
    max_response_tokens: int = 150  # Keep voice responses short
    filler_enabled: bool = True  # Enable filler phrases for System 2 calls
    punctuation_pause_ms: int = 200  # Pause duration at punctuation


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION GETTERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_hybrid_config() -> HybridConfig:
    """Get hybrid system configuration from environment or defaults."""
    return HybridConfig(
        fast_brain=FastBrainConfig(
            model=os.getenv("FAST_BRAIN_MODEL", FAST_MODEL),
            max_tokens=int(os.getenv("FAST_BRAIN_MAX_TOKENS", str(FAST_MAX_TOKENS))),
            temperature=float(os.getenv("FAST_BRAIN_TEMPERATURE", str(FAST_TEMPERATURE))),
        ),
        deep_brain=DeepBrainConfig(
            model=os.getenv("DEEP_BRAIN_MODEL", DEEP_MODEL),
            max_tokens=int(os.getenv("DEEP_BRAIN_MAX_TOKENS", str(DEEP_MAX_TOKENS))),
            temperature=float(os.getenv("DEEP_BRAIN_TEMPERATURE", str(DEEP_TEMPERATURE))),
        ),
        region=os.getenv("MODAL_REGION", MODAL_REGION),
        keep_warm=int(os.getenv("KEEP_WARM", str(KEEP_WARM))),
    )


def get_server_config() -> ServerConfig:
    """Get server configuration from environment or defaults."""
    return ServerConfig(
        min_containers=int(os.getenv("FAST_BRAIN_MIN_CONTAINERS", "1")),
        max_containers=int(os.getenv("FAST_BRAIN_MAX_CONTAINERS", "10")),
    )


def get_voice_config() -> VoiceAgentConfig:
    """Get voice agent configuration from environment or defaults."""
    return VoiceAgentConfig(
        max_response_tokens=int(os.getenv("FAST_BRAIN_MAX_RESPONSE_TOKENS", "150")),
        filler_enabled=os.getenv("FAST_BRAIN_FILLER_ENABLED", "true").lower() == "true",
    )


def get_config() -> tuple[HybridConfig, ServerConfig, VoiceAgentConfig]:
    """Get all configurations."""
    return get_hybrid_config(), get_server_config(), get_voice_config()


# ═══════════════════════════════════════════════════════════════════════════════
# COST ESTIMATES
# ═══════════════════════════════════════════════════════════════════════════════

COST_ESTIMATES = {
    "system1_per_call": 0.0005,  # Modal compute (Groq API is free)
    "system2_per_call": 0.01,    # Claude API average cost
    "system1_percentage": 90,    # Expected % handled by System 1
    "system2_percentage": 10,    # Expected % escalated to System 2
}


def estimate_monthly_cost(calls_per_day: int) -> dict:
    """
    Estimate monthly costs based on expected call volume.

    Args:
        calls_per_day: Expected number of API calls per day

    Returns:
        Dict with cost breakdown
    """
    calls_per_month = calls_per_day * 30

    system1_calls = calls_per_month * (COST_ESTIMATES["system1_percentage"] / 100)
    system2_calls = calls_per_month * (COST_ESTIMATES["system2_percentage"] / 100)

    system1_cost = system1_calls * COST_ESTIMATES["system1_per_call"]
    system2_cost = system2_calls * COST_ESTIMATES["system2_per_call"]
    total_cost = system1_cost + system2_cost

    # Compare to using Claude for everything
    all_claude_cost = calls_per_month * COST_ESTIMATES["system2_per_call"]
    savings = all_claude_cost - total_cost
    savings_percentage = (savings / all_claude_cost) * 100 if all_claude_cost > 0 else 0

    return {
        "calls_per_month": calls_per_month,
        "system1_calls": int(system1_calls),
        "system2_calls": int(system2_calls),
        "system1_cost": round(system1_cost, 2),
        "system2_cost": round(system2_cost, 2),
        "total_cost": round(total_cost, 2),
        "all_claude_cost": round(all_claude_cost, 2),
        "savings": round(savings, 2),
        "savings_percentage": round(savings_percentage, 1),
    }
