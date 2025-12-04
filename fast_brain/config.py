"""
Fast Brain Configuration

Model and deployment configuration for BitNet LPU.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class ModelConfig:
    """BitNet model configuration"""
    # Model paths
    model_name: str = "HF1BitLLM/Llama3-8B-1.58-100B-tokens"
    model_path: str = "/root/BitNet/models/Llama3-8B-1.58-100B-tokens/ggml-model-i2_s.gguf"
    exec_path: str = "/root/BitNet/run_inference.py"
    skills_path: str = "/root/skills"

    # Generation defaults
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1

    # Inference settings
    num_threads: int = 4
    batch_size: int = 1
    context_length: int = 4096

    # Quantization
    quantization: str = "i2_s"  # BitNet 1.58-bit quantization


@dataclass
class ServerConfig:
    """Server and deployment configuration"""
    # Modal settings
    app_name: str = "fast-brain"
    gpu_type: str = "T4"  # T4 for cost-effective, A10G for higher performance
    min_containers: int = 1  # Keep warm for low latency
    max_containers: int = 10
    container_idle_timeout: int = 300  # 5 minutes
    request_timeout: int = 120

    # API settings
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list = field(default_factory=lambda: ["*"])

    # Performance targets
    target_ttfb_ms: float = 50.0
    target_throughput_tps: float = 500.0
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
    filler_enabled: bool = True  # Enable "hmm", "let me think" fillers
    punctuation_pause_ms: int = 200  # Pause duration at punctuation


def get_config() -> tuple[ModelConfig, ServerConfig, VoiceAgentConfig]:
    """Get configuration from environment or defaults"""
    model_config = ModelConfig(
        model_path=os.getenv("FAST_BRAIN_MODEL_PATH", ModelConfig.model_path),
        skills_path=os.getenv("FAST_BRAIN_SKILLS_PATH", ModelConfig.skills_path),
        max_tokens=int(os.getenv("FAST_BRAIN_MAX_TOKENS", "256")),
        temperature=float(os.getenv("FAST_BRAIN_TEMPERATURE", "0.7")),
    )

    server_config = ServerConfig(
        gpu_type=os.getenv("FAST_BRAIN_GPU", "T4"),
        min_containers=int(os.getenv("FAST_BRAIN_MIN_CONTAINERS", "1")),
        max_containers=int(os.getenv("FAST_BRAIN_MAX_CONTAINERS", "10")),
    )

    voice_config = VoiceAgentConfig(
        max_response_tokens=int(os.getenv("FAST_BRAIN_MAX_RESPONSE_TOKENS", "150")),
        filler_enabled=os.getenv("FAST_BRAIN_FILLER_ENABLED", "true").lower() == "true",
    )

    return model_config, server_config, voice_config
