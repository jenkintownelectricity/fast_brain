"""
Premier Voice Assistant - LiveKit Voice Agent

Voice agent worker with LLM fallback chain:
Fast Brain (primary) -> Groq (fallback 1) -> Anthropic Claude (fallback 2)

Environment Variables:
    FAST_BRAIN_URL: URL to Fast Brain LPU (primary)
    GROQ_API_KEY: Groq API key (fallback 1)
    ANTHROPIC_API_KEY: Anthropic API key (fallback 2)
    LIVEKIT_URL: LiveKit server URL
    LIVEKIT_API_KEY: LiveKit API key
    LIVEKIT_API_SECRET: LiveKit API secret

Usage:
    python -m worker.voice_agent
"""

import asyncio
import logging
import os
import sys
from typing import AsyncGenerator, Optional

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.agents.stt import STT
from livekit.agents.tts import TTS
from livekit.plugins import deepgram, silero, cartesia

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fast_brain.client import FastBrainClient, StreamingResponse

logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)


# =============================================================================
# LLM Providers with Fallback Chain
# =============================================================================

class FastBrainLLM(llm.LLM):
    """
    Fast Brain LLM adapter for LiveKit.

    Primary LLM in the fallback chain.
    Provides streaming responses with <50ms TTFB target.
    """

    def __init__(
        self,
        url: str,
        skill_adapter: Optional[str] = None,
    ):
        super().__init__()
        self.client = FastBrainClient(url, skill_adapter=skill_adapter)
        self._is_healthy = False

    async def check_health(self) -> bool:
        """Check if Fast Brain is available"""
        try:
            health = await self.client.health_check()
            self._is_healthy = health.get("status") == "healthy"
            return self._is_healthy
        except Exception as e:
            logger.warning(f"Fast Brain health check failed: {e}")
            self._is_healthy = False
            return False

    async def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: Optional[llm.FunctionContext] = None,
        temperature: float = 0.7,
        n: int = 1,
        parallel_tool_calls: bool = True,
    ) -> "llm.LLMStream":
        """Generate streaming chat response"""
        return FastBrainLLMStream(
            client=self.client,
            chat_ctx=chat_ctx,
            temperature=temperature,
        )


class FastBrainLLMStream(llm.LLMStream):
    """Streaming response from Fast Brain"""

    def __init__(
        self,
        client: FastBrainClient,
        chat_ctx: llm.ChatContext,
        temperature: float = 0.7,
    ):
        super().__init__(
            chat_ctx=chat_ctx,
            fnc_ctx=None,
        )
        self.client = client
        self.temperature = temperature
        self._task: Optional[asyncio.Task] = None

    async def _run(self) -> None:
        """Run the streaming generation"""
        # Convert ChatContext to messages
        messages = []
        for msg in self._chat_ctx.messages:
            if msg.role == llm.ChatRole.SYSTEM:
                messages.append({"role": "system", "content": msg.content})
            elif msg.role == llm.ChatRole.USER:
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == llm.ChatRole.ASSISTANT:
                messages.append({"role": "assistant", "content": msg.content})

        # Get last user message
        last_user_msg = ""
        system_prompt = None
        history = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                if last_user_msg:
                    history.append({"role": "user", "content": last_user_msg})
                last_user_msg = msg["content"]
            elif msg["role"] == "assistant":
                history.append({"role": "assistant", "content": msg["content"]})

        try:
            async for chunk in self.client.stream_chat(
                message=last_user_msg,
                system_prompt=system_prompt,
                conversation_history=history if history else None,
                temperature=self.temperature,
            ):
                if chunk.text:
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            choices=[
                                llm.Choice(
                                    delta=llm.ChoiceDelta(
                                        role=llm.ChatRole.ASSISTANT,
                                        content=chunk.text,
                                    ),
                                    index=0,
                                )
                            ]
                        )
                    )

                if chunk.is_done:
                    # Log performance metrics
                    if chunk.ttfb_ms:
                        logger.info(f"Fast Brain TTFB: {chunk.ttfb_ms:.1f}ms")
                    if chunk.tokens_per_sec:
                        logger.info(f"Fast Brain throughput: {chunk.tokens_per_sec:.1f} tok/s")

        except Exception as e:
            logger.error(f"Fast Brain stream error: {e}")
            raise


class GroqLLM(llm.LLM):
    """
    Groq LLM adapter for LiveKit.

    Fallback 1 in the chain - fast inference on Groq hardware.
    """

    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        super().__init__()
        self.api_key = api_key
        self.model = model

    async def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: Optional[llm.FunctionContext] = None,
        temperature: float = 0.7,
        n: int = 1,
        parallel_tool_calls: bool = True,
    ) -> "llm.LLMStream":
        """Use LiveKit's Groq plugin"""
        from livekit.plugins import openai as lk_openai

        groq_llm = lk_openai.LLM(
            model=self.model,
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        return await groq_llm.chat(
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            temperature=temperature,
        )


class AnthropicLLM(llm.LLM):
    """
    Anthropic Claude adapter for LiveKit.

    Fallback 2 in the chain - highest quality responses.
    """

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        super().__init__()
        self.api_key = api_key
        self.model = model

    async def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: Optional[llm.FunctionContext] = None,
        temperature: float = 0.7,
        n: int = 1,
        parallel_tool_calls: bool = True,
    ) -> "llm.LLMStream":
        """Use LiveKit's Anthropic plugin"""
        from livekit.plugins import anthropic as lk_anthropic

        claude_llm = lk_anthropic.LLM(
            model=self.model,
            api_key=self.api_key,
        )

        return await claude_llm.chat(
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            temperature=temperature,
        )


class FallbackLLM(llm.LLM):
    """
    LLM with automatic fallback chain.

    Tries providers in order:
    1. Fast Brain (if FAST_BRAIN_URL is set)
    2. Groq (if GROQ_API_KEY is set)
    3. Anthropic (if ANTHROPIC_API_KEY is set)
    """

    def __init__(self):
        super().__init__()
        self.providers: list[tuple[str, llm.LLM]] = []
        self._setup_providers()

    def _setup_providers(self):
        """Initialize available LLM providers"""
        # Fast Brain (primary)
        fast_brain_url = os.getenv("FAST_BRAIN_URL")
        if fast_brain_url:
            logger.info(f"Fast Brain configured: {fast_brain_url}")
            self.providers.append(("fast_brain", FastBrainLLM(fast_brain_url)))

        # Groq (fallback 1)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            logger.info("Groq configured as fallback")
            self.providers.append(("groq", GroqLLM(groq_key)))

        # Anthropic (fallback 2)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            logger.info("Anthropic configured as fallback")
            self.providers.append(("anthropic", AnthropicLLM(anthropic_key)))

        if not self.providers:
            raise RuntimeError(
                "No LLM providers configured. Set FAST_BRAIN_URL, "
                "GROQ_API_KEY, or ANTHROPIC_API_KEY"
            )

    async def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: Optional[llm.FunctionContext] = None,
        temperature: float = 0.7,
        n: int = 1,
        parallel_tool_calls: bool = True,
    ) -> "llm.LLMStream":
        """Try providers in fallback order"""
        last_error = None

        for name, provider in self.providers:
            try:
                logger.debug(f"Trying LLM provider: {name}")

                # Health check for Fast Brain
                if isinstance(provider, FastBrainLLM):
                    if not await provider.check_health():
                        logger.warning(f"Fast Brain unhealthy, trying fallback")
                        continue

                stream = await provider.chat(
                    chat_ctx=chat_ctx,
                    fnc_ctx=fnc_ctx,
                    temperature=temperature,
                    n=n,
                    parallel_tool_calls=parallel_tool_calls,
                )

                logger.info(f"Using LLM provider: {name}")
                return stream

            except Exception as e:
                logger.warning(f"Provider {name} failed: {e}")
                last_error = e
                continue

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


# =============================================================================
# Voice Agent Configuration
# =============================================================================

SYSTEM_PROMPT = """You are a helpful voice assistant for Premier Voice Assistant.
Your responses should be:
- Concise and conversational
- Natural for spoken delivery
- Helpful and friendly

Keep responses brief (1-3 sentences) unless more detail is requested.
If you don't understand something, ask for clarification politely.
"""


async def entrypoint(ctx: JobContext):
    """
    Main voice agent entrypoint.

    Called when a new participant joins the LiveKit room.
    """
    logger.info(f"Voice agent starting for room: {ctx.room.name}")

    # Connect to the room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for first participant
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # Initialize components
    llm_provider = FallbackLLM()

    # STT: Deepgram for accurate transcription
    stt = deepgram.STT(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        language="en-US",
    )

    # TTS: Cartesia for low-latency voice synthesis
    tts = cartesia.TTS(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice="79a125e8-cd45-4c13-8a67-188112f4dd22",  # Sonic voice
    )

    # VAD: Silero for voice activity detection
    vad = silero.VAD.load()

    # Create the voice pipeline
    agent = VoicePipelineAgent(
        vad=vad,
        stt=stt,
        llm=llm_provider,
        tts=tts,
        chat_ctx=llm.ChatContext().append(
            role="system",
            text=SYSTEM_PROMPT,
        ),
        # Performance settings for low latency
        min_endpointing_delay=0.5,  # Faster turn detection
        interrupt_speech_duration=0.5,  # Allow interruptions
        preemptive_synthesis=True,  # Start TTS early
    )

    # Start the agent
    agent.start(ctx.room, participant)

    logger.info("Voice agent started successfully")

    # Keep running until disconnected
    await agent.aclose()


def prewarm(proc: JobProcess):
    """
    Prewarm the worker process.

    Called before accepting jobs to reduce cold start latency.
    """
    logger.info("Prewarming voice agent worker...")

    # Prewarm VAD model
    proc.userdata["vad"] = silero.VAD.load()

    logger.info("Prewarm complete")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
