"""
HIVE215 Voice Agent - LiveKit Worker with Turn Detector
========================================================

Voice agent worker with:
- LiveKit turn detector plugin (context-aware turn detection)
- Hybrid System 1 + System 2 Fast Brain (Groq + Claude)
- Filler phrase strategy for zero-latency perception
- Noise cancellation (LiveKit Cloud)

Architecture (Thinking, Fast and Slow):
    User speaks → Deepgram STT → Turn Detector → Fast Brain (Groq)
                                                      │
                                               ┌──────┴──────┐
                                               │             │
                                            [SIMPLE]     [COMPLEX]
                                               │             │
                                               ▼             ▼
                                           Answer      Filler + Claude
                                           (~80ms)        (~2s hidden)

Environment Variables:
    FAST_BRAIN_URL: URL to Fast Brain LPU (Modal)
    LIVEKIT_URL: LiveKit server URL
    LIVEKIT_API_KEY: LiveKit API key
    LIVEKIT_API_SECRET: LiveKit API secret
    DEEPGRAM_API_KEY: Deepgram API key (YOUR key, not LiveKit's)
    CARTESIA_API_KEY: Cartesia API key (YOUR key, not LiveKit's)

Usage:
    python -m worker.voice_agent dev
"""

import asyncio
import logging
import os
import sys
import httpx
from typing import Optional, AsyncGenerator
from dataclasses import dataclass

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
from livekit.plugins import deepgram, silero, cartesia

# Turn detector - the key plugin!
from livekit.plugins.turn_detector import EOUModel

logger = logging.getLogger("hive215-voice-agent")
logger.setLevel(logging.INFO)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Fast Brain endpoint (Modal)
FAST_BRAIN_URL = os.getenv("FAST_BRAIN_URL", "https://your-username--fast-brain-lpu.modal.run")

# Skill routing by phone number (instant, no LLM needed)
SKILL_BY_NUMBER = {
    "+12156100085": "electrician",  # Jenkintown Electricity
    "+12155551234": "plumber",
    "+12155555678": "lawyer",
    # Add more mappings as needed
}
DEFAULT_SKILL = os.getenv("DEFAULT_SKILL", "receptionist")

# Filler phrases for hiding Claude's latency
FILLER_PHRASES = {
    "analysis": "That's a good question. Let me pull up your information and analyze that for you, just a moment...",
    "calculation": "Let me run those numbers for you real quick...",
    "research": "That's a detailed question. Let me look into that for you...",
    "complex": "Hmm, let me think through that carefully...",
    "default": "Let me check on that for you, just a moment...",
}


# =============================================================================
# FAST BRAIN CLIENT (Hybrid System 1 + System 2)
# =============================================================================

@dataclass
class HybridResponse:
    """Response from Fast Brain hybrid endpoint"""
    content: str
    filler: Optional[str] = None
    system_used: str = "fast"
    fast_latency_ms: int = 0
    deep_latency_ms: Optional[int] = None
    total_latency_ms: int = 0


class FastBrainClient:
    """
    Client for Fast Brain hybrid LPU.

    Uses /v1/chat/hybrid endpoint which automatically routes:
    - Simple questions → Groq (~80ms)
    - Complex questions → Filler phrase + Claude (~2s hidden)
    """

    def __init__(self, base_url: str, skill: str = "general"):
        self.base_url = base_url.rstrip("/")
        self.skill = skill
        self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        await self._client.aclose()

    async def health_check(self) -> dict:
        """Check Fast Brain health"""
        response = await self._client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    async def think_hybrid(
        self,
        messages: list[dict],
        user_context: Optional[dict] = None,
    ) -> HybridResponse:
        """
        Call Fast Brain hybrid endpoint.

        Returns both the answer and an optional filler phrase
        to speak while waiting for complex reasoning.
        """
        payload = {
            "messages": messages,
            "skill": self.skill,
            "user_context": user_context,
        }

        response = await self._client.post(
            f"{self.base_url}/v1/chat/hybrid",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return HybridResponse(
            content=data.get("content", ""),
            filler=data.get("filler"),
            system_used=data.get("system_used", "fast"),
            fast_latency_ms=data.get("fast_latency_ms", 0),
            deep_latency_ms=data.get("deep_latency_ms"),
            total_latency_ms=data.get("total_latency_ms", 0),
        )

    async def get_greeting(self) -> dict:
        """Get the greeting for the current skill"""
        response = await self._client.get(f"{self.base_url}/v1/greeting/{self.skill}")
        response.raise_for_status()
        return response.json()


# =============================================================================
# LIVEKIT LLM ADAPTER (Hybrid with Filler Support)
# =============================================================================

class HybridFastBrainLLM(llm.LLM):
    """
    LiveKit LLM adapter for Fast Brain hybrid endpoint.

    Handles the filler phrase strategy automatically:
    1. Sends user message to Fast Brain
    2. If complex (needs Claude), immediately yields filler phrase
    3. Then yields the actual answer when ready
    """

    def __init__(self, client: FastBrainClient):
        super().__init__()
        self.client = client
        self._conversation_history: list[dict] = []

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: Optional[llm.FunctionContext] = None,
        temperature: float = 0.7,
        n: int = 1,
        parallel_tool_calls: bool = True,
    ) -> "HybridLLMStream":
        return HybridLLMStream(
            client=self.client,
            chat_ctx=chat_ctx,
            conversation_history=self._conversation_history,
        )


class HybridLLMStream(llm.LLMStream):
    """
    Streaming response from Fast Brain hybrid endpoint.

    The magic: If a filler phrase is returned, we yield it FIRST
    so TTS can start speaking while Claude thinks.
    """

    def __init__(
        self,
        client: FastBrainClient,
        chat_ctx: llm.ChatContext,
        conversation_history: list[dict],
    ):
        super().__init__(chat_ctx=chat_ctx, fnc_ctx=None)
        self.client = client
        self.conversation_history = conversation_history

    async def _run(self) -> None:
        # Convert ChatContext to messages
        messages = []
        for msg in self._chat_ctx.messages:
            if msg.role == llm.ChatRole.SYSTEM:
                messages.append({"role": "system", "content": msg.content})
            elif msg.role == llm.ChatRole.USER:
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == llm.ChatRole.ASSISTANT:
                messages.append({"role": "assistant", "content": msg.content})

        try:
            # Call Fast Brain hybrid endpoint
            result = await self.client.think_hybrid(messages)

            # Log which system was used
            if result.system_used == "deep":
                logger.info(
                    f"[Hybrid] Used Claude. Fast: {result.fast_latency_ms}ms, "
                    f"Deep: {result.deep_latency_ms}ms, Total: {result.total_latency_ms}ms"
                )
            else:
                logger.info(f"[Hybrid] Used Groq. Latency: {result.fast_latency_ms}ms")

            # THE MAGIC: If there's a filler, yield it first!
            # This lets TTS start speaking while the real answer arrives
            if result.filler:
                logger.info(f"[Filler] Speaking: {result.filler[:50]}...")
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        choices=[
                            llm.Choice(
                                delta=llm.ChoiceDelta(
                                    role=llm.ChatRole.ASSISTANT,
                                    content=result.filler + " ",
                                ),
                                index=0,
                            )
                        ]
                    )
                )

            # Now yield the actual answer
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(
                                role=llm.ChatRole.ASSISTANT,
                                content=result.content,
                            ),
                            index=0,
                        )
                    ]
                )
            )

            # Update conversation history
            if messages:
                self.conversation_history.extend(messages[-2:])  # Keep last exchange

        except Exception as e:
            logger.error(f"Fast Brain error: {e}")
            # Fallback response
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(
                                role=llm.ChatRole.ASSISTANT,
                                content="I'm sorry, I'm having trouble right now. Can you try again?",
                            ),
                            index=0,
                        )
                    ]
                )
            )


# =============================================================================
# SKILL SELECTOR (Logic-based, ~0ms - NOT LLM-based!)
# =============================================================================

def get_skill_from_sip_metadata(ctx: JobContext) -> str:
    """
    Get skill from SIP trunk metadata.

    CRITICAL: This is logic-based (instant), not LLM-based (slow).
    Uses the phone number that was dialed to determine the skill.
    """
    # Try to get phone number from room metadata or SIP info
    room_name = ctx.room.name

    # Check if room name contains skill hint (e.g., "room_electrician_cust123")
    if "_" in room_name:
        parts = room_name.split("_")
        if len(parts) >= 2 and parts[1] in SKILL_BY_NUMBER.values():
            logger.info(f"Skill from room name: {parts[1]}")
            return parts[1]

    # Check participant metadata for SIP info
    # LiveKit provides sip.trunkPhoneNumber in metadata
    try:
        for participant in ctx.room.remote_participants.values():
            metadata = participant.metadata
            if metadata:
                import json
                meta = json.loads(metadata)
                phone = meta.get("sip.trunkPhoneNumber") or meta.get("phone_number")
                if phone and phone in SKILL_BY_NUMBER:
                    skill = SKILL_BY_NUMBER[phone]
                    logger.info(f"Skill from phone {phone}: {skill}")
                    return skill
    except Exception as e:
        logger.warning(f"Could not parse SIP metadata: {e}")

    logger.info(f"Using default skill: {DEFAULT_SKILL}")
    return DEFAULT_SKILL


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

def get_system_prompt(skill: str) -> str:
    """Get the system prompt for a skill"""

    prompts = {
        "receptionist": """You are a professional receptionist. Be warm, helpful, and efficient.
- Answer calls warmly and professionally
- Collect caller's name and reason for calling
- For complex questions, use the ask_expert tool

VOICE RULES:
- Use contractions (I'm, you're, we'll)
- Keep sentences under 20 words
- No markdown or formatting
- Be conversational, not robotic""",

        "electrician": """You are Sparky, a friendly receptionist for Jenkintown Electricity.

HANDLE DIRECTLY:
- Greetings and basic questions
- Collecting customer info (name, phone, address)
- Understanding their issue (outlets, panels, lighting)
- Scheduling callbacks

USE ask_expert FOR:
- Electrical code questions
- Cost estimates
- Technical troubleshooting
- Bill analysis

VOICE RULES:
- Use contractions (I'm, you're, we'll)
- Keep sentences under 20 words
- Be conversational""",

        "plumber": """You are a receptionist for a plumbing company.

HANDLE DIRECTLY:
- Greetings and basic questions
- Collecting customer info
- Understanding plumbing issues
- Scheduling callbacks

USE ask_expert FOR:
- Plumbing code questions
- Cost estimates
- Technical advice

VOICE RULES:
- Use contractions
- Keep sentences short
- Be friendly and down-to-earth""",

        "lawyer": """You are an intake specialist for a law firm.

HANDLE DIRECTLY:
- Greetings
- Collecting contact info
- Understanding general nature of legal matter
- Scheduling consultations

USE ask_expert FOR:
- ANY legal questions or advice
- Case evaluation

CRITICAL: You are NOT an attorney. NEVER give legal advice.

VOICE RULES:
- Professional tone
- Use contractions
- Keep sentences short""",
    }

    return prompts.get(skill, prompts["receptionist"])


# =============================================================================
# VOICE AGENT ENTRYPOINT
# =============================================================================

async def entrypoint(ctx: JobContext):
    """
    Main voice agent entrypoint with turn detector.

    Pipeline:
    1. Phone call → LiveKit Cloud (noise cancellation FREE)
    2. Audio → Deepgram Nova-3 (YOUR key)
    3. Transcript → Turn Detector (FREE, runs locally)
    4. Text → Fast Brain Hybrid (Groq + Claude)
    5. Response → Cartesia Sonic (YOUR key)
    6. Audio → Caller
    """
    logger.info(f"Voice agent starting for room: {ctx.room.name}")

    # Connect to the room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for participant
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # Get skill from SIP metadata (instant, ~0ms)
    skill = get_skill_from_sip_metadata(ctx)
    logger.info(f"Selected skill: {skill}")

    # Initialize Fast Brain client
    brain_client = FastBrainClient(
        base_url=FAST_BRAIN_URL,
        skill=skill,
    )

    # Verify Fast Brain is healthy
    try:
        health = await brain_client.health_check()
        logger.info(f"Fast Brain: {health.get('architecture', 'connected')}")
        logger.info(f"  System 1: {health.get('system1', {}).get('model', 'unknown')}")
        logger.info(f"  System 2: {health.get('system2', {}).get('model', 'unknown')}")
    except Exception as e:
        logger.warning(f"Fast Brain health check failed: {e}")

    # Create LLM adapter
    llm_adapter = HybridFastBrainLLM(brain_client)

    # Initialize components (YOUR API KEYS - not LiveKit's!)
    # This avoids the inference gateway limits

    # VAD: Silero (runs locally, FREE)
    vad = silero.VAD.load()

    # STT: Deepgram Nova-3 (YOUR key = unlimited, ~$0.006/min)
    stt = deepgram.STT(
        api_key=os.environ.get("DEEPGRAM_API_KEY"),
        model="nova-2",
        language="en",
    )

    # TTS: Cartesia Sonic (YOUR key = unlimited, ~$0.0225/min)
    tts = cartesia.TTS(
        api_key=os.environ.get("CARTESIA_API_KEY"),
        model="sonic-english",
        voice="79a125e8-cd45-4c13-8a67-188112f4dd22",  # Professional female
    )

    # TURN DETECTOR: LiveKit's plugin (FREE, runs locally!)
    # This is the key improvement over basic VAD
    turn_detector = EOUModel()

    # Get system prompt for skill
    system_prompt = get_system_prompt(skill)

    # Create the voice pipeline with turn detector
    agent = VoicePipelineAgent(
        vad=vad,
        stt=stt,
        llm=llm_adapter,
        tts=tts,
        turn_detector=turn_detector,  # <-- THE KEY ADDITION!
        chat_ctx=llm.ChatContext().append(
            role="system",
            text=system_prompt,
        ),
        # Performance settings
        min_endpointing_delay=0.5,
        max_endpointing_delay=6.0,  # Wait longer if turn detector thinks user will continue
        interrupt_speech_duration=0.5,
        preemptive_synthesis=True,
    )

    # Event handlers
    @agent.on("user_speech_committed")
    def on_user_speech(transcript: str):
        logger.info(f"User: {transcript}")

    @agent.on("agent_speech_committed")
    def on_agent_speech(text: str):
        logger.info(f"Agent: {text[:100]}...")

    # Start the agent
    agent.start(ctx.room, participant)

    # Get and speak greeting
    try:
        greeting = await brain_client.get_greeting()
        greeting_text = greeting.get("text", "Hello! How can I help you today?")
        logger.info(f"Speaking greeting: {greeting_text}")
        await agent.say(greeting_text, allow_interruptions=True)
    except Exception as e:
        logger.warning(f"Could not get greeting: {e}")
        await agent.say("Hello! How can I help you today?", allow_interruptions=True)

    logger.info("Voice agent started successfully")

    # Keep running until disconnected
    await agent.aclose()

    # Cleanup
    await brain_client.close()


def prewarm(proc: JobProcess):
    """
    Prewarm the worker process.

    Called before accepting jobs to reduce cold start latency.
    """
    logger.info("Prewarming voice agent worker...")

    # Prewarm VAD model (runs locally)
    proc.userdata["vad"] = silero.VAD.load()

    # Prewarm turn detector model (runs locally, ~200MB)
    proc.userdata["turn_detector"] = EOUModel()

    logger.info("Prewarm complete - VAD and turn detector loaded")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
