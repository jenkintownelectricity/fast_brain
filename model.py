"""
Fast Brain Model Wrapper

BitNet model wrapper with streaming generation for voice AI.
Optimized for sub-50ms time-to-first-byte.
"""

import asyncio
import subprocess
import time
import os
from typing import AsyncGenerator, Generator, Optional
from dataclasses import dataclass

from .config import ModelConfig, VoiceAgentConfig


@dataclass
class GenerationResult:
    """Result from text generation"""
    text: str
    tokens: int
    ttfb_ms: float
    total_time_ms: float
    tokens_per_sec: float


@dataclass
class TokenChunk:
    """A chunk of generated text"""
    text: str
    is_final: bool = False
    stats: Optional[GenerationResult] = None


class BitNetModel:
    """
    BitNet model wrapper with streaming support.

    Designed for ultra-low latency voice AI inference:
    - <50ms time-to-first-byte
    - >500 tokens/second throughput
    - Efficient 1.58-bit quantization
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.is_loaded = False
        self._process = None

    def load(self) -> bool:
        """
        Verify model files exist and are ready.

        Returns True if model is ready for inference.
        """
        if not os.path.exists(self.config.model_path):
            print(f"Model not found at {self.config.model_path}")
            return False

        if not os.path.exists(self.config.exec_path):
            print(f"Inference script not found at {self.config.exec_path}")
            return False

        self.is_loaded = True
        print(f"BitNet model loaded: {self.config.model_name}")
        return True

    def format_prompt(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[list] = None,
    ) -> str:
        """
        Format input into model prompt.

        Args:
            user_input: The user's current message
            system_prompt: Optional system instructions
            conversation_history: List of {"role": str, "content": str} dicts
        """
        parts = []

        if system_prompt:
            parts.append(f"System: {system_prompt}")

        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")

        parts.append(f"User: {user_input}")
        parts.append("Assistant:")

        return "\n".join(parts)

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        skill_adapter: Optional[str] = None,
    ) -> GenerationResult:
        """
        Synchronous generation (blocking).

        Use for non-streaming scenarios.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        cmd = [
            "python3", self.config.exec_path,
            "-m", self.config.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "-t", str(self.config.num_threads),
        ]

        if skill_adapter:
            adapter_path = os.path.join(self.config.skills_path, skill_adapter)
            if os.path.exists(adapter_path):
                cmd.extend(["--lora", adapter_path])

        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout.strip()

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000
        token_count = len(output.split())

        return GenerationResult(
            text=output,
            tokens=token_count,
            ttfb_ms=total_time,  # For sync, TTFB = total time
            total_time_ms=total_time,
            tokens_per_sec=token_count / (total_time / 1000) if total_time > 0 else 0,
        )

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        skill_adapter: Optional[str] = None,
    ) -> Generator[TokenChunk, None, None]:
        """
        Synchronous streaming generation.

        Yields TokenChunk objects as tokens are generated.
        Final chunk has is_final=True and includes stats.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()
        first_token_time = None
        total_tokens = 0

        max_tokens = max_tokens or self.config.max_tokens

        cmd = [
            "python3", self.config.exec_path,
            "-m", self.config.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "-t", str(self.config.num_threads),
        ]

        if skill_adapter:
            adapter_path = os.path.join(self.config.skills_path, skill_adapter)
            if os.path.exists(adapter_path):
                cmd.extend(["--lora", adapter_path])

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        full_text = ""
        try:
            for line in process.stdout:
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                line = line.rstrip('\n')
                if line:
                    total_tokens += len(line.split())
                    full_text += line
                    yield TokenChunk(text=line, is_final=False)

        finally:
            process.terminate()
            process.wait()

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000
        ttfb = ((first_token_time - start_time) * 1000) if first_token_time else total_time

        stats = GenerationResult(
            text=full_text,
            tokens=total_tokens,
            ttfb_ms=ttfb,
            total_time_ms=total_time,
            tokens_per_sec=total_tokens / (total_time / 1000) if total_time > 0 else 0,
        )

        yield TokenChunk(text="", is_final=True, stats=stats)

    async def generate_stream_async(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        skill_adapter: Optional[str] = None,
    ) -> AsyncGenerator[TokenChunk, None]:
        """
        Async streaming generation.

        Optimized for voice AI with lowest possible TTFB.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()
        first_token_time = None
        total_tokens = 0

        max_tokens = max_tokens or self.config.max_tokens

        cmd = [
            "python3", self.config.exec_path,
            "-m", self.config.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "-t", str(self.config.num_threads),
        ]

        if skill_adapter:
            adapter_path = os.path.join(self.config.skills_path, skill_adapter)
            if os.path.exists(adapter_path):
                cmd.extend(["--lora", adapter_path])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        full_text = ""
        buffer = ""

        try:
            while True:
                # Read in small chunks for lowest latency
                chunk = await process.stdout.read(32)
                if not chunk:
                    break

                text = chunk.decode('utf-8', errors='ignore')
                buffer += text

                # Yield complete words for better TTS
                while ' ' in buffer or '\n' in buffer:
                    if '\n' in buffer:
                        word, buffer = buffer.split('\n', 1)
                        word += '\n'
                    else:
                        word, buffer = buffer.split(' ', 1)
                        word += ' '

                    if word.strip():
                        if first_token_time is None:
                            first_token_time = time.perf_counter()

                        total_tokens += 1
                        full_text += word
                        yield TokenChunk(text=word, is_final=False)

            # Yield remaining buffer
            if buffer.strip():
                total_tokens += 1
                full_text += buffer
                yield TokenChunk(text=buffer, is_final=False)

        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000
        ttfb = ((first_token_time - start_time) * 1000) if first_token_time else total_time

        stats = GenerationResult(
            text=full_text,
            tokens=total_tokens,
            ttfb_ms=ttfb,
            total_time_ms=total_time,
            tokens_per_sec=total_tokens / (total_time / 1000) if total_time > 0 else 0,
        )

        yield TokenChunk(text="", is_final=True, stats=stats)


class FastBrainLLM:
    """
    High-level interface for voice agent integration.

    Wraps BitNetModel with voice-specific features:
    - Automatic prompt formatting
    - System prompt management
    - Filler word injection for latency masking
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        voice_config: Optional[VoiceAgentConfig] = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.voice_config = voice_config or VoiceAgentConfig()
        self.model = BitNetModel(self.model_config)
        self.conversation_history: list = []

    def load(self) -> bool:
        """Load the model"""
        return self.model.load()

    def set_system_prompt(self, prompt: str):
        """Set the system prompt for conversations"""
        self.voice_config.default_system_prompt = prompt

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    async def respond(
        self,
        user_input: str,
        include_history: bool = True,
        skill_adapter: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response to user input.

        This is the main entry point for voice agent integration.

        Args:
            user_input: The user's spoken input (transcribed)
            include_history: Whether to include conversation history
            skill_adapter: Optional skill adapter for specialized responses

        Yields:
            Text chunks suitable for TTS
        """
        # Build prompt
        history = self.conversation_history if include_history else None
        prompt = self.model.format_prompt(
            user_input,
            system_prompt=self.voice_config.default_system_prompt,
            conversation_history=history,
        )

        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
        })

        # Generate response
        full_response = ""
        async for chunk in self.model.generate_stream_async(
            prompt,
            max_tokens=self.voice_config.max_response_tokens,
            skill_adapter=skill_adapter,
        ):
            if not chunk.is_final:
                full_response += chunk.text
                yield chunk.text

        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response.strip(),
        })
