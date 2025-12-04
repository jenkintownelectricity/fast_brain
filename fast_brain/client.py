"""
Fast Brain Client

HTTP and WebSocket client for connecting to Fast Brain LPU.
Used by the LiveKit voice agent worker.
"""

import asyncio
import json
import time
import httpx
from typing import AsyncGenerator, Optional
from dataclasses import dataclass


@dataclass
class StreamingResponse:
    """Response chunk from Fast Brain"""
    text: str
    is_done: bool = False
    ttfb_ms: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    total_tokens: Optional[int] = None


class FastBrainClient:
    """
    Client for Fast Brain LPU HTTP API.

    Usage:
        client = FastBrainClient(url="https://your-modal-url.modal.run")

        # Streaming (recommended for voice)
        async for chunk in client.stream_chat("Hello!"):
            send_to_tts(chunk.text)

        # Non-streaming
        response = await client.chat("Hello!")
    """

    def __init__(
        self,
        url: str,
        timeout: float = 30.0,
        skill_adapter: Optional[str] = None,
    ):
        """
        Initialize the Fast Brain client.

        Args:
            url: Base URL of Fast Brain endpoint (FAST_BRAIN_URL env var)
            timeout: Request timeout in seconds
            skill_adapter: Default skill adapter to use
        """
        self.base_url = url.rstrip('/')
        self.timeout = timeout
        self.default_skill_adapter = skill_adapter
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                http2=True,  # HTTP/2 for better streaming
            )
        return self._client

    async def close(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> dict:
        """
        Check if Fast Brain is healthy.

        Returns dict with status, model_loaded, skills_available.
        """
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    async def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        skill_adapter: Optional[str] = None,
    ) -> str:
        """
        Send a chat message and get complete response.

        For voice applications, prefer stream_chat() for lower latency.
        """
        client = await self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        response = await client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                "skill_adapter": skill_adapter or self.default_skill_adapter,
            },
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def stream_chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[list] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        skill_adapter: Optional[str] = None,
    ) -> AsyncGenerator[StreamingResponse, None]:
        """
        Stream chat response for ultra-low latency.

        This is the recommended method for voice applications.

        Args:
            message: User's message
            system_prompt: Optional system instructions
            conversation_history: List of {"role": str, "content": str}
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            skill_adapter: Skill adapter for specialized responses

        Yields:
            StreamingResponse objects with text chunks
        """
        client = await self._get_client()
        start_time = time.perf_counter()
        first_token_time = None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": message})

        async with client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "skill_adapter": skill_adapter or self.default_skill_adapter,
            },
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Extract content from delta
                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                finish_reason = choices[0].get("finish_reason")

                # Track timing
                if content and first_token_time is None:
                    first_token_time = time.perf_counter()

                # Check for metrics in final chunk
                metrics = data.get("metrics", {})

                if content:
                    yield StreamingResponse(
                        text=content,
                        is_done=False,
                    )

                if finish_reason == "stop":
                    ttfb = metrics.get("ttfb_ms")
                    if ttfb is None and first_token_time:
                        ttfb = (first_token_time - start_time) * 1000

                    yield StreamingResponse(
                        text="",
                        is_done=True,
                        ttfb_ms=ttfb,
                        tokens_per_sec=metrics.get("tokens_per_sec"),
                        total_tokens=metrics.get("total_tokens"),
                    )

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        skill_adapter: Optional[str] = None,
    ) -> str:
        """
        Raw completion endpoint (not chat formatted).

        Use this when you've already formatted the prompt.
        """
        client = await self._get_client()

        response = await client.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                "skill_adapter": skill_adapter or self.default_skill_adapter,
            },
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["text"]

    async def stream_complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        skill_adapter: Optional[str] = None,
    ) -> AsyncGenerator[StreamingResponse, None]:
        """
        Stream raw completion for ultra-low latency.
        """
        client = await self._get_client()
        start_time = time.perf_counter()
        first_token_time = None

        async with client.stream(
            "POST",
            f"{self.base_url}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "skill_adapter": skill_adapter or self.default_skill_adapter,
            },
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                if not choices:
                    continue

                text = choices[0].get("text", "")
                finish_reason = choices[0].get("finish_reason")

                if text and first_token_time is None:
                    first_token_time = time.perf_counter()

                metrics = data.get("metrics", {})

                if text:
                    yield StreamingResponse(text=text, is_done=False)

                if finish_reason == "stop":
                    ttfb = metrics.get("ttfb_ms")
                    if ttfb is None and first_token_time:
                        ttfb = (first_token_time - start_time) * 1000

                    yield StreamingResponse(
                        text="",
                        is_done=True,
                        ttfb_ms=ttfb,
                        tokens_per_sec=metrics.get("tokens_per_sec"),
                        total_tokens=metrics.get("total_tokens"),
                    )


class FastBrainWebSocket:
    """
    WebSocket client for real-time streaming.

    Lower overhead than HTTP for continuous conversation.
    """

    def __init__(self, url: str):
        """
        Initialize WebSocket client.

        Args:
            url: Fast Brain WebSocket URL (ws:// or wss://)
        """
        # Convert HTTP URL to WebSocket
        if url.startswith("https://"):
            self.ws_url = url.replace("https://", "wss://") + "/v1/stream"
        elif url.startswith("http://"):
            self.ws_url = url.replace("http://", "ws://") + "/v1/stream"
        else:
            self.ws_url = url

        self._ws = None

    async def connect(self):
        """Establish WebSocket connection"""
        import websockets
        self._ws = await websockets.connect(self.ws_url)

    async def close(self):
        """Close WebSocket connection"""
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        skill_adapter: Optional[str] = None,
    ) -> AsyncGenerator[StreamingResponse, None]:
        """
        Stream generation over WebSocket.

        Maintains connection for multiple requests.
        """
        if not self._ws:
            await self.connect()

        # Send request
        await self._ws.send(json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "skill_adapter": skill_adapter,
        }))

        # Receive streaming response
        while True:
            message = await self._ws.recv()
            data = json.loads(message)

            if "error" in data:
                raise RuntimeError(data["error"])

            token = data.get("token", "")
            is_done = data.get("done", False)
            stats = data.get("stats", {})

            yield StreamingResponse(
                text=token,
                is_done=is_done,
                ttfb_ms=stats.get("ttfb_ms"),
                tokens_per_sec=stats.get("tokens_per_sec"),
                total_tokens=stats.get("total_tokens"),
            )

            if is_done:
                break

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience function for quick usage
async def quick_chat(
    url: str,
    message: str,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Quick one-shot chat with Fast Brain.

    Usage:
        response = await quick_chat(
            os.getenv("FAST_BRAIN_URL"),
            "Hello, how are you?"
        )
    """
    client = FastBrainClient(url)
    try:
        return await client.chat(message, system_prompt=system_prompt)
    finally:
        await client.close()
