"""
Fast Brain - Modal Deployment with HTTP/WebSocket Endpoints

Optimized BitNet LPU deployment for voice assistant integration.
Provides streaming inference with <50ms TTFB target.

Usage:
    modal deploy fast_brain/deploy.py

Endpoints:
    GET  /health          - Health check
    POST /v1/completions  - OpenAI-compatible completions
    POST /v1/chat         - Chat completions with streaming
    WS   /v1/stream       - WebSocket streaming endpoint
"""

import modal
import asyncio
import json
import os
import subprocess
import time
from typing import Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# Modal Configuration
# =============================================================================

# GPU-optimized image for fast inference
fast_brain_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "cmake", "ninja-build", "clang", "build-essential",
        "wget", "curl", "libssl-dev"
    )
    .pip_install(
        "huggingface_hub",
        "fastapi",
        "uvicorn",
        "websockets",
        "httpx",
        "pydantic>=2.0",
        "numpy",
    )
    # Clone BitNet repository
    .run_commands(
        "git clone --recursive https://github.com/microsoft/BitNet /root/BitNet",
    )
    # Download the BitNet Llama3-8B model
    .run_commands(
        'python3 -c "from huggingface_hub import snapshot_download; '
        "snapshot_download('HF1BitLLM/Llama3-8B-1.58-100B-tokens', "
        "local_dir='/root/BitNet/models/Llama3-8B-1.58-100B-tokens')\""
    )
    # Compile optimized kernels
    .run_commands(
        "cd /root/BitNet && python3 setup_env.py -md models/Llama3-8B-1.58-100B-tokens -q i2_s"
    )
)

app = modal.App("fast-brain-lpu")

# Persistent volume for model cache and skill adapters
model_volume = modal.Volume.from_name("fast-brain-models", create_if_missing=True)
skills_volume = modal.Volume.from_name("fast-brain-skills", create_if_missing=True)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = True
    skill_adapter: Optional[str] = None


@dataclass
class InferenceStats:
    """Statistics for inference request"""
    ttfb_ms: float = 0.0
    total_tokens: int = 0
    tokens_per_sec: float = 0.0
    total_time_ms: float = 0.0


# =============================================================================
# BitNet Inference Engine
# =============================================================================

class BitNetEngine:
    """
    High-performance BitNet inference engine optimized for streaming.

    Designed for:
    - <50ms time-to-first-byte
    - >500 tokens/second throughput
    - Efficient memory usage with 1-bit weights
    """

    def __init__(
        self,
        model_path: str,
        exec_path: str,
        skills_path: str,
    ):
        self.model_path = model_path
        self.exec_path = exec_path
        self.skills_path = skills_path
        self.is_ready = False
        self._process = None

    def initialize(self):
        """Initialize the engine and verify model availability"""
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"Model not found at {self.model_path}")
        if not os.path.exists(self.exec_path):
            raise RuntimeError(f"Inference script not found at {self.exec_path}")
        self.is_ready = True

    def list_skills(self) -> list[str]:
        """List available skill adapters"""
        if os.path.exists(self.skills_path):
            return [f for f in os.listdir(self.skills_path) if f.endswith('.lora')]
        return []

    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> AsyncGenerator[tuple[str, Optional[InferenceStats]], None]:
        """
        Generate tokens with streaming output.

        Yields tuples of (token, stats) where stats is only populated
        on the final yield.
        """
        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0

        # Build command
        cmd = [
            "python3", self.exec_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(config.max_tokens),
            "-t", "4",  # CPU threads for hybrid execution
        ]

        # Add skill adapter if specified
        if config.skill_adapter:
            adapter_path = os.path.join(self.skills_path, config.skill_adapter)
            if os.path.exists(adapter_path):
                cmd.extend(["--lora", adapter_path])

        # Start subprocess with line-buffered output
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            # Stream tokens as they arrive
            buffer = ""
            while True:
                chunk = await process.stdout.read(64)
                if not chunk:
                    break

                text = chunk.decode('utf-8', errors='ignore')
                buffer += text

                # Yield complete tokens/words
                while '\n' in buffer or ' ' in buffer:
                    if '\n' in buffer:
                        token, buffer = buffer.split('\n', 1)
                    elif ' ' in buffer:
                        token, buffer = buffer.split(' ', 1)
                        token += ' '
                    else:
                        break

                    if token:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1
                        yield token, None

            # Yield remaining buffer
            if buffer:
                token_count += 1
                yield buffer, None

        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()

        # Calculate final stats
        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000
        ttfb = ((first_token_time - start_time) * 1000) if first_token_time else total_time

        stats = InferenceStats(
            ttfb_ms=ttfb,
            total_tokens=token_count,
            tokens_per_sec=(token_count / (total_time / 1000)) if total_time > 0 else 0,
            total_time_ms=total_time,
        )

        yield "", stats

    def generate_sync(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> tuple[str, InferenceStats]:
        """Synchronous generation for non-streaming requests"""
        start_time = time.perf_counter()

        cmd = [
            "python3", self.exec_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(config.max_tokens),
            "-t", "4",
        ]

        if config.skill_adapter:
            adapter_path = os.path.join(self.skills_path, config.skill_adapter)
            if os.path.exists(adapter_path):
                cmd.extend(["--lora", adapter_path])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        end_time = time.perf_counter()
        output = result.stdout

        stats = InferenceStats(
            ttfb_ms=(end_time - start_time) * 1000,
            total_tokens=len(output.split()),
            tokens_per_sec=len(output.split()) / (end_time - start_time) if (end_time - start_time) > 0 else 0,
            total_time_ms=(end_time - start_time) * 1000,
        )

        return output, stats


# =============================================================================
# FastAPI Application
# =============================================================================

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional as OptionalType

web_app = FastAPI(
    title="Fast Brain LPU",
    description="BitNet inference endpoint for Premier Voice Assistant",
    version="0.1.0",
)


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request"""
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    skill_adapter: OptionalType[str] = None


class ChatMessage(BaseModel):
    """Chat message format"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat completion request"""
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = True
    skill_adapter: OptionalType[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    skills_available: List[str]
    version: str


# Global engine instance (initialized in container)
_engine: Optional[BitNetEngine] = None


def get_engine() -> BitNetEngine:
    """Get the initialized engine instance"""
    global _engine
    if _engine is None or not _engine.is_ready:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return _engine


@web_app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers"""
    global _engine
    return HealthResponse(
        status="healthy" if _engine and _engine.is_ready else "initializing",
        model_loaded=_engine.is_ready if _engine else False,
        skills_available=_engine.list_skills() if _engine else [],
        version="0.1.0",
    )


@web_app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "bitnet-llama3-8b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "fast-brain",
            }
        ]
    }


@web_app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """
    Create a completion (OpenAI-compatible).

    For voice assistant integration, use stream=True for lowest latency.
    """
    engine = get_engine()

    config = GenerationConfig(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream,
        skill_adapter=request.skill_adapter,
    )

    if request.stream:
        async def stream_response():
            async for token, stats in engine.generate_stream(request.prompt, config):
                if token:
                    chunk = {
                        "id": f"cmpl-{int(time.time()*1000)}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": "bitnet-llama3-8b",
                        "choices": [{"text": token, "index": 0, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                if stats:
                    # Final chunk with stats
                    final = {
                        "id": f"cmpl-{int(time.time()*1000)}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": "bitnet-llama3-8b",
                        "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": len(request.prompt.split()),
                            "completion_tokens": stats.total_tokens,
                            "total_tokens": len(request.prompt.split()) + stats.total_tokens,
                        },
                        "metrics": {
                            "ttfb_ms": stats.ttfb_ms,
                            "tokens_per_sec": stats.tokens_per_sec,
                            "total_time_ms": stats.total_time_ms,
                        }
                    }
                    yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        output, stats = engine.generate_sync(request.prompt, config)
        return {
            "id": f"cmpl-{int(time.time()*1000)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "bitnet-llama3-8b",
            "choices": [{"text": output, "index": 0, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": stats.total_tokens,
                "total_tokens": len(request.prompt.split()) + stats.total_tokens,
            },
            "metrics": {
                "ttfb_ms": stats.ttfb_ms,
                "tokens_per_sec": stats.tokens_per_sec,
                "total_time_ms": stats.total_time_ms,
            }
        }


@web_app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatRequest):
    """
    Create a chat completion (OpenAI-compatible).

    Converts chat messages to prompt format for BitNet.
    """
    engine = get_engine()

    # Convert messages to prompt
    prompt_parts = []
    for msg in request.messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    prompt_parts.append("Assistant:")
    prompt = "\n".join(prompt_parts)

    config = GenerationConfig(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=request.stream,
        skill_adapter=request.skill_adapter,
    )

    if request.stream:
        async def stream_response():
            full_response = ""
            async for token, stats in engine.generate_stream(prompt, config):
                if token:
                    full_response += token
                    chunk = {
                        "id": f"chatcmpl-{int(time.time()*1000)}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "bitnet-llama3-8b",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                if stats:
                    final = {
                        "id": f"chatcmpl-{int(time.time()*1000)}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "bitnet-llama3-8b",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                        "metrics": {
                            "ttfb_ms": stats.ttfb_ms,
                            "tokens_per_sec": stats.tokens_per_sec,
                            "total_time_ms": stats.total_time_ms,
                        }
                    }
                    yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    else:
        output, stats = engine.generate_sync(prompt, config)
        return {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "bitnet-llama3-8b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": stats.total_tokens,
                "total_tokens": len(prompt.split()) + stats.total_tokens,
            },
            "metrics": {
                "ttfb_ms": stats.ttfb_ms,
                "tokens_per_sec": stats.tokens_per_sec,
                "total_time_ms": stats.total_time_ms,
            }
        }


@web_app.websocket("/v1/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming.

    Protocol:
    1. Client sends JSON: {"prompt": "...", "max_tokens": 256, ...}
    2. Server streams JSON: {"token": "...", "done": false}
    3. Server sends final: {"token": "", "done": true, "stats": {...}}
    """
    await websocket.accept()
    engine = get_engine()

    try:
        while True:
            # Receive request
            data = await websocket.receive_json()
            prompt = data.get("prompt", "")

            if not prompt:
                await websocket.send_json({"error": "No prompt provided"})
                continue

            config = GenerationConfig(
                max_tokens=data.get("max_tokens", 256),
                temperature=data.get("temperature", 0.7),
                stream=True,
                skill_adapter=data.get("skill_adapter"),
            )

            # Stream response
            async for token, stats in engine.generate_stream(prompt, config):
                if token:
                    await websocket.send_json({
                        "token": token,
                        "done": False,
                    })
                if stats:
                    await websocket.send_json({
                        "token": "",
                        "done": True,
                        "stats": {
                            "ttfb_ms": stats.ttfb_ms,
                            "tokens_per_sec": stats.tokens_per_sec,
                            "total_time_ms": stats.total_time_ms,
                            "total_tokens": stats.total_tokens,
                        }
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"error": str(e)})


# =============================================================================
# Modal Deployment
# =============================================================================

@app.cls(
    image=fast_brain_image,
    gpu="T4",  # NVIDIA T4 for cost-effective inference
    min_containers=1,  # Keep one warm for <5s cold start
    max_containers=10,  # Scale up for high load
    timeout=300,
    volumes={
        "/root/models": model_volume,
        "/root/skills": skills_volume,
    },
)
class FastBrainLPU:
    """
    Fast Brain Language Processing Unit.

    Modal-deployed BitNet inference service with:
    - GPU acceleration for fast inference
    - Warm containers for low latency
    - HTTP/WebSocket endpoints
    """

    @modal.enter()
    def initialize(self):
        """Initialize on container startup"""
        global _engine

        model_path = "/root/BitNet/models/Llama3-8B-1.58-100B-tokens/ggml-model-i2_s.gguf"
        exec_path = "/root/BitNet/run_inference.py"
        skills_path = "/root/skills"

        _engine = BitNetEngine(
            model_path=model_path,
            exec_path=exec_path,
            skills_path=skills_path,
        )
        _engine.initialize()
        print("Fast Brain LPU initialized and ready")

    @modal.asgi_app()
    def serve(self):
        """Serve the FastAPI application"""
        return web_app

    @modal.method()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        skill_adapter: Optional[str] = None,
    ):
        """
        Direct Modal method for streaming generation.

        Use this for Modal-native integrations.
        """
        global _engine
        config = GenerationConfig(
            max_tokens=max_tokens,
            skill_adapter=skill_adapter,
        )

        for token, stats in asyncio.run(
            self._collect_stream(_engine.generate_stream(prompt, config))
        ):
            if token:
                yield token

    async def _collect_stream(self, gen):
        """Helper to collect async generator results"""
        results = []
        async for item in gen:
            results.append(item)
        return results

    @modal.method()
    def health(self) -> dict:
        """Health check method"""
        global _engine
        return {
            "status": "healthy" if _engine and _engine.is_ready else "initializing",
            "model_loaded": _engine.is_ready if _engine else False,
            "skills": _engine.list_skills() if _engine else [],
        }


# =============================================================================
# Local Development Entrypoint
# =============================================================================

@app.local_entrypoint()
def main():
    """Local testing entrypoint"""
    print("Testing Fast Brain LPU...")

    lpu = FastBrainLPU()

    # Test health
    health = lpu.health.remote()
    print(f"Health: {health}")

    # Test generation
    prompt = "User: What is the capital of France?\nAssistant:"
    print(f"\nPrompt: {prompt}")
    print("Response: ", end="", flush=True)

    for chunk in lpu.generate.remote_gen(prompt):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    # For local testing without Modal
    import uvicorn

    # Initialize mock engine for local testing
    _engine = BitNetEngine(
        model_path="/tmp/model",
        exec_path="/tmp/inference.py",
        skills_path="/tmp/skills",
    )

    uvicorn.run(web_app, host="0.0.0.0", port=8000)
