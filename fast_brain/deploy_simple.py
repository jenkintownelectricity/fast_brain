"""
Fast Brain - Simple Modal Deployment (Stub for Dashboard Testing)

A lightweight FastAPI deployment that provides the same API interface
as the full BitNet LPU, but uses simulated responses for testing.

Usage:
    modal deploy fast_brain/deploy_simple.py
"""

import modal
import time
import json
from typing import Optional, List

# Simple image without BitNet compilation
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "uvicorn",
        "pydantic>=2.0",
    )
)

app = modal.App("fast-brain-lpu")

# =============================================================================
# FastAPI Application
# =============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

web_app = FastAPI(
    title="Fast Brain LPU",
    description="BitNet inference endpoint for Premier Voice Assistant",
    version="0.1.0",
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False
    skill_adapter: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    skills_available: List[str]
    version: str


@web_app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        skills_available=["general", "coding", "creative"],
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


@web_app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatRequest):
    """
    Create a chat completion (OpenAI-compatible).

    This is a stub implementation that returns simulated responses.
    """
    start_time = time.perf_counter()

    # Get the last user message
    user_message = ""
    for msg in request.messages:
        if msg.role == "user":
            user_message = msg.content

    # Simulate processing time for realistic TTFB
    time.sleep(0.03)  # 30ms simulated TTFB

    # Generate a simple response
    response_text = f"Hello! I received your message: '{user_message[:50]}...'. This is a test response from Fast Brain LPU. The full BitNet model will be integrated soon for real inference."

    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000

    if request.stream:
        async def stream_response():
            words = response_text.split()
            for i, word in enumerate(words):
                chunk = {
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "bitnet-llama3-8b",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": word + " "},
                        "finish_reason": None if i < len(words) - 1 else "stop",
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                time.sleep(0.01)  # Simulate token generation
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
        )
    else:
        return {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "bitnet-llama3-8b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(user_message.split()) + len(response_text.split()),
            },
            "metrics": {
                "ttfb_ms": 30.0,
                "tokens_per_sec": len(response_text.split()) / (total_time_ms / 1000),
                "total_time_ms": total_time_ms,
            }
        }


# =============================================================================
# Modal Deployment
# =============================================================================

@app.function(
    image=image,
    cpu=1,
    memory=512,
    timeout=300,
)
@modal.asgi_app()
def serve():
    """Serve the FastAPI application"""
    return web_app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(web_app, host="0.0.0.0", port=8000)
