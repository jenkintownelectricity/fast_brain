"""
Fast Brain - Real BitNet LPU Deployment

Uses HuggingFace transformers for BitNet inference with streaming.
Target: <50ms TTFB, >500 tok/s throughput.

Usage:
    modal deploy fast_brain/deploy_bitnet.py
"""

import modal
import time
import json
from typing import Optional, List, AsyncGenerator

# GPU image with transformers and BitNet support
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers>=4.37.0",
        "accelerate",
        "bitsandbytes",
        "fastapi",
        "uvicorn",
        "pydantic>=2.0",
        "huggingface_hub",
    )
)

app = modal.App("fast-brain-lpu")

# =============================================================================
# BitNet Model Wrapper
# =============================================================================

class BitNetInference:
    """BitNet model wrapper with streaming generation."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"  # Use CPU for BitNet compatibility
        self.model_id = "HF1BitLLM/Llama3-8B-1.58-100B-tokens"

    def load(self):
        """Load the BitNet model."""
        import os
        # Completely disable CUDA to avoid initialization errors
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading BitNet model: {self.model_id} (CPU only)")
        start = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        # Load on CPU for BitNet 1.58-bit compatibility
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_time = time.time() - start
        print(f"Model loaded in {load_time:.2f}s on CPU")

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncGenerator[tuple[str, dict], None]:
        """Generate tokens with streaming."""
        import torch

        start_time = time.perf_counter()
        first_token_time = None

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        input_length = inputs.input_ids.shape[1]

        # Generate with streaming
        generated_tokens = 0
        full_response = ""

        with torch.no_grad():
            # Use generate with streaming
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )

            # Decode the generated tokens
            generated_ids = outputs.sequences[0][input_length:]

            # Stream tokens one by one
            for i, token_id in enumerate(generated_ids):
                if token_id == self.tokenizer.eos_token_id:
                    break

                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)

                if first_token_time is None and token_text.strip():
                    first_token_time = time.perf_counter()

                generated_tokens += 1
                full_response += token_text

                yield token_text, None

        # Calculate final stats
        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000
        ttfb = ((first_token_time - start_time) * 1000) if first_token_time else total_time

        stats = {
            "ttfb_ms": round(ttfb, 2),
            "total_time_ms": round(total_time, 2),
            "tokens": generated_tokens,
            "tokens_per_sec": round(generated_tokens / (total_time / 1000), 2) if total_time > 0 else 0,
        }

        yield "", stats

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> tuple[str, dict]:
        """Non-streaming generation."""
        full_response = ""
        stats = {}

        for token, token_stats in self.generate_stream(prompt, max_tokens, temperature):
            if token:
                full_response += token
            if token_stats:
                stats = token_stats

        return full_response, stats


# =============================================================================
# FastAPI Application
# =============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

web_app = FastAPI(
    title="Fast Brain LPU",
    description="BitNet inference endpoint for Premier Voice Assistant",
    version="1.0.0",
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    skills_available: List[str]
    version: str
    model_id: str


# Global model instance
_model: Optional[BitNetInference] = None


@web_app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global _model
    return HealthResponse(
        status="healthy" if _model and _model.model else "initializing",
        model_loaded=_model is not None and _model.model is not None,
        skills_available=["general", "receptionist", "coding"],
        version="1.0.0",
        model_id="HF1BitLLM/Llama3-8B-1.58-100B-tokens",
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
    """Create a chat completion (OpenAI-compatible)."""
    global _model

    if not _model or not _model.model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.perf_counter()

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

    if request.stream:
        async def stream_response():
            for token, stats in _model.generate_stream(
                prompt,
                request.max_tokens,
                request.temperature
            ):
                if token:
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
                        "metrics": stats,
                    }
                    yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
        )
    else:
        response_text, stats = _model.generate(
            prompt,
            request.max_tokens,
            request.temperature,
        )

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
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": stats.get("tokens", 0),
                "total_tokens": len(prompt.split()) + stats.get("tokens", 0),
            },
            "metrics": stats,
        }


# =============================================================================
# Modal Deployment
# =============================================================================

@app.cls(
    image=image,
    cpu=4,  # Use more CPU cores for inference
    memory=32768,  # 32GB RAM for 8B model
    timeout=600,
    container_idle_timeout=300,  # Keep warm for 5 minutes
)
class FastBrainLPU:
    """Fast Brain Language Processing Unit with real BitNet model."""

    @modal.enter()
    def load_model(self):
        """Load model on container startup."""
        global _model
        _model = BitNetInference()
        _model.load()
        print("Fast Brain LPU ready with BitNet model!")

    @modal.asgi_app()
    def serve(self):
        """Serve the FastAPI application."""
        return web_app


# =============================================================================
# Local Testing
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # Initialize model for local testing
    _model = BitNetInference()
    print("Note: Run with GPU for actual inference")

    uvicorn.run(web_app, host="0.0.0.0", port=8000)
