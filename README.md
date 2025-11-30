# fast_brain

Virtual Chip - A serverless BitNet.cpp LPU (Language Processing Unit) on Modal.

## Overview

This project creates a "Virtual Chip" that:
- Compiles specialized 1-bit kernels (BitNet.cpp) in a serverless environment
- Downloads and serves the BitNet Llama3-8B model
- Exposes a high-speed inference endpoint
- Supports hot-swappable skill adapters (LoRA)

## Quick Start

### 1. Install Modal

```bash
pip install modal
```

### 2. Setup Token

```bash
modal setup
```

### 3. Deploy the Virtual Chip

```bash
modal deploy modal_lpu.py
```

## Training Expert Skills

Use `training_the_expert.py` to create specialized LoRA adapters:

```bash
# Prepare your dataset as JSONL:
# {"instruction": "How do I connect?", "input": "", "output": "Use ctx.connect()..."}

python training_the_expert.py
```

## Upload Skills to the LPU

After training, upload adapters to the Modal volume:

```bash
modal volume put lpu-skills adapters/livekit_architect /root/skills/livekit_architect.lora
```

## Integration Guide

### Connect from Your Application

```python
import modal

# Connect to the remote class
lpu = modal.Cls.lookup("bitnet-lpu-v1", "VirtualLPU")()

# Generate with base model
for chunk in lpu.chat.remote_gen("User: Hello! Assistant:"):
    print(chunk, end="", flush=True)

# Generate with a skill adapter
for chunk in lpu.chat.remote_gen(
    "User: Help me build a LiveKit agent! Assistant:",
    skill_adapter="livekit_architect.lora"
):
    print(chunk, end="", flush=True)

# List available skills
skills = lpu.list_skills.remote()
print(f"Available skills: {skills}")
```

### LiveKit Agent Integration

```python
import modal
from livekit.agents import AutoSubscribe, JobContext, llm

# Get the Virtual LPU
lpu = modal.Cls.lookup("bitnet-lpu-v1", "VirtualLPU")()

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Use the LPU for fast inference
    async def generate_response(prompt: str) -> str:
        response = ""
        for chunk in lpu.chat.remote_gen(prompt, skill_adapter="livekit_architect.lora"):
            response += chunk
        return response

    # Your agent logic here...
```

## Architecture

```
+-------------------+     +------------------+     +------------------+
|   Your App        | --> |   Modal Cloud    | --> |   Virtual LPU    |
|   (LiveKit Agent) |     |   (Serverless)   |     |   (BitNet.cpp)   |
+-------------------+     +------------------+     +------------------+
                                   |
                          +--------v--------+
                          |  Skills Volume  |
                          |  (LoRA Adapters)|
                          +-----------------+
```

## Configuration

- `keep_warm=1`: One instance always ready (zero cold start)
- `keep_warm=0`: Standard serverless (~2s startup)
- `timeout=600`: 10 minute max inference time

## Files

- `modal_lpu.py` - Virtual Chip deployment code
- `training_the_expert.py` - LoRA adapter training script
- `bitnet_lpu_roadmap.html` - Project roadmap
- `lora_swarm.html` - Multi-skill swarm documentation
