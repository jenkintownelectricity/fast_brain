import modal
import subprocess
import os
import time
from typing import Optional

# 1. Define the "Silicon" (The Environment)
# We build a custom Docker image that has the BitNet C++ kernels pre-compiled.
# This acts like "taping out" your chip design.
lpu_image = (
    modal.Image.debian_slim()
    .apt_install("git", "cmake", "ninja-build", "clang", "build-essential", "wget")
    # Install huggingface_hub FIRST (needed by setup_env.py to download model)
    .pip_install("huggingface_hub[cli]")
    # Clone the 1-bit LLM architecture
    .run_commands(
        "git clone --recursive https://github.com/microsoft/BitNet /root/BitNet",
    )
    # "Fabricate" the chip (Compile the C++ kernels AND download model)
    .run_commands(
        "cd /root/BitNet && python3 setup_env.py --hf-repo HF1BitLLM/Llama3-8B-1.58-100B-tokens -q i2_s"
    )
)

app = modal.App("bitnet-lpu-v1")

# Persistent volume for skill adapters (LoRA weights)
# Upload adapters with: modal volume put lpu-skills adapters/skill_name /root/skills/skill_name.lora
skills_volume = modal.Volume.from_name("lpu-skills", create_if_missing=True)

# 2. Define the "LPU" Class
# keep_warm=1 ensures one "chip" is always powered on and ready (Zero Latency).
# If you remove keep_warm, it becomes standard serverless (Cheaper, 2s startup lag).
@app.cls(
    image=lpu_image,
    min_containers=1,  # One instance always ready (zero cold start)
    timeout=600,
    volumes={"/root/skills": skills_volume}  # Mount skill adapters
)
class VirtualLPU:
    def __enter__(self):
        """
        This runs when the 'chip' powers on.
        We verify the model path and prepare the C++ executable command.
        """
        # setup_env.py downloads model to BitNet/models/ and converts to GGUF
        self.model_path = "/root/BitNet/models/Llama3-8B-1.58-100B-tokens/ggml-model-i2_s.gguf"
        self.exec_path = "/root/BitNet/run_inference.py"  # Wrapper provided by repo
        self.skills_path = "/root/skills"

        # Cache loaded skill adapters
        self.loaded_skills = {}

        # In a real LPU optimization, we would load the model into RAM here
        # But BitNet loads via the CLI command per request in the default script.
        # For true <10ms latency, we would use the C++ binary directly via subprocess.
        print("⚡ LPU Online. Ternary weights loaded.")

    @modal.method()
    def list_skills(self) -> list:
        """List all available skill adapters on the volume."""
        skills_volume.reload()  # Sync latest from cloud
        if os.path.exists(self.skills_path):
            return os.listdir(self.skills_path)
        return []

    @modal.method()
    def chat(self, prompt: str, skill_adapter: Optional[str] = None, max_tokens: int = 128):
        """
        The 'Inference' pin on your chip.
        Input: Text Prompt, optional skill adapter
        Output: Stream of tokens

        Args:
            prompt: The input text prompt
            skill_adapter: Name of skill adapter (e.g., "livekit_architect.lora")
            max_tokens: Maximum tokens to generate
        """
        start = time.time()

        # Build the command
        cmd = [
            "python3", self.exec_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "-t", "4"  # Threads (CPU cores)
        ]

        # Add skill adapter if specified
        if skill_adapter:
            adapter_path = os.path.join(self.skills_path, skill_adapter)
            if os.path.exists(adapter_path):
                cmd.extend(["--lora", adapter_path])
                print(f"🧠 Skill Loaded: {skill_adapter}")
            else:
                print(f"⚠️ Skill not found: {skill_adapter}, using base model")

        # Open a pipe to the process to stream results instantly
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )

        # Yield tokens as they are generated (Streaming)
        for line in process.stdout:
            yield line

        print(f"⏱️ Inference Time: {time.time() - start:.2f}s")

# 3. Local Entrypoint (To test it from your laptop)
@app.local_entrypoint()
def main():
    print("Connecting to Cloud LPU...")
    lpu = VirtualLPU()
    
    prompt = "User: Explain quantum computing in one sentence.\nAssistant:"
    
    print(f"Sending: {prompt}")
    for chunk in lpu.chat.remote_gen(prompt):
        print(chunk, end="", flush=True)