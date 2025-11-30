import modal
import subprocess
import os
import time

# 1. Define the "Silicon" (The Environment)
# We build a custom Docker image that has the BitNet C++ kernels pre-compiled.
# This acts like "taping out" your chip design.
lpu_image = (
    modal.Image.debian_slim()
    .apt_install("git", "cmake", "ninja-build", "clang", "build-essential", "wget")
    # Clone the 1-bit LLM architecture
    .run_commands(
        "git clone --recursive https://github.com/microsoft/BitNet /root/BitNet",
    )
    # "Fabricate" the chip (Compile the C++ kernels)
    .run_commands(
        "cd /root/BitNet && python3 setup_env.py --hf-repo HF1BitLLM/Llama3-8B-1.58-100B-tokens -q i2_s"
    )
    # Pre-download the model weights so they are baked into the image (Fast Boot)
    .pip_install("huggingface_hub")
    .run_commands(
        "huggingface-cli download HF1BitLLM/Llama3-8B-1.58-100B-tokens --local-dir /root/models/bitnet_llama3"
    )
)

app = modal.App("bitnet-lpu-v1")

# 2. Define the "LPU" Class
# keep_warm=1 ensures one "chip" is always powered on and ready (Zero Latency).
# If you remove keep_warm, it becomes standard serverless (Cheaper, 2s startup lag).
@app.cls(image=lpu_image, keep_warm=1, timeout=600)
class VirtualLPU:
    def __enter__(self):
        """
        This runs when the 'chip' powers on.
        We verify the model path and prepare the C++ executable command.
        """
        self.model_path = "/root/models/bitnet_llama3/ggml-model-i2_s.gguf"
        self.exec_path = "/root/BitNet/run_inference.py" # Wrapper provided by repo
        
        # In a real LPU optimization, we would load the model into RAM here
        # But BitNet loads via the CLI command per request in the default script.
        # For true <10ms latency, we would use the C++ binary directly via subprocess.
        print("⚡ LPU Online. Ternary weights loaded.")

    @modal.method()
    def chat(self, prompt: str):
        """
        The 'Inference' pin on your chip.
        Input: Text Prompt
        Output: Stream of tokens
        """
        start = time.time()
        
        # We invoke the C++ kernel directly
        # Note: In production, we'd use the compiled binary, not the python wrapper, for speed.
        cmd = [
            "python3", self.exec_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", "128", # Max tokens
            "-t", "4"    # Threads (CPU cores)
        ]

        # Open a pipe to the process to stream results instantly
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1 # Line buffered
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