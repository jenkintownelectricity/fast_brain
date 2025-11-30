from unsloth import FastLanguageModel
import torch
from datasets import load_dataset

# 1. Configuration
max_seq_length = 2048 # Supports long documentation
dtype = None # Auto detection
load_in_4bit = True # Use 4bit quantization to fit on free GPUs (Colab T4)

# 2. Load Base Model (Llama-3 is the best base for BitNet compatibility)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instructbnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Add LoRA Adapters (This is the "Empty Cartridge")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Optimized for 0
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 4. The "Brain Juice" (Your Dataset)
# Format your LiveKit/WebRTC docs into a JSONL file:
# {"instruction": "How do I connect to a room?", "input": "", "output": "Use ctx.connect()..."}
dataset = load_dataset("json", data_files="livekit_docs_cleaned.jsonl", split="train")

# 5. Train the Expert
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "output",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Quick run
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer.train()

# 6. Save the Cartridge
# This produces a ~50MB file you can load into your LPU
model.save_pretrained("adapters/livekit_architect")
print("✅ Expert Adapter Created: adapters/livekit_architect")