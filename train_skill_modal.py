"""
HIVE215 Skill Training - Modal Deployment
==========================================

One-click LoRA fine-tuning for custom skills using Unsloth + QLoRA.

Features:
- 2-5x faster training with 80% less VRAM (Unsloth)
- QLoRA 4-bit quantization for efficiency
- Real-time progress streaming
- Auto-export to Modal volume
- Clear error messages

Usage:
    # Train a specific skill
    modal run train_skill_modal.py --skill-id molasses-master-expert

    # Train with custom settings
    modal run train_skill_modal.py --skill-id my-skill --epochs 3 --lr 2e-4

Requirements:
    - Training data in database (created via dashboard)
    - Modal account with GPU access

Cost Estimate:
    - ~$0.50-2.00 per training run (A10G, 10-30 min)
"""

import modal
import os
import json
from datetime import datetime
from pathlib import Path

# =============================================================================
# MODAL APP SETUP
# =============================================================================

app = modal.App("hive215-skill-trainer")

# Shared volume for trained adapters
adapters_volume = modal.Volume.from_name("hive215-adapters", create_if_missing=True)

# Shared volume for database access
data_volume = modal.Volume.from_name("hive215-data", create_if_missing=True)

# Training image with Unsloth + dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Required for unsloth installation
    .pip_install(
        # Core PyTorch first
        "torch>=2.1.0",
        "torchvision",  # Required by unsloth
        "triton",
    )
    .pip_install(
        # Unsloth for fast training (install after torch)
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    )
    .pip_install(
        # Training dependencies
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "trl>=0.7.0",
        "bitsandbytes>=0.41.0",
        # Utilities
        "huggingface_hub",
        "safetensors",
        "sentencepiece",
        "xformers",
    )
    .add_local_file("database.py", "/root/database.py")
)

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Default training config (optimized for voice skills)
DEFAULT_CONFIG = {
    # Model
    "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "max_seq_length": 2048,

    # LoRA
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],

    # Training
    "epochs": 3,
    "batch_size": 2,
    "gradient_accumulation": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 5,
    "weight_decay": 0.01,
    "max_steps": -1,  # -1 = use epochs

    # Optimization
    "use_gradient_checkpointing": True,
    "use_flash_attention": True,
    "optim": "adamw_8bit",

    # Output
    "output_format": "lora",  # lora, merged, gguf
    "push_to_hub": False,
}

# =============================================================================
# TRAINING CLASS
# =============================================================================

@app.cls(
    image=training_image,
    gpu="A10G",  # 24GB VRAM - good balance of cost/performance
    timeout=3600,  # 1 hour max
    volumes={
        "/data": data_volume,
        "/adapters": adapters_volume,
    },
)
class SkillTrainer:
    """
    Fine-tune a skill using Unsloth + QLoRA.

    Optimized for voice AI skills with:
    - Fast training (10-30 min typical)
    - Low memory usage (fits on A10G)
    - High quality results
    """

    @modal.enter()
    def setup(self):
        """Initialize on container start."""
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TRAINER] Device: {self.device}")
        print(f"[TRAINER] CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[TRAINER] GPU: {torch.cuda.get_device_name()}")
            print(f"[TRAINER] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def _get_training_data(self, skill_id: str) -> dict:
        """
        Load training data for a skill from the database.

        Returns:
            dict with 'data' (list of examples) and 'metadata'
        """
        import sys
        sys.path.insert(0, "/root")
        os.environ['HIVE215_DB_PATH'] = '/data/hive215.db'

        try:
            import database as db

            # Get skill info
            skill = db.get_skill(skill_id)
            if not skill:
                return {"error": f"Skill '{skill_id}' not found in database"}

            # Get training data
            training_records = db.get_training_data(skill_id)

            if not training_records:
                # Try to generate from skill knowledge
                knowledge = skill.get('knowledge', [])
                system_prompt = skill.get('system_prompt', '')

                if not knowledge and not system_prompt:
                    return {"error": f"No training data or knowledge found for skill '{skill_id}'"}

                # Auto-generate training examples from knowledge
                training_data = self._generate_from_knowledge(skill)
            else:
                training_data = [
                    {
                        "instruction": r.get('system_prompt', skill.get('system_prompt', '')),
                        "input": r.get('user_input', ''),
                        "output": r.get('assistant_output', ''),
                    }
                    for r in training_records
                ]

            return {
                "data": training_data,
                "metadata": {
                    "skill_id": skill_id,
                    "skill_name": skill.get('name', skill_id),
                    "examples": len(training_data),
                    "system_prompt": skill.get('system_prompt', ''),
                }
            }

        except Exception as e:
            return {"error": f"Failed to load training data: {str(e)}"}

    def _generate_from_knowledge(self, skill: dict) -> list:
        """Generate training examples from skill knowledge."""
        training_data = []
        system_prompt = skill.get('system_prompt', 'You are a helpful assistant.')
        knowledge = skill.get('knowledge', [])
        name = skill.get('name', 'Assistant')

        # Greeting examples
        greetings = ["Hello", "Hi", "Hey there", "Good morning", "I need help"]
        for greet in greetings:
            training_data.append({
                "instruction": system_prompt,
                "input": greet,
                "output": f"Hello! I'm {name}. How can I help you today?"
            })

        # Knowledge-based examples
        for item in knowledge:
            if isinstance(item, str) and item.strip():
                # Create Q&A from knowledge item
                training_data.append({
                    "instruction": system_prompt,
                    "input": f"Tell me about {item.split(':')[0] if ':' in item else item[:50]}",
                    "output": item
                })

        return training_data

    @modal.method()
    def train(
        self,
        skill_id: str,
        config: dict = None,
        callback_url: str = None,
    ) -> dict:
        """
        Train a LoRA adapter for a skill.

        Args:
            skill_id: ID of the skill to train
            config: Training config (uses defaults if not provided)
            callback_url: Optional webhook for progress updates

        Returns:
            dict with training results and adapter path
        """
        import torch
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
        import time

        start_time = time.time()
        config = {**DEFAULT_CONFIG, **(config or {})}

        print(f"\n{'='*60}")
        print(f"🚀 HIVE215 SKILL TRAINER")
        print(f"{'='*60}")
        print(f"Skill ID: {skill_id}")
        print(f"Base Model: {config['base_model']}")
        print(f"LoRA Rank: {config['lora_r']}")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Epochs: {config['epochs']}")
        print(f"{'='*60}\n")

        # Step 1: Load training data
        print("[1/5] Loading training data...")
        data_result = self._get_training_data(skill_id)

        if "error" in data_result:
            return {"success": False, "error": data_result["error"]}

        training_data = data_result["data"]
        metadata = data_result["metadata"]

        print(f"  ✓ Loaded {len(training_data)} examples")
        print(f"  ✓ Skill: {metadata['skill_name']}")

        if len(training_data) < 10:
            print(f"  ⚠ Warning: Only {len(training_data)} examples. Consider adding more for better results.")

        # Step 2: Load base model
        print("\n[2/5] Loading base model...")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config['base_model'],
                max_seq_length=config['max_seq_length'],
                dtype=None,  # Auto-detect
                load_in_4bit=True,
            )
            print(f"  ✓ Model loaded: {config['base_model']}")
        except Exception as e:
            return {"success": False, "error": f"Failed to load model: {str(e)}"}

        # Step 3: Add LoRA adapters
        print("\n[3/5] Adding LoRA adapters...")
        try:
            model = FastLanguageModel.get_peft_model(
                model,
                r=config['lora_r'],
                target_modules=config['target_modules'],
                lora_alpha=config['lora_alpha'],
                lora_dropout=config['lora_dropout'],
                bias="none",
                use_gradient_checkpointing="unsloth" if config['use_gradient_checkpointing'] else False,
                random_state=42,
            )
            print(f"  ✓ LoRA configured (r={config['lora_r']}, alpha={config['lora_alpha']})")

            # Count trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  ✓ Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

        except Exception as e:
            return {"success": False, "error": f"Failed to add LoRA: {str(e)}"}

        # Step 4: Prepare dataset
        print("\n[4/5] Preparing dataset...")
        try:
            # Format for Llama 3 Instruct
            def format_example(example):
                system = example.get('instruction', metadata['system_prompt'])
                user = example.get('input', '')
                assistant = example.get('output', '')

                # Llama 3 format
                text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant}<|eot_id|>"""
                return {"text": text}

            formatted_data = [format_example(ex) for ex in training_data]
            dataset = Dataset.from_list(formatted_data)

            print(f"  ✓ Dataset prepared: {len(dataset)} examples")

        except Exception as e:
            return {"success": False, "error": f"Failed to prepare dataset: {str(e)}"}

        # Step 5: Train
        print("\n[5/5] Training...")
        try:
            # Calculate max steps
            if config['max_steps'] > 0:
                max_steps = config['max_steps']
            else:
                steps_per_epoch = len(dataset) // (config['batch_size'] * config['gradient_accumulation'])
                max_steps = steps_per_epoch * config['epochs']

            print(f"  → Max steps: {max_steps}")
            print(f"  → Batch size: {config['batch_size']} x {config['gradient_accumulation']} gradient accumulation")

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=config['max_seq_length'],
                dataset_num_proc=2,
                packing=False,
                args=TrainingArguments(
                    per_device_train_batch_size=config['batch_size'],
                    gradient_accumulation_steps=config['gradient_accumulation'],
                    warmup_steps=config['warmup_steps'],
                    max_steps=max_steps,
                    learning_rate=config['learning_rate'],
                    weight_decay=config['weight_decay'],
                    fp16=not torch.cuda.is_bf16_supported(),
                    bf16=torch.cuda.is_bf16_supported(),
                    logging_steps=1,
                    optim=config['optim'],
                    seed=42,
                    output_dir=f"/tmp/training_{skill_id}",
                    report_to="none",  # Disable W&B for now
                ),
            )

            # Train with progress tracking
            print("\n  Training progress:")
            print("  " + "-" * 50)

            train_result = trainer.train()

            print("  " + "-" * 50)
            print(f"  ✓ Training complete!")
            print(f"  → Final loss: {train_result.training_loss:.4f}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"Training failed: {str(e)}"}

        # Step 6: Save adapter
        print("\n[6/5] Saving adapter...")
        try:
            adapter_path = f"/adapters/{skill_id}"
            os.makedirs(adapter_path, exist_ok=True)

            # Save LoRA adapter
            model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)

            # Save training metadata
            training_meta = {
                "skill_id": skill_id,
                "skill_name": metadata['skill_name'],
                "base_model": config['base_model'],
                "training_examples": len(training_data),
                "final_loss": train_result.training_loss,
                "epochs": config['epochs'],
                "lora_r": config['lora_r'],
                "lora_alpha": config['lora_alpha'],
                "learning_rate": config['learning_rate'],
                "training_time_seconds": time.time() - start_time,
                "trained_at": datetime.now().isoformat(),
            }

            with open(f"{adapter_path}/training_metadata.json", "w") as f:
                json.dump(training_meta, f, indent=2)

            # Commit volume changes
            adapters_volume.commit()

            print(f"  ✓ Adapter saved to: {adapter_path}")
            print(f"  ✓ Training metadata saved")

        except Exception as e:
            return {"success": False, "error": f"Failed to save adapter: {str(e)}"}

        # Summary
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Skill: {metadata['skill_name']} ({skill_id})")
        print(f"Examples: {len(training_data)}")
        print(f"Final Loss: {train_result.training_loss:.4f}")
        print(f"Time: {total_time/60:.1f} minutes")
        print(f"Adapter: {adapter_path}")
        print(f"{'='*60}\n")

        return {
            "success": True,
            "skill_id": skill_id,
            "adapter_path": adapter_path,
            "training_examples": len(training_data),
            "final_loss": train_result.training_loss,
            "training_time_seconds": total_time,
            "metadata": training_meta,
        }

    @modal.method()
    def test_adapter(self, skill_id: str, prompt: str) -> str:
        """
        Test a trained adapter with a prompt.

        Args:
            skill_id: ID of the trained skill
            prompt: User message to test

        Returns:
            Model response
        """
        from unsloth import FastLanguageModel

        adapter_path = f"/adapters/{skill_id}"

        if not os.path.exists(adapter_path):
            return f"Error: No adapter found for skill '{skill_id}'"

        # Load metadata
        meta_path = f"{adapter_path}/training_metadata.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            system_prompt = meta.get('system_prompt', 'You are a helpful assistant.')
        else:
            system_prompt = 'You are a helpful assistant.'

        # Load model with adapter
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(model)

        # Format prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()

        return response

    @modal.method()
    def list_adapters(self) -> list:
        """List all trained adapters."""
        adapters = []

        if os.path.exists("/adapters"):
            for name in os.listdir("/adapters"):
                adapter_path = f"/adapters/{name}"
                if os.path.isdir(adapter_path):
                    meta_path = f"{adapter_path}/training_metadata.json"
                    if os.path.exists(meta_path):
                        with open(meta_path) as f:
                            meta = json.load(f)
                        adapters.append(meta)
                    else:
                        adapters.append({"skill_id": name, "status": "unknown"})

        return adapters


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main(
    skill_id: str = None,
    epochs: int = 3,
    lr: float = 2e-4,
    lora_r: int = 16,
    test_prompt: str = None,
    list_adapters: bool = False,
):
    """
    Train or test a skill adapter.

    Examples:
        # Train a skill
        modal run train_skill_modal.py --skill-id my-skill

        # Train with custom settings
        modal run train_skill_modal.py --skill-id my-skill --epochs 5 --lr 1e-4

        # Test a trained adapter
        modal run train_skill_modal.py --skill-id my-skill --test-prompt "Hello"

        # List all adapters
        modal run train_skill_modal.py --list-adapters
    """
    trainer = SkillTrainer()

    if list_adapters:
        print("\n📦 Trained Adapters:")
        print("-" * 50)
        adapters = trainer.list_adapters.remote()
        if adapters:
            for a in adapters:
                print(f"  • {a.get('skill_id', 'unknown')}")
                if 'skill_name' in a:
                    print(f"    Name: {a['skill_name']}")
                if 'final_loss' in a:
                    print(f"    Loss: {a['final_loss']:.4f}")
                if 'trained_at' in a:
                    print(f"    Trained: {a['trained_at']}")
                print()
        else:
            print("  No adapters found.")
        return

    if not skill_id:
        print("Error: --skill-id is required")
        print("Usage: modal run train_skill_modal.py --skill-id your-skill-id")
        return

    if test_prompt:
        print(f"\n🧪 Testing adapter for skill: {skill_id}")
        print(f"Prompt: {test_prompt}")
        print("-" * 50)
        response = trainer.test_adapter.remote(skill_id, test_prompt)
        print(f"Response: {response}")
        return

    # Train
    config = {
        "epochs": epochs,
        "learning_rate": lr,
        "lora_r": lora_r,
    }

    result = trainer.train.remote(skill_id, config)

    if result["success"]:
        print("\n✅ Training successful!")
        print(f"Adapter saved to Modal volume: hive215-adapters/{skill_id}")
    else:
        print(f"\n❌ Training failed: {result['error']}")
