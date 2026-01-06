"""
Continuous Learner - Set and Forget Self-Improvement System

This script automatically:
1. Collects feedback/corrections from production logs
2. Generates DPO training pairs (chosen vs rejected responses)
3. Trains improved LoRA adapters
4. Hot-swaps them into production

Run via cron for "set and forget" operation:
    0 3 * * 0 python continuous_learner.py  # Weekly at 3 AM Sunday

Requirements:
    pip install modal unsloth datasets trl transformers
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import hashlib

# Configuration - Use absolute paths for Modal volume persistence
FEEDBACK_LOG_PATH = Path("/data/logs/feedback.jsonl")
CORRECTIONS_LOG_PATH = Path("/data/logs/corrections.jsonl")
TRAINING_DATA_DIR = Path("/data/training_data")
ADAPTERS_DIR = Path("/data/adapters")
IMPROVEMENT_HISTORY = Path("/data/logs/improvement_history.json")

# Ensure directories exist on startup
for d in [TRAINING_DATA_DIR, ADAPTERS_DIR, Path("/data/logs")]:
    os.makedirs(d, exist_ok=True)


# =============================================================================
# FEEDBACK COLLECTION
# =============================================================================

class FeedbackCollector:
    """Collect and process user feedback for continuous learning."""

    def __init__(self, log_path: Path = FEEDBACK_LOG_PATH):
        self.log_path = log_path
        self.corrections_path = CORRECTIONS_LOG_PATH

    def log_interaction(
        self,
        user_input: str,
        agent_response: str,
        skill_used: str,
        rating: Optional[int] = None,
        correction: Optional[str] = None,
        metadata: Optional[dict] = None
    ):
        """Log an interaction for future learning."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "agent_response": agent_response,
            "skill_used": skill_used,
            "rating": rating,  # 1-5 or thumbs up/down
            "correction": correction,  # User-provided better response
            "metadata": metadata or {}
        }

        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        # If there's a correction, also log to corrections file
        if correction:
            self.log_correction(user_input, agent_response, correction, skill_used)

    def log_correction(
        self,
        user_input: str,
        rejected_response: str,
        chosen_response: str,
        skill: str
    ):
        """Log a correction as a DPO training pair."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": user_input,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "skill": skill
        }

        with open(self.corrections_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def get_recent_corrections(self, days: int = 7) -> list:
        """Get corrections from the last N days."""
        if not self.corrections_path.exists():
            return []

        cutoff = datetime.now() - timedelta(days=days)
        corrections = []

        with open(self.corrections_path) as f:
            for line in f:
                entry = json.loads(line)
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time > cutoff:
                    corrections.append(entry)

        return corrections

    def get_low_rated_interactions(self, threshold: int = 3, days: int = 7) -> list:
        """Get interactions with low ratings for potential improvement."""
        if not self.log_path.exists():
            return []

        cutoff = datetime.now() - timedelta(days=days)
        low_rated = []

        with open(self.log_path) as f:
            for line in f:
                entry = json.loads(line)
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time > cutoff and entry.get('rating') and entry['rating'] <= threshold:
                    low_rated.append(entry)

        return low_rated


# =============================================================================
# DPO DATA GENERATION
# =============================================================================

class DPODataGenerator:
    """Generate Direct Preference Optimization training data."""

    def __init__(self, collector: FeedbackCollector):
        self.collector = collector

    def generate_training_data(self, skill: str, days: int = 7) -> Path:
        """Generate DPO training data from recent feedback."""

        # Get corrections (explicit preference pairs)
        corrections = self.collector.get_recent_corrections(days)
        skill_corrections = [c for c in corrections if c['skill'] == skill]

        # Get low-rated interactions (implicit negative examples)
        low_rated = self.collector.get_low_rated_interactions(days=days)
        skill_low_rated = [lr for lr in low_rated if lr['skill_used'] == skill]

        training_data = []

        # Add explicit corrections
        for c in skill_corrections:
            training_data.append({
                "prompt": c['prompt'],
                "chosen": c['chosen'],
                "rejected": c['rejected']
            })

        # For low-rated responses, we could generate better alternatives
        # using a larger model, but for now we just flag them
        print(f"Found {len(skill_low_rated)} low-rated interactions to review")

        # Save training data
        output_path = TRAINING_DATA_DIR / f"{skill}_dpo_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(output_path, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')

        print(f"Generated {len(training_data)} DPO training pairs: {output_path}")
        return output_path


# =============================================================================
# AUTOMATIC TRAINER
# =============================================================================

class AutoTrainer:
    """Automatically train improved LoRA adapters."""

    def __init__(self):
        self.history_path = IMPROVEMENT_HISTORY

    def should_train(self, skill: str, min_examples: int = 10) -> bool:
        """Check if we have enough new data to warrant training."""
        dpo_files = list(TRAINING_DATA_DIR.glob(f"{skill}_dpo_*.jsonl"))

        total_examples = 0
        for f in dpo_files:
            with open(f) as file:
                total_examples += sum(1 for _ in file)

        return total_examples >= min_examples

    def train_improved_adapter(self, skill: str, base_adapter: Optional[str] = None):
        """Train an improved adapter using DPO."""

        # Merge all DPO data for this skill
        dpo_files = list(TRAINING_DATA_DIR.glob(f"{skill}_dpo_*.jsonl"))
        all_data = []

        for f in dpo_files:
            with open(f) as file:
                for line in file:
                    all_data.append(json.loads(line))

        if not all_data:
            print(f"No DPO data found for skill: {skill}")
            return None

        # Create merged training file
        merged_path = TRAINING_DATA_DIR / f"{skill}_dpo_merged.jsonl"
        with open(merged_path, 'w') as f:
            for item in all_data:
                f.write(json.dumps(item) + '\n')

        # Generate version number
        version = datetime.now().strftime('%Y%m%d_%H%M')
        new_adapter_name = f"{skill}_v{version}"

        # Generate training script
        script = self._generate_dpo_training_script(
            skill=skill,
            data_path=str(merged_path),
            output_name=new_adapter_name,
            base_adapter=base_adapter
        )

        script_path = Path(f"train_{skill}_dpo.py")
        with open(script_path, 'w') as f:
            f.write(script)

        print(f"Training script generated: {script_path}")
        print(f"Run with: python {script_path}")

        # Log improvement attempt
        self._log_improvement(skill, new_adapter_name, len(all_data))

        return script_path

    def _generate_dpo_training_script(
        self,
        skill: str,
        data_path: str,
        output_name: str,
        base_adapter: Optional[str] = None
    ) -> str:
        """Generate a DPO training script."""

        return f'''
"""
Auto-generated DPO Training Script
Skill: {skill}
Generated: {datetime.now().isoformat()}
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
import torch

# 1. Load Base Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instructbnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True
)

# 2. Load existing adapter if specified
{"" if not base_adapter else f'''
from peft import PeftModel
model = PeftModel.from_pretrained(model, "/data/adapters/{base_adapter}")
model = model.merge_and_unload()  # Merge for continued training
'''}

# 3. Add fresh LoRA adapters for DPO training
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth"
)

# 4. Load DPO training data
dataset = load_dataset("json", data_files="{data_path}", split="train")

# 5. DPO Training
training_args = DPOConfig(
    output_dir="/data/outputs_{output_name}",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    max_steps=100,
    logging_steps=10,
    save_steps=50,
    beta=0.1,  # DPO temperature
    max_length=1024,
    max_prompt_length=512,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

# 6. Save improved adapter
model.save_pretrained("/data/adapters/{output_name}")
print("✅ Improved adapter saved: /data/adapters/{output_name}")

# 7. Upload to Modal (uncomment to auto-deploy)
# import subprocess
# subprocess.run(["modal", "volume", "put", "lpu-skills",
#                 "/data/adapters/{output_name}", "/root/skills/{output_name}.lora"])
# print("🚀 Deployed to Modal!")
'''

    def _log_improvement(self, skill: str, adapter_name: str, num_examples: int):
        """Log improvement attempt to history."""
        history = []
        if self.history_path.exists():
            with open(self.history_path) as f:
                history = json.load(f)

        history.append({
            "timestamp": datetime.now().isoformat(),
            "skill": skill,
            "adapter_name": adapter_name,
            "training_examples": num_examples
        })

        with open(self.history_path, 'w') as f:
            json.dump(history, f, indent=2)


# =============================================================================
# MODAL DEPLOYMENT
# =============================================================================

def deploy_to_modal(adapter_path: str, skill_name: str):
    """Deploy trained adapter to Modal volume."""
    import subprocess

    result = subprocess.run([
        "modal", "volume", "put", "lpu-skills",
        adapter_path, f"/root/skills/{skill_name}.lora"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Deployed {skill_name} to Modal")
        return True
    else:
        print(f"❌ Deployment failed: {result.stderr}")
        return False


# =============================================================================
# MAIN CONTINUOUS LEARNING LOOP
# =============================================================================

def continuous_learning_cycle(skills: list[str], min_examples: int = 10):
    """Run a complete continuous learning cycle for all skills."""

    collector = FeedbackCollector()
    generator = DPODataGenerator(collector)
    trainer = AutoTrainer()

    print(f"🔄 Starting continuous learning cycle at {datetime.now()}")
    print(f"Skills to check: {skills}")

    for skill in skills:
        print(f"\n--- Processing: {skill} ---")

        # 1. Generate DPO data from recent feedback
        dpo_path = generator.generate_training_data(skill)

        # 2. Check if we have enough data to train
        if trainer.should_train(skill, min_examples=min_examples):
            print(f"✅ Sufficient data for {skill}, generating training script...")

            # 3. Generate training script
            script_path = trainer.train_improved_adapter(skill)

            print(f"""
📋 Next steps for {skill}:
   1. Review training script: {script_path}
   2. Run training: python {script_path}
   3. Deploy: modal volume put lpu-skills adapters/{skill}_vXXX /root/skills/{skill}.lora
""")
        else:
            print(f"⏳ Not enough data yet for {skill} (need {min_examples} examples)")

    print(f"\n✅ Continuous learning cycle complete!")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Set up logging in your voice agent
    collector = FeedbackCollector()

    # Log a sample interaction with correction
    collector.log_interaction(
        user_input="What time do you close?",
        agent_response="We close at 5pm.",
        skill_used="plumber",
        rating=2,  # User rated it poorly
        correction="We're open 24/7 for emergencies! Regular hours are 8am to 6pm Monday through Friday."
    )

    # Log another interaction
    collector.log_interaction(
        user_input="Do you fix water heaters?",
        agent_response="Yes, we fix water heaters.",
        skill_used="plumber",
        rating=5  # Good response
    )

    # Run the continuous learning cycle
    skills_to_improve = ["plumber", "restaurant", "tech_support"]
    continuous_learning_cycle(skills_to_improve, min_examples=5)
