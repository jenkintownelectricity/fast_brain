"""
Skill Factory UI - Self-Service Expert Voice Agent Creator

A Gradio-based interface for creating specialized LoRA adapters
from business documents, FAQs, and training data.

Usage:
    pip install gradio pandas PyPDF2
    python skill_factory.py
"""

import gradio as gr
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
import hashlib

# Storage paths
BUSINESS_PROFILES_DIR = Path("business_profiles")
TRAINING_DATA_DIR = Path("training_data")
ADAPTERS_DIR = Path("adapters")

# Ensure directories exist
for dir_path in [BUSINESS_PROFILES_DIR, TRAINING_DATA_DIR, ADAPTERS_DIR]:
    dir_path.mkdir(exist_ok=True)


# =============================================================================
# BUSINESS PROFILE MANAGEMENT
# =============================================================================

def save_business_profile(
    business_name: str,
    business_type: str,
    description: str,
    greeting: str,
    personality: str,
    key_services: str,
    faq_text: str,
    custom_instructions: str
) -> str:
    """Save a business profile to JSON."""
    if not business_name.strip():
        return "❌ Business name is required"

    # Create safe filename
    safe_name = re.sub(r'[^\w\-]', '_', business_name.lower())
    profile_path = BUSINESS_PROFILES_DIR / f"{safe_name}.json"

    profile = {
        "business_name": business_name,
        "business_type": business_type,
        "description": description,
        "greeting": greeting,
        "personality": personality,
        "key_services": [s.strip() for s in key_services.split('\n') if s.strip()],
        "faq": parse_faq(faq_text),
        "custom_instructions": custom_instructions,
        "created_at": datetime.now().isoformat(),
        "skill_adapter": f"{safe_name}.lora"
    }

    with open(profile_path, 'w') as f:
        json.dump(profile, f, indent=2)

    return f"✅ Business profile saved: {profile_path}"


def parse_faq(faq_text: str) -> list:
    """Parse FAQ text into Q&A pairs."""
    faqs = []
    lines = faq_text.strip().split('\n')
    current_q = None

    for line in lines:
        line = line.strip()
        if line.lower().startswith('q:') or line.startswith('?'):
            current_q = line.lstrip('qQ:? ').strip()
        elif line.lower().startswith('a:') and current_q:
            answer = line.lstrip('aA: ').strip()
            faqs.append({"question": current_q, "answer": answer})
            current_q = None
        elif current_q and line:
            # Continuation of answer
            faqs.append({"question": current_q, "answer": line})
            current_q = None

    return faqs


def load_business_profiles() -> list:
    """Load all saved business profiles."""
    profiles = []
    for path in BUSINESS_PROFILES_DIR.glob("*.json"):
        with open(path) as f:
            profiles.append(json.load(f))
    return profiles


def get_profile_choices() -> list:
    """Get list of profile names for dropdown."""
    return [p["business_name"] for p in load_business_profiles()]


# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================

def process_document(file, business_name: str) -> str:
    """Process uploaded document into training data."""
    if file is None:
        return "❌ No file uploaded"

    if not business_name:
        return "❌ Please select or create a business profile first"

    safe_name = re.sub(r'[^\w\-]', '_', business_name.lower())

    # Read file content
    try:
        if file.name.endswith('.pdf'):
            text = extract_pdf_text(file.name)
        elif file.name.endswith('.txt') or file.name.endswith('.md'):
            with open(file.name, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file.name.endswith('.json') or file.name.endswith('.jsonl'):
            # Already in training format
            with open(file.name, 'r') as f:
                content = f.read()
            output_path = TRAINING_DATA_DIR / f"{safe_name}_imported.jsonl"
            with open(output_path, 'w') as f:
                f.write(content)
            return f"✅ Training data imported: {output_path}"
        else:
            return f"❌ Unsupported file type: {file.name}"
    except Exception as e:
        return f"❌ Error reading file: {e}"

    # Convert to training format
    training_data = text_to_training_data(text, business_name)

    # Save as JSONL
    output_path = TRAINING_DATA_DIR / f"{safe_name}_docs.jsonl"
    with open(output_path, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')

    return f"✅ Created {len(training_data)} training examples: {output_path}"


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        return "ERROR: PyPDF2 not installed. Run: pip install PyPDF2"
    except Exception as e:
        return f"ERROR: {e}"


def text_to_training_data(text: str, business_name: str) -> list:
    """Convert raw text to instruction-following training data."""
    training_data = []

    # Split into chunks
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]

    for i, para in enumerate(paragraphs[:100]):  # Limit to 100 examples
        # Create Q&A style training examples
        training_data.append({
            "instruction": f"You are an expert assistant for {business_name}. Answer the following question based on your knowledge.",
            "input": f"Tell me about: {para[:100]}...",
            "output": para
        })

    return training_data


# =============================================================================
# TRAINING DATA GENERATION FROM PROFILE
# =============================================================================

def generate_training_from_profile(business_name: str) -> str:
    """Generate training data from business profile."""
    if not business_name:
        return "❌ Please select a business profile"

    safe_name = re.sub(r'[^\w\-]', '_', business_name.lower())
    profile_path = BUSINESS_PROFILES_DIR / f"{safe_name}.json"

    if not profile_path.exists():
        return f"❌ Profile not found: {business_name}"

    with open(profile_path) as f:
        profile = json.load(f)

    training_data = []

    # System context
    system_prompt = f"""You are {profile['business_name']}, a {profile['business_type']} assistant.
Personality: {profile['personality']}
Description: {profile['description']}
Always be helpful, professional, and knowledgeable about our services."""

    # Greeting examples
    greetings = [
        "Hello", "Hi there", "Hey", "Good morning", "Good afternoon",
        "I need help", "Can you help me?", "Is anyone there?"
    ]
    for greet in greetings:
        training_data.append({
            "instruction": system_prompt,
            "input": greet,
            "output": profile['greeting']
        })

    # Service inquiries
    for service in profile['key_services']:
        training_data.append({
            "instruction": system_prompt,
            "input": f"Tell me about {service}",
            "output": f"Absolutely! {service} is one of our key services. {profile['description']}"
        })
        training_data.append({
            "instruction": system_prompt,
            "input": f"Do you offer {service}?",
            "output": f"Yes! We specialize in {service}. How can I help you with that today?"
        })

    # FAQ training
    for faq in profile['faq']:
        training_data.append({
            "instruction": system_prompt,
            "input": faq['question'],
            "output": faq['answer']
        })

    # Save training data
    output_path = TRAINING_DATA_DIR / f"{safe_name}_profile.jsonl"
    with open(output_path, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')

    return f"✅ Generated {len(training_data)} training examples from profile: {output_path}"


# =============================================================================
# SKILL TRAINING (LoRA)
# =============================================================================

def train_skill(business_name: str, training_steps: int = 60) -> str:
    """Train a LoRA adapter for the business."""
    if not business_name:
        return "❌ Please select a business profile"

    safe_name = re.sub(r'[^\w\-]', '_', business_name.lower())

    # Find all training data for this business
    training_files = list(TRAINING_DATA_DIR.glob(f"{safe_name}*.jsonl"))

    if not training_files:
        return f"❌ No training data found for {business_name}. Upload documents or generate from profile first."

    # Merge training files
    merged_data = []
    for tf in training_files:
        with open(tf) as f:
            for line in f:
                merged_data.append(json.loads(line))

    merged_path = TRAINING_DATA_DIR / f"{safe_name}_merged.jsonl"
    with open(merged_path, 'w') as f:
        for item in merged_data:
            f.write(json.dumps(item) + '\n')

    # Generate training script
    script = f'''
# Auto-generated training script for {business_name}
# Run this on a GPU machine (Colab, Modal, etc.)

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# 1. Load Base Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instructbnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True
)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth"
)

# 3. Load Training Data
dataset = load_dataset("json", data_files="{merged_path}", split="train")

# 4. Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="output",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps={training_steps},
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        output_dir="outputs",
    ),
)

trainer.train()

# 5. Save Adapter
model.save_pretrained("adapters/{safe_name}")
print("✅ Skill trained: adapters/{safe_name}")
'''

    script_path = Path(f"train_{safe_name}.py")
    with open(script_path, 'w') as f:
        f.write(script)

    return f"""✅ Training script generated: {script_path}

📊 Training data: {len(merged_data)} examples from {len(training_files)} files

To train the skill, run:
  python {script_path}

Or on Google Colab:
  1. Upload {merged_path}
  2. Run the script with GPU runtime

After training, upload to Modal:
  modal volume put lpu-skills adapters/{safe_name} /root/skills/{safe_name}.lora
"""


# =============================================================================
# GRADIO UI
# =============================================================================

def create_ui():
    """Create the Skill Factory Gradio interface."""

    with gr.Blocks(
        title="🧠 Skill Factory - Voice Agent Creator",
        theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="blue")
    ) as app:

        gr.Markdown("""
        # 🧠 Skill Factory
        ### Create Expert Voice Agents from Your Business Data

        Transform your business documents, FAQs, and knowledge into specialized AI skills
        that can be hot-swapped into your Voice Agent.
        """)

        with gr.Tabs():
            # =================================================================
            # TAB 1: BUSINESS PROFILE
            # =================================================================
            with gr.Tab("📋 Business Profile"):
                gr.Markdown("### Define Your Business Identity")

                with gr.Row():
                    with gr.Column():
                        business_name = gr.Textbox(
                            label="Business Name",
                            placeholder="Joe's Plumbing Services"
                        )
                        business_type = gr.Dropdown(
                            label="Business Type",
                            choices=[
                                "Plumbing Services",
                                "Electrical Services",
                                "HVAC Services",
                                "Restaurant/Food Service",
                                "Medical/Healthcare",
                                "Legal Services",
                                "Real Estate",
                                "Retail Store",
                                "Tech Support",
                                "General Customer Service",
                                "Other"
                            ],
                            value="General Customer Service"
                        )
                        description = gr.Textbox(
                            label="Business Description",
                            placeholder="We provide 24/7 emergency plumbing services...",
                            lines=3
                        )

                    with gr.Column():
                        greeting = gr.Textbox(
                            label="Default Greeting",
                            placeholder="Hello! Thanks for calling Joe's Plumbing. How can I help you today?",
                            lines=2
                        )
                        personality = gr.Dropdown(
                            label="Agent Personality",
                            choices=[
                                "Professional and formal",
                                "Friendly and casual",
                                "Warm and empathetic",
                                "Efficient and direct",
                                "Technical and precise"
                            ],
                            value="Friendly and casual"
                        )

                key_services = gr.Textbox(
                    label="Key Services (one per line)",
                    placeholder="Emergency repairs\nDrain cleaning\nWater heater installation\nPipe replacement",
                    lines=5
                )

                faq_text = gr.Textbox(
                    label="FAQ (Q: question / A: answer format)",
                    placeholder="""Q: What are your hours?
A: We're available 24/7 for emergencies, regular hours are 8am-6pm.

Q: Do you offer free estimates?
A: Yes, we provide free estimates for all non-emergency work.""",
                    lines=8
                )

                custom_instructions = gr.Textbox(
                    label="Custom Instructions (Optional)",
                    placeholder="Always ask for the customer's address first. Never quote prices over the phone...",
                    lines=3
                )

                save_profile_btn = gr.Button("💾 Save Business Profile", variant="primary")
                profile_status = gr.Textbox(label="Status", interactive=False)

                save_profile_btn.click(
                    fn=save_business_profile,
                    inputs=[
                        business_name, business_type, description, greeting,
                        personality, key_services, faq_text, custom_instructions
                    ],
                    outputs=profile_status
                )

            # =================================================================
            # TAB 2: DOCUMENT UPLOAD
            # =================================================================
            with gr.Tab("📄 Upload Documents"):
                gr.Markdown("""
                ### Feed Your Agent Knowledge

                Upload PDFs, manuals, documentation, or text files.
                The system will convert them into training data.
                """)

                profile_selector = gr.Dropdown(
                    label="Select Business Profile",
                    choices=get_profile_choices(),
                    interactive=True
                )

                refresh_btn = gr.Button("🔄 Refresh Profiles")
                refresh_btn.click(
                    fn=lambda: gr.update(choices=get_profile_choices()),
                    outputs=profile_selector
                )

                file_upload = gr.File(
                    label="Upload Document",
                    file_types=[".pdf", ".txt", ".md", ".json", ".jsonl"]
                )

                process_btn = gr.Button("⚙️ Process Document", variant="primary")
                process_status = gr.Textbox(label="Processing Status", interactive=False)

                process_btn.click(
                    fn=process_document,
                    inputs=[file_upload, profile_selector],
                    outputs=process_status
                )

                gr.Markdown("---")

                generate_btn = gr.Button("🔄 Generate Training Data from Profile")
                generate_status = gr.Textbox(label="Generation Status", interactive=False)

                generate_btn.click(
                    fn=generate_training_from_profile,
                    inputs=[profile_selector],
                    outputs=generate_status
                )

            # =================================================================
            # TAB 3: TRAIN SKILL
            # =================================================================
            with gr.Tab("🎓 Train Skill"):
                gr.Markdown("""
                ### Create Your Expert Adapter

                Train a LoRA adapter from your business data.
                This creates a small (~50MB) "skill cartridge" that can be
                hot-swapped into your Voice Agent.
                """)

                train_profile = gr.Dropdown(
                    label="Select Business Profile",
                    choices=get_profile_choices(),
                    interactive=True
                )

                refresh_train_btn = gr.Button("🔄 Refresh Profiles")
                refresh_train_btn.click(
                    fn=lambda: gr.update(choices=get_profile_choices()),
                    outputs=train_profile
                )

                training_steps = gr.Slider(
                    label="Training Steps",
                    minimum=20,
                    maximum=200,
                    value=60,
                    step=10,
                    info="More steps = better quality but longer training"
                )

                train_btn = gr.Button("🚀 Generate Training Script", variant="primary")
                train_status = gr.Textbox(
                    label="Training Instructions",
                    interactive=False,
                    lines=15
                )

                train_btn.click(
                    fn=train_skill,
                    inputs=[train_profile, training_steps],
                    outputs=train_status
                )

            # =================================================================
            # TAB 4: MANAGE SKILLS
            # =================================================================
            with gr.Tab("🎯 Manage Skills"):
                gr.Markdown("""
                ### Your Skill Library

                View and manage your trained skills.
                """)

                def list_skills():
                    profiles = load_business_profiles()
                    skills_info = []
                    for p in profiles:
                        adapter_path = ADAPTERS_DIR / p['business_name'].lower().replace(' ', '_')
                        status = "✅ Trained" if adapter_path.exists() else "⏳ Not trained"
                        skills_info.append({
                            "Business": p['business_name'],
                            "Type": p['business_type'],
                            "Status": status,
                            "Created": p.get('created_at', 'Unknown')[:10]
                        })
                    return skills_info

                skills_table = gr.Dataframe(
                    headers=["Business", "Type", "Status", "Created"],
                    label="Registered Skills"
                )

                refresh_skills_btn = gr.Button("🔄 Refresh")
                refresh_skills_btn.click(
                    fn=list_skills,
                    outputs=skills_table
                )

                gr.Markdown("""
                ### Deploy to Modal

                After training, upload your skill to the cloud:
                ```bash
                modal volume put lpu-skills adapters/your_skill /root/skills/your_skill.lora
                ```

                Then use in your agent:
                ```python
                lpu.chat.remote_gen(prompt, skill_adapter="your_skill.lora")
                ```
                """)

        gr.Markdown("""
        ---
        ### 🔄 Continuous Learning

        Set up automatic skill improvement by logging user corrections
        and retraining periodically. See `continuous_learner.py` for the automation script.
        """)

    return app


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set to True to create public link
    )
