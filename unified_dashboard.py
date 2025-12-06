"""
Unified Skill Command Center Dashboard - COMPREHENSIVE VERSION

Combines ALL features from:
- Skill Command Center (API Keys, LLM Testing, Compare Providers, Latency Masking, Stats, Voice Integration)
- Skill Factory (Business Profile, Upload Documents, Train Skill, Manage Skills with filterable table)
- LPU Inference (BitNet model testing)
- Dashboard (Stats, Server Status, Training Pipeline, Activity Feed)

All in one beautiful cyberpunk-themed interface.

Usage:
    pip install flask flask-cors
    python unified_dashboard.py
    # Opens at http://localhost:5000
"""

from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
import json
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
import random

app = Flask(__name__)
CORS(app)

# Data paths
BUSINESS_PROFILES_DIR = Path("business_profiles")
TRAINING_DATA_DIR = Path("training_data")
ADAPTERS_DIR = Path("adapters")
LOGS_DIR = Path("logs")
DOCUMENTS_DIR = Path("uploaded_documents")

# Ensure directories exist
for d in [BUSINESS_PROFILES_DIR, TRAINING_DATA_DIR, ADAPTERS_DIR, LOGS_DIR, DOCUMENTS_DIR]:
    d.mkdir(exist_ok=True)

# In-memory state for API keys and activity
API_KEYS = {}
ACTIVITY_LOG = []

# Voice configuration and platform connections
VOICE_CONFIG = {
    "selected_voice": "en-US-JennyNeural",
    "selected_provider": "edge_tts",
    "voice_settings": {
        "speed": 1.0,
        "pitch": 1.0,
        "volume": 1.0
    }
}

VOICE_CHOICES = {
    "edge_tts": {
        "name": "Edge TTS",
        "provider": "Microsoft (Free)",
        "voices": [
            {"id": "en-US-JennyNeural", "name": "Jenny", "gender": "female", "style": "friendly"},
            {"id": "en-US-AriaNeural", "name": "Aria", "gender": "female", "style": "professional"},
            {"id": "en-US-SaraNeural", "name": "Sara", "gender": "female", "style": "warm"},
            {"id": "en-US-AnaNeural", "name": "Ana", "gender": "female", "style": "youthful"},
            {"id": "en-US-GuyNeural", "name": "Guy", "gender": "male", "style": "conversational"},
            {"id": "en-US-DavisNeural", "name": "Davis", "gender": "male", "style": "confident"},
            {"id": "en-US-TonyNeural", "name": "Tony", "gender": "male", "style": "friendly"},
            {"id": "en-US-JasonNeural", "name": "Jason", "gender": "male", "style": "neutral"},
            {"id": "en-GB-SoniaNeural", "name": "Sonia (UK)", "gender": "female", "style": "british"},
            {"id": "en-GB-RyanNeural", "name": "Ryan (UK)", "gender": "male", "style": "british"},
            {"id": "en-AU-NatashaNeural", "name": "Natasha (AU)", "gender": "female", "style": "australian"},
            {"id": "en-AU-WilliamNeural", "name": "William (AU)", "gender": "male", "style": "australian"},
        ]
    },
    "kokoro": {
        "name": "Kokoro",
        "provider": "Kokoro TTS (Free)",
        "voices": [
            {"id": "af_bella", "name": "Bella", "gender": "female", "style": "american"},
            {"id": "af_nicole", "name": "Nicole", "gender": "female", "style": "american"},
            {"id": "af_sarah", "name": "Sarah", "gender": "female", "style": "american"},
            {"id": "af_sky", "name": "Sky", "gender": "female", "style": "american"},
            {"id": "am_adam", "name": "Adam", "gender": "male", "style": "american"},
            {"id": "am_michael", "name": "Michael", "gender": "male", "style": "american"},
            {"id": "bf_emma", "name": "Emma", "gender": "female", "style": "british"},
            {"id": "bf_isabella", "name": "Isabella", "gender": "female", "style": "british"},
            {"id": "bm_george", "name": "George", "gender": "male", "style": "british"},
            {"id": "bm_lewis", "name": "Lewis", "gender": "male", "style": "british"},
        ]
    },
    "chatterbox": {
        "name": "Chatterbox",
        "provider": "Resemble AI",
        "voices": [
            {"id": "chatterbox_default", "name": "Default", "gender": "neutral", "style": "conversational"},
            {"id": "chatterbox_warm", "name": "Warm", "gender": "female", "style": "friendly"},
            {"id": "chatterbox_professional", "name": "Professional", "gender": "male", "style": "business"},
            {"id": "chatterbox_energetic", "name": "Energetic", "gender": "female", "style": "upbeat"},
        ]
    },
    "kokoro": {
        "name": "Kokoro",
        "provider": "Kokoro TTS",
        "voices": [
            {"id": "af_bella", "name": "Bella", "gender": "female", "style": "american"},
            {"id": "af_nicole", "name": "Nicole", "gender": "female", "style": "american"},
            {"id": "am_adam", "name": "Adam", "gender": "male", "style": "american"},
            {"id": "am_michael", "name": "Michael", "gender": "male", "style": "american"},
            {"id": "bf_emma", "name": "Emma", "gender": "female", "style": "british"},
            {"id": "bm_george", "name": "George", "gender": "male", "style": "british"},
        ]
    },
    "xtts": {
        "name": "XTTS-v2",
        "provider": "Coqui AI",
        "voices": [
            {"id": "xtts_clone", "name": "Custom Clone", "gender": "custom", "style": "cloned"},
            {"id": "xtts_en_female", "name": "English Female", "gender": "female", "style": "neutral"},
            {"id": "xtts_en_male", "name": "English Male", "gender": "male", "style": "neutral"},
        ]
    },
    "openvoice": {
        "name": "OpenVoice",
        "provider": "MyShell AI",
        "voices": [
            {"id": "openvoice_default", "name": "Default", "gender": "neutral", "style": "neutral"},
            {"id": "openvoice_clone", "name": "Instant Clone", "gender": "custom", "style": "cloned"},
        ]
    },
    "elevenlabs": {
        "name": "ElevenLabs",
        "provider": "ElevenLabs (Paid)",
        "voices": [
            {"id": "eleven_rachel", "name": "Rachel", "gender": "female", "style": "conversational"},
            {"id": "eleven_drew", "name": "Drew", "gender": "male", "style": "neutral"},
            {"id": "eleven_clyde", "name": "Clyde", "gender": "male", "style": "character"},
            {"id": "eleven_paul", "name": "Paul", "gender": "male", "style": "news"},
            {"id": "eleven_domi", "name": "Domi", "gender": "female", "style": "strong"},
            {"id": "eleven_bella", "name": "Bella", "gender": "female", "style": "soft"},
            {"id": "eleven_antoni", "name": "Antoni", "gender": "male", "style": "warm"},
            {"id": "eleven_custom", "name": "Custom Clone", "gender": "custom", "style": "cloned"},
        ]
    },
    "openai": {
        "name": "OpenAI TTS",
        "provider": "OpenAI (Paid)",
        "voices": [
            {"id": "openai_alloy", "name": "Alloy", "gender": "neutral", "style": "balanced"},
            {"id": "openai_echo", "name": "Echo", "gender": "male", "style": "neutral"},
            {"id": "openai_fable", "name": "Fable", "gender": "neutral", "style": "storytelling"},
            {"id": "openai_onyx", "name": "Onyx", "gender": "male", "style": "deep"},
            {"id": "openai_nova", "name": "Nova", "gender": "female", "style": "warm"},
            {"id": "openai_shimmer", "name": "Shimmer", "gender": "female", "style": "expressive"},
        ]
    },
    "azure": {
        "name": "Azure TTS",
        "provider": "Microsoft Azure (Paid)",
        "voices": [
            {"id": "azure_jenny", "name": "Jenny", "gender": "female", "style": "neural"},
            {"id": "azure_guy", "name": "Guy", "gender": "male", "style": "neural"},
            {"id": "azure_aria", "name": "Aria", "gender": "female", "style": "conversational"},
            {"id": "azure_davis", "name": "Davis", "gender": "male", "style": "conversational"},
        ]
    },
    "cartesia": {
        "name": "Cartesia",
        "provider": "Cartesia (Paid)",
        "voices": [
            {"id": "cartesia_default", "name": "Default", "gender": "neutral", "style": "natural"},
            {"id": "cartesia_custom", "name": "Custom Voice", "gender": "custom", "style": "cloned"},
        ]
    }
}

PLATFORM_CONNECTIONS = {
    "livekit": {
        "name": "LiveKit",
        "status": "disconnected",
        "config": {
            "url": "",
            "api_key": "",
            "api_secret": ""
        },
        "description": "Real-time voice and video with agents SDK"
    },
    "vapi": {
        "name": "Vapi",
        "status": "disconnected",
        "config": {
            "api_key": "",
            "assistant_id": ""
        },
        "description": "Voice AI platform for building phone agents"
    },
    "twilio": {
        "name": "Twilio",
        "status": "disconnected",
        "config": {
            "account_sid": "",
            "auth_token": "",
            "phone_number": ""
        },
        "description": "Cloud communications platform for voice calls"
    },
    "retell": {
        "name": "Retell AI",
        "status": "disconnected",
        "config": {
            "api_key": "",
            "agent_id": ""
        },
        "description": "Conversational voice AI for customer interactions"
    },
    "bland": {
        "name": "Bland AI",
        "status": "disconnected",
        "config": {
            "api_key": "",
            "pathway_id": ""
        },
        "description": "AI phone agents for enterprises"
    },
    "vocode": {
        "name": "Vocode",
        "status": "disconnected",
        "config": {
            "api_key": ""
        },
        "description": "Open-source voice agent framework"
    },
    "daily": {
        "name": "Daily.co",
        "status": "disconnected",
        "config": {
            "api_key": "",
            "room_url": ""
        },
        "description": "Real-time video/audio platform with Pipecat integration"
    },
    "websocket": {
        "name": "Custom WebSocket",
        "status": "disconnected",
        "config": {
            "url": "",
            "auth_header": ""
        },
        "description": "Connect to any WebSocket-based voice service"
    }
}

def add_activity(message, icon=""):
    """Add an activity to the log."""
    ACTIVITY_LOG.insert(0, {
        "message": message,
        "icon": icon,
        "timestamp": datetime.now().isoformat(),
        "ago": "just now"
    })
    # Keep only last 20 activities
    if len(ACTIVITY_LOG) > 20:
        ACTIVITY_LOG.pop()


# =============================================================================
# API ENDPOINTS - SKILLS
# =============================================================================

@app.route('/api/skills')
def get_skills():
    """Get all registered skills with their status."""
    skills = []
    for profile_path in BUSINESS_PROFILES_DIR.glob("*.json"):
        try:
            with open(profile_path) as f:
                profile = json.load(f)
            safe_name = profile_path.stem
            adapter_path = ADAPTERS_DIR / safe_name
            has_adapter = adapter_path.exists()
            training_files = list(TRAINING_DATA_DIR.glob(f"{safe_name}*.jsonl"))
            training_examples = sum(sum(1 for _ in open(tf)) for tf in training_files) if training_files else 0

            skills.append({
                "id": safe_name,
                "name": profile.get("business_name", safe_name),
                "type": profile.get("business_type", "Unknown"),
                "status": "deployed" if has_adapter else "training" if training_examples > 0 else "draft",
                "personality": profile.get("personality", ""),
                "description": profile.get("description", ""),
                "training_examples": training_examples,
                "created_at": profile.get("created_at", ""),
                "requests_today": random.randint(10, 500),
                "avg_latency_ms": random.randint(80, 200),
                "satisfaction_rate": random.randint(85, 99),
            })
        except Exception as e:
            print(f"Error loading {profile_path}: {e}")
    return jsonify(skills)


@app.route('/api/skills-table')
def get_skills_table():
    """Get skills in table format for filtering."""
    skills = []
    for profile_path in BUSINESS_PROFILES_DIR.glob("*.json"):
        try:
            with open(profile_path) as f:
                profile = json.load(f)
            safe_name = profile_path.stem
            adapter_path = ADAPTERS_DIR / safe_name
            has_adapter = adapter_path.exists()
            training_files = list(TRAINING_DATA_DIR.glob(f"{safe_name}*.jsonl"))
            training_examples = sum(sum(1 for _ in open(tf)) for tf in training_files) if training_files else 0

            skills.append({
                "id": safe_name,
                "business": profile.get("business_name", safe_name),
                "type": profile.get("business_type", "Unknown"),
                "status": "Deployed" if has_adapter else "Training" if training_examples > 0 else "Draft",
                "examples": training_examples,
                "created": profile.get("created_at", "")[:10] if profile.get("created_at") else "Unknown",
            })
        except Exception as e:
            print(f"Error loading {profile_path}: {e}")
    return jsonify(skills)


@app.route('/api/server-status')
def get_server_status():
    """Get warm server status."""
    return jsonify({
        "status": "online",
        "warm_containers": 1,
        "region": "us-east-1",
        "uptime_hours": random.randint(1, 720),
        "total_requests": random.randint(1000, 50000),
        "avg_cold_start_ms": 2100,
        "avg_warm_latency_ms": 89,
        "memory_usage_mb": random.randint(4000, 7000),
        "active_skill": "plumber_expert",
        "skills_loaded": ["plumber_expert", "restaurant_host", "tech_support"],
        "cost_today_usd": round(random.uniform(0.5, 5.0), 2),
    })


@app.route('/api/metrics')
def get_metrics():
    """Get performance metrics over time."""
    now = datetime.now()
    metrics = []
    for i in range(24):
        timestamp = now - timedelta(hours=23-i)
        metrics.append({
            "timestamp": timestamp.isoformat(),
            "requests": random.randint(50, 300),
            "latency_p50": random.randint(70, 120),
            "latency_p99": random.randint(150, 300),
        })
    return jsonify(metrics)


@app.route('/api/activity')
def get_activity():
    """Get recent activity."""
    # Update "ago" times
    now = datetime.now()
    for act in ACTIVITY_LOG:
        try:
            ts = datetime.fromisoformat(act["timestamp"])
            delta = now - ts
            if delta.seconds < 60:
                act["ago"] = "just now"
            elif delta.seconds < 3600:
                act["ago"] = f"{delta.seconds // 60} minutes ago"
            elif delta.seconds < 86400:
                act["ago"] = f"{delta.seconds // 3600} hours ago"
            else:
                act["ago"] = f"{delta.days} days ago"
        except:
            pass
    return jsonify(ACTIVITY_LOG[:10])


@app.route('/api/training-status')
def get_training_status():
    """Get current training pipeline status."""
    # Simulated training status
    return jsonify({
        "current_skill": "plumber_expert",
        "stage": "train",  # ingest, process, train, deploy
        "progress": random.randint(40, 90),
        "stages": {
            "ingest": "complete",
            "process": "complete",
            "train": "in_progress",
            "deploy": "pending"
        }
    })


@app.route('/api/create-skill', methods=['POST'])
def create_skill():
    """Create a new skill from the quick form."""
    data = request.json
    safe_name = data['name'].lower().replace(' ', '_')
    safe_name = re.sub(r'[^\w\-]', '_', safe_name)
    profile_path = BUSINESS_PROFILES_DIR / f"{safe_name}.json"

    profile = {
        "business_name": data['name'],
        "business_type": data.get('type', 'General'),
        "description": data.get('description', ''),
        "greeting": data.get('greeting', f"Hello! How can I help you with {data['name']}?"),
        "personality": data.get('personality', 'Friendly and helpful'),
        "key_services": data.get('services', '').split('\n') if data.get('services') else [],
        "faq": [],
        "custom_instructions": data.get('customInstructions', ''),
        "created_at": datetime.now().isoformat(),
    }

    with open(profile_path, 'w') as f:
        json.dump(profile, f, indent=2)

    add_activity(f"New skill created: {data['name']}", "")
    return jsonify({"success": True, "id": safe_name})


@app.route('/api/delete-skill/<skill_id>', methods=['DELETE'])
def delete_skill(skill_id):
    """Delete a skill."""
    profile_path = BUSINESS_PROFILES_DIR / f"{skill_id}.json"
    if profile_path.exists():
        profile_path.unlink()
        add_activity(f"Skill deleted: {skill_id}", "")
        return jsonify({"success": True})
    return jsonify({"error": "Skill not found"}), 404


@app.route('/api/upload-document', methods=['POST'])
def upload_document():
    """Handle document upload."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    skill_id = request.form.get('skill_id', 'general')

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the file
    safe_name = re.sub(r'[^\w\-\.]', '_', file.filename)
    file_path = DOCUMENTS_DIR / f"{skill_id}_{safe_name}"
    file.save(file_path)

    # Process based on file type
    examples = 0
    try:
        if safe_name.endswith('.txt') or safe_name.endswith('.md'):
            with open(file_path, 'r') as f:
                text = f.read()
            examples = process_text_to_training(text, skill_id)
        elif safe_name.endswith('.json') or safe_name.endswith('.jsonl'):
            # Copy directly to training data
            training_path = TRAINING_DATA_DIR / f"{skill_id}_imported.jsonl"
            with open(file_path, 'r') as src:
                with open(training_path, 'w') as dst:
                    dst.write(src.read())
            examples = sum(1 for _ in open(training_path))
        else:
            examples = 10  # Placeholder for PDF processing
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    add_activity(f"Document processed: {examples} examples", "")
    return jsonify({"success": True, "examples": examples})


def process_text_to_training(text, skill_id):
    """Convert text to training data."""
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
    training_data = []

    for para in paragraphs[:100]:
        training_data.append({
            "instruction": f"You are an expert assistant for {skill_id}.",
            "input": f"Tell me about: {para[:100]}...",
            "output": para
        })

    output_path = TRAINING_DATA_DIR / f"{skill_id}_docs.jsonl"
    with open(output_path, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')

    return len(training_data)


@app.route('/api/generate-training/<skill_id>', methods=['POST'])
def generate_training(skill_id):
    """Generate training data from profile."""
    profile_path = BUSINESS_PROFILES_DIR / f"{skill_id}.json"
    if not profile_path.exists():
        return jsonify({"error": "Profile not found"}), 404

    with open(profile_path) as f:
        profile = json.load(f)

    training_data = []
    system_prompt = f"You are {profile['business_name']}, a {profile['business_type']} assistant. {profile.get('personality', '')}"

    # Greeting examples
    for greet in ["Hello", "Hi", "Hey", "Good morning", "I need help"]:
        training_data.append({
            "instruction": system_prompt,
            "input": greet,
            "output": profile.get('greeting', 'Hello! How can I help you?')
        })

    # Service examples
    for service in profile.get('key_services', []):
        if service.strip():
            training_data.append({
                "instruction": system_prompt,
                "input": f"Tell me about {service}",
                "output": f"Absolutely! {service} is one of our key services. {profile.get('description', '')}"
            })

    # Save
    output_path = TRAINING_DATA_DIR / f"{skill_id}_profile.jsonl"
    with open(output_path, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')

    add_activity(f"Training data generated: {len(training_data)} examples", "")
    return jsonify({"success": True, "examples": len(training_data)})


@app.route('/api/save-api-keys', methods=['POST'])
def save_api_keys():
    """Save API keys."""
    data = request.json
    saved = []

    if data.get('groq'):
        API_KEYS['groq'] = data['groq']
        saved.append('Groq')
    if data.get('openai'):
        API_KEYS['openai'] = data['openai']
        saved.append('OpenAI')
    if data.get('anthropic'):
        API_KEYS['anthropic'] = data['anthropic']
        saved.append('Anthropic')

    if saved:
        add_activity(f"API keys saved: {', '.join(saved)}", "")
        return jsonify({"success": True, "saved": saved})
    return jsonify({"success": True, "message": "No keys provided"})


@app.route('/api/test-llm', methods=['POST'])
def test_llm():
    """Test a single LLM provider."""
    data = request.json
    provider = data.get('provider', 'mock')
    prompt = data.get('prompt', '')

    # Simulated response for now
    import time
    start = time.time()

    responses = {
        'groq': f"[Groq - Llama 3.1 8B] This is a fast response to: {prompt[:50]}...",
        'openai': f"[OpenAI - GPT-4o-mini] I'd be happy to help with: {prompt[:50]}...",
        'anthropic': f"[Anthropic - Claude 3 Haiku] Let me address: {prompt[:50]}...",
        'bitnet': f"[BitNet LPU] Processing: {prompt[:50]}...",
    }

    latencies = {'groq': 100, 'openai': 300, 'anthropic': 400, 'bitnet': 800}

    response = responses.get(provider, f"[{provider}] Response to: {prompt[:50]}...")
    latency = latencies.get(provider, 500) + random.randint(-50, 50)

    return jsonify({
        "success": True,
        "response": response,
        "latency_ms": latency,
        "provider": provider
    })


@app.route('/api/compare-llms', methods=['POST'])
def compare_llms():
    """Compare multiple LLM providers."""
    data = request.json
    providers = data.get('providers', ['groq', 'openai'])
    prompt = data.get('prompt', '')

    results = []
    for provider in providers:
        responses = {
            'groq': f"[Groq] Fast response to: {prompt[:30]}...",
            'openai': f"[OpenAI] Quality response to: {prompt[:30]}...",
            'anthropic': f"[Anthropic] Thoughtful response to: {prompt[:30]}...",
            'bitnet': f"[BitNet] Efficient response to: {prompt[:30]}...",
        }
        latencies = {'groq': 100, 'openai': 300, 'anthropic': 400, 'bitnet': 800}

        results.append({
            "provider": provider,
            "response": responses.get(provider, f"[{provider}] Response..."),
            "first_token_ms": latencies.get(provider, 500) + random.randint(-30, 30),
            "total_ms": latencies.get(provider, 500) * 2 + random.randint(-50, 100),
        })

    return jsonify({"success": True, "results": results})


@app.route('/api/lpu-inference', methods=['POST'])
def lpu_inference():
    """Run inference on the Modal LPU."""
    data = request.json
    prompt = data.get('prompt', '')

    try:
        import modal
        lpu = modal.Cls.from_name("bitnet-lpu-v1", "VirtualLPU")()
        response = ""
        for chunk in lpu.chat.remote_gen(prompt, max_tokens=data.get('maxTokens', 128)):
            response += chunk
        add_activity("LPU inference completed", "")
        return jsonify({"success": True, "response": response})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/stats')
def get_stats():
    """Get system statistics."""
    return jsonify({
        "total_queries": random.randint(1000, 10000),
        "by_provider": {
            "groq": random.randint(100, 500),
            "openai": random.randint(50, 200),
            "anthropic": random.randint(30, 150),
            "bitnet": random.randint(200, 800),
        },
        "avg_latency_ms": random.randint(100, 300),
        "cache_hits": random.randint(50, 200),
        "skills_active": len(list(BUSINESS_PROFILES_DIR.glob("*.json"))),
        "adapters_trained": len(list(ADAPTERS_DIR.glob("*"))),
    })


# =============================================================================
# API ENDPOINTS - VOICE CONFIGURATION
# =============================================================================

@app.route('/api/voice/providers')
def get_voice_providers():
    """Get all available voice providers and their voices."""
    return jsonify(VOICE_CHOICES)


@app.route('/api/voice/config')
def get_voice_config():
    """Get current voice configuration."""
    return jsonify(VOICE_CONFIG)


@app.route('/api/voice/config', methods=['POST'])
def save_voice_config():
    """Save voice configuration."""
    global VOICE_CONFIG
    data = request.json

    if data.get('selected_voice'):
        VOICE_CONFIG['selected_voice'] = data['selected_voice']
    if data.get('selected_provider'):
        VOICE_CONFIG['selected_provider'] = data['selected_provider']
    if data.get('voice_settings'):
        VOICE_CONFIG['voice_settings'].update(data['voice_settings'])

    add_activity(f"Voice changed to {VOICE_CONFIG['selected_voice']}", "")
    return jsonify({"success": True, "config": VOICE_CONFIG})


@app.route('/api/voice/test', methods=['POST'])
def test_voice():
    """Test a voice with sample text - generates real audio with Edge TTS."""
    import asyncio
    import base64
    import io
    import time

    data = request.json
    voice_id = data.get('voice_id', 'en-US-JennyNeural')
    text = data.get('text', 'Hello! This is a test of the voice synthesis system.')
    provider = data.get('provider', 'edge_tts')

    start_time = time.time()

    # Edge TTS - Free Microsoft voices
    if provider == 'edge_tts' or voice_id.startswith('en-'):
        try:
            import edge_tts

            async def generate_edge_audio():
                communicate = edge_tts.Communicate(text, voice_id)
                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]
                return audio_data

            audio_bytes = asyncio.run(generate_edge_audio())
            duration_ms = int((time.time() - start_time) * 1000)

            # Return base64 encoded audio
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            add_activity(f"Voice test: {voice_id} ({duration_ms}ms)", "")
            return jsonify({
                "success": True,
                "voice_id": voice_id,
                "provider": "edge_tts",
                "text": text,
                "duration_ms": duration_ms,
                "audio_base64": audio_base64,
                "audio_format": "audio/mpeg",
                "message": f"Generated audio with Edge TTS voice '{voice_id}'"
            })
        except ImportError:
            return jsonify({
                "success": False,
                "error": "edge-tts not installed. Run: pip install edge-tts"
            }), 500
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    # Kokoro TTS - Free local voices
    elif provider == 'kokoro' or voice_id.startswith('af_') or voice_id.startswith('am_') or voice_id.startswith('bf_') or voice_id.startswith('bm_'):
        try:
            from kokoro_onnx import Kokoro
            import soundfile as sf

            kokoro = Kokoro('kokoro-v1.0.onnx', 'voices-v1.0.bin')
            samples, sr = kokoro.create(text, voice=voice_id)

            # Convert to WAV bytes
            buffer = io.BytesIO()
            sf.write(buffer, samples, sr, format='WAV')
            audio_bytes = buffer.getvalue()
            duration_ms = int((time.time() - start_time) * 1000)

            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            add_activity(f"Voice test: {voice_id} ({duration_ms}ms)", "")
            return jsonify({
                "success": True,
                "voice_id": voice_id,
                "provider": "kokoro",
                "text": text,
                "duration_ms": duration_ms,
                "audio_base64": audio_base64,
                "audio_format": "audio/wav",
                "message": f"Generated audio with Kokoro voice '{voice_id}'"
            })
        except ImportError:
            return jsonify({
                "success": False,
                "error": "kokoro-onnx not installed. Run: pip install kokoro-onnx soundfile"
            }), 500
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    # Fallback for other providers (simulated)
    else:
        add_activity(f"Voice test (simulated): {voice_id}", "")
        return jsonify({
            "success": True,
            "voice_id": voice_id,
            "provider": provider,
            "text": text,
            "duration_ms": random.randint(800, 2000),
            "audio_base64": None,
            "message": f"Voice '{voice_id}' tested (simulated - provider not configured)"
        })


# =============================================================================
# API ENDPOINTS - PLATFORM CONNECTIONS
# =============================================================================

@app.route('/api/platforms')
def get_platforms():
    """Get all platform connections and their status."""
    return jsonify(PLATFORM_CONNECTIONS)


@app.route('/api/platforms/<platform_id>')
def get_platform(platform_id):
    """Get a specific platform connection."""
    if platform_id in PLATFORM_CONNECTIONS:
        return jsonify(PLATFORM_CONNECTIONS[platform_id])
    return jsonify({"error": "Platform not found"}), 404


@app.route('/api/platforms/<platform_id>/connect', methods=['POST'])
def connect_platform(platform_id):
    """Connect to a voice platform."""
    global PLATFORM_CONNECTIONS

    if platform_id not in PLATFORM_CONNECTIONS:
        return jsonify({"error": "Platform not found"}), 404

    data = request.json
    config = data.get('config', {})

    # Update the platform config
    PLATFORM_CONNECTIONS[platform_id]['config'].update(config)

    # Simulate connection test
    has_required_fields = all(v for v in PLATFORM_CONNECTIONS[platform_id]['config'].values())

    if has_required_fields:
        PLATFORM_CONNECTIONS[platform_id]['status'] = 'connected'
        add_activity(f"Connected to {PLATFORM_CONNECTIONS[platform_id]['name']}", "")
        return jsonify({
            "success": True,
            "status": "connected",
            "message": f"Successfully connected to {PLATFORM_CONNECTIONS[platform_id]['name']}"
        })
    else:
        PLATFORM_CONNECTIONS[platform_id]['status'] = 'error'
        return jsonify({
            "success": False,
            "status": "error",
            "message": "Missing required configuration fields"
        })


@app.route('/api/platforms/<platform_id>/disconnect', methods=['POST'])
def disconnect_platform(platform_id):
    """Disconnect from a voice platform."""
    global PLATFORM_CONNECTIONS

    if platform_id not in PLATFORM_CONNECTIONS:
        return jsonify({"error": "Platform not found"}), 404

    PLATFORM_CONNECTIONS[platform_id]['status'] = 'disconnected'
    add_activity(f"Disconnected from {PLATFORM_CONNECTIONS[platform_id]['name']}", "")

    return jsonify({
        "success": True,
        "status": "disconnected",
        "message": f"Disconnected from {PLATFORM_CONNECTIONS[platform_id]['name']}"
    })


@app.route('/api/platforms/<platform_id>/test', methods=['POST'])
def test_platform(platform_id):
    """Test a platform connection."""
    if platform_id not in PLATFORM_CONNECTIONS:
        return jsonify({"error": "Platform not found"}), 404

    platform = PLATFORM_CONNECTIONS[platform_id]

    if platform['status'] != 'connected':
        return jsonify({
            "success": False,
            "message": "Platform is not connected"
        })

    # Simulate connection test
    add_activity(f"Testing connection to {platform['name']}", "")
    return jsonify({
        "success": True,
        "message": f"Connection to {platform['name']} is working",
        "latency_ms": random.randint(50, 200)
    })


# =============================================================================
# API ENDPOINTS - FAST BRAIN
# =============================================================================

FAST_BRAIN_CONFIG = {
    "url": os.getenv("FAST_BRAIN_URL", ""),
    "status": "disconnected",
    "model": "groq-llama-3.3-70b",
    "backend": "groq",
    "min_containers": 1,
    "max_containers": 10,
    "selected_skill": "general",
    "stats": {
        "total_requests": 0,
        "avg_ttfb_ms": 0,
        "avg_throughput_tps": 0,
        "uptime_hours": 0,
        "errors": [],
    }
}

# Local skills cache (synced from Fast Brain LPU)
FAST_BRAIN_SKILLS = {
    "general": {
        "id": "general",
        "name": "General Assistant",
        "description": "Helpful general-purpose assistant",
        "system_prompt": "You are a helpful AI assistant. Be friendly, concise, and helpful. Respond in 1-2 sentences unless more detail is needed.",
        "knowledge": [],
        "is_builtin": True,
    },
    "receptionist": {
        "id": "receptionist",
        "name": "Professional Receptionist",
        "description": "Expert phone answering and call handling",
        "system_prompt": "You are a professional AI receptionist. Respond in 1-2 short sentences maximum. Be warm, helpful, and conversational—never robotic.",
        "knowledge": [],
        "is_builtin": True,
    },
    "electrician": {
        "id": "electrician",
        "name": "Electrician Assistant",
        "description": "Expert in electrical services and scheduling",
        "system_prompt": "You are an AI assistant for an electrical contracting business. Be professional, safety-conscious, and helpful.",
        "knowledge": ["Panel upgrades: $1,500-$3,000", "EV charger: $500-$2,000", "Emergency: 24/7 with $150 fee"],
        "is_builtin": True,
    },
    "plumber": {
        "id": "plumber",
        "name": "Plumber Assistant",
        "description": "Expert in plumbing services",
        "system_prompt": "You are an AI assistant for a plumbing company. Be helpful and knowledgeable about plumbing services.",
        "knowledge": ["Drain cleaning: $150-$300", "Water heater: $1,000-$3,000"],
        "is_builtin": True,
    },
    "lawyer": {
        "id": "lawyer",
        "name": "Legal Intake Assistant",
        "description": "Professional legal intake and scheduling",
        "system_prompt": "You are a legal intake assistant. Be professional, confidential, and thorough. DO NOT provide legal advice.",
        "knowledge": ["Initial consultations typically free", "All communications confidential"],
        "is_builtin": True,
    },
}

# System status tracking for comprehensive monitoring
SYSTEM_STATUS = {
    "fast_brain": {"status": "unknown", "last_check": None, "error": None, "latency_ms": None},
    "hive215": {"status": "unknown", "last_check": None, "error": None, "latency_ms": None},
    "groq": {"status": "unknown", "last_check": None, "error": None},
    "modal": {"status": "unknown", "last_check": None, "error": None},
}


@app.route('/api/fast-brain/config')
def get_fast_brain_config():
    """Get Fast Brain configuration and status."""
    return jsonify(FAST_BRAIN_CONFIG)


@app.route('/api/fast-brain/config', methods=['POST'])
def update_fast_brain_config():
    """Update Fast Brain configuration."""
    global FAST_BRAIN_CONFIG
    data = request.json

    if 'url' in data:
        # Strip trailing slash to avoid double-slash issues
        clean_url = data['url'].rstrip('/')
        FAST_BRAIN_CONFIG['url'] = clean_url
        os.environ['FAST_BRAIN_URL'] = clean_url

    if 'min_containers' in data:
        FAST_BRAIN_CONFIG['min_containers'] = data['min_containers']
    if 'max_containers' in data:
        FAST_BRAIN_CONFIG['max_containers'] = data['max_containers']

    add_activity(f"Fast Brain config updated", "")
    return jsonify({"success": True, "config": FAST_BRAIN_CONFIG})


@app.route('/api/fast-brain/health')
def fast_brain_health():
    """Check Fast Brain health."""
    global FAST_BRAIN_CONFIG

    url = FAST_BRAIN_CONFIG.get('url')
    if not url:
        return jsonify({
            "status": "not_configured",
            "message": "FAST_BRAIN_URL not set"
        })

    # Try to connect to Fast Brain LPU
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{url}/health")
            if response.status_code == 200:
                health_data = response.json()
                api_status = health_data.get("status", "unknown")
                FAST_BRAIN_CONFIG['status'] = 'connected' if api_status in ['healthy', 'initializing'] else 'error'
                return jsonify({
                    "status": api_status,
                    "model_loaded": health_data.get("model_loaded", False),
                    "skills": health_data.get("skills_available", []),
                    "version": health_data.get("version", "unknown"),
                })
    except Exception as e:
        FAST_BRAIN_CONFIG['status'] = 'error'
        return jsonify({
            "status": "error",
            "message": str(e)
        })

    FAST_BRAIN_CONFIG['status'] = 'disconnected'
    return jsonify({"status": "disconnected"})


@app.route('/api/fast-brain/chat', methods=['POST'])
def fast_brain_chat():
    """Send a chat request to Fast Brain."""
    data = request.json
    message = data.get('message', '')
    system_prompt = data.get('system_prompt', '')
    max_tokens = data.get('max_tokens', 256)
    stream = data.get('stream', False)

    url = FAST_BRAIN_CONFIG.get('url')
    if not url:
        return jsonify({"success": False, "error": "Fast Brain not configured"})

    try:
        import httpx
        import time

        start_time = time.time()

        # Build messages array for OpenAI-compatible API
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        # Use Fast Brain LPU /v1/chat/completions endpoint
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            # If 422, get the validation error details
            if response.status_code == 422:
                error_detail = response.json()
                return jsonify({"success": False, "error": f"Validation error: {error_detail}"})
            response.raise_for_status()
            result = response.json()

        elapsed_ms = (time.time() - start_time) * 1000

        # Update stats
        FAST_BRAIN_CONFIG['stats']['total_requests'] += 1

        # Extract content from OpenAI-compatible response
        choices = result.get('choices', [])
        content = choices[0].get('message', {}).get('content', '') if choices else ''
        metrics = result.get('metrics', {"ttfb_ms": elapsed_ms, "tokens_per_sec": 0})

        add_activity(f"Fast Brain chat: {message[:30]}...", "")

        return jsonify({
            "success": True,
            "response": content,
            "metrics": {
                "ttfb_ms": metrics.get('ttfb_ms', elapsed_ms),
                "total_time_ms": elapsed_ms,
                "tokens_per_sec": metrics.get('tokens_per_sec', 0),
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/fast-brain/benchmark', methods=['POST'])
def fast_brain_benchmark():
    """Run a benchmark on Fast Brain."""
    data = request.json
    num_requests = data.get('num_requests', 5)
    prompt = data.get('prompt', 'Hello, how are you?')

    url = FAST_BRAIN_CONFIG.get('url')
    if not url:
        return jsonify({"success": False, "error": "Fast Brain not configured"})

    results = []
    try:
        import httpx
        import time

        with httpx.Client(timeout=30.0) as client:
            for i in range(num_requests):
                start = time.time()
                # Use Fast Brain LPU /v1/chat/completions endpoint
                response = client.post(
                    f"{url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 50,
                        "stream": False,
                    },
                )
                elapsed = (time.time() - start) * 1000
                result = response.json()
                metrics = result.get('metrics', {})
                results.append({
                    "request": i + 1,
                    "ttfb_ms": metrics.get('ttfb_ms', elapsed),
                    "total_ms": elapsed,
                    "tokens_per_sec": metrics.get('tokens_per_sec', 0),
                })

        avg_ttfb = sum(r['ttfb_ms'] for r in results) / len(results)
        avg_total = sum(r['total_ms'] for r in results) / len(results)
        avg_tps = sum(r['tokens_per_sec'] for r in results) / len(results)

        return jsonify({
            "success": True,
            "results": results,
            "summary": {
                "avg_ttfb_ms": round(avg_ttfb, 2),
                "avg_total_ms": round(avg_total, 2),
                "avg_tokens_per_sec": round(avg_tps, 2),
                "ttfb_target_met": avg_ttfb < 50,
                "throughput_target_met": avg_tps > 500,
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# =============================================================================
# API ENDPOINTS - FAST BRAIN SKILLS
# =============================================================================

@app.route('/api/fast-brain/skills')
def get_fast_brain_skills():
    """Get all available skills."""
    url = FAST_BRAIN_CONFIG.get('url')

    # Try to fetch from deployed LPU first
    if url:
        try:
            import httpx
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{url}/v1/skills")
                if response.status_code == 200:
                    remote_skills = response.json().get("skills", [])
                    # Merge remote skills with local cache
                    for skill in remote_skills:
                        if skill["id"] not in FAST_BRAIN_SKILLS:
                            FAST_BRAIN_SKILLS[skill["id"]] = {
                                **skill,
                                "is_builtin": False,
                            }
        except:
            pass  # Fall back to local cache

    return jsonify({
        "skills": list(FAST_BRAIN_SKILLS.values()),
        "selected": FAST_BRAIN_CONFIG.get("selected_skill", "general")
    })


@app.route('/api/fast-brain/skills', methods=['POST'])
def create_fast_brain_skill():
    """Create a new custom skill."""
    global FAST_BRAIN_SKILLS
    data = request.json

    skill_id = data.get('skill_id', '').lower().replace(' ', '_')
    if not skill_id:
        return jsonify({"success": False, "error": "Skill ID is required"})

    if skill_id in FAST_BRAIN_SKILLS and FAST_BRAIN_SKILLS[skill_id].get('is_builtin'):
        return jsonify({"success": False, "error": "Cannot overwrite built-in skill"})

    skill = {
        "id": skill_id,
        "name": data.get('name', skill_id.title()),
        "description": data.get('description', ''),
        "system_prompt": data.get('system_prompt', ''),
        "knowledge": data.get('knowledge', []),
        "is_builtin": False,
    }

    FAST_BRAIN_SKILLS[skill_id] = skill

    # Try to sync to deployed LPU
    url = FAST_BRAIN_CONFIG.get('url')
    if url:
        try:
            import httpx
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    f"{url}/v1/skills",
                    json={
                        "skill_id": skill_id,
                        "name": skill["name"],
                        "description": skill["description"],
                        "system_prompt": skill["system_prompt"],
                        "knowledge": skill["knowledge"],
                    }
                )
                if response.status_code == 200:
                    add_activity(f"Skill '{skill['name']}' synced to LPU", "")
        except Exception as e:
            add_activity(f"Skill created locally (sync failed: {str(e)[:50]})", "")

    add_activity(f"Created skill: {skill['name']}", "")
    return jsonify({"success": True, "skill": skill})


@app.route('/api/fast-brain/skills/<skill_id>', methods=['DELETE'])
def delete_fast_brain_skill(skill_id):
    """Delete a custom skill."""
    global FAST_BRAIN_SKILLS

    if skill_id not in FAST_BRAIN_SKILLS:
        return jsonify({"success": False, "error": "Skill not found"})

    if FAST_BRAIN_SKILLS[skill_id].get('is_builtin'):
        return jsonify({"success": False, "error": "Cannot delete built-in skill"})

    del FAST_BRAIN_SKILLS[skill_id]
    add_activity(f"Deleted skill: {skill_id}", "")
    return jsonify({"success": True})


@app.route('/api/fast-brain/skills/select', methods=['POST'])
def select_fast_brain_skill():
    """Select the active skill for chat."""
    global FAST_BRAIN_CONFIG
    data = request.json
    skill_id = data.get('skill_id', 'general')

    if skill_id not in FAST_BRAIN_SKILLS:
        return jsonify({"success": False, "error": "Skill not found"})

    FAST_BRAIN_CONFIG['selected_skill'] = skill_id
    add_activity(f"Selected skill: {FAST_BRAIN_SKILLS[skill_id]['name']}", "")
    return jsonify({"success": True, "skill_id": skill_id})


# =============================================================================
# API ENDPOINTS - SYSTEM STATUS & HIVE215 INTEGRATION
# =============================================================================

HIVE215_CONFIG = {
    "url": os.getenv("HIVE215_URL", ""),
    "api_key": os.getenv("HIVE215_API_KEY", ""),
    "status": "disconnected",
    "last_sync": None,
}

INTEGRATION_CHECKLIST = [
    {"id": "fast_brain_url", "name": "Fast Brain URL configured", "status": "pending", "category": "Fast Brain"},
    {"id": "fast_brain_health", "name": "Fast Brain health check passing", "status": "pending", "category": "Fast Brain"},
    {"id": "groq_api_key", "name": "Groq API key in Modal secrets", "status": "pending", "category": "Fast Brain"},
    {"id": "skills_synced", "name": "Skills synced to LPU", "status": "pending", "category": "Fast Brain"},
    {"id": "hive215_url", "name": "Hive215 dashboard URL configured", "status": "pending", "category": "Hive215"},
    {"id": "hive215_api_key", "name": "Hive215 API key configured", "status": "pending", "category": "Hive215"},
    {"id": "hive215_webhooks", "name": "Webhooks configured for events", "status": "pending", "category": "Hive215"},
    {"id": "livekit_connected", "name": "LiveKit agent connected", "status": "pending", "category": "Voice"},
    {"id": "voice_provider", "name": "Voice provider selected", "status": "pending", "category": "Voice"},
    {"id": "fallback_chain", "name": "LLM fallback chain configured", "status": "pending", "category": "LLM"},
]


@app.route('/api/system/status')
def get_system_status():
    """Get comprehensive system status."""
    global SYSTEM_STATUS
    from datetime import datetime

    # Update Fast Brain status
    url = FAST_BRAIN_CONFIG.get('url')
    if url:
        try:
            import httpx
            import time
            start = time.time()
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{url}/health")
                latency = (time.time() - start) * 1000
                if response.status_code == 200:
                    health = response.json()
                    SYSTEM_STATUS["fast_brain"] = {
                        "status": "online" if health.get("status") == "healthy" else "degraded",
                        "last_check": datetime.now().isoformat(),
                        "error": None,
                        "latency_ms": round(latency, 1),
                        "model": health.get("backend", "unknown"),
                        "skills_count": len(health.get("skills_available", [])),
                    }
                else:
                    SYSTEM_STATUS["fast_brain"] = {
                        "status": "error",
                        "last_check": datetime.now().isoformat(),
                        "error": f"HTTP {response.status_code}",
                        "latency_ms": None,
                    }
        except Exception as e:
            SYSTEM_STATUS["fast_brain"] = {
                "status": "offline",
                "last_check": datetime.now().isoformat(),
                "error": str(e)[:100],
                "latency_ms": None,
            }
    else:
        SYSTEM_STATUS["fast_brain"] = {
            "status": "not_configured",
            "last_check": datetime.now().isoformat(),
            "error": "URL not set",
            "latency_ms": None,
        }

    return jsonify({
        "systems": SYSTEM_STATUS,
        "timestamp": datetime.now().isoformat(),
    })


@app.route('/api/system/checklist')
def get_integration_checklist():
    """Get integration checklist with current status."""
    checklist = []

    for item in INTEGRATION_CHECKLIST:
        status = "pending"

        # Auto-check status based on current config
        if item["id"] == "fast_brain_url":
            status = "complete" if FAST_BRAIN_CONFIG.get("url") else "pending"
        elif item["id"] == "fast_brain_health":
            status = "complete" if SYSTEM_STATUS.get("fast_brain", {}).get("status") == "online" else "pending"
        elif item["id"] == "groq_api_key":
            status = "complete" if SYSTEM_STATUS.get("fast_brain", {}).get("status") in ["online", "degraded"] else "pending"
        elif item["id"] == "skills_synced":
            status = "complete" if len(FAST_BRAIN_SKILLS) > 0 else "pending"
        elif item["id"] == "hive215_url":
            status = "complete" if HIVE215_CONFIG.get("url") else "pending"
        elif item["id"] == "hive215_api_key":
            status = "complete" if HIVE215_CONFIG.get("api_key") else "pending"
        elif item["id"] == "voice_provider":
            status = "complete" if VOICE_CONFIG.get("selected_provider") else "pending"
        elif item["id"] == "livekit_connected":
            status = "complete" if PLATFORM_CONNECTIONS.get("livekit", {}).get("status") == "connected" else "pending"

        checklist.append({**item, "status": status})

    complete_count = sum(1 for c in checklist if c["status"] == "complete")
    return jsonify({
        "checklist": checklist,
        "complete": complete_count,
        "total": len(checklist),
        "percent": round(complete_count / len(checklist) * 100),
    })


@app.route('/api/hive215/config')
def get_hive215_config():
    """Get Hive215 configuration."""
    return jsonify(HIVE215_CONFIG)


@app.route('/api/hive215/config', methods=['POST'])
def update_hive215_config():
    """Update Hive215 configuration."""
    global HIVE215_CONFIG
    data = request.json

    if 'url' in data:
        HIVE215_CONFIG['url'] = data['url'].rstrip('/')
    if 'api_key' in data:
        HIVE215_CONFIG['api_key'] = data['api_key']

    add_activity("Hive215 config updated", "")
    return jsonify({"success": True, "config": HIVE215_CONFIG})


@app.route('/api/fast-brain/errors')
def get_fast_brain_errors():
    """Get recent errors for debugging."""
    return jsonify({
        "errors": FAST_BRAIN_CONFIG['stats'].get('errors', [])[-20:],  # Last 20 errors
        "total": len(FAST_BRAIN_CONFIG['stats'].get('errors', [])),
    })


# =============================================================================
# API ENDPOINTS - VOICE LAB
# =============================================================================

VOICE_PROJECTS = {}
VOICE_TRAINING_QUEUE = []


@app.route('/api/voice-lab/projects')
def get_voice_projects():
    """Get all voice projects."""
    return jsonify(list(VOICE_PROJECTS.values()))


@app.route('/api/voice-lab/projects', methods=['POST'])
def create_voice_project():
    """Create a new voice project."""
    data = request.json
    project_id = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    project = {
        "id": project_id,
        "name": data.get("name", "Untitled Voice"),
        "description": data.get("description", ""),
        "base_voice": data.get("base_voice", "chatterbox_default"),
        "provider": data.get("provider", "chatterbox"),
        "status": "draft",
        "samples": [],
        "settings": {
            "pitch": 1.0,
            "speed": 1.0,
            "emotion": "neutral",
            "style": "conversational",
        },
        "training_data": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    VOICE_PROJECTS[project_id] = project
    add_activity(f"Voice project created: {project['name']}", "")
    return jsonify({"success": True, "project": project})


@app.route('/api/voice-lab/projects/<project_id>')
def get_voice_project(project_id):
    """Get a specific voice project."""
    if project_id in VOICE_PROJECTS:
        return jsonify(VOICE_PROJECTS[project_id])
    return jsonify({"error": "Project not found"}), 404


@app.route('/api/voice-lab/projects/<project_id>', methods=['PUT'])
def update_voice_project(project_id):
    """Update a voice project."""
    if project_id not in VOICE_PROJECTS:
        return jsonify({"error": "Project not found"}), 404

    data = request.json
    project = VOICE_PROJECTS[project_id]

    if 'name' in data:
        project['name'] = data['name']
    if 'description' in data:
        project['description'] = data['description']
    if 'settings' in data:
        project['settings'].update(data['settings'])

    project['updated_at'] = datetime.now().isoformat()

    return jsonify({"success": True, "project": project})


@app.route('/api/voice-lab/projects/<project_id>/samples', methods=['POST'])
def add_voice_sample(project_id):
    """Add a training sample to a voice project."""
    if project_id not in VOICE_PROJECTS:
        return jsonify({"error": "Project not found"}), 404

    data = request.json
    sample = {
        "id": f"sample_{len(VOICE_PROJECTS[project_id]['samples'])+1}",
        "text": data.get("text", ""),
        "audio_url": data.get("audio_url", ""),
        "duration_ms": data.get("duration_ms", 0),
        "emotion": data.get("emotion", "neutral"),
        "created_at": datetime.now().isoformat(),
    }

    VOICE_PROJECTS[project_id]['samples'].append(sample)
    VOICE_PROJECTS[project_id]['updated_at'] = datetime.now().isoformat()

    return jsonify({"success": True, "sample": sample})


@app.route('/api/voice-lab/projects/<project_id>/train', methods=['POST'])
def train_voice(project_id):
    """Start training a custom voice."""
    if project_id not in VOICE_PROJECTS:
        return jsonify({"error": "Project not found"}), 404

    project = VOICE_PROJECTS[project_id]

    if len(project['samples']) < 3:
        return jsonify({
            "success": False,
            "error": "Need at least 3 audio samples to train"
        })

    project['status'] = 'training'
    project['training_started'] = datetime.now().isoformat()

    VOICE_TRAINING_QUEUE.append({
        "project_id": project_id,
        "started_at": datetime.now().isoformat(),
        "progress": 0,
    })

    add_activity(f"Voice training started: {project['name']}", "")

    return jsonify({
        "success": True,
        "message": "Training started",
        "estimated_time_minutes": len(project['samples']) * 5
    })


@app.route('/api/voice-lab/projects/<project_id>/test', methods=['POST'])
def test_voice_project(project_id):
    """Test a trained voice with sample text."""
    if project_id not in VOICE_PROJECTS:
        return jsonify({"error": "Project not found"}), 404

    data = request.json
    text = data.get("text", "Hello, this is a test of my custom voice.")

    project = VOICE_PROJECTS[project_id]

    # Simulate voice synthesis
    add_activity(f"Voice test: {project['name']}", "")

    return jsonify({
        "success": True,
        "project_id": project_id,
        "text": text,
        "duration_ms": len(text) * 50,  # Rough estimate
        "audio_url": None,  # Would be real audio URL
        "message": f"Voice '{project['name']}' synthesized successfully"
    })


@app.route('/api/voice-lab/training-status')
def get_voice_training_status():
    """Get status of all voice training jobs."""
    # Simulate progress
    for job in VOICE_TRAINING_QUEUE:
        if job['progress'] < 100:
            job['progress'] = min(100, job['progress'] + random.randint(5, 15))

        if job['progress'] >= 100:
            project = VOICE_PROJECTS.get(job['project_id'])
            if project:
                project['status'] = 'trained'

    return jsonify(VOICE_TRAINING_QUEUE)


# =============================================================================
# API ENDPOINTS - SKILL FINE-TUNING
# =============================================================================

SKILL_TRAINING_JOBS = {}


@app.route('/api/skills/<skill_id>/fine-tune', methods=['POST'])
def fine_tune_skill(skill_id):
    """Start fine-tuning a skill with new examples."""
    data = request.json
    examples = data.get('examples', [])

    if not examples:
        return jsonify({"success": False, "error": "No examples provided"})

    job_id = f"ft_{skill_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    SKILL_TRAINING_JOBS[job_id] = {
        "id": job_id,
        "skill_id": skill_id,
        "status": "running",
        "progress": 0,
        "examples_count": len(examples),
        "started_at": datetime.now().isoformat(),
        "metrics": {
            "loss": 2.5,
            "accuracy": 0.0,
        }
    }

    # Save examples to training data
    training_file = TRAINING_DATA_DIR / f"{skill_id}_finetune.jsonl"
    with open(training_file, 'a') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    add_activity(f"Fine-tuning started for {skill_id} with {len(examples)} examples", "")

    return jsonify({
        "success": True,
        "job_id": job_id,
        "message": f"Fine-tuning started with {len(examples)} examples"
    })


@app.route('/api/skills/<skill_id>/fine-tune/status')
def get_fine_tune_status(skill_id):
    """Get fine-tuning status for a skill."""
    jobs = [j for j in SKILL_TRAINING_JOBS.values() if j['skill_id'] == skill_id]

    # Simulate progress
    for job in jobs:
        if job['status'] == 'running' and job['progress'] < 100:
            job['progress'] = min(100, job['progress'] + random.randint(10, 25))
            job['metrics']['loss'] = max(0.1, job['metrics']['loss'] - 0.3)
            job['metrics']['accuracy'] = min(0.99, job['metrics']['accuracy'] + 0.1)

            if job['progress'] >= 100:
                job['status'] = 'completed'
                job['completed_at'] = datetime.now().isoformat()

    return jsonify(jobs)


@app.route('/api/skills/<skill_id>/feedback', methods=['POST'])
def add_skill_feedback(skill_id):
    """Add feedback for a skill response."""
    data = request.json

    feedback = {
        "skill_id": skill_id,
        "query": data.get("query", ""),
        "response": data.get("response", ""),
        "rating": data.get("rating", 0),  # 1-5 or thumbs up/down
        "corrected_response": data.get("corrected_response"),
        "timestamp": datetime.now().isoformat(),
    }

    # Save to feedback file
    feedback_file = TRAINING_DATA_DIR / f"{skill_id}_feedback.jsonl"
    with open(feedback_file, 'a') as f:
        f.write(json.dumps(feedback) + '\n')

    add_activity(f"Feedback added for {skill_id}", "")

    return jsonify({"success": True, "feedback": feedback})


@app.route('/api/skills/<skill_id>/auto-improve', methods=['POST'])
def auto_improve_skill(skill_id):
    """Automatically improve skill based on collected feedback."""
    feedback_file = TRAINING_DATA_DIR / f"{skill_id}_feedback.jsonl"

    if not feedback_file.exists():
        return jsonify({"success": False, "error": "No feedback collected yet"})

    # Count feedback
    with open(feedback_file, 'r') as f:
        feedback_count = sum(1 for _ in f)

    if feedback_count < 10:
        return jsonify({
            "success": False,
            "error": f"Need at least 10 feedback items, have {feedback_count}"
        })

    # Trigger auto-improvement (simulated)
    add_activity(f"Auto-improvement started for {skill_id} with {feedback_count} feedback items", "")

    return jsonify({
        "success": True,
        "message": f"Auto-improvement started with {feedback_count} feedback items",
        "estimated_time_minutes": 15
    })


# =============================================================================
# UNIFIED DASHBOARD HTML
# =============================================================================

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Skill Command Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&family=Rajdhani:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --neon-cyan: #00fff2;
            --neon-pink: #ff00ff;
            --neon-purple: #b400ff;
            --neon-blue: #00a2ff;
            --neon-green: #00ff88;
            --neon-orange: #ff6b00;
            --neon-yellow: #ffee00;
            --bg-dark: #0a0a0f;
            --card-bg: rgba(15, 15, 25, 0.8);
            --glass-surface: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
            --text-primary: #ffffff;
            --text-secondary: #a0a0b0;
            --font-display: 'Orbitron', monospace;
            --font-body: 'Space Grotesk', sans-serif;
            --font-accent: 'Rajdhani', sans-serif;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: var(--font-body);
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Background Effects */
        .bg-effects {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            pointer-events: none;
            z-index: 0;
        }

        .grid-overlay {
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background-image:
                linear-gradient(rgba(0, 255, 242, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 255, 242, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
        }

        .gradient-blob {
            position: absolute;
            border-radius: 50%;
            filter: blur(100px);
            animation: pulse 8s ease-in-out infinite;
        }

        .blob-1 { width: 600px; height: 600px; background: radial-gradient(circle, rgba(0, 255, 242, 0.15) 0%, transparent 70%); top: -200px; right: -200px; }
        .blob-2 { width: 500px; height: 500px; background: radial-gradient(circle, rgba(180, 0, 255, 0.1) 0%, transparent 70%); bottom: -150px; left: -150px; animation-delay: -4s; }

        @keyframes pulse { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.2); opacity: 0.7; } }

        /* Container */
        .container {
            position: relative;
            z-index: 1;
            max-width: 1800px;
            margin: 0 auto;
            padding: 1.5rem;
        }

        /* Header */
        header {
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .logo {
            font-family: var(--font-display);
            font-size: 2rem;
            font-weight: 900;
            background: linear-gradient(135deg, var(--neon-cyan) 0%, var(--neon-purple) 50%, var(--neon-pink) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .subtitle {
            font-family: var(--font-accent);
            font-size: 0.9rem;
            color: var(--text-secondary);
            letter-spacing: 0.3em;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 1.2rem;
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.2) 0%, rgba(0, 255, 242, 0.1) 100%);
            border: 1px solid var(--neon-green);
            border-radius: 50px;
            font-family: var(--font-display);
            font-size: 0.7rem;
            color: var(--neon-green);
            margin-top: 0.75rem;
        }

        .status-dot { width: 8px; height: 8px; background: var(--neon-green); border-radius: 50%; animation: blink 1s infinite; }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

        /* Main Tabs */
        .main-tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
            justify-content: center;
        }

        .main-tab-btn {
            padding: 0.6rem 1.2rem;
            background: var(--glass-surface);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            color: var(--text-secondary);
            font-family: var(--font-display);
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .main-tab-btn:hover {
            border-color: var(--neon-cyan);
            color: var(--neon-cyan);
        }

        .main-tab-btn.active {
            background: linear-gradient(135deg, rgba(0, 255, 242, 0.2), rgba(180, 0, 255, 0.1));
            border-color: var(--neon-cyan);
            color: var(--neon-cyan);
            box-shadow: 0 0 20px rgba(0, 255, 242, 0.2);
        }

        /* Sub Tabs */
        .sub-tabs {
            display: flex;
            gap: 0.25rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
            padding: 0.5rem;
            background: var(--glass-surface);
            border-radius: 8px;
        }

        .sub-tab-btn {
            padding: 0.5rem 1rem;
            background: transparent;
            border: 1px solid transparent;
            border-radius: 6px;
            color: var(--text-secondary);
            font-family: var(--font-accent);
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .sub-tab-btn:hover {
            background: var(--glass-surface);
            color: var(--text-primary);
        }

        .sub-tab-btn.active {
            background: linear-gradient(135deg, rgba(0, 162, 255, 0.2), rgba(0, 255, 242, 0.1));
            border-color: var(--neon-blue);
            color: var(--neon-blue);
        }

        .tab-content, .sub-tab-content { display: none; }
        .tab-content.active, .sub-tab-content.active { display: block; }

        /* Glass Card */
        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }

        .glass-card:hover {
            border-color: rgba(0, 255, 242, 0.3);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }

        /* Dashboard Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
        }

        .card-full { grid-column: span 4; }
        .card-three { grid-column: span 3; }
        .card-half { grid-column: span 2; }
        .card-quarter { grid-column: span 1; }

        @media (max-width: 1200px) {
            .dashboard-grid { grid-template-columns: repeat(2, 1fr); }
            .card-full, .card-three, .card-half { grid-column: span 2; }
        }

        @media (max-width: 768px) {
            .dashboard-grid { grid-template-columns: 1fr; }
            .card-full, .card-three, .card-half, .card-quarter { grid-column: span 1; }
        }

        /* Stat Card */
        .stat-card { text-align: center; padding: 1rem; }
        .stat-value {
            font-family: var(--font-display);
            font-size: 1.75rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--neon-cyan), var(--neon-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stat-value.green { background: linear-gradient(135deg, var(--neon-green), var(--neon-cyan)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .stat-value.pink { background: linear-gradient(135deg, var(--neon-pink), var(--neon-purple)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .stat-value.orange { background: linear-gradient(135deg, var(--neon-orange), var(--neon-yellow)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .stat-label { font-family: var(--font-accent); font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; margin-top: 0.25rem; }

        /* Section Header */
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .section-title {
            font-family: var(--font-display);
            font-size: 1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .section-icon {
            width: 28px; height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple));
            border-radius: 6px;
            font-size: 0.9rem;
        }

        /* Form Elements */
        .form-group { margin-bottom: 1rem; }
        .form-label {
            display: block;
            font-family: var(--font-accent);
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 0.4rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .form-input, .form-select, .form-textarea {
            width: 100%;
            padding: 0.6rem 0.9rem;
            background: var(--glass-surface);
            border: 1px solid var(--glass-border);
            border-radius: 6px;
            color: var(--text-primary);
            font-family: var(--font-body);
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }

        .form-input:focus, .form-select:focus, .form-textarea:focus {
            outline: none;
            border-color: var(--neon-cyan);
            box-shadow: 0 0 15px rgba(0, 255, 242, 0.2);
        }

        /* Fix dropdown option styling for dark theme */
        .form-select option,
        select option {
            background: #1a1a2e;
            color: #ffffff;
            padding: 0.5rem;
        }

        .form-select option:hover,
        select option:hover,
        .form-select option:checked,
        select option:checked {
            background: #00fff2;
            color: #0a0a0f;
        }

        .form-textarea { resize: vertical; min-height: 80px; }

        .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        @media (max-width: 768px) { .form-row { grid-template-columns: 1fr; } }

        /* Buttons */
        .btn {
            padding: 0.6rem 1.2rem;
            border: none;
            border-radius: 6px;
            font-family: var(--font-display);
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--neon-cyan), var(--neon-blue));
            color: var(--bg-dark);
            box-shadow: 0 0 20px rgba(0, 255, 242, 0.3);
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 30px rgba(0, 255, 242, 0.5);
        }

        .btn-secondary {
            background: transparent;
            border: 1px solid var(--neon-purple);
            color: var(--neon-purple);
        }

        .btn-secondary:hover {
            background: rgba(180, 0, 255, 0.1);
            box-shadow: 0 0 20px rgba(180, 0, 255, 0.3);
        }

        .btn-success {
            background: linear-gradient(135deg, var(--neon-green), var(--neon-cyan));
            color: var(--bg-dark);
        }

        .btn-danger {
            background: linear-gradient(135deg, var(--neon-pink), var(--neon-orange));
            color: var(--bg-dark);
        }

        .btn-sm { padding: 0.4rem 0.8rem; font-size: 0.7rem; }

        /* Skills Grid */
        .skills-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 0.75rem;
        }

        .skill-card {
            background: var(--glass-surface);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .skill-card:hover {
            border-color: var(--neon-cyan);
            transform: scale(1.02);
        }

        .skill-card.deployed { border-left: 3px solid var(--neon-green); }
        .skill-card.training { border-left: 3px solid var(--neon-orange); }
        .skill-card.draft { border-left: 3px solid var(--text-secondary); }

        .skill-header { display: flex; justify-content: space-between; margin-bottom: 0.75rem; }
        .skill-name { font-family: var(--font-display); font-size: 0.9rem; }
        .skill-type { font-size: 0.7rem; color: var(--text-secondary); }
        .skill-status {
            padding: 0.2rem 0.6rem;
            border-radius: 20px;
            font-size: 0.65rem;
            text-transform: uppercase;
        }
        .skill-status.deployed { background: rgba(0, 255, 136, 0.2); color: var(--neon-green); }
        .skill-status.training { background: rgba(255, 107, 0, 0.2); color: var(--neon-orange); }
        .skill-status.draft { background: rgba(160, 160, 176, 0.2); color: var(--text-secondary); }

        .skill-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--glass-border); }
        .skill-metric { text-align: center; }
        .metric-value { font-family: var(--font-display); font-size: 0.8rem; color: var(--neon-cyan); }
        .metric-label { font-size: 0.6rem; color: var(--text-secondary); text-transform: uppercase; }

        /* Skills Table */
        .skills-table {
            width: 100%;
            border-collapse: collapse;
        }

        .skills-table th, .skills-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--glass-border);
        }

        .skills-table th {
            font-family: var(--font-display);
            font-size: 0.75rem;
            color: var(--neon-cyan);
            text-transform: uppercase;
            background: var(--glass-surface);
        }

        .skills-table tr:hover {
            background: var(--glass-surface);
        }

        .table-status {
            padding: 0.2rem 0.6rem;
            border-radius: 20px;
            font-size: 0.7rem;
        }

        .table-status.deployed { background: rgba(0, 255, 136, 0.2); color: var(--neon-green); }
        .table-status.training { background: rgba(255, 107, 0, 0.2); color: var(--neon-orange); }
        .table-status.draft { background: rgba(160, 160, 176, 0.2); color: var(--text-secondary); }

        .table-filter {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }

        .table-filter input, .table-filter select {
            padding: 0.5rem;
            background: var(--glass-surface);
            border: 1px solid var(--glass-border);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 0.85rem;
        }

        /* Console */
        .console {
            background: #0d0d12;
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            padding: 1rem;
            font-family: monospace;
            font-size: 0.85rem;
            min-height: 250px;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            color: var(--neon-green);
        }

        .console .prompt { color: var(--neon-cyan); }
        .console .response { color: var(--text-primary); }
        .console .error { color: var(--neon-pink); }
        .console .info { color: var(--text-secondary); }

        /* Training Pipeline */
        .pipeline {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 0;
        }

        .pipeline-step {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
            flex: 1;
        }

        .pipeline-icon {
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            font-size: 1.25rem;
            background: var(--glass-surface);
            border: 2px solid var(--glass-border);
            transition: all 0.3s ease;
        }

        .pipeline-icon.complete {
            background: rgba(0, 255, 136, 0.2);
            border-color: var(--neon-green);
        }

        .pipeline-icon.active {
            background: rgba(0, 162, 255, 0.2);
            border-color: var(--neon-blue);
            animation: glow 1.5s ease-in-out infinite;
        }

        @keyframes glow {
            0%, 100% { box-shadow: 0 0 10px rgba(0, 162, 255, 0.3); }
            50% { box-shadow: 0 0 25px rgba(0, 162, 255, 0.6); }
        }

        .pipeline-label {
            font-family: var(--font-accent);
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        .pipeline-connector {
            flex: 0.5;
            height: 2px;
            background: var(--glass-border);
        }

        .pipeline-connector.complete {
            background: var(--neon-green);
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: var(--glass-surface);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 1rem;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--neon-cyan), var(--neon-blue));
            border-radius: 4px;
            transition: width 0.5s ease;
        }

        .progress-text {
            text-align: center;
            font-family: var(--font-display);
            font-size: 0.9rem;
            margin-top: 0.5rem;
            color: var(--neon-cyan);
        }

        /* Activity Feed */
        .activity-feed {
            max-height: 300px;
            overflow-y: auto;
        }

        .activity-item {
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--glass-border);
        }

        .activity-icon {
            font-size: 1rem;
            width: 24px;
            text-align: center;
        }

        .activity-content { flex: 1; }
        .activity-message { font-size: 0.85rem; }
        .activity-time { font-size: 0.7rem; color: var(--text-secondary); }

        /* Chart */
        .chart-container {
            height: 120px;
            display: flex;
            align-items: flex-end;
            gap: 3px;
            padding: 0.5rem 0;
        }

        .chart-bar {
            flex: 1;
            background: linear-gradient(180deg, var(--neon-cyan) 0%, rgba(0, 255, 242, 0.3) 100%);
            border-radius: 3px 3px 0 0;
            transition: all 0.3s ease;
        }

        .chart-bar:hover { background: linear-gradient(180deg, var(--neon-pink) 0%, rgba(255, 0, 255, 0.3) 100%); }

        /* Server Panel */
        .server-panel { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; }
        .server-stat { text-align: center; padding: 0.75rem; background: var(--glass-surface); border-radius: 10px; }

        /* Compare Results */
        .compare-results {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }

        .compare-card {
            background: var(--glass-surface);
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            padding: 1rem;
        }

        .compare-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--glass-border);
        }

        .compare-provider {
            font-family: var(--font-display);
            font-size: 0.9rem;
            color: var(--neon-cyan);
        }

        .compare-latency {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        /* Code Block */
        .code-block {
            background: #0d0d12;
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            padding: 1rem;
            font-family: monospace;
            font-size: 0.8rem;
            overflow-x: auto;
            white-space: pre;
            color: var(--text-primary);
        }

        .code-block .keyword { color: var(--neon-pink); }
        .code-block .string { color: var(--neon-green); }
        .code-block .comment { color: var(--text-secondary); }

        /* Loading */
        .loading { opacity: 0.5; pointer-events: none; }
        .spinner {
            display: inline-block;
            width: 18px; height: 18px;
            border: 2px solid var(--glass-border);
            border-top-color: var(--neon-cyan);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Messages */
        .message {
            padding: 0.75rem 1rem;
            border-radius: 6px;
            margin-bottom: 0.75rem;
            font-family: var(--font-accent);
            font-size: 0.9rem;
        }
        .message.success { background: rgba(0, 255, 136, 0.1); border: 1px solid var(--neon-green); color: var(--neon-green); }
        .message.error { background: rgba(255, 0, 85, 0.1); border: 1px solid var(--neon-pink); color: var(--neon-pink); }
        .message.info { background: rgba(0, 162, 255, 0.1); border: 1px solid var(--neon-blue); color: var(--neon-blue); }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-dark); }
        ::-webkit-scrollbar-thumb { background: var(--glass-border); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--neon-cyan); }

        /* File Upload */
        .file-upload {
            border: 2px dashed var(--glass-border);
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .file-upload:hover {
            border-color: var(--neon-cyan);
            background: var(--glass-surface);
        }

        .file-upload input { display: none; }
        .file-upload-icon { font-size: 2rem; margin-bottom: 0.5rem; }
        .file-upload-text { color: var(--text-secondary); font-size: 0.9rem; }
    </style>
</head>
<body>
    <div class="bg-effects">
        <div class="grid-overlay"></div>
        <div class="gradient-blob blob-1"></div>
        <div class="gradient-blob blob-2"></div>
    </div>

    <div class="container">
        <header>
            <h1 class="logo">Skill Command Center</h1>
            <p class="subtitle">Voice Agent Intelligence Hub</p>
            <div class="status-badge">
                <span class="status-dot"></span>
                <span id="server-status-text">LPU Online - 1 Warm Container</span>
            </div>
        </header>

        <!-- Main Tabs -->
        <div class="main-tabs">
            <button class="main-tab-btn active" onclick="showMainTab('dashboard')">Dashboard</button>
            <button class="main-tab-btn" onclick="showMainTab('fastbrain')">Fast Brain</button>
            <button class="main-tab-btn" onclick="showMainTab('voicelab')">Voice Lab</button>
            <button class="main-tab-btn" onclick="showMainTab('factory')">Skill Factory</button>
            <button class="main-tab-btn" onclick="showMainTab('command')">Command Center</button>
            <button class="main-tab-btn" onclick="showMainTab('lpu')">LPU Inference</button>
        </div>

        <!-- ================================================================ -->
        <!-- DASHBOARD TAB -->
        <!-- ================================================================ -->
        <div id="tab-dashboard" class="tab-content active">
            <div class="dashboard-grid">
                <div class="glass-card stat-card card-quarter">
                    <div class="stat-value" id="total-skills">0</div>
                    <div class="stat-label">Active Skills</div>
                </div>
                <div class="glass-card stat-card card-quarter">
                    <div class="stat-value green" id="requests-today">0</div>
                    <div class="stat-label">Requests Today</div>
                </div>
                <div class="glass-card stat-card card-quarter">
                    <div class="stat-value pink" id="avg-latency">0ms</div>
                    <div class="stat-label">Avg Latency</div>
                </div>
                <div class="glass-card stat-card card-quarter">
                    <div class="stat-value orange" id="satisfaction">0%</div>
                    <div class="stat-label">Satisfaction</div>
                </div>

                <div class="glass-card card-full">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Server</span> Server Status</div>
                        <button class="btn btn-secondary btn-sm" onclick="refreshServerStatus()">Refresh</button>
                    </div>
                    <div class="server-panel">
                        <div class="server-stat">
                            <div class="stat-value green" style="font-size: 1.25rem;">Online</div>
                            <div class="stat-label">Status</div>
                        </div>
                        <div class="server-stat">
                            <div class="stat-value" style="font-size: 1.25rem;" id="warm-containers">1</div>
                            <div class="stat-label">Warm Containers</div>
                        </div>
                        <div class="server-stat">
                            <div class="stat-value" style="font-size: 1.25rem;" id="memory-usage">0GB</div>
                            <div class="stat-label">Memory</div>
                        </div>
                        <div class="server-stat">
                            <div class="stat-value orange" style="font-size: 1.25rem;" id="cost-today">$0.00</div>
                            <div class="stat-label">Cost Today</div>
                        </div>
                    </div>
                </div>

                <div class="glass-card card-three">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Skills</span> Skill Library</div>
                        <div>
                            <button class="btn btn-secondary btn-sm" onclick="showMainTab('factory')">Open Factory</button>
                            <button class="btn btn-primary btn-sm" onclick="showMainTab('factory'); showFactoryTab('profile')">+ New Skill</button>
                        </div>
                    </div>
                    <div class="skills-grid" id="skills-grid"></div>
                </div>

                <div class="glass-card card-quarter">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Train</span> Training Pipeline</div>
                    </div>
                    <div class="pipeline" style="flex-direction: column; gap: 0.75rem;">
                        <div class="pipeline-step" style="flex-direction: row; width: 100%; justify-content: flex-start; gap: 1rem;">
                            <div class="pipeline-icon complete" id="pipe-ingest">Ingest</div>
                            <div class="pipeline-label">Ingest</div>
                        </div>
                        <div class="pipeline-step" style="flex-direction: row; width: 100%; justify-content: flex-start; gap: 1rem;">
                            <div class="pipeline-icon complete" id="pipe-process">Process</div>
                            <div class="pipeline-label">Process</div>
                        </div>
                        <div class="pipeline-step" style="flex-direction: row; width: 100%; justify-content: flex-start; gap: 1rem;">
                            <div class="pipeline-icon active" id="pipe-train">Train</div>
                            <div class="pipeline-label">Train</div>
                        </div>
                        <div class="pipeline-step" style="flex-direction: row; width: 100%; justify-content: flex-start; gap: 1rem;">
                            <div class="pipeline-icon" id="pipe-deploy">Deploy</div>
                            <div class="pipeline-label">Deploy</div>
                        </div>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="training-progress" style="width: 67%;"></div>
                    </div>
                    <div class="progress-text" id="training-text">Training: plumber_expert 67%</div>
                </div>

                <div class="glass-card card-quarter">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Feed</span> Activity Feed</div>
                    </div>
                    <div class="activity-feed" id="activity-feed">
                        <div class="activity-item">
                            <span class="activity-icon">Done</span>
                            <div class="activity-content">
                                <div class="activity-message">plumber_expert deployed successfully</div>
                                <div class="activity-time">2 minutes ago</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="glass-card card-half">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Chart</span> Request Volume (24h)</div>
                    </div>
                    <div class="chart-container" id="request-chart"></div>
                </div>

                <div class="glass-card card-half">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Speed</span> Latency (24h)</div>
                    </div>
                    <div class="chart-container" id="latency-chart"></div>
                </div>
            </div>
        </div>

        <!-- ================================================================ -->
        <!-- SKILL FACTORY TAB -->
        <!-- ================================================================ -->
        <div id="tab-factory" class="tab-content">
            <div class="sub-tabs">
                <button class="sub-tab-btn active" onclick="showFactoryTab('profile')">Business Profile</button>
                <button class="sub-tab-btn" onclick="showFactoryTab('documents')">Upload Documents</button>
                <button class="sub-tab-btn" onclick="showFactoryTab('train')">Train Skill</button>
                <button class="sub-tab-btn" onclick="showFactoryTab('manage')">Manage Skills</button>
            </div>

            <!-- Business Profile -->
            <div id="factory-profile" class="sub-tab-content active">
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Profile</span> Create Business Profile</div>
                    </div>
                    <div id="profile-message"></div>
                    <form id="profile-form" onsubmit="saveProfile(event)">
                        <div class="form-row">
                            <div class="form-group">
                                <label class="form-label">Business Name *</label>
                                <input type="text" class="form-input" id="p-name" placeholder="Joe's Plumbing Services" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Business Type</label>
                                <select class="form-select" id="p-type">
                                    <option value="General Customer Service">General Customer Service</option>
                                    <option value="Plumbing Services">Plumbing Services</option>
                                    <option value="Electrical Services">Electrical Services</option>
                                    <option value="HVAC Services">HVAC Services</option>
                                    <option value="Restaurant/Food Service">Restaurant/Food Service</option>
                                    <option value="Medical/Healthcare">Medical/Healthcare</option>
                                    <option value="Legal Services">Legal Services</option>
                                    <option value="Real Estate">Real Estate</option>
                                    <option value="Tech Support">Tech Support</option>
                                    <option value="Retail Store">Retail Store</option>
                                </select>
                            </div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Business Description</label>
                            <textarea class="form-textarea" id="p-description" placeholder="We provide 24/7 emergency plumbing services in the greater metro area..."></textarea>
                        </div>
                        <div class="form-row">
                            <div class="form-group">
                                <label class="form-label">Default Greeting</label>
                                <input type="text" class="form-input" id="p-greeting" placeholder="Hello! Thanks for calling. How can I help you today?">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Personality</label>
                                <select class="form-select" id="p-personality">
                                    <option value="Friendly and casual">Friendly and casual</option>
                                    <option value="Professional and formal">Professional and formal</option>
                                    <option value="Warm and empathetic">Warm and empathetic</option>
                                    <option value="Efficient and direct">Efficient and direct</option>
                                    <option value="Technical and precise">Technical and precise</option>
                                </select>
                            </div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Key Services (one per line)</label>
                            <textarea class="form-textarea" id="p-services" placeholder="Emergency repairs&#10;Drain cleaning&#10;Water heater installation"></textarea>
                        </div>
                        <div class="form-group">
                            <label class="form-label">FAQ (Q: question / A: answer)</label>
                            <textarea class="form-textarea" id="p-faq" placeholder="Q: What are your hours?&#10;A: We're open 24/7 for emergencies.&#10;&#10;Q: Do you offer free estimates?&#10;A: Yes!"></textarea>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Custom Instructions</label>
                            <textarea class="form-textarea" id="p-instructions" placeholder="Always ask for the customer's address first..."></textarea>
                        </div>
                        <div style="display: flex; gap: 1rem;">
                            <button type="submit" class="btn btn-primary">Save Profile</button>
                            <button type="button" class="btn btn-success" onclick="generateFromProfile()">Generate Training Data</button>
                        </div>
                    </form>
                </div>
            </div>

            <!-- Upload Documents -->
            <div id="factory-documents" class="sub-tab-content">
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Docs</span> Upload Documents</div>
                    </div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">Upload PDFs, text files, or training data. The system will convert them into training examples.</p>
                    <div id="upload-message"></div>
                    <div class="form-group">
                        <label class="form-label">Select Business Profile</label>
                        <select class="form-select" id="doc-profile"></select>
                    </div>
                    <div class="file-upload" onclick="document.getElementById('file-input').click()">
                        <div class="file-upload-icon">Upload</div>
                        <div class="file-upload-text">Click to upload PDF, TXT, MD, JSON, or JSONL</div>
                        <input type="file" id="file-input" accept=".pdf,.txt,.md,.json,.jsonl" onchange="handleFileUpload(this)">
                    </div>
                    <div id="file-preview" style="margin-top: 1rem;"></div>
                </div>
            </div>

            <!-- Train Skill -->
            <div id="factory-train" class="sub-tab-content">
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Train</span> Train LoRA Adapter</div>
                    </div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">Training requires a GPU. Generate a script to run on Colab, Modal, or your own machine.</p>
                    <div class="form-group">
                        <label class="form-label">Select Skill to Train</label>
                        <select class="form-select" id="train-profile"></select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Training Steps: <span id="steps-value">60</span></label>
                        <input type="range" id="train-steps" min="20" max="200" value="60" style="width: 100%;" oninput="document.getElementById('steps-value').textContent = this.value">
                        <div style="display: flex; justify-content: space-between; color: var(--text-secondary); font-size: 0.75rem;">
                            <span>20 (Quick)</span>
                            <span>200 (Thorough)</span>
                        </div>
                    </div>
                    <button class="btn btn-primary" onclick="generateTrainingScript()">Generate Training Script</button>
                    <div id="training-output" style="margin-top: 1rem; display: none;">
                        <label class="form-label">Training Script</label>
                        <pre class="code-block" id="training-script"></pre>
                    </div>
                </div>
            </div>

            <!-- Manage Skills -->
            <div id="factory-manage" class="sub-tab-content">
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Table</span> Skill Library</div>
                        <button class="btn btn-secondary btn-sm" onclick="loadSkillsTable()">Refresh</button>
                    </div>
                    <div class="table-filter">
                        <input type="text" id="table-search" placeholder="Search skills..." oninput="filterSkillsTable()">
                        <select id="table-type-filter" onchange="filterSkillsTable()">
                            <option value="">All Types</option>
                            <option value="Plumbing">Plumbing</option>
                            <option value="Restaurant">Restaurant</option>
                            <option value="Tech Support">Tech Support</option>
                            <option value="Healthcare">Healthcare</option>
                        </select>
                        <select id="table-status-filter" onchange="filterSkillsTable()">
                            <option value="">All Status</option>
                            <option value="Deployed">Deployed</option>
                            <option value="Training">Training</option>
                            <option value="Draft">Draft</option>
                        </select>
                    </div>
                    <table class="skills-table">
                        <thead>
                            <tr>
                                <th>Business</th>
                                <th>Type</th>
                                <th>Status</th>
                                <th>Examples</th>
                                <th>Created</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="skills-table-body"></tbody>
                    </table>
                    <div style="margin-top: 1.5rem; padding: 1rem; background: var(--glass-surface); border-radius: 8px;">
                        <p style="color: var(--text-secondary); font-size: 0.85rem;">
                            <strong>Deploy to Modal:</strong><br>
                            <code style="color: var(--neon-cyan);">modal volume put lpu-skills adapters/your_skill /root/skills/your_skill.lora</code>
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <!-- ================================================================ -->
        <!-- COMMAND CENTER TAB -->
        <!-- ================================================================ -->
        <div id="tab-command" class="tab-content">
            <div class="sub-tabs">
                <button class="sub-tab-btn active" onclick="showCommandTab('keys')">API Keys</button>
                <button class="sub-tab-btn" onclick="showCommandTab('test')">Test LLM</button>
                <button class="sub-tab-btn" onclick="showCommandTab('compare')">Compare</button>
                <button class="sub-tab-btn" onclick="showCommandTab('masking')">Latency Masking</button>
                <button class="sub-tab-btn" onclick="showCommandTab('stats')">Stats</button>
                <button class="sub-tab-btn" onclick="showCommandTab('voices')">Voice Library</button>
                <button class="sub-tab-btn" onclick="showCommandTab('voice')">Voice Integration</button>
            </div>

            <!-- API Keys -->
            <div id="command-keys" class="sub-tab-content active">
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Keys</span> Configure API Keys</div>
                    </div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">Leave blank to use mock clients for testing.</p>
                    <div id="keys-message"></div>
                    <div class="form-row">
                        <div class="form-group">
                            <label class="form-label">Groq API Key</label>
                            <input type="password" class="form-input" id="key-groq" placeholder="gsk_...">
                        </div>
                        <div class="form-group">
                            <label class="form-label">OpenAI API Key</label>
                            <input type="password" class="form-input" id="key-openai" placeholder="sk-...">
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Anthropic API Key</label>
                        <input type="password" class="form-input" id="key-anthropic" placeholder="sk-ant-...">
                    </div>
                    <button class="btn btn-primary" onclick="saveApiKeys()">Save API Keys</button>
                </div>
            </div>

            <!-- Test LLM -->
            <div id="command-test" class="sub-tab-content">
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Test</span> Test Single LLM</div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label class="form-label">Prompt</label>
                            <textarea class="form-textarea" id="test-prompt" placeholder="What are your hours?"></textarea>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Provider</label>
                            <select class="form-select" id="test-provider">
                                <option value="groq">Groq (Llama 3.1 8B)</option>
                                <option value="openai">OpenAI (GPT-4o-mini)</option>
                                <option value="anthropic">Anthropic (Claude 3 Haiku)</option>
                                <option value="bitnet">BitNet LPU</option>
                            </select>
                            <div style="margin-top: 0.5rem;">
                                <label><input type="checkbox" id="test-masking"> Enable Latency Masking</label>
                            </div>
                        </div>
                    </div>
                    <button class="btn btn-primary" onclick="testLLM()" id="test-btn">Run Test</button>
                    <div style="margin-top: 1rem;">
                        <label class="form-label">Response</label>
                        <div class="console" id="test-output">Waiting for test...</div>
                    </div>
                </div>
            </div>

            <!-- Compare -->
            <div id="command-compare" class="sub-tab-content">
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Compare</span> Compare Providers</div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Prompt</label>
                        <textarea class="form-textarea" id="compare-prompt" placeholder="Explain quantum computing in one sentence."></textarea>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Providers to Compare</label>
                        <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                            <label><input type="checkbox" value="groq" class="compare-provider" checked> Groq</label>
                            <label><input type="checkbox" value="openai" class="compare-provider" checked> OpenAI</label>
                            <label><input type="checkbox" value="anthropic" class="compare-provider"> Anthropic</label>
                            <label><input type="checkbox" value="bitnet" class="compare-provider"> BitNet</label>
                        </div>
                    </div>
                    <button class="btn btn-primary" onclick="compareLLMs()" id="compare-btn">Compare</button>
                    <div class="compare-results" id="compare-output" style="margin-top: 1rem;"></div>
                </div>
            </div>

            <!-- Latency Masking -->
            <div id="command-masking" class="sub-tab-content">
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Mask</span> Latency Masking Config</div>
                    </div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">Configure filler sounds and phrases that make slow LLMs feel natural.</p>
                    <div class="form-row">
                        <div>
                            <h4 style="color: var(--neon-cyan); margin-bottom: 0.5rem;">Filler Sounds</h4>
                            <ul style="color: var(--text-secondary); list-style: none;">
                                <li>Hmm...</li>
                                <li>Mmm...</li>
                                <li>Umm...</li>
                                <li>Ah...</li>
                                <li>Well...</li>
                            </ul>
                        </div>
                        <div>
                            <h4 style="color: var(--neon-cyan); margin-bottom: 0.5rem;">Thinking Phrases</h4>
                            <ul style="color: var(--text-secondary); list-style: none;">
                                <li>Let me think about that...</li>
                                <li>That's a good question...</li>
                                <li>Let me check...</li>
                                <li>One moment...</li>
                            </ul>
                        </div>
                    </div>
                    <div style="margin-top: 1rem;">
                        <h4 style="color: var(--neon-cyan); margin-bottom: 0.5rem;">Skill-Specific Fillers</h4>
                        <table class="skills-table">
                            <tr><th>Skill Type</th><th>Custom Fillers</th></tr>
                            <tr><td>Technical</td><td>"Let me look that up...", "Checking the docs..."</td></tr>
                            <tr><td>Customer Service</td><td>"I understand.", "Let me help with that..."</td></tr>
                            <tr><td>Scheduling</td><td>"Let me check the calendar...", "One moment..."</td></tr>
                            <tr><td>Sales</td><td>"Great question!", "Let me find the best option..."</td></tr>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Stats -->
            <div id="command-stats" class="sub-tab-content">
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Stats</span> Performance Statistics</div>
                        <button class="btn btn-secondary btn-sm" onclick="refreshStats()">Refresh</button>
                    </div>
                    <pre class="code-block" id="stats-output">Loading stats...</pre>
                </div>
            </div>

            <!-- Voice Library -->
            <div id="command-voices" class="sub-tab-content">
                <!-- Voice Selection Card -->
                <div class="glass-card" style="margin-bottom: 1rem;">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Voices</span> Voice Selection</div>
                    </div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">Select your preferred TTS provider and voice for your agent.</p>

                    <div class="form-row">
                        <div class="form-group">
                            <label class="form-label">TTS Provider</label>
                            <select class="form-select" id="voice-provider" onchange="loadProviderVoices()">
                                <option value="">Select a provider...</option>
                                <optgroup label="Free / Open Source">
                                    <option value="edge_tts">Edge TTS (Microsoft - Free)</option>
                                    <option value="kokoro">Kokoro (Ultra-fast Local)</option>
                                    <option value="chatterbox">Chatterbox (Resemble AI)</option>
                                    <option value="xtts">XTTS-v2 (Coqui AI)</option>
                                    <option value="openvoice">OpenVoice (MyShell)</option>
                                </optgroup>
                                <optgroup label="Paid Services">
                                    <option value="elevenlabs">ElevenLabs</option>
                                    <option value="openai">OpenAI TTS</option>
                                    <option value="azure">Azure TTS</option>
                                    <option value="cartesia">Cartesia</option>
                                </optgroup>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Voice</label>
                            <select class="form-select" id="voice-select" onchange="selectVoice()">
                                <option value="">Select a voice...</option>
                            </select>
                        </div>
                    </div>

                    <div id="voice-details" style="display: none; margin-bottom: 1rem; padding: 1rem; background: var(--glass-surface); border-radius: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong id="voice-name" style="color: var(--neon-cyan);"></strong>
                                <span id="voice-meta" style="color: var(--text-secondary); margin-left: 0.5rem;"></span>
                            </div>
                            <span id="voice-status" class="table-status deployed">Selected</span>
                        </div>
                    </div>

                    <div class="form-row">
                        <div class="form-group" style="flex: 2;">
                            <label class="form-label">Test Text</label>
                            <input type="text" class="form-input" id="voice-test-text" value="Hello! How can I help you today?" placeholder="Enter text to test...">
                        </div>
                        <div class="form-group" style="flex: 1;">
                            <label class="form-label">Speed</label>
                            <input type="range" id="voice-speed" min="0.5" max="2" step="0.1" value="1" style="width: 100%;" oninput="document.getElementById('voice-speed-val').textContent = this.value + 'x'">
                            <small style="color: var(--text-secondary);">Speed: <span id="voice-speed-val">1x</span></small>
                        </div>
                    </div>

                    <div style="display: flex; gap: 0.5rem;">
                        <button class="btn btn-primary" onclick="testVoice()">Test Voice</button>
                        <button class="btn btn-secondary" onclick="saveVoiceConfig()">Save as Default</button>
                    </div>
                    <div id="voice-test-result" style="margin-top: 1rem;"></div>
                </div>

                <!-- Voice Library Card -->
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Lib</span> Free TTS Voice Library</div>
                    </div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">Open-source Text-to-Speech models with voice cloning capabilities. Replace paid services like ElevenLabs and Cartesia.</p>

                    <table class="skills-table" style="margin-bottom: 1.5rem;">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Parameters</th>
                                <th>Voice Cloning</th>
                                <th>License</th>
                                <th>Quality</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong style="color: var(--neon-cyan);">Chatterbox</strong><br><small style="color: var(--text-secondary);">Resemble AI</small></td>
                                <td>0.5B</td>
                                <td><span class="table-status deployed">Yes (few seconds)</span></td>
                                <td>MIT</td>
                                <td>Outperforms ElevenLabs in blind tests</td>
                            </tr>
                            <tr>
                                <td><strong style="color: var(--neon-green);">Kokoro</strong><br><small style="color: var(--text-secondary);">Fastest option</small></td>
                                <td>82M</td>
                                <td><span class="table-status draft">No</span></td>
                                <td>Apache 2.0</td>
                                <td>Comparable to larger models, very fast</td>
                            </tr>
                            <tr>
                                <td><strong style="color: var(--neon-purple);">XTTS-v2</strong><br><small style="color: var(--text-secondary);">Coqui AI</small></td>
                                <td>~500M</td>
                                <td><span class="table-status deployed">Yes (6 sec audio)</span></td>
                                <td>CPML</td>
                                <td>Multi-language cloning, 17+ languages</td>
                            </tr>
                            <tr>
                                <td><strong style="color: var(--neon-orange);">F5-TTS</strong><br><small style="color: var(--text-secondary);">E2 TTS variant</small></td>
                                <td>~300M</td>
                                <td><span class="table-status deployed">Yes</span></td>
                                <td>MIT</td>
                                <td>Comparable to ElevenLabs, 12GB VRAM</td>
                            </tr>
                            <tr>
                                <td><strong style="color: var(--neon-pink);">Tortoise-TTS</strong><br><small style="color: var(--text-secondary);">Most realistic</small></td>
                                <td>~1B</td>
                                <td><span class="table-status deployed">Yes</span></td>
                                <td>Apache 2.0</td>
                                <td>Best prosody/intonation, slower</td>
                            </tr>
                            <tr>
                                <td><strong style="color: var(--neon-blue);">Piper</strong><br><small style="color: var(--text-secondary);">Lightweight</small></td>
                                <td>~20M</td>
                                <td><span class="table-status draft">No</span></td>
                                <td>MIT</td>
                                <td>Ultra-fast, runs on Raspberry Pi</td>
                            </tr>
                            <tr>
                                <td><strong style="color: var(--neon-yellow);">Bark</strong><br><small style="color: var(--text-secondary);">Suno AI</small></td>
                                <td>~350M</td>
                                <td><span class="table-status training">Prompt-based</span></td>
                                <td>MIT</td>
                                <td>Can generate music, laughter, effects</td>
                            </tr>
                            <tr>
                                <td><strong style="color: var(--text-primary);">OpenVoice</strong><br><small style="color: var(--text-secondary);">MyShell AI</small></td>
                                <td>~300M</td>
                                <td><span class="table-status deployed">Yes (instant)</span></td>
                                <td>MIT</td>
                                <td>Instant voice cloning, style control</td>
                            </tr>
                        </tbody>
                    </table>

                    <h4 style="color: var(--neon-cyan); margin-bottom: 0.75rem;">Recommended Setup: Chatterbox</h4>
                    <p style="color: var(--text-secondary); margin-bottom: 0.75rem;">Best balance of quality, speed, and ease of use. MIT licensed with 23 languages support.</p>
                    <pre class="code-block"># Install Chatterbox
pip install chatterbox-tts

# Basic usage
from chatterbox import ChatterboxTTS

tts = ChatterboxTTS()

# Generate speech
audio = tts.generate("Hello! How can I help you today?")
audio.save("output.wav")

# Clone a voice (just needs a few seconds of audio)
tts.clone_voice("reference_audio.wav", voice_name="custom_voice")
audio = tts.generate("Now speaking in cloned voice!", voice="custom_voice")</pre>

                    <h4 style="color: var(--neon-green); margin: 1.5rem 0 0.75rem;">Quick Setup: Kokoro (Fastest)</h4>
                    <pre class="code-block"># Install Kokoro - ultra-fast TTS
pip install kokoro-onnx soundfile

from kokoro_onnx import Kokoro

kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")
samples, sr = kokoro.create("Hello, this is super fast TTS!", voice="af_bella", speed=1.0)

import soundfile as sf
sf.write("output.wav", samples, sr)</pre>

                    <h4 style="color: var(--neon-purple); margin: 1.5rem 0 0.75rem;">Multi-Language: XTTS-v2</h4>
                    <pre class="code-block"># Install XTTS-v2 (Coqui)
pip install TTS

from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Clone voice and generate in different language
tts.tts_to_file(
    text="Bonjour! Comment puis-je vous aider?",
    speaker_wav="reference.wav",  # 6 seconds of audio
    language="fr",
    file_path="output.wav"
)</pre>

                    <div style="margin-top: 1.5rem; padding: 1rem; background: var(--glass-surface); border-radius: 8px;">
                        <h4 style="color: var(--neon-orange); margin-bottom: 0.5rem;">Cost Savings</h4>
                        <p style="color: var(--text-secondary); font-size: 0.9rem;">
                            <strong>ElevenLabs:</strong> $0.30/1K chars | <strong>Cartesia:</strong> $0.015/min<br>
                            <strong>Self-hosted:</strong> $0 (just compute costs)<br><br>
                            At 10,000 minutes/month: Save ~$150/month by self-hosting Chatterbox or XTTS-v2
                        </p>
                    </div>
                </div>
            </div>

            <!-- Voice Integration -->
            <div id="command-voice" class="sub-tab-content">
                <!-- Platform Connections -->
                <div class="glass-card" style="margin-bottom: 1rem;">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Connect</span> Voice Platform Connections</div>
                        <button class="btn btn-secondary btn-sm" onclick="loadPlatforms()">Refresh</button>
                    </div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">Connect your Skill Command Center to voice platforms for real-time voice interactions.</p>

                    <div id="platforms-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem;">
                        <!-- LiveKit -->
                        <div class="platform-card" id="platform-livekit" style="padding: 1rem; background: var(--glass-surface); border-radius: 8px; border: 1px solid var(--glass-border);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                                <strong style="color: var(--neon-cyan);">LiveKit</strong>
                                <span id="status-livekit" class="table-status draft">Disconnected</span>
                            </div>
                            <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.75rem;">Real-time voice and video with agents SDK</p>
                            <div class="form-group" style="margin-bottom: 0.5rem;">
                                <input type="text" class="form-input" id="livekit-url" placeholder="wss://your-app.livekit.cloud" style="font-size: 0.85rem;">
                            </div>
                            <div class="form-group" style="margin-bottom: 0.5rem;">
                                <input type="password" class="form-input" id="livekit-api-key" placeholder="API Key" style="font-size: 0.85rem;">
                            </div>
                            <div class="form-group" style="margin-bottom: 0.75rem;">
                                <input type="password" class="form-input" id="livekit-api-secret" placeholder="API Secret" style="font-size: 0.85rem;">
                            </div>
                            <div style="display: flex; gap: 0.5rem;">
                                <button class="btn btn-primary btn-sm" onclick="connectPlatform('livekit')">Connect</button>
                                <button class="btn btn-secondary btn-sm" onclick="testPlatform('livekit')">Test</button>
                            </div>
                        </div>

                        <!-- Vapi -->
                        <div class="platform-card" id="platform-vapi" style="padding: 1rem; background: var(--glass-surface); border-radius: 8px; border: 1px solid var(--glass-border);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                                <strong style="color: var(--neon-green);">Vapi</strong>
                                <span id="status-vapi" class="table-status draft">Disconnected</span>
                            </div>
                            <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.75rem;">Voice AI platform for building phone agents</p>
                            <div class="form-group" style="margin-bottom: 0.5rem;">
                                <input type="password" class="form-input" id="vapi-api-key" placeholder="Vapi API Key" style="font-size: 0.85rem;">
                            </div>
                            <div class="form-group" style="margin-bottom: 0.75rem;">
                                <input type="text" class="form-input" id="vapi-assistant-id" placeholder="Assistant ID (optional)" style="font-size: 0.85rem;">
                            </div>
                            <div style="display: flex; gap: 0.5rem;">
                                <button class="btn btn-primary btn-sm" onclick="connectPlatform('vapi')">Connect</button>
                                <button class="btn btn-secondary btn-sm" onclick="testPlatform('vapi')">Test</button>
                            </div>
                        </div>

                        <!-- Twilio -->
                        <div class="platform-card" id="platform-twilio" style="padding: 1rem; background: var(--glass-surface); border-radius: 8px; border: 1px solid var(--glass-border);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                                <strong style="color: var(--neon-purple);">Twilio</strong>
                                <span id="status-twilio" class="table-status draft">Disconnected</span>
                            </div>
                            <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.75rem;">Cloud communications for voice calls</p>
                            <div class="form-group" style="margin-bottom: 0.5rem;">
                                <input type="text" class="form-input" id="twilio-account-sid" placeholder="Account SID" style="font-size: 0.85rem;">
                            </div>
                            <div class="form-group" style="margin-bottom: 0.5rem;">
                                <input type="password" class="form-input" id="twilio-auth-token" placeholder="Auth Token" style="font-size: 0.85rem;">
                            </div>
                            <div class="form-group" style="margin-bottom: 0.75rem;">
                                <input type="text" class="form-input" id="twilio-phone" placeholder="+1234567890" style="font-size: 0.85rem;">
                            </div>
                            <div style="display: flex; gap: 0.5rem;">
                                <button class="btn btn-primary btn-sm" onclick="connectPlatform('twilio')">Connect</button>
                                <button class="btn btn-secondary btn-sm" onclick="testPlatform('twilio')">Test</button>
                            </div>
                        </div>

                        <!-- Retell AI -->
                        <div class="platform-card" id="platform-retell" style="padding: 1rem; background: var(--glass-surface); border-radius: 8px; border: 1px solid var(--glass-border);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                                <strong style="color: var(--neon-orange);">Retell AI</strong>
                                <span id="status-retell" class="table-status draft">Disconnected</span>
                            </div>
                            <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.75rem;">Conversational voice AI for customer interactions</p>
                            <div class="form-group" style="margin-bottom: 0.5rem;">
                                <input type="password" class="form-input" id="retell-api-key" placeholder="Retell API Key" style="font-size: 0.85rem;">
                            </div>
                            <div class="form-group" style="margin-bottom: 0.75rem;">
                                <input type="text" class="form-input" id="retell-agent-id" placeholder="Agent ID" style="font-size: 0.85rem;">
                            </div>
                            <div style="display: flex; gap: 0.5rem;">
                                <button class="btn btn-primary btn-sm" onclick="connectPlatform('retell')">Connect</button>
                                <button class="btn btn-secondary btn-sm" onclick="testPlatform('retell')">Test</button>
                            </div>
                        </div>

                        <!-- Bland AI -->
                        <div class="platform-card" id="platform-bland" style="padding: 1rem; background: var(--glass-surface); border-radius: 8px; border: 1px solid var(--glass-border);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                                <strong style="color: var(--neon-pink);">Bland AI</strong>
                                <span id="status-bland" class="table-status draft">Disconnected</span>
                            </div>
                            <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.75rem;">AI phone agents for enterprises</p>
                            <div class="form-group" style="margin-bottom: 0.5rem;">
                                <input type="password" class="form-input" id="bland-api-key" placeholder="Bland API Key" style="font-size: 0.85rem;">
                            </div>
                            <div class="form-group" style="margin-bottom: 0.75rem;">
                                <input type="text" class="form-input" id="bland-pathway-id" placeholder="Pathway ID" style="font-size: 0.85rem;">
                            </div>
                            <div style="display: flex; gap: 0.5rem;">
                                <button class="btn btn-primary btn-sm" onclick="connectPlatform('bland')">Connect</button>
                                <button class="btn btn-secondary btn-sm" onclick="testPlatform('bland')">Test</button>
                            </div>
                        </div>

                        <!-- Daily.co / Pipecat -->
                        <div class="platform-card" id="platform-daily" style="padding: 1rem; background: var(--glass-surface); border-radius: 8px; border: 1px solid var(--glass-border);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                                <strong style="color: var(--neon-blue);">Daily.co</strong>
                                <span id="status-daily" class="table-status draft">Disconnected</span>
                            </div>
                            <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.75rem;">Real-time video/audio with Pipecat integration</p>
                            <div class="form-group" style="margin-bottom: 0.5rem;">
                                <input type="password" class="form-input" id="daily-api-key" placeholder="Daily API Key" style="font-size: 0.85rem;">
                            </div>
                            <div class="form-group" style="margin-bottom: 0.75rem;">
                                <input type="text" class="form-input" id="daily-room-url" placeholder="Room URL (optional)" style="font-size: 0.85rem;">
                            </div>
                            <div style="display: flex; gap: 0.5rem;">
                                <button class="btn btn-primary btn-sm" onclick="connectPlatform('daily')">Connect</button>
                                <button class="btn btn-secondary btn-sm" onclick="testPlatform('daily')">Test</button>
                            </div>
                        </div>

                        <!-- Vocode -->
                        <div class="platform-card" id="platform-vocode" style="padding: 1rem; background: var(--glass-surface); border-radius: 8px; border: 1px solid var(--glass-border);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                                <strong style="color: var(--neon-yellow);">Vocode</strong>
                                <span id="status-vocode" class="table-status draft">Disconnected</span>
                            </div>
                            <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.75rem;">Open-source voice agent framework</p>
                            <div class="form-group" style="margin-bottom: 0.75rem;">
                                <input type="password" class="form-input" id="vocode-api-key" placeholder="Vocode API Key" style="font-size: 0.85rem;">
                            </div>
                            <div style="display: flex; gap: 0.5rem;">
                                <button class="btn btn-primary btn-sm" onclick="connectPlatform('vocode')">Connect</button>
                                <button class="btn btn-secondary btn-sm" onclick="testPlatform('vocode')">Test</button>
                            </div>
                        </div>

                        <!-- Custom WebSocket -->
                        <div class="platform-card" id="platform-websocket" style="padding: 1rem; background: var(--glass-surface); border-radius: 8px; border: 1px solid var(--glass-border);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                                <strong style="color: var(--text-primary);">Custom WebSocket</strong>
                                <span id="status-websocket" class="table-status draft">Disconnected</span>
                            </div>
                            <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.75rem;">Connect to any WebSocket-based service</p>
                            <div class="form-group" style="margin-bottom: 0.5rem;">
                                <input type="text" class="form-input" id="websocket-url" placeholder="wss://your-server.com/ws" style="font-size: 0.85rem;">
                            </div>
                            <div class="form-group" style="margin-bottom: 0.75rem;">
                                <input type="password" class="form-input" id="websocket-auth" placeholder="Auth Header (optional)" style="font-size: 0.85rem;">
                            </div>
                            <div style="display: flex; gap: 0.5rem;">
                                <button class="btn btn-primary btn-sm" onclick="connectPlatform('websocket')">Connect</button>
                                <button class="btn btn-secondary btn-sm" onclick="testPlatform('websocket')">Test</button>
                            </div>
                        </div>
                    </div>
                    <div id="platform-message" style="margin-top: 1rem;"></div>
                </div>

                <!-- Integration Code Examples -->
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Code</span> Integration Examples</div>
                    </div>
                    <h4 style="color: var(--neon-cyan); margin-bottom: 0.5rem;">Basic Integration</h4>
                    <pre class="code-block">from skill_command_center import SkillCommandCenter

center = SkillCommandCenter()

# In your voice handler:
async def on_user_speech(text: str):
    async for chunk in center.process_query(text, use_latency_masking=True):
        await tts.speak(chunk)</pre>
                    <h4 style="color: var(--neon-cyan); margin: 1rem 0 0.5rem;">LiveKit Agents Integration</h4>
                    <pre class="code-block">from livekit.agents import llm, Agent
from livekit.plugins import silero

class SkillCenterLLM(llm.LLM):
    def __init__(self):
        self.center = SkillCommandCenter()

    async def chat(self, messages):
        prompt = messages[-1].content
        async for chunk in self.center.process_query(prompt):
            yield llm.ChatChunk(content=chunk)

# Use with LiveKit Agent
agent = Agent(
    vad=silero.VAD(),
    llm=SkillCenterLLM(),
    tts=your_tts_provider
)</pre>
                    <h4 style="color: var(--neon-green); margin: 1rem 0 0.5rem;">Vapi Custom LLM</h4>
                    <pre class="code-block"># Set your server URL in Vapi dashboard as Custom LLM endpoint
from flask import Flask, request, jsonify

@app.route('/vapi/chat', methods=['POST'])
async def vapi_handler():
    data = request.json
    message = data['messages'][-1]['content']

    response = ""
    async for chunk in center.process_query(message):
        response += chunk

    return jsonify({
        "message": {"role": "assistant", "content": response}
    })</pre>
                    <h4 style="color: var(--neon-purple); margin: 1rem 0 0.5rem;">Daily.co Pipecat Integration</h4>
                    <pre class="code-block">from pipecat.pipeline import Pipeline
from pipecat.transports.services.daily import DailyTransport

class SkillCenterProcessor:
    async def process(self, text):
        async for chunk in center.process_query(text):
            yield chunk

pipeline = Pipeline([
    DailyTransport(room_url, token),
    your_stt,
    SkillCenterProcessor(),
    your_tts
])</pre>
                </div>
            </div>
        </div>

        <!-- ================================================================ -->
        <!-- LPU INFERENCE TAB -->
        <!-- ================================================================ -->
        <div id="tab-lpu" class="tab-content">
            <div class="glass-card">
                <div class="section-header">
                    <div class="section-title"><span class="section-icon">LPU</span> BitNet LPU Console</div>
                    <button class="btn btn-secondary btn-sm" onclick="clearLpuConsole()">Clear</button>
                </div>
                <div class="form-group">
                    <label class="form-label">Prompt</label>
                    <textarea class="form-textarea" id="lpu-prompt" placeholder="User: What are your hours?&#10;Assistant:"></textarea>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Max Tokens</label>
                        <input type="number" class="form-input" id="lpu-tokens" value="128" min="16" max="512">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Skill Adapter</label>
                        <select class="form-select" id="lpu-skill">
                            <option value="">None (Base Model)</option>
                        </select>
                    </div>
                </div>
                <button class="btn btn-primary" onclick="runLpuInference()" id="lpu-btn">Run Inference</button>
                <div style="margin-top: 1rem;">
                    <label class="form-label">Output</label>
                    <div class="console" id="lpu-console">
<span class="info">// BitNet LPU Console Ready
// Model: Llama3-8B-1.58-100B-tokens (3.58 GiB)
// Quantization: I2_S - 2 bpw ternary
// Speed: ~8 tokens/second on CPU

</span>Waiting for input...</div>
                </div>
            </div>
        </div>

        <!-- ================================================================ -->
        <!-- FAST BRAIN TAB -->
        <!-- ================================================================ -->
        <div id="tab-fastbrain" class="tab-content">
            <!-- Sub-tabs for Fast Brain -->
            <div class="sub-tabs">
                <button class="sub-tab-btn active" onclick="showFastBrainTab('dashboard')">Dashboard</button>
                <button class="sub-tab-btn" onclick="showFastBrainTab('skills')">Skills Manager</button>
                <button class="sub-tab-btn" onclick="showFastBrainTab('chat')">Test Chat</button>
                <button class="sub-tab-btn" onclick="showFastBrainTab('integration')">Hive215 Integration</button>
            </div>

            <!-- Dashboard Sub-tab -->
            <div id="fb-tab-dashboard" class="sub-tab-content active">
                <div class="dashboard-grid">
                    <!-- System Status Panel -->
                    <div class="glass-card card-full" style="background: linear-gradient(135deg, rgba(0,255,136,0.1) 0%, rgba(0,212,255,0.1) 100%);">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">SYS</span> System Status</div>
                            <button class="btn btn-sm btn-secondary" onclick="refreshSystemStatus()">Refresh</button>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                            <!-- Fast Brain Status -->
                            <div class="status-card" id="status-fast-brain" style="padding: 1rem; background: var(--glass-surface); border-radius: 8px; border-left: 4px solid var(--text-secondary);">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <strong>Fast Brain LPU</strong>
                                    <span class="status-indicator" id="fb-status-indicator">--</span>
                                </div>
                                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.5rem;">
                                    <div>Latency: <span id="fb-status-latency">--</span></div>
                                    <div>Model: <span id="fb-status-model">--</span></div>
                                    <div>Skills: <span id="fb-status-skills">--</span></div>
                                </div>
                            </div>
                            <!-- Groq Status -->
                            <div class="status-card" id="status-groq" style="padding: 1rem; background: var(--glass-surface); border-radius: 8px; border-left: 4px solid var(--text-secondary);">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <strong>Groq API</strong>
                                    <span class="status-indicator" id="groq-status-indicator">--</span>
                                </div>
                                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.5rem;">
                                    Backend for Fast Brain inference
                                </div>
                            </div>
                            <!-- Hive215 Status -->
                            <div class="status-card" id="status-hive215" style="padding: 1rem; background: var(--glass-surface); border-radius: 8px; border-left: 4px solid var(--text-secondary);">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <strong>Hive215</strong>
                                    <span class="status-indicator" id="hive-status-indicator">--</span>
                                </div>
                                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.5rem;">
                                    Voice assistant platform
                                </div>
                            </div>
                            <!-- Modal Status -->
                            <div class="status-card" id="status-modal" style="padding: 1rem; background: var(--glass-surface); border-radius: 8px; border-left: 4px solid var(--text-secondary);">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <strong>Modal</strong>
                                    <span class="status-indicator" id="modal-status-indicator">--</span>
                                </div>
                                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.5rem;">
                                    Serverless deployment
                                </div>
                            </div>
                        </div>
                        <div id="system-error-display" style="margin-top: 1rem; display: none;">
                            <div style="color: var(--neon-orange); font-weight: bold; margin-bottom: 0.5rem;">Recent Errors:</div>
                            <div id="system-errors" class="console" style="max-height: 150px; overflow-y: auto;"></div>
                        </div>
                    </div>

                    <!-- Status Cards -->
                    <div class="glass-card stat-card card-quarter">
                        <div class="stat-value" id="fb-status">Offline</div>
                        <div class="stat-label">Status</div>
                    </div>
                    <div class="glass-card stat-card card-quarter">
                        <div class="stat-value green" id="fb-ttfb">--ms</div>
                        <div class="stat-label">Avg TTFB</div>
                    </div>
                    <div class="glass-card stat-card card-quarter">
                        <div class="stat-value pink" id="fb-throughput">--</div>
                        <div class="stat-label">Tokens/sec</div>
                    </div>
                    <div class="glass-card stat-card card-quarter">
                        <div class="stat-value orange" id="fb-requests">0</div>
                        <div class="stat-label">Total Requests</div>
                    </div>

                    <!-- Configuration -->
                    <div class="glass-card card-half">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">CFG</span> Fast Brain Configuration</div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Fast Brain URL</label>
                            <input type="text" class="form-input" id="fb-url" placeholder="https://your-username--fast-brain-lpu.modal.run">
                        </div>
                        <div class="form-row">
                            <div class="form-group">
                                <label class="form-label">Min Containers</label>
                                <input type="number" class="form-input" id="fb-min-containers" value="1" min="0" max="10">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Max Containers</label>
                                <input type="number" class="form-input" id="fb-max-containers" value="10" min="1" max="100">
                            </div>
                        </div>
                        <div class="form-row">
                            <button class="btn btn-primary" onclick="saveFastBrainConfig()">Save Config</button>
                            <button class="btn btn-secondary" onclick="checkFastBrainHealth()">Check Health</button>
                        </div>
                        <div id="fb-config-message" style="margin-top: 1rem;"></div>
                    </div>

                    <!-- Performance Targets -->
                    <div class="glass-card card-half">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">TGT</span> Performance Targets</div>
                        </div>
                        <table class="skills-table">
                            <tr>
                                <td>Time to First Byte (TTFB)</td>
                                <td><span id="fb-ttfb-target" class="table-status draft">&lt;100ms</span></td>
                            </tr>
                            <tr>
                                <td>Throughput</td>
                                <td><span id="fb-throughput-target" class="table-status draft">&gt;200 tok/s</span></td>
                            </tr>
                            <tr>
                                <td>Cold Start</td>
                                <td><span class="table-status deployed">&lt;5s (warm)</span></td>
                            </tr>
                            <tr>
                                <td>Backend</td>
                                <td>Groq Llama 3.3 70B</td>
                            </tr>
                            <tr>
                                <td>Skills</td>
                                <td><span id="fb-skills-count">5</span> available</td>
                            </tr>
                        </table>
                        <button class="btn btn-success" onclick="runFastBrainBenchmark()" style="margin-top: 1rem;">Run Benchmark</button>
                    </div>

                    <!-- Benchmark Results -->
                    <div class="glass-card card-full" id="fb-benchmark-results" style="display: none;">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">BNK</span> Benchmark Results</div>
                        </div>
                        <div id="fb-benchmark-output"></div>
                    </div>
                </div>
            </div>

            <!-- Skills Manager Sub-tab -->
            <div id="fb-tab-skills" class="sub-tab-content">
                <div class="dashboard-grid">
                    <!-- Skills List -->
                    <div class="glass-card card-full">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">SKL</span> Available Skills</div>
                            <button class="btn btn-primary btn-sm" onclick="showCreateSkillModal()">+ Create Skill</button>
                        </div>
                        <div id="fb-skills-list" class="skills-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem;">
                            <!-- Skills will be loaded here -->
                            <div style="color: var(--text-secondary); padding: 2rem; text-align: center;">
                                Loading skills...
                            </div>
                        </div>
                    </div>

                    <!-- Create Skill Form (Modal-like) -->
                    <div class="glass-card card-full" id="fb-create-skill-form" style="display: none;">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">NEW</span> Create Custom Skill</div>
                            <button class="btn btn-sm" onclick="hideCreateSkillModal()" style="background: transparent; color: var(--text-secondary);">✕</button>
                        </div>
                        <div class="form-row">
                            <div class="form-group">
                                <label class="form-label">Skill ID (lowercase, no spaces)</label>
                                <input type="text" class="form-input" id="new-skill-id" placeholder="my_custom_skill">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Display Name</label>
                                <input type="text" class="form-input" id="new-skill-name" placeholder="My Custom Skill">
                            </div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Description</label>
                            <input type="text" class="form-input" id="new-skill-description" placeholder="Brief description of what this skill does">
                        </div>
                        <div class="form-group">
                            <label class="form-label">System Prompt</label>
                            <textarea class="form-textarea" id="new-skill-prompt" rows="6" placeholder="You are an AI assistant specialized in..."></textarea>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Knowledge Base (one item per line)</label>
                            <textarea class="form-textarea" id="new-skill-knowledge" rows="4" placeholder="Pricing: $100-500&#10;Hours: Mon-Fri 9am-5pm&#10;Service area: Philadelphia metro"></textarea>
                        </div>
                        <div class="form-row">
                            <button class="btn btn-primary" onclick="createCustomSkill()">Create Skill</button>
                            <button class="btn btn-secondary" onclick="hideCreateSkillModal()">Cancel</button>
                        </div>
                        <div id="create-skill-message" style="margin-top: 1rem;"></div>
                    </div>
                </div>
            </div>

            <!-- Test Chat Sub-tab -->
            <div id="fb-tab-chat" class="sub-tab-content">
                <div class="dashboard-grid">
                    <div class="glass-card card-full">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">CHT</span> Test Fast Brain</div>
                        </div>

                        <!-- Skill Selector -->
                        <div class="form-row" style="margin-bottom: 1rem;">
                            <div class="form-group" style="flex: 1;">
                                <label class="form-label">Select Skill</label>
                                <select class="form-select" id="fb-skill-selector" onchange="onSkillSelect()">
                                    <option value="general">General Assistant</option>
                                    <option value="receptionist">Professional Receptionist</option>
                                    <option value="electrician">Electrician Assistant</option>
                                    <option value="plumber">Plumber Assistant</option>
                                    <option value="lawyer">Legal Intake Assistant</option>
                                </select>
                            </div>
                            <div class="form-group" style="flex: 0 0 auto; display: flex; align-items: flex-end;">
                                <button class="btn btn-secondary btn-sm" onclick="loadSkillPrompt()">Load Skill Prompt</button>
                            </div>
                        </div>

                        <!-- Active Skill Indicator -->
                        <div id="fb-active-skill-info" style="background: var(--glass-surface); padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid var(--neon-cyan);">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong style="color: var(--neon-cyan);">Active Skill:</strong>
                                    <span id="fb-active-skill-name" style="margin-left: 0.5rem;">General Assistant</span>
                                </div>
                                <span id="fb-skill-status" class="table-status deployed">Ready</span>
                            </div>
                            <div id="fb-active-skill-desc" style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.25rem;">
                                Helpful general-purpose assistant
                            </div>
                        </div>

                        <div class="form-group">
                            <label class="form-label">System Prompt (or use skill default)</label>
                            <textarea class="form-textarea" id="fb-system-prompt" rows="4" placeholder="Leave empty to use selected skill's default prompt..."></textarea>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Message</label>
                            <textarea class="form-textarea" id="fb-message" placeholder="Enter your message to test Fast Brain..."></textarea>
                        </div>
                        <div class="form-row">
                            <div class="form-group">
                                <label class="form-label">Max Tokens</label>
                                <input type="number" class="form-input" id="fb-max-tokens" value="256" min="16" max="1024">
                            </div>
                        </div>
                        <button class="btn btn-primary" onclick="testFastBrainChat()" id="fb-chat-btn">Send to Fast Brain</button>
                        <div style="margin-top: 1rem;">
                            <label class="form-label">Response</label>
                            <div class="console" id="fb-response" style="min-height: 100px;">Waiting for request...</div>
                        </div>
                        <div id="fb-metrics" style="margin-top: 0.5rem; color: var(--text-secondary); font-size: 0.85rem;"></div>
                    </div>
                </div>
            </div>

            <!-- Hive215 Integration Sub-tab -->
            <div id="fb-tab-integration" class="sub-tab-content">
                <div class="dashboard-grid">
                    <!-- Integration Checklist -->
                    <div class="glass-card card-full">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">CHK</span> Integration Checklist</div>
                            <div>
                                <span id="integration-progress" style="color: var(--neon-cyan); font-weight: bold;">0%</span> Complete
                            </div>
                        </div>
                        <div style="background: var(--glass-surface); border-radius: 8px; padding: 0.5rem; margin-bottom: 1rem;">
                            <div id="integration-progress-bar" style="height: 8px; background: var(--neon-cyan); border-radius: 4px; width: 0%; transition: width 0.3s;"></div>
                        </div>
                        <div id="integration-checklist">
                            <!-- Checklist items will be loaded here -->
                            <div style="color: var(--text-secondary); padding: 1rem; text-align: center;">Loading checklist...</div>
                        </div>
                    </div>

                    <!-- Hive215 Configuration -->
                    <div class="glass-card card-half">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">H215</span> Hive215 Configuration</div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Hive215 Dashboard URL</label>
                            <input type="text" class="form-input" id="hive-url" placeholder="https://hive215.vercel.app">
                        </div>
                        <div class="form-group">
                            <label class="form-label">API Key</label>
                            <input type="password" class="form-input" id="hive-api-key" placeholder="Your Hive215 API key">
                        </div>
                        <div class="form-row">
                            <button class="btn btn-primary" onclick="saveHive215Config()">Save Config</button>
                            <button class="btn btn-secondary" onclick="testHive215Connection()">Test Connection</button>
                        </div>
                        <div id="hive-config-message" style="margin-top: 1rem;"></div>
                    </div>

                    <!-- Quick Setup Guide -->
                    <div class="glass-card card-half">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">DOC</span> Quick Setup Guide</div>
                        </div>
                        <div style="font-size: 0.9rem; line-height: 1.6;">
                            <h4 style="color: var(--neon-cyan); margin-bottom: 0.5rem;">1. Deploy Fast Brain</h4>
                            <pre class="code-block" style="font-size: 0.8rem; margin-bottom: 1rem;">modal deploy fast_brain/deploy_groq.py</pre>

                            <h4 style="color: var(--neon-cyan); margin-bottom: 0.5rem;">2. Set URL in Hive215</h4>
                            <p style="color: var(--text-secondary); margin-bottom: 1rem;">Add FAST_BRAIN_URL to Railway worker environment</p>

                            <h4 style="color: var(--neon-cyan); margin-bottom: 0.5rem;">3. Sync Skills</h4>
                            <p style="color: var(--text-secondary); margin-bottom: 1rem;">Skills created here auto-sync to both Fast Brain and Hive215</p>

                            <h4 style="color: var(--neon-cyan); margin-bottom: 0.5rem;">4. Test Voice Flow</h4>
                            <p style="color: var(--text-secondary);">Call your Twilio number to test the full pipeline</p>
                        </div>
                    </div>

                    <!-- Connection Diagram -->
                    <div class="glass-card card-full">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">MAP</span> System Architecture</div>
                        </div>
                        <pre class="code-block" style="font-size: 0.75rem; overflow-x: auto;">
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              HIVE215 VOICE PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   [Caller] ──► [Twilio] ──► [LiveKit Worker] ──► [Fast Brain LPU] ──► [Response] │
│                                    │                    │                        │
│                                    │              ┌─────┴─────┐                  │
│                              [Deepgram STT]      │   Groq    │                  │
│                                    │             │ Llama 3.3 │                  │
│                              [Cartesia TTS]      │   70B     │                  │
│                                    │             └───────────┘                  │
│                                    ▼                    │                        │
│                           [Audio Response]    [Skills Database]                  │
│                                                         │                        │
│   ┌──────────────────────────────────────────────────────┘                      │
│   │                                                                              │
│   │  Skills: receptionist │ electrician │ plumber │ lawyer │ custom...         │
│   │                                                                              │
└───┴──────────────────────────────────────────────────────────────────────────────┘
                        </pre>
                    </div>
                </div>
            </div>
        </div>

        <!-- ================================================================ -->
        <!-- VOICE LAB TAB -->
        <!-- ================================================================ -->
        <div id="tab-voicelab" class="tab-content">
            <!-- Sub-tabs -->
            <div class="sub-tabs">
                <button class="sub-tab-btn active" onclick="showVoiceLabTab('projects')">Voice Projects</button>
                <button class="sub-tab-btn" onclick="showVoiceLabTab('create')">Create Voice</button>
                <button class="sub-tab-btn" onclick="showVoiceLabTab('training')">Training Queue</button>
                <button class="sub-tab-btn" onclick="showVoiceLabTab('skills')">Skill Training</button>
            </div>

            <!-- Voice Projects List -->
            <div id="voicelab-projects" class="sub-tab-content active">
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Voice</span> Your Voice Projects</div>
                        <button class="btn btn-primary btn-sm" onclick="showVoiceLabTab('create')">+ New Voice</button>
                    </div>
                    <div id="voice-projects-list" class="skills-grid">
                        <div style="color: var(--text-secondary); padding: 2rem; text-align: center;">
                            No voice projects yet. Click "New Voice" to create one.
                        </div>
                    </div>
                </div>
            </div>

            <!-- Create Voice -->
            <div id="voicelab-create" class="sub-tab-content">
                <div class="dashboard-grid">
                    <div class="glass-card card-half">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">New</span> Create Custom Voice</div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Voice Name</label>
                            <input type="text" class="form-input" id="vl-name" placeholder="My Custom Voice">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Description</label>
                            <textarea class="form-textarea" id="vl-description" placeholder="Describe the voice characteristics..."></textarea>
                        </div>
                        <div class="form-row">
                            <div class="form-group">
                                <label class="form-label">Base Voice Provider</label>
                                <select class="form-select" id="vl-provider" onchange="loadVLProviderVoices()">
                                    <optgroup label="Free / Open Source">
                                        <option value="edge_tts">Edge TTS (Microsoft - Free)</option>
                                        <option value="kokoro">Kokoro (Ultra-fast Local)</option>
                                        <option value="chatterbox">Chatterbox</option>
                                        <option value="xtts">XTTS-v2 (Coqui)</option>
                                        <option value="openvoice">OpenVoice</option>
                                    </optgroup>
                                    <optgroup label="Paid Services">
                                        <option value="elevenlabs">ElevenLabs</option>
                                        <option value="cartesia">Cartesia</option>
                                    </optgroup>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Base Voice</label>
                                <select class="form-select" id="vl-base-voice">
                                    <option value="default">Default</option>
                                </select>
                            </div>
                        </div>
                        <button class="btn btn-primary" onclick="createVoiceProject()">Create Project</button>
                        <div id="vl-create-message" style="margin-top: 1rem;"></div>
                    </div>

                    <div class="glass-card card-half">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">Settings</span> Voice Settings</div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Pitch: <span id="vl-pitch-value">1.0</span></label>
                            <input type="range" class="form-input" id="vl-pitch" min="0.5" max="2.0" step="0.1" value="1.0" oninput="document.getElementById('vl-pitch-value').textContent = this.value">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Speed: <span id="vl-speed-value">1.0</span></label>
                            <input type="range" class="form-input" id="vl-speed" min="0.5" max="2.0" step="0.1" value="1.0" oninput="document.getElementById('vl-speed-value').textContent = this.value">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Emotion</label>
                            <select class="form-select" id="vl-emotion">
                                <option value="neutral">Neutral</option>
                                <option value="happy">Happy</option>
                                <option value="sad">Sad</option>
                                <option value="angry">Angry</option>
                                <option value="excited">Excited</option>
                                <option value="calm">Calm</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Speaking Style</label>
                            <select class="form-select" id="vl-style">
                                <option value="conversational">Conversational</option>
                                <option value="professional">Professional</option>
                                <option value="friendly">Friendly</option>
                                <option value="formal">Formal</option>
                                <option value="casual">Casual</option>
                            </select>
                        </div>
                    </div>

                    <div class="glass-card card-full">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">Audio</span> Upload Training Samples</div>
                        </div>
                        <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                            Upload 3+ audio samples (5-30 seconds each) to train your custom voice. Clearer recordings produce better results.
                        </p>
                        <div class="file-upload" onclick="document.getElementById('vl-audio-input').click()">
                            <div class="file-upload-icon">Upload Audio</div>
                            <div class="file-upload-text">Click to upload audio files (WAV, MP3, M4A)</div>
                            <input type="file" id="vl-audio-input" accept="audio/*" multiple style="display: none;">
                        </div>
                        <div id="vl-samples-list" style="margin-top: 1rem;"></div>
                    </div>
                </div>
            </div>

            <!-- Training Queue -->
            <div id="voicelab-training" class="sub-tab-content">
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Train</span> Voice Training Queue</div>
                        <button class="btn btn-secondary btn-sm" onclick="refreshVoiceTrainingStatus()">Refresh</button>
                    </div>
                    <div id="voice-training-queue">
                        <div style="color: var(--text-secondary); padding: 2rem; text-align: center;">
                            No voice training jobs in progress.
                        </div>
                    </div>
                </div>
            </div>

            <!-- Skill Training -->
            <div id="voicelab-skills" class="sub-tab-content">
                <div class="dashboard-grid">
                    <div class="glass-card card-half">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">Tune</span> Fine-tune Skills</div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Select Skill</label>
                            <select class="form-select" id="ft-skill-select">
                                <option value="">Select a skill...</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Training Examples (JSONL format)</label>
                            <textarea class="form-textarea" id="ft-examples" placeholder='{"instruction": "...", "input": "...", "output": "..."}'></textarea>
                        </div>
                        <button class="btn btn-primary" onclick="startSkillFineTune()">Start Fine-tuning</button>
                        <div id="ft-message" style="margin-top: 1rem;"></div>
                    </div>

                    <div class="glass-card card-half">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">Feedback</span> Add Training Feedback</div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Select Skill</label>
                            <select class="form-select" id="feedback-skill-select">
                                <option value="">Select a skill...</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Query</label>
                            <input type="text" class="form-input" id="feedback-query" placeholder="What the user asked...">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Response (from skill)</label>
                            <textarea class="form-textarea" id="feedback-response" placeholder="What the skill responded..."></textarea>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Rating</label>
                            <select class="form-select" id="feedback-rating">
                                <option value="1">1 - Very Poor</option>
                                <option value="2">2 - Poor</option>
                                <option value="3">3 - Acceptable</option>
                                <option value="4">4 - Good</option>
                                <option value="5">5 - Excellent</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Corrected Response (optional)</label>
                            <textarea class="form-textarea" id="feedback-corrected" placeholder="What the response should have been..."></textarea>
                        </div>
                        <button class="btn btn-success" onclick="submitSkillFeedback()">Submit Feedback</button>
                    </div>

                    <div class="glass-card card-full">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">Auto</span> Auto-Improvement Status</div>
                        </div>
                        <table class="skills-table">
                            <thead>
                                <tr>
                                    <th>Skill</th>
                                    <th>Feedback Count</th>
                                    <th>Last Trained</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody id="skill-improvement-table">
                                <tr>
                                    <td colspan="5" style="text-align: center; color: var(--text-secondary);">
                                        Select a skill and add feedback to enable auto-improvement
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // ============================================================
        // TAB NAVIGATION
        // ============================================================
        function showMainTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.main-tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('tab-' + tabId).classList.add('active');
            event.target.classList.add('active');

            if (tabId === 'dashboard') { loadSkills(); loadMetrics(); }
            if (tabId === 'fastbrain') { loadFastBrainConfig(); loadFastBrainSkills(); refreshSystemStatus(); loadHive215Config(); }
            if (tabId === 'voicelab') { loadVoiceProjects(); loadSkillsForDropdowns(); }
            if (tabId === 'factory') { loadProfileDropdowns(); }
            if (tabId === 'command') { refreshStats(); }
        }

        function showFactoryTab(tabId) {
            document.querySelectorAll('#tab-factory .sub-tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('#tab-factory .sub-tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('factory-' + tabId).classList.add('active');
            event.target.classList.add('active');

            if (tabId === 'manage') loadSkillsTable();
        }

        function showCommandTab(tabId) {
            document.querySelectorAll('#tab-command .sub-tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('#tab-command .sub-tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('command-' + tabId).classList.add('active');
            event.target.classList.add('active');
        }

        // ============================================================
        // DASHBOARD
        // ============================================================
        async function loadSkills() {
            try {
                const res = await fetch('/api/skills');
                const skills = await res.json();
                const grid = document.getElementById('skills-grid');
                grid.innerHTML = '';

                let totalReq = 0, totalLat = 0, totalSat = 0;

                if (skills.length === 0) {
                    grid.innerHTML = '<p style="color: var(--text-secondary); text-align: center; grid-column: span 4;">No skills yet. Create one in Skill Factory!</p>';
                }

                skills.forEach(s => {
                    totalReq += s.requests_today;
                    totalLat += s.avg_latency_ms;
                    totalSat += s.satisfaction_rate;

                    grid.innerHTML += `
                        <div class="skill-card ${s.status}">
                            <div class="skill-header">
                                <div>
                                    <div class="skill-name">${s.name}</div>
                                    <div class="skill-type">${s.type}</div>
                                </div>
                                <span class="skill-status ${s.status}">${s.status}</span>
                            </div>
                            <div class="skill-metrics">
                                <div class="skill-metric"><div class="metric-value">${s.requests_today}</div><div class="metric-label">Requests</div></div>
                                <div class="skill-metric"><div class="metric-value">${s.avg_latency_ms}ms</div><div class="metric-label">Latency</div></div>
                                <div class="skill-metric"><div class="metric-value">${s.satisfaction_rate}%</div><div class="metric-label">Satisfaction</div></div>
                            </div>
                        </div>
                    `;
                });

                document.getElementById('total-skills').textContent = skills.length || 0;
                document.getElementById('requests-today').textContent = totalReq.toLocaleString();
                document.getElementById('avg-latency').textContent = skills.length ? Math.round(totalLat / skills.length) + 'ms' : '0ms';
                document.getElementById('satisfaction').textContent = skills.length ? Math.round(totalSat / skills.length) + '%' : '0%';
            } catch (e) {
                console.error(e);
            }
        }

        async function refreshServerStatus() {
            try {
                const res = await fetch('/api/server-status');
                const s = await res.json();
                document.getElementById('server-status-text').textContent = `LPU ${s.status === 'online' ? 'Online' : 'Offline'} - ${s.warm_containers} Warm Container`;
                document.getElementById('warm-containers').textContent = s.warm_containers;
                document.getElementById('memory-usage').textContent = (s.memory_usage_mb / 1000).toFixed(1) + 'GB';
                document.getElementById('cost-today').textContent = '$' + s.cost_today_usd.toFixed(2);
            } catch (e) { console.error(e); }
        }

        async function loadMetrics() {
            try {
                const res = await fetch('/api/metrics');
                const metrics = await res.json();

                const reqChart = document.getElementById('request-chart');
                const latChart = document.getElementById('latency-chart');
                reqChart.innerHTML = '';
                latChart.innerHTML = '';

                const maxReq = Math.max(...metrics.map(m => m.requests));
                const maxLat = Math.max(...metrics.map(m => m.latency_p99));

                metrics.forEach(m => {
                    reqChart.innerHTML += `<div class="chart-bar" style="height: ${m.requests / maxReq * 100}%" title="${m.requests} requests"></div>`;
                    latChart.innerHTML += `<div class="chart-bar" style="height: ${m.latency_p50 / maxLat * 100}%; background: linear-gradient(180deg, var(--neon-green), rgba(0,255,136,0.3));" title="${m.latency_p50}ms"></div>`;
                });
            } catch (e) { console.error(e); }
        }

        async function loadActivity() {
            try {
                const res = await fetch('/api/activity');
                const activities = await res.json();
                const feed = document.getElementById('activity-feed');

                if (activities.length === 0) {
                    feed.innerHTML = '<p style="color: var(--text-secondary); text-align: center;">No recent activity</p>';
                    return;
                }

                feed.innerHTML = activities.map(a => `
                    <div class="activity-item">
                        <span class="activity-icon">${a.icon || 'i'}</span>
                        <div class="activity-content">
                            <div class="activity-message">${a.message}</div>
                            <div class="activity-time">${a.ago}</div>
                        </div>
                    </div>
                `).join('');
            } catch (e) { console.error(e); }
        }

        // ============================================================
        // SKILL FACTORY
        // ============================================================
        async function loadProfileDropdowns() {
            try {
                const res = await fetch('/api/skills');
                const skills = await res.json();
                const options = skills.map(s => `<option value="${s.id}">${s.name}</option>`).join('');
                document.getElementById('doc-profile').innerHTML = options || '<option value="">No profiles yet</option>';
                document.getElementById('train-profile').innerHTML = options || '<option value="">No profiles yet</option>';
            } catch (e) { console.error(e); }
        }

        async function saveProfile(e) {
            e.preventDefault();
            const data = {
                name: document.getElementById('p-name').value,
                type: document.getElementById('p-type').value,
                description: document.getElementById('p-description').value,
                greeting: document.getElementById('p-greeting').value,
                personality: document.getElementById('p-personality').value,
                services: document.getElementById('p-services').value,
                customInstructions: document.getElementById('p-instructions').value
            };

            try {
                const res = await fetch('/api/create-skill', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await res.json();
                if (result.success) {
                    document.getElementById('profile-message').innerHTML = '<div class="message success">Profile saved successfully!</div>';
                    document.getElementById('profile-form').reset();
                    loadProfileDropdowns();
                }
            } catch (e) {
                document.getElementById('profile-message').innerHTML = '<div class="message error">Error: ' + e.message + '</div>';
            }
        }

        async function generateFromProfile() {
            const name = document.getElementById('p-name').value;
            if (!name) {
                alert('Please enter a business name first');
                return;
            }
            const skillId = name.toLowerCase().replace(/[^\\w-]/g, '_');
            try {
                const res = await fetch(`/api/generate-training/${skillId}`, { method: 'POST' });
                const result = await res.json();
                if (result.success) {
                    document.getElementById('profile-message').innerHTML = `<div class="message success">Generated ${result.examples} training examples!</div>`;
                }
            } catch (e) {
                document.getElementById('profile-message').innerHTML = '<div class="message error">Error: ' + e.message + '</div>';
            }
        }

        async function handleFileUpload(input) {
            const file = input.files[0];
            if (!file) return;

            const skillId = document.getElementById('doc-profile').value;
            if (!skillId) {
                alert('Please select a profile first');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);
            formData.append('skill_id', skillId);

            try {
                const res = await fetch('/api/upload-document', {
                    method: 'POST',
                    body: formData
                });
                const result = await res.json();
                if (result.success) {
                    document.getElementById('upload-message').innerHTML = `<div class="message success">Processed ${result.examples} training examples!</div>`;
                } else {
                    document.getElementById('upload-message').innerHTML = `<div class="message error">${result.error}</div>`;
                }
            } catch (e) {
                document.getElementById('upload-message').innerHTML = '<div class="message error">Error: ' + e.message + '</div>';
            }
        }

        function generateTrainingScript() {
            const skillId = document.getElementById('train-profile').value;
            const steps = document.getElementById('train-steps').value;

            const script = `# Training script for ${skillId}
# Run on a GPU machine (Colab, Modal, etc.)

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3-8b-Instructbnb-4bit",
    max_seq_length=2048, load_in_4bit=True
)

model = FastLanguageModel.get_peft_model(model, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, use_gradient_checkpointing="unsloth")

dataset = load_dataset("json",
    data_files="training_data/${skillId}_merged.jsonl",
    split="train")

trainer = SFTTrainer(model, tokenizer, train_dataset=dataset,
    dataset_text_field="output", max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=${steps},
        learning_rate=2e-4,
        output_dir="outputs"))

trainer.train()
model.save_pretrained("adapters/${skillId}")
print("Training complete: adapters/${skillId}")`;

            document.getElementById('training-script').textContent = script;
            document.getElementById('training-output').style.display = 'block';
        }

        // ============================================================
        // SKILLS TABLE
        // ============================================================
        let allSkillsData = [];

        async function loadSkillsTable() {
            try {
                const res = await fetch('/api/skills-table');
                allSkillsData = await res.json();
                renderSkillsTable(allSkillsData);
            } catch (e) { console.error(e); }
        }

        function renderSkillsTable(skills) {
            const tbody = document.getElementById('skills-table-body');
            if (skills.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-secondary);">No skills found</td></tr>';
                return;
            }
            tbody.innerHTML = skills.map(s => `
                <tr>
                    <td>${s.business}</td>
                    <td>${s.type}</td>
                    <td><span class="table-status ${s.status.toLowerCase()}">${s.status}</span></td>
                    <td>${s.examples}</td>
                    <td>${s.created}</td>
                    <td>
                        <button class="btn btn-secondary btn-sm" onclick="editSkill('${s.id}')">Edit</button>
                        <button class="btn btn-danger btn-sm" onclick="deleteSkill('${s.id}')">Delete</button>
                    </td>
                </tr>
            `).join('');
        }

        function filterSkillsTable() {
            const search = document.getElementById('table-search').value.toLowerCase();
            const typeFilter = document.getElementById('table-type-filter').value;
            const statusFilter = document.getElementById('table-status-filter').value;

            const filtered = allSkillsData.filter(s => {
                const matchSearch = s.business.toLowerCase().includes(search) || s.type.toLowerCase().includes(search);
                const matchType = !typeFilter || s.type.includes(typeFilter);
                const matchStatus = !statusFilter || s.status === statusFilter;
                return matchSearch && matchType && matchStatus;
            });

            renderSkillsTable(filtered);
        }

        function editSkill(id) {
            alert('Edit skill: ' + id);
        }

        async function deleteSkill(id) {
            if (!confirm('Delete this skill?')) return;
            try {
                await fetch(`/api/delete-skill/${id}`, { method: 'DELETE' });
                loadSkillsTable();
            } catch (e) { console.error(e); }
        }

        // ============================================================
        // COMMAND CENTER
        // ============================================================
        async function saveApiKeys() {
            const data = {
                groq: document.getElementById('key-groq').value,
                openai: document.getElementById('key-openai').value,
                anthropic: document.getElementById('key-anthropic').value
            };

            try {
                const res = await fetch('/api/save-api-keys', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await res.json();
                if (result.saved && result.saved.length > 0) {
                    document.getElementById('keys-message').innerHTML = `<div class="message success">Saved: ${result.saved.join(', ')}</div>`;
                } else {
                    document.getElementById('keys-message').innerHTML = '<div class="message info">No keys provided. Using mock clients.</div>';
                }
            } catch (e) {
                document.getElementById('keys-message').innerHTML = '<div class="message error">Error: ' + e.message + '</div>';
            }
        }

        async function testLLM() {
            const btn = document.getElementById('test-btn');
            const output = document.getElementById('test-output');
            const prompt = document.getElementById('test-prompt').value;
            const provider = document.getElementById('test-provider').value;

            if (!prompt) {
                alert('Please enter a prompt');
                return;
            }

            btn.innerHTML = '<span class="spinner"></span> Testing...';
            btn.disabled = true;

            try {
                const res = await fetch('/api/test-llm', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, provider })
                });
                const result = await res.json();
                output.innerHTML = `<span class="response">${result.response}</span>\\n\\n<span class="info">Latency: ${result.latency_ms}ms</span>`;
            } catch (e) {
                output.innerHTML = `<span class="error">Error: ${e.message}</span>`;
            }

            btn.innerHTML = 'Run Test';
            btn.disabled = false;
        }

        async function compareLLMs() {
            const btn = document.getElementById('compare-btn');
            const output = document.getElementById('compare-output');
            const prompt = document.getElementById('compare-prompt').value;
            const providers = Array.from(document.querySelectorAll('.compare-provider:checked')).map(c => c.value);

            if (!prompt) {
                alert('Please enter a prompt');
                return;
            }

            if (providers.length === 0) {
                alert('Please select at least one provider');
                return;
            }

            btn.innerHTML = '<span class="spinner"></span> Comparing...';
            btn.disabled = true;

            try {
                const res = await fetch('/api/compare-llms', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, providers })
                });
                const result = await res.json();

                output.innerHTML = result.results.map(r => `
                    <div class="compare-card">
                        <div class="compare-header">
                            <span class="compare-provider">${r.provider.toUpperCase()}</span>
                            <span class="compare-latency">First: ${r.first_token_ms}ms | Total: ${r.total_ms}ms</span>
                        </div>
                        <div>${r.response}</div>
                    </div>
                `).join('');
            } catch (e) {
                output.innerHTML = `<div class="message error">Error: ${e.message}</div>`;
            }

            btn.innerHTML = 'Compare';
            btn.disabled = false;
        }

        async function refreshStats() {
            try {
                const res = await fetch('/api/stats');
                const stats = await res.json();
                document.getElementById('stats-output').textContent = JSON.stringify(stats, null, 2);
            } catch (e) { console.error(e); }
        }

        // ============================================================
        // LPU INFERENCE
        // ============================================================
        async function runLpuInference() {
            const btn = document.getElementById('lpu-btn');
            const consoleEl = document.getElementById('lpu-console');
            const prompt = document.getElementById('lpu-prompt').value;
            const maxTokens = parseInt(document.getElementById('lpu-tokens').value);

            if (!prompt) {
                alert('Please enter a prompt');
                return;
            }

            btn.innerHTML = '<span class="spinner"></span> Running...';
            btn.disabled = true;

            consoleEl.innerHTML += `\\n<span class="prompt">&gt; ${prompt}</span>\\n`;

            try {
                const res = await fetch('/api/lpu-inference', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, maxTokens })
                });
                const result = await res.json();

                if (result.success) {
                    consoleEl.innerHTML += `<span class="response">${result.response}</span>\\n`;
                } else {
                    consoleEl.innerHTML += `<span class="error">Error: ${result.error}</span>\\n`;
                }
            } catch (e) {
                consoleEl.innerHTML += `<span class="error">Error: ${e.message}</span>\\n`;
            }

            btn.innerHTML = 'Run Inference';
            btn.disabled = false;
            consoleEl.scrollTop = consoleEl.scrollHeight;
        }

        function clearLpuConsole() {
            document.getElementById('lpu-console').innerHTML = '<span class="info">// Console cleared\\n</span>';
        }

        // ============================================================
        // VOICE MANAGEMENT
        // ============================================================
        let voiceProviders = {};

        async function loadVoiceProviders() {
            try {
                const res = await fetch('/api/voice/providers');
                voiceProviders = await res.json();
            } catch (e) {
                console.error('Failed to load voice providers:', e);
            }
        }

        function loadProviderVoices() {
            const provider = document.getElementById('voice-provider').value;
            const voiceSelect = document.getElementById('voice-select');

            voiceSelect.innerHTML = '<option value="">Select a voice...</option>';
            document.getElementById('voice-details').style.display = 'none';

            if (!provider || !voiceProviders[provider]) return;

            const providerData = voiceProviders[provider];
            providerData.voices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice.id;
                option.textContent = `${voice.name} (${voice.gender}, ${voice.style})`;
                voiceSelect.appendChild(option);
            });
        }

        function selectVoice() {
            const provider = document.getElementById('voice-provider').value;
            const voiceId = document.getElementById('voice-select').value;

            if (!provider || !voiceId || !voiceProviders[provider]) {
                document.getElementById('voice-details').style.display = 'none';
                return;
            }

            const voice = voiceProviders[provider].voices.find(v => v.id === voiceId);
            if (voice) {
                document.getElementById('voice-details').style.display = 'block';
                document.getElementById('voice-name').textContent = voice.name;
                document.getElementById('voice-meta').textContent = `${voiceProviders[provider].name} | ${voice.gender} | ${voice.style}`;
            }
        }

        async function testVoice() {
            const provider = document.getElementById('voice-provider').value;
            const voiceId = document.getElementById('voice-select').value;
            const text = document.getElementById('voice-test-text').value || 'Hello! This is a test of the voice synthesis system.';
            const resultEl = document.getElementById('voice-test-result');

            if (!voiceId) {
                resultEl.innerHTML = '<div style="color: var(--neon-orange);">Please select a voice first</div>';
                return;
            }

            resultEl.innerHTML = '<div style="color: var(--neon-cyan);">Generating audio with Edge TTS...</div>';

            try {
                const res = await fetch('/api/voice/test', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ voice_id: voiceId, text, provider })
                });
                const result = await res.json();

                if (result.success && result.audio_base64) {
                    // Create audio element and play
                    const audio = new Audio('data:' + result.audio_format + ';base64,' + result.audio_base64);
                    audio.onended = () => {
                        resultEl.innerHTML = `<div style="color: var(--neon-green);">&#9658; Audio played! (${result.duration_ms}ms to generate)</div>`;
                    };
                    audio.onerror = (e) => {
                        resultEl.innerHTML = `<div style="color: var(--neon-orange);">Playback error</div>`;
                    };
                    audio.play();
                    resultEl.innerHTML = `<div style="color: var(--neon-cyan);">&#9658; Playing ${result.provider} audio...</div>`;
                } else if (result.success) {
                    resultEl.innerHTML = `<div style="color: var(--neon-green);">Voice test completed in ${result.duration_ms}ms</div>`;
                } else {
                    resultEl.innerHTML = `<div style="color: var(--neon-orange);">Error: ${result.error || result.message}</div>`;
                }
            } catch (e) {
                resultEl.innerHTML = `<div style="color: var(--neon-orange);">Error: ${e.message}</div>`;
            }
        }

        async function saveVoiceConfig() {
            const provider = document.getElementById('voice-provider').value;
            const voiceId = document.getElementById('voice-select').value;
            const speed = document.getElementById('voice-speed').value;
            const resultEl = document.getElementById('voice-test-result');

            if (!voiceId) {
                resultEl.innerHTML = '<div style="color: var(--neon-orange);">Please select a voice first</div>';
                return;
            }

            try {
                const res = await fetch('/api/voice/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        selected_provider: provider,
                        selected_voice: voiceId,
                        voice_settings: { speed: parseFloat(speed) }
                    })
                });
                const result = await res.json();

                if (result.success) {
                    resultEl.innerHTML = '<div style="color: var(--neon-green);">Voice configuration saved as default!</div>';
                } else {
                    resultEl.innerHTML = '<div style="color: var(--neon-orange);">Failed to save configuration</div>';
                }
            } catch (e) {
                resultEl.innerHTML = `<div style="color: var(--neon-orange);">Error: ${e.message}</div>`;
            }
        }

        // ============================================================
        // PLATFORM CONNECTIONS
        // ============================================================
        async function loadPlatforms() {
            try {
                const res = await fetch('/api/platforms');
                const platforms = await res.json();

                for (const [id, platform] of Object.entries(platforms)) {
                    updatePlatformStatus(id, platform.status);
                }
            } catch (e) {
                console.error('Failed to load platforms:', e);
            }
        }

        function updatePlatformStatus(platformId, status) {
            const statusEl = document.getElementById(`status-${platformId}`);
            if (!statusEl) return;

            statusEl.className = 'table-status';
            switch (status) {
                case 'connected':
                    statusEl.classList.add('deployed');
                    statusEl.textContent = 'Connected';
                    break;
                case 'error':
                    statusEl.classList.add('training');
                    statusEl.textContent = 'Error';
                    break;
                default:
                    statusEl.classList.add('draft');
                    statusEl.textContent = 'Disconnected';
            }
        }

        function getPlatformConfig(platformId) {
            const config = {};
            switch (platformId) {
                case 'livekit':
                    config.url = document.getElementById('livekit-url').value;
                    config.api_key = document.getElementById('livekit-api-key').value;
                    config.api_secret = document.getElementById('livekit-api-secret').value;
                    break;
                case 'vapi':
                    config.api_key = document.getElementById('vapi-api-key').value;
                    config.assistant_id = document.getElementById('vapi-assistant-id').value;
                    break;
                case 'twilio':
                    config.account_sid = document.getElementById('twilio-account-sid').value;
                    config.auth_token = document.getElementById('twilio-auth-token').value;
                    config.phone_number = document.getElementById('twilio-phone').value;
                    break;
                case 'retell':
                    config.api_key = document.getElementById('retell-api-key').value;
                    config.agent_id = document.getElementById('retell-agent-id').value;
                    break;
                case 'bland':
                    config.api_key = document.getElementById('bland-api-key').value;
                    config.pathway_id = document.getElementById('bland-pathway-id').value;
                    break;
                case 'daily':
                    config.api_key = document.getElementById('daily-api-key').value;
                    config.room_url = document.getElementById('daily-room-url').value;
                    break;
                case 'vocode':
                    config.api_key = document.getElementById('vocode-api-key').value;
                    break;
                case 'websocket':
                    config.url = document.getElementById('websocket-url').value;
                    config.auth_header = document.getElementById('websocket-auth').value;
                    break;
            }
            return config;
        }

        async function connectPlatform(platformId) {
            const config = getPlatformConfig(platformId);
            const messageEl = document.getElementById('platform-message');

            messageEl.innerHTML = '<div style="color: var(--neon-cyan);">Connecting...</div>';

            try {
                const res = await fetch(`/api/platforms/${platformId}/connect`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ config })
                });
                const result = await res.json();

                updatePlatformStatus(platformId, result.status);

                if (result.success) {
                    messageEl.innerHTML = `<div style="color: var(--neon-green);">${result.message}</div>`;
                } else {
                    messageEl.innerHTML = `<div style="color: var(--neon-orange);">${result.message}</div>`;
                }
            } catch (e) {
                messageEl.innerHTML = `<div style="color: var(--neon-orange);">Connection failed: ${e.message}</div>`;
            }
        }

        async function testPlatform(platformId) {
            const messageEl = document.getElementById('platform-message');

            messageEl.innerHTML = '<div style="color: var(--neon-cyan);">Testing connection...</div>';

            try {
                const res = await fetch(`/api/platforms/${platformId}/test`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const result = await res.json();

                if (result.success) {
                    messageEl.innerHTML = `<div style="color: var(--neon-green);">${result.message} (latency: ${result.latency_ms}ms)</div>`;
                } else {
                    messageEl.innerHTML = `<div style="color: var(--neon-orange);">${result.message}</div>`;
                }
            } catch (e) {
                messageEl.innerHTML = `<div style="color: var(--neon-orange);">Test failed: ${e.message}</div>`;
            }
        }

        async function disconnectPlatform(platformId) {
            const messageEl = document.getElementById('platform-message');

            try {
                const res = await fetch(`/api/platforms/${platformId}/disconnect`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const result = await res.json();

                updatePlatformStatus(platformId, result.status);
                messageEl.innerHTML = `<div style="color: var(--text-secondary);">${result.message}</div>`;
            } catch (e) {
                messageEl.innerHTML = `<div style="color: var(--neon-orange);">Disconnect failed: ${e.message}</div>`;
            }
        }

        // ============================================================
        // FAST BRAIN FUNCTIONS
        // ============================================================
        async function loadFastBrainConfig() {
            try {
                const res = await fetch('/api/fast-brain/config');
                const config = await res.json();

                document.getElementById('fb-url').value = config.url || '';
                document.getElementById('fb-min-containers').value = config.min_containers || 1;
                document.getElementById('fb-max-containers').value = config.max_containers || 10;
                document.getElementById('fb-status').textContent = config.status || 'Offline';
                document.getElementById('fb-requests').textContent = config.stats?.total_requests || 0;

                if (config.status === 'connected') {
                    document.getElementById('fb-status').style.color = 'var(--neon-green)';
                } else {
                    document.getElementById('fb-status').style.color = 'var(--text-secondary)';
                }
            } catch (e) {
                console.error('Failed to load Fast Brain config:', e);
            }
        }

        async function saveFastBrainConfig() {
            const messageEl = document.getElementById('fb-config-message');
            messageEl.innerHTML = '<div style="color: var(--neon-cyan);">Saving...</div>';

            try {
                const res = await fetch('/api/fast-brain/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        url: document.getElementById('fb-url').value,
                        min_containers: parseInt(document.getElementById('fb-min-containers').value),
                        max_containers: parseInt(document.getElementById('fb-max-containers').value),
                    })
                });
                const result = await res.json();

                if (result.success) {
                    messageEl.innerHTML = '<div style="color: var(--neon-green);">Configuration saved!</div>';
                    loadFastBrainConfig();
                } else {
                    messageEl.innerHTML = `<div style="color: var(--neon-orange);">Error: ${result.error}</div>`;
                }
            } catch (e) {
                messageEl.innerHTML = `<div style="color: var(--neon-orange);">Error: ${e.message}</div>`;
            }
        }

        async function checkFastBrainHealth() {
            const messageEl = document.getElementById('fb-config-message');
            messageEl.innerHTML = '<div style="color: var(--neon-cyan);">Checking health...</div>';

            try {
                const res = await fetch('/api/fast-brain/health');
                const result = await res.json();

                if (result.status === 'healthy') {
                    messageEl.innerHTML = `<div style="color: var(--neon-green);">Connected! Model loaded: ${result.model_loaded}</div>`;
                    document.getElementById('fb-status').textContent = 'Online';
                    document.getElementById('fb-status').style.color = 'var(--neon-green)';
                } else if (result.status === 'not_configured') {
                    messageEl.innerHTML = '<div style="color: var(--neon-orange);">Not configured - enter URL above</div>';
                } else {
                    messageEl.innerHTML = `<div style="color: var(--neon-orange);">${result.message || 'Connection failed'}</div>`;
                }
            } catch (e) {
                messageEl.innerHTML = `<div style="color: var(--neon-orange);">Error: ${e.message}</div>`;
            }
        }

        async function testFastBrainChat() {
            const btn = document.getElementById('fb-chat-btn');
            const responseEl = document.getElementById('fb-response');
            const metricsEl = document.getElementById('fb-metrics');

            const message = document.getElementById('fb-message').value;
            const systemPrompt = document.getElementById('fb-system-prompt').value;
            const maxTokens = parseInt(document.getElementById('fb-max-tokens').value);

            if (!message) {
                alert('Please enter a message');
                return;
            }

            btn.innerHTML = '<span class="spinner"></span> Sending...';
            btn.disabled = true;
            responseEl.textContent = 'Processing...';
            metricsEl.textContent = '';

            try {
                const res = await fetch('/api/fast-brain/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, system_prompt: systemPrompt, max_tokens: maxTokens })
                });
                const result = await res.json();

                if (result.success) {
                    responseEl.textContent = result.response;
                    const metrics = result.metrics;
                    metricsEl.innerHTML = `TTFB: <span style="color: ${metrics.ttfb_ms < 50 ? 'var(--neon-green)' : 'var(--neon-orange)'}">${metrics.ttfb_ms.toFixed(1)}ms</span> | Total: ${metrics.total_time_ms.toFixed(1)}ms | Throughput: ${metrics.tokens_per_sec.toFixed(1)} tok/s`;

                    // Update targets
                    document.getElementById('fb-ttfb-target').className = `table-status ${metrics.ttfb_ms < 50 ? 'deployed' : 'draft'}`;
                    document.getElementById('fb-throughput-target').className = `table-status ${metrics.tokens_per_sec > 500 ? 'deployed' : 'draft'}`;
                } else {
                    responseEl.innerHTML = `<span style="color: var(--neon-orange);">Error: ${result.error}</span>`;
                }
            } catch (e) {
                responseEl.innerHTML = `<span style="color: var(--neon-orange);">Error: ${e.message}</span>`;
            }

            btn.innerHTML = 'Send to Fast Brain';
            btn.disabled = false;
        }

        async function runFastBrainBenchmark() {
            const resultsDiv = document.getElementById('fb-benchmark-results');
            const outputDiv = document.getElementById('fb-benchmark-output');

            resultsDiv.style.display = 'block';
            outputDiv.innerHTML = '<div style="color: var(--neon-cyan);">Running benchmark (5 requests)...</div>';

            try {
                const res = await fetch('/api/fast-brain/benchmark', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ num_requests: 5, prompt: 'Hello, how are you?' })
                });
                const result = await res.json();

                if (result.success) {
                    const summary = result.summary;
                    outputDiv.innerHTML = `
                        <div style="margin-bottom: 1rem;">
                            <h4 style="color: var(--neon-cyan); margin-bottom: 0.5rem;">Summary</h4>
                            <p>Avg TTFB: <span style="color: ${summary.ttfb_target_met ? 'var(--neon-green)' : 'var(--neon-orange)'}">${summary.avg_ttfb_ms}ms</span> ${summary.ttfb_target_met ? '✓' : '✗'} (target: <50ms)</p>
                            <p>Avg Total: ${summary.avg_total_ms}ms</p>
                            <p>Avg Throughput: <span style="color: ${summary.throughput_target_met ? 'var(--neon-green)' : 'var(--neon-orange)'}">${summary.avg_tokens_per_sec} tok/s</span> ${summary.throughput_target_met ? '✓' : '✗'} (target: >500)</p>
                        </div>
                        <h4 style="color: var(--neon-cyan); margin-bottom: 0.5rem;">Individual Requests</h4>
                        <table class="skills-table">
                            <tr><th>#</th><th>TTFB</th><th>Total</th><th>Tokens/sec</th></tr>
                            ${result.results.map(r => `<tr><td>${r.request}</td><td>${r.ttfb_ms.toFixed(1)}ms</td><td>${r.total_ms.toFixed(1)}ms</td><td>${r.tokens_per_sec.toFixed(1)}</td></tr>`).join('')}
                        </table>
                    `;

                    // Update status cards
                    document.getElementById('fb-ttfb').textContent = `${summary.avg_ttfb_ms}ms`;
                    document.getElementById('fb-throughput').textContent = `${summary.avg_tokens_per_sec}`;
                } else {
                    outputDiv.innerHTML = `<div style="color: var(--neon-orange);">Error: ${result.error}</div>`;
                }
            } catch (e) {
                outputDiv.innerHTML = `<div style="color: var(--neon-orange);">Error: ${e.message}</div>`;
            }
        }

        // ============================================================
        // FAST BRAIN SUB-TABS & SKILLS MANAGEMENT
        // ============================================================
        let fbSkillsCache = {};

        function showFastBrainTab(tabId) {
            document.querySelectorAll('#tab-fastbrain .sub-tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('#tab-fastbrain .sub-tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('fb-tab-' + tabId).classList.add('active');
            event.target.classList.add('active');

            // Load data based on tab
            if (tabId === 'skills') loadFastBrainSkills();
            if (tabId === 'dashboard') refreshSystemStatus();
            if (tabId === 'integration') loadIntegrationChecklist();
        }

        async function loadFastBrainSkills() {
            const container = document.getElementById('fb-skills-list');
            container.innerHTML = '<div style="color: var(--text-secondary); padding: 2rem; text-align: center;">Loading skills...</div>';

            try {
                const res = await fetch('/api/fast-brain/skills');
                const data = await res.json();
                const skills = data.skills || [];
                fbSkillsCache = {};
                skills.forEach(s => fbSkillsCache[s.id] = s);

                // Update skill selector dropdown
                updateSkillSelector(skills);
                document.getElementById('fb-skills-count').textContent = skills.length;

                if (skills.length === 0) {
                    container.innerHTML = '<div style="color: var(--text-secondary); padding: 2rem; text-align: center;">No skills available.</div>';
                    return;
                }

                container.innerHTML = skills.map(skill => `
                    <div class="skill-card" style="padding: 1rem; background: var(--glass-surface); border-radius: 8px; border: 1px solid var(--glass-border); border-left: 4px solid ${skill.is_builtin ? 'var(--neon-cyan)' : 'var(--neon-green)'};">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                            <div>
                                <strong style="color: var(--text-primary);">${skill.name}</strong>
                                <span style="font-size: 0.75rem; color: var(--text-secondary); margin-left: 0.5rem;">${skill.id}</span>
                            </div>
                            <span class="table-status ${skill.is_builtin ? 'deployed' : 'active'}" style="font-size: 0.7rem;">
                                ${skill.is_builtin ? 'Built-in' : 'Custom'}
                            </span>
                        </div>
                        <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.75rem;">${skill.description}</p>
                        <div style="font-size: 0.8rem; color: var(--text-secondary); background: var(--glass-bg); padding: 0.5rem; border-radius: 4px; max-height: 80px; overflow: hidden;">
                            ${skill.system_prompt?.substring(0, 150)}${skill.system_prompt?.length > 150 ? '...' : ''}
                        </div>
                        ${skill.knowledge && skill.knowledge.length > 0 ? `
                            <div style="margin-top: 0.5rem; font-size: 0.75rem; color: var(--neon-cyan);">
                                ${skill.knowledge.length} knowledge items
                            </div>
                        ` : ''}
                        <div style="margin-top: 0.75rem; display: flex; gap: 0.5rem;">
                            <button class="btn btn-sm btn-secondary" onclick="selectSkillForChat('${skill.id}')">Use in Chat</button>
                            ${!skill.is_builtin ? `<button class="btn btn-sm" onclick="deleteSkill('${skill.id}')" style="background: var(--neon-orange); color: white;">Delete</button>` : ''}
                        </div>
                    </div>
                `).join('');
            } catch (e) {
                container.innerHTML = `<div style="color: var(--neon-orange); padding: 2rem; text-align: center;">Error loading skills: ${e.message}</div>`;
            }
        }

        function updateSkillSelector(skills) {
            const selector = document.getElementById('fb-skill-selector');
            if (!selector) return;

            selector.innerHTML = skills.map(s => `
                <option value="${s.id}">${s.name}</option>
            `).join('');
        }

        function selectSkillForChat(skillId) {
            showFastBrainTab('chat');
            document.getElementById('fb-skill-selector').value = skillId;
            onSkillSelect();
        }

        function onSkillSelect() {
            const skillId = document.getElementById('fb-skill-selector').value;
            const skill = fbSkillsCache[skillId];

            if (skill) {
                document.getElementById('fb-active-skill-name').textContent = skill.name;
                document.getElementById('fb-active-skill-desc').textContent = skill.description;

                // Also update on server
                fetch('/api/fast-brain/skills/select', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ skill_id: skillId })
                });
            }
        }

        function loadSkillPrompt() {
            const skillId = document.getElementById('fb-skill-selector').value;
            const skill = fbSkillsCache[skillId];

            if (skill) {
                document.getElementById('fb-system-prompt').value = skill.system_prompt;
            }
        }

        function showCreateSkillModal() {
            document.getElementById('fb-create-skill-form').style.display = 'block';
        }

        function hideCreateSkillModal() {
            document.getElementById('fb-create-skill-form').style.display = 'none';
            // Clear form
            document.getElementById('new-skill-id').value = '';
            document.getElementById('new-skill-name').value = '';
            document.getElementById('new-skill-description').value = '';
            document.getElementById('new-skill-prompt').value = '';
            document.getElementById('new-skill-knowledge').value = '';
            document.getElementById('create-skill-message').innerHTML = '';
        }

        async function createCustomSkill() {
            const messageEl = document.getElementById('create-skill-message');
            messageEl.innerHTML = '<div style="color: var(--neon-cyan);">Creating skill...</div>';

            const knowledge = document.getElementById('new-skill-knowledge').value
                .split('\\n')
                .map(s => s.trim())
                .filter(s => s.length > 0);

            try {
                const res = await fetch('/api/fast-brain/skills', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        skill_id: document.getElementById('new-skill-id').value,
                        name: document.getElementById('new-skill-name').value,
                        description: document.getElementById('new-skill-description').value,
                        system_prompt: document.getElementById('new-skill-prompt').value,
                        knowledge: knowledge,
                    })
                });
                const result = await res.json();

                if (result.success) {
                    messageEl.innerHTML = '<div style="color: var(--neon-green);">Skill created successfully!</div>';
                    setTimeout(() => {
                        hideCreateSkillModal();
                        loadFastBrainSkills();
                    }, 1000);
                } else {
                    messageEl.innerHTML = `<div style="color: var(--neon-orange);">Error: ${result.error}</div>`;
                }
            } catch (e) {
                messageEl.innerHTML = `<div style="color: var(--neon-orange);">Error: ${e.message}</div>`;
            }
        }

        async function deleteSkill(skillId) {
            if (!confirm(`Delete skill "${skillId}"? This cannot be undone.`)) return;

            try {
                const res = await fetch(`/api/fast-brain/skills/${skillId}`, { method: 'DELETE' });
                const result = await res.json();

                if (result.success) {
                    loadFastBrainSkills();
                } else {
                    alert(`Error: ${result.error}`);
                }
            } catch (e) {
                alert(`Error: ${e.message}`);
            }
        }

        // ============================================================
        // SYSTEM STATUS & HIVE215 INTEGRATION
        // ============================================================
        async function refreshSystemStatus() {
            const statusIndicators = {
                'fb-status-indicator': { element: null, card: 'status-fast-brain' },
                'groq-status-indicator': { element: null, card: 'status-groq' },
                'hive-status-indicator': { element: null, card: 'status-hive215' },
                'modal-status-indicator': { element: null, card: 'status-modal' },
            };

            // Set all to loading
            Object.keys(statusIndicators).forEach(id => {
                const el = document.getElementById(id);
                if (el) el.textContent = 'Checking...';
            });

            try {
                const res = await fetch('/api/system/status');
                const data = await res.json();
                const systems = data.systems || {};

                // Update Fast Brain status
                const fb = systems.fast_brain || {};
                const fbIndicator = document.getElementById('fb-status-indicator');
                const fbCard = document.getElementById('status-fast-brain');
                if (fbIndicator) {
                    fbIndicator.textContent = fb.status === 'online' ? 'Online' : fb.status === 'degraded' ? 'Degraded' : fb.status || 'Offline';
                    fbIndicator.style.color = fb.status === 'online' ? 'var(--neon-green)' : fb.status === 'degraded' ? 'var(--neon-yellow)' : 'var(--neon-orange)';
                }
                if (fbCard) {
                    fbCard.style.borderLeftColor = fb.status === 'online' ? 'var(--neon-green)' : fb.status === 'degraded' ? 'var(--neon-yellow)' : 'var(--text-secondary)';
                }
                document.getElementById('fb-status-latency').textContent = fb.latency_ms ? `${fb.latency_ms}ms` : '--';
                document.getElementById('fb-status-model').textContent = fb.model || '--';
                document.getElementById('fb-status-skills').textContent = fb.skills_count || '--';

                // Update Groq status (inferred from Fast Brain)
                const groqIndicator = document.getElementById('groq-status-indicator');
                const groqCard = document.getElementById('status-groq');
                if (groqIndicator) {
                    groqIndicator.textContent = fb.status === 'online' ? 'Connected' : 'Unknown';
                    groqIndicator.style.color = fb.status === 'online' ? 'var(--neon-green)' : 'var(--text-secondary)';
                }
                if (groqCard) {
                    groqCard.style.borderLeftColor = fb.status === 'online' ? 'var(--neon-green)' : 'var(--text-secondary)';
                }

                // Update Modal status
                const modalIndicator = document.getElementById('modal-status-indicator');
                const modalCard = document.getElementById('status-modal');
                if (modalIndicator) {
                    modalIndicator.textContent = fb.status === 'online' ? 'Running' : 'Unknown';
                    modalIndicator.style.color = fb.status === 'online' ? 'var(--neon-green)' : 'var(--text-secondary)';
                }
                if (modalCard) {
                    modalCard.style.borderLeftColor = fb.status === 'online' ? 'var(--neon-green)' : 'var(--text-secondary)';
                }

                // Update Hive215 status
                const hive = systems.hive215 || {};
                const hiveIndicator = document.getElementById('hive-status-indicator');
                const hiveCard = document.getElementById('status-hive215');
                if (hiveIndicator) {
                    hiveIndicator.textContent = hive.status === 'online' ? 'Connected' : hive.status || 'Not configured';
                    hiveIndicator.style.color = hive.status === 'online' ? 'var(--neon-green)' : 'var(--text-secondary)';
                }
                if (hiveCard) {
                    hiveCard.style.borderLeftColor = hive.status === 'online' ? 'var(--neon-green)' : 'var(--text-secondary)';
                }

                // Show errors if any
                if (fb.error) {
                    document.getElementById('system-error-display').style.display = 'block';
                    document.getElementById('system-errors').textContent = fb.error;
                } else {
                    document.getElementById('system-error-display').style.display = 'none';
                }

            } catch (e) {
                console.error('Failed to refresh system status:', e);
            }
        }

        async function loadIntegrationChecklist() {
            const container = document.getElementById('integration-checklist');

            try {
                const res = await fetch('/api/system/checklist');
                const data = await res.json();

                // Update progress
                document.getElementById('integration-progress').textContent = `${data.percent}%`;
                document.getElementById('integration-progress-bar').style.width = `${data.percent}%`;

                // Group by category
                const byCategory = {};
                data.checklist.forEach(item => {
                    if (!byCategory[item.category]) byCategory[item.category] = [];
                    byCategory[item.category].push(item);
                });

                container.innerHTML = Object.entries(byCategory).map(([category, items]) => `
                    <div style="margin-bottom: 1.5rem;">
                        <h4 style="color: var(--neon-cyan); margin-bottom: 0.75rem; font-size: 0.9rem;">${category}</h4>
                        ${items.map(item => `
                            <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0; border-bottom: 1px solid var(--glass-border);">
                                <span style="font-size: 1.2rem;">${item.status === 'complete' ? '✅' : '⬜'}</span>
                                <span style="color: ${item.status === 'complete' ? 'var(--text-primary)' : 'var(--text-secondary)'};">
                                    ${item.name}
                                </span>
                            </div>
                        `).join('')}
                    </div>
                `).join('');
            } catch (e) {
                container.innerHTML = `<div style="color: var(--neon-orange);">Error loading checklist: ${e.message}</div>`;
            }
        }

        async function saveHive215Config() {
            const messageEl = document.getElementById('hive-config-message');
            messageEl.innerHTML = '<div style="color: var(--neon-cyan);">Saving...</div>';

            try {
                const res = await fetch('/api/hive215/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        url: document.getElementById('hive-url').value,
                        api_key: document.getElementById('hive-api-key').value,
                    })
                });
                const result = await res.json();

                if (result.success) {
                    messageEl.innerHTML = '<div style="color: var(--neon-green);">Configuration saved!</div>';
                    loadIntegrationChecklist();
                } else {
                    messageEl.innerHTML = `<div style="color: var(--neon-orange);">Error: ${result.error}</div>`;
                }
            } catch (e) {
                messageEl.innerHTML = `<div style="color: var(--neon-orange);">Error: ${e.message}</div>`;
            }
        }

        async function testHive215Connection() {
            const messageEl = document.getElementById('hive-config-message');
            messageEl.innerHTML = '<div style="color: var(--neon-cyan);">Testing connection...</div>';

            // For now, just verify URL is set
            const url = document.getElementById('hive-url').value;
            if (!url) {
                messageEl.innerHTML = '<div style="color: var(--neon-orange);">Please enter the Hive215 URL first</div>';
                return;
            }

            messageEl.innerHTML = '<div style="color: var(--neon-green);">URL configured! Full connection test requires deployed services.</div>';
        }

        async function loadHive215Config() {
            try {
                const res = await fetch('/api/hive215/config');
                const config = await res.json();

                document.getElementById('hive-url').value = config.url || '';
                document.getElementById('hive-api-key').value = config.api_key || '';
            } catch (e) {
                console.error('Failed to load Hive215 config:', e);
            }
        }

        // ============================================================
        // VOICE LAB FUNCTIONS
        // ============================================================
        function showVoiceLabTab(tabId) {
            document.querySelectorAll('#tab-voicelab .sub-tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('#tab-voicelab .sub-tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('voicelab-' + tabId).classList.add('active');
            event.target.classList.add('active');
        }

        async function loadVoiceProjects() {
            try {
                const res = await fetch('/api/voice-lab/projects');
                const projects = await res.json();

                const container = document.getElementById('voice-projects-list');
                if (projects.length === 0) {
                    container.innerHTML = '<div style="color: var(--text-secondary); padding: 2rem; text-align: center;">No voice projects yet. Click "New Voice" to create one.</div>';
                    return;
                }

                container.innerHTML = projects.map(p => `
                    <div class="skill-card ${p.status}" onclick="openVoiceProject('${p.id}')">
                        <div class="skill-header">
                            <div>
                                <div class="skill-name">${p.name}</div>
                                <div class="skill-type">${p.provider} | ${p.base_voice}</div>
                            </div>
                            <span class="skill-status ${p.status}">${p.status}</span>
                        </div>
                        <div class="skill-metrics">
                            <div class="skill-metric">
                                <div class="metric-value">${p.samples?.length || 0}</div>
                                <div class="metric-label">Samples</div>
                            </div>
                            <div class="skill-metric">
                                <div class="metric-value">${p.settings?.speed || 1.0}x</div>
                                <div class="metric-label">Speed</div>
                            </div>
                            <div class="skill-metric">
                                <div class="metric-value">${p.settings?.emotion || 'neutral'}</div>
                                <div class="metric-label">Emotion</div>
                            </div>
                        </div>
                    </div>
                `).join('');
            } catch (e) {
                console.error('Failed to load voice projects:', e);
            }
        }

        // Voice Lab - Load provider voices for dropdown
        function loadVLProviderVoices() {
            const provider = document.getElementById('vl-provider').value;
            const voiceSelect = document.getElementById('vl-base-voice');
            voiceSelect.innerHTML = '<option value="">Select a voice...</option>';

            if (!provider || !voiceProviders[provider]) {
                // Load providers if not already loaded
                fetch('/api/voice/providers').then(r => r.json()).then(data => {
                    voiceProviders = data;
                    populateVLVoices(provider);
                });
                return;
            }
            populateVLVoices(provider);
        }

        function populateVLVoices(provider) {
            const voiceSelect = document.getElementById('vl-base-voice');
            if (!voiceProviders[provider]) return;

            voiceProviders[provider].voices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice.id;
                option.textContent = `${voice.name} (${voice.gender}, ${voice.style})`;
                voiceSelect.appendChild(option);
            });
        }

        async function createVoiceProject() {
            const messageEl = document.getElementById('vl-create-message');
            messageEl.innerHTML = '<div style="color: var(--neon-cyan);">Creating project...</div>';

            try {
                const res = await fetch('/api/voice-lab/projects', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: document.getElementById('vl-name').value || 'Untitled Voice',
                        description: document.getElementById('vl-description').value,
                        provider: document.getElementById('vl-provider').value,
                        base_voice: document.getElementById('vl-base-voice').value,
                        settings: {
                            pitch: parseFloat(document.getElementById('vl-pitch').value),
                            speed: parseFloat(document.getElementById('vl-speed').value),
                            emotion: document.getElementById('vl-emotion').value,
                            style: document.getElementById('vl-style').value,
                        }
                    })
                });
                const result = await res.json();

                if (result.success) {
                    messageEl.innerHTML = `<div style="color: var(--neon-green);">Project "${result.project.name}" created!</div>`;
                    loadVoiceProjects();
                    showVoiceLabTab('projects');
                } else {
                    messageEl.innerHTML = `<div style="color: var(--neon-orange);">Error: ${result.error}</div>`;
                }
            } catch (e) {
                messageEl.innerHTML = `<div style="color: var(--neon-orange);">Error: ${e.message}</div>`;
            }
        }

        function openVoiceProject(projectId) {
            alert('Opening project: ' + projectId + '\\n\\nFull project editor coming soon!');
        }

        async function refreshVoiceTrainingStatus() {
            const container = document.getElementById('voice-training-queue');

            try {
                const res = await fetch('/api/voice-lab/training-status');
                const jobs = await res.json();

                if (jobs.length === 0) {
                    container.innerHTML = '<div style="color: var(--text-secondary); padding: 2rem; text-align: center;">No voice training jobs in progress.</div>';
                    return;
                }

                container.innerHTML = `
                    <table class="skills-table">
                        <tr><th>Project</th><th>Progress</th><th>Started</th><th>Status</th></tr>
                        ${jobs.map(j => `
                            <tr>
                                <td>${j.project_id}</td>
                                <td>
                                    <div style="background: var(--glass-surface); border-radius: 4px; overflow: hidden; height: 20px;">
                                        <div style="background: linear-gradient(90deg, var(--neon-cyan), var(--neon-green)); height: 100%; width: ${j.progress}%;"></div>
                                    </div>
                                    <span style="font-size: 0.75rem;">${j.progress}%</span>
                                </td>
                                <td>${new Date(j.started_at).toLocaleString()}</td>
                                <td><span class="table-status ${j.progress >= 100 ? 'deployed' : 'training'}">${j.progress >= 100 ? 'Complete' : 'Training'}</span></td>
                            </tr>
                        `).join('')}
                    </table>
                `;
            } catch (e) {
                container.innerHTML = `<div style="color: var(--neon-orange);">Error loading training status</div>`;
            }
        }

        // ============================================================
        // SKILL FINE-TUNING FUNCTIONS
        // ============================================================
        async function loadSkillsForDropdowns() {
            try {
                const res = await fetch('/api/skills');
                const skills = await res.json();

                const ftSelect = document.getElementById('ft-skill-select');
                const fbSelect = document.getElementById('feedback-skill-select');

                const options = '<option value="">Select a skill...</option>' +
                    skills.map(s => `<option value="${s.id}">${s.name}</option>`).join('');

                ftSelect.innerHTML = options;
                fbSelect.innerHTML = options;
            } catch (e) {
                console.error('Failed to load skills for dropdowns:', e);
            }
        }

        async function startSkillFineTune() {
            const messageEl = document.getElementById('ft-message');
            const skillId = document.getElementById('ft-skill-select').value;
            const examplesText = document.getElementById('ft-examples').value;

            if (!skillId) {
                messageEl.innerHTML = '<div style="color: var(--neon-orange);">Please select a skill</div>';
                return;
            }

            // Parse JSONL examples
            let examples = [];
            try {
                const lines = examplesText.trim().split('\\n').filter(l => l.trim());
                examples = lines.map(l => JSON.parse(l));
            } catch (e) {
                messageEl.innerHTML = '<div style="color: var(--neon-orange);">Invalid JSONL format</div>';
                return;
            }

            if (examples.length === 0) {
                messageEl.innerHTML = '<div style="color: var(--neon-orange);">Please add at least one training example</div>';
                return;
            }

            messageEl.innerHTML = '<div style="color: var(--neon-cyan);">Starting fine-tuning...</div>';

            try {
                const res = await fetch(`/api/skills/${skillId}/fine-tune`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ examples })
                });
                const result = await res.json();

                if (result.success) {
                    messageEl.innerHTML = `<div style="color: var(--neon-green);">${result.message}</div>`;
                } else {
                    messageEl.innerHTML = `<div style="color: var(--neon-orange);">Error: ${result.error}</div>`;
                }
            } catch (e) {
                messageEl.innerHTML = `<div style="color: var(--neon-orange);">Error: ${e.message}</div>`;
            }
        }

        async function submitSkillFeedback() {
            const skillId = document.getElementById('feedback-skill-select').value;
            const query = document.getElementById('feedback-query').value;
            const response = document.getElementById('feedback-response').value;
            const rating = document.getElementById('feedback-rating').value;
            const corrected = document.getElementById('feedback-corrected').value;

            if (!skillId || !query || !response) {
                alert('Please fill in skill, query, and response');
                return;
            }

            try {
                const res = await fetch(`/api/skills/${skillId}/feedback`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query,
                        response,
                        rating: parseInt(rating),
                        corrected_response: corrected || null
                    })
                });
                const result = await res.json();

                if (result.success) {
                    alert('Feedback submitted successfully!');
                    document.getElementById('feedback-query').value = '';
                    document.getElementById('feedback-response').value = '';
                    document.getElementById('feedback-corrected').value = '';
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }

        // ============================================================
        // INITIALIZATION
        // ============================================================
        document.addEventListener('DOMContentLoaded', () => {
            loadSkills();
            refreshServerStatus();
            loadMetrics();
            loadActivity();
            loadVoiceProviders();
            loadPlatforms();

            setInterval(loadSkills, 30000);
            setInterval(refreshServerStatus, 10000);
            setInterval(loadActivity, 15000);
        });
    </script>
</body>
</html>
'''


@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)


if __name__ == '__main__':
    # Add some initial activity
    add_activity("plumber_expert deployed successfully", "")
    add_activity("Training started for restaurant_host", "")
    add_activity("Low satisfaction detected on tech_support", "")
    add_activity("New documents processed: 47 examples", "")

    print("""
    ===============================================================
             UNIFIED SKILL COMMAND CENTER
             Voice Agent Intelligence Hub
    ===============================================================

    Dashboard: http://localhost:5000

    TABS:
      Dashboard       - Stats, Server, Skills, Training Pipeline, Activity
      Skill Factory   - Business Profile, Upload Docs, Train, Manage Skills
      Command Center  - API Keys, Test LLM, Compare, Masking, Stats, Voice
      LPU Inference   - BitNet model testing

    ===============================================================
    """)
    app.run(host='0.0.0.0', port=5000, debug=True)
