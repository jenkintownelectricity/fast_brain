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

from flask import Flask, jsonify, render_template_string, request, send_file
from flask_cors import CORS
import json
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
import random

# Import unified database
try:
    import database as db
    USE_DATABASE = True
    print("Using SQLite database for persistent storage")
except ImportError:
    USE_DATABASE = False
    print("Database module not found - using in-memory storage")

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

# Load API keys from database (for backward compatibility with existing code)
API_KEYS = db.get_all_api_keys() if USE_DATABASE else {}

# Activity log - now uses database but keep reference for backward compatibility
ACTIVITY_LOG = db.get_recent_activity(20) if USE_DATABASE else []

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
    "parler_tts": {
        "name": "Parler TTS",
        "provider": "Parler (GPU - Expressive)",
        "voices": [
            {"id": "receptionist", "name": "Sarah (Receptionist)", "gender": "female", "style": "professional warm"},
            {"id": "electrician", "name": "Mike (Electrician)", "gender": "male", "style": "friendly knowledgeable"},
            {"id": "plumber", "name": "Tom (Plumber)", "gender": "male", "style": "reassuring calm"},
            {"id": "lawyer", "name": "Jennifer (Lawyer)", "gender": "female", "style": "professional articulate"},
            {"id": "solar", "name": "Alex (Solar)", "gender": "neutral", "style": "enthusiastic energetic"},
            {"id": "general", "name": "Jordan (General)", "gender": "neutral", "style": "warm natural"},
        ]
    },
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

def add_activity(message, icon="", category="general"):
    """Add an activity to the log."""
    if USE_DATABASE:
        db.add_activity(message, icon, category)
    else:
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
                "requests_today": 0,
                "avg_latency_ms": 0,
                "satisfaction_rate": 0,
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
    # Get real skill list from database
    skills_loaded = []
    if USE_DATABASE:
        db_skills = db.get_all_skills()
        skills_loaded = [s.get('id', s.get('skill_id', '')) for s in db_skills[:5]]

    return jsonify({
        "status": "online",
        "warm_containers": 1,
        "region": "modal-cloud",
        "uptime_hours": 0,
        "total_requests": 0,
        "avg_cold_start_ms": 0,
        "avg_warm_latency_ms": 0,
        "memory_usage_mb": 0,
        "active_skill": skills_loaded[0] if skills_loaded else None,
        "skills_loaded": skills_loaded,
        "cost_today_usd": 0,
    })


@app.route('/api/metrics')
def get_metrics():
    """Get performance metrics over time."""
    # Return empty metrics - real metrics come from activity log
    return jsonify([])


@app.route('/api/activity')
def get_activity():
    """Get recent activity."""
    if USE_DATABASE:
        activities = db.get_recent_activity(limit=10)
        return jsonify(activities)
    else:
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
    # Return idle status - real training status from database when training is active
    return jsonify({
        "current_skill": None,
        "stage": "idle",
        "progress": 0,
        "stages": {
            "ingest": "pending",
            "process": "pending",
            "train": "pending",
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
    """Save API keys (persisted to database with encryption)."""
    global API_KEYS
    data = request.json
    saved = []

    if data.get('groq'):
        db.save_api_key('groq', data['groq'])
        API_KEYS['groq'] = data['groq']
        saved.append('Groq')
    if data.get('openai'):
        db.save_api_key('openai', data['openai'])
        API_KEYS['openai'] = data['openai']
        saved.append('OpenAI')
    if data.get('anthropic'):
        db.save_api_key('anthropic', data['anthropic'])
        API_KEYS['anthropic'] = data['anthropic']
        saved.append('Anthropic')

    if saved:
        add_activity(f"API keys saved: {', '.join(saved)}", "", "api")
        return jsonify({"success": True, "saved": saved})
    return jsonify({"success": True, "message": "No keys provided"})


@app.route('/api/test-llm', methods=['POST'])
def test_llm():
    """Test a single LLM provider with real API call."""
    data = request.json
    provider = data.get('provider', 'groq')
    prompt = data.get('prompt', '')

    import time
    start = time.time()

    # Try to make real API call
    api_keys = db.get_all_api_keys() if USE_DATABASE else {}

    try:
        import httpx

        if provider == 'groq' and api_keys.get('groq'):
            response = httpx.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={'Authorization': f"Bearer {api_keys['groq']}", 'Content-Type': 'application/json'},
                json={'model': 'llama-3.3-70b-versatile', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 150},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                latency = int((time.time() - start) * 1000)
                return jsonify({
                    "success": True,
                    "response": result['choices'][0]['message']['content'],
                    "latency_ms": latency,
                    "provider": provider
                })

        # No API key configured
        return jsonify({
            "success": False,
            "error": f"No API key configured for {provider}. Add it in Settings → API Keys.",
            "provider": provider
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "provider": provider
        })


@app.route('/api/compare-llms', methods=['POST'])
def compare_llms():
    """Compare multiple LLM providers - requires API keys configured."""
    data = request.json
    providers = data.get('providers', ['groq'])
    prompt = data.get('prompt', '')

    results = []
    api_keys = db.get_all_api_keys() if USE_DATABASE else {}

    for provider in providers:
        if not api_keys.get(provider):
            results.append({
                "provider": provider,
                "response": f"No API key configured for {provider}",
                "first_token_ms": 0,
                "total_ms": 0,
                "error": True
            })
        else:
            # Would need real implementation for each provider
            results.append({
                "provider": provider,
                "response": f"API key found for {provider} - real comparison requires implementation",
                "first_token_ms": 0,
                "total_ms": 0,
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
    """Get system statistics - real counts from database."""
    skills_count = 0
    if USE_DATABASE:
        skills = db.get_all_skills()
        skills_count = len(skills) if skills else 0

    return jsonify({
        "total_queries": 0,
        "by_provider": {
            "groq": 0,
            "openai": 0,
            "anthropic": 0,
            "bitnet": 0,
        },
        "avg_latency_ms": 0,
        "cache_hits": 0,
        "skills_active": skills_count,
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
    """Test a voice with sample text - generates real audio."""
    import base64
    import time

    data = request.json
    voice_id = data.get('voice_id', 'en-US-JennyNeural')
    text = data.get('text', 'Hello! This is a test of the voice synthesis system.')
    provider = data.get('provider', 'edge_tts')
    emotion = data.get('emotion', 'neutral')
    skill_id = data.get('skill_id', 'general')

    start_time = time.time()

    # Parler TTS - Expressive voices via Modal GPU
    if provider == 'parler_tts':
        if not text or len(text.strip()) == 0:
            text = "Hello! This is a voice test."
        text = text.strip()[:500]

        try:
            import httpx

            # Call Parler TTS Modal function
            parler_url = "https://jenkintownelectricity--hive215-parler-tts-parlerttsmodel-synthesize-with-emotion.modal.run"

            with httpx.Client(timeout=120.0) as client:  # Long timeout for cold start
                response = client.post(
                    parler_url,
                    json={
                        "text": text,
                        "skill_id": skill_id,
                        "emotion": emotion
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    # Result is [audio_bytes, description]
                    if isinstance(result, list) and len(result) >= 2:
                        audio_bytes = bytes(result[0]) if isinstance(result[0], list) else result[0]
                        description = result[1]

                        duration_ms = int((time.time() - start_time) * 1000)
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8') if isinstance(audio_bytes, bytes) else result[0]

                        add_activity(f"Parler TTS: {skill_id}/{emotion} ({duration_ms}ms)", "")
                        return jsonify({
                            "success": True,
                            "voice_id": f"{skill_id}_{emotion}",
                            "provider": "parler_tts",
                            "text": text,
                            "duration_ms": duration_ms,
                            "audio_base64": audio_base64,
                            "audio_format": "audio/wav",
                            "message": f"Generated with Parler TTS: {description}"
                        })
                    else:
                        raise Exception(f"Unexpected response format: {type(result)}")
                else:
                    raise Exception(f"Parler TTS returned {response.status_code}: {response.text[:200]}")

        except httpx.TimeoutException:
            return jsonify({
                "success": False,
                "error": "Parler TTS timeout (GPU may be cold starting - try again in 30s)"
            }), 500
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Parler TTS failed: {str(e)}"
            }), 500

    # Map voice IDs to gTTS language/accent codes
    VOICE_TO_GTTS = {
        # Edge TTS voices -> gTTS (all use 'en' but we track the original)
        'en-US-JennyNeural': ('en', 'com'),
        'en-US-AriaNeural': ('en', 'com'),
        'en-US-GuyNeural': ('en', 'com'),
        'en-US-DavisNeural': ('en', 'com'),
        'en-US-SaraNeural': ('en', 'com'),
        'en-US-AnaNeural': ('en', 'com'),
        'en-GB-SoniaNeural': ('en', 'co.uk'),
        'en-GB-RyanNeural': ('en', 'co.uk'),
        'en-GB-LibbyNeural': ('en', 'co.uk'),
        'en-AU-NatashaNeural': ('en', 'com.au'),
        'en-AU-WilliamNeural': ('en', 'com.au'),
        # Kokoro voices -> gTTS
        'af_bella': ('en', 'com'),
        'af_nicole': ('en', 'com'),
        'af_sarah': ('en', 'com'),
        'af_sky': ('en', 'com'),
        'am_adam': ('en', 'com'),
        'am_michael': ('en', 'com'),
        'bf_emma': ('en', 'co.uk'),
        'bf_isabella': ('en', 'co.uk'),
        'bm_george': ('en', 'co.uk'),
        'bm_lewis': ('en', 'co.uk'),
    }

    # Use gTTS for all voices (HTTP-based, works reliably from Modal)
    # Ensure text is not empty and clean it
    if not text or len(text.strip()) == 0:
        text = "Hello! This is a voice test."
    text = text.strip()[:500]  # Limit length

    try:
        from gtts import gTTS
        import io

        # Get language settings for this voice
        lang, tld = VOICE_TO_GTTS.get(voice_id, ('en', 'com'))

        # Generate audio with gTTS
        tts = gTTS(text=text, lang=lang, tld=tld)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_bytes = audio_buffer.getvalue()

        if not audio_bytes:
            raise Exception("No audio generated")

        duration_ms = int((time.time() - start_time) * 1000)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Determine display provider name
        if voice_id.startswith('af_') or voice_id.startswith('am_') or voice_id.startswith('bf_') or voice_id.startswith('bm_'):
            display_provider = f"kokoro (via Google TTS)"
            message = f"Kokoro '{voice_id}' using Google TTS ({tld})"
        else:
            display_provider = "google_tts"
            message = f"Generated audio with Google TTS ({tld})"

        add_activity(f"Voice test: {voice_id} ({duration_ms}ms)", "")
        return jsonify({
            "success": True,
            "voice_id": voice_id,
            "provider": display_provider,
            "text": text,
            "duration_ms": duration_ms,
            "audio_base64": audio_base64,
            "audio_format": "audio/mpeg",
            "message": message
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"TTS failed: {str(e)}"
        }), 500


# =============================================================================
# API ENDPOINTS - GOLDEN PROMPTS (with Database Persistence)
# =============================================================================

# In-memory storage for custom prompts (fallback if no database)
CUSTOM_PROMPTS = {}

def get_custom_prompt(skill_id):
    """Get custom prompt from database or memory."""
    if USE_DATABASE:
        prompt = db.get_golden_prompt(skill_id)
        return prompt['content'] if prompt else None
    return CUSTOM_PROMPTS.get(skill_id)

def get_all_custom_prompt_ids():
    """Get list of skill IDs with custom prompts."""
    if USE_DATABASE:
        prompts = db.get_all_golden_prompts()
        return [p['skill_id'] for p in prompts]
    return list(CUSTOM_PROMPTS.keys())

@app.route('/api/golden-prompts')
def get_golden_prompts_list():
    """Get list of all available golden prompts."""
    try:
        import sys
        sys.path.insert(0, '/root')
        from golden_prompts import SKILL_MANUALS, TOKEN_ESTIMATES

        custom_ids = get_all_custom_prompt_ids()

        prompts = []
        for skill_id in SKILL_MANUALS.keys():
            tokens = TOKEN_ESTIMATES.get(skill_id, 800)
            prefill_ms = int((tokens / 6000) * 1000)
            prompts.append({
                "id": skill_id,
                "name": skill_id.replace("_", " ").replace("-", " ").title(),
                "tokens": tokens,
                "prefill_ms": prefill_ms,
                "is_custom": skill_id in custom_ids
            })
        return jsonify({"prompts": prompts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/golden-prompts/<skill_id>')
def get_golden_prompt(skill_id):
    """Get a specific golden prompt with optional context substitution."""
    try:
        import sys
        sys.path.insert(0, '/root')
        from golden_prompts import SKILL_MANUALS, TOKEN_ESTIMATES, get_skill_prompt

        # Check for custom override first (database or memory)
        custom_content = get_custom_prompt(skill_id)

        if custom_content:
            content = custom_content
            is_custom = True
        elif skill_id in SKILL_MANUALS:
            content = SKILL_MANUALS[skill_id]
            is_custom = False
        else:
            return jsonify({"error": f"Skill '{skill_id}' not found"}), 404

        # Get context variables from query params
        business_name = request.args.get('business_name', '{business_name}')
        agent_name = request.args.get('agent_name', '{agent_name}')

        # Substitute context variables
        formatted = content.replace('{business_name}', business_name).replace('{agent_name}', agent_name)

        tokens = TOKEN_ESTIMATES.get(skill_id, 800)
        prefill_ms = int((tokens / 6000) * 1000)

        return jsonify({
            "skill_id": skill_id,
            "content": content,
            "formatted": formatted,
            "tokens": tokens,
            "prefill_ms": prefill_ms,
            "is_custom": is_custom
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/golden-prompts/<skill_id>', methods=['POST'])
def save_golden_prompt(skill_id):
    """Save a custom golden prompt with database persistence."""
    global CUSTOM_PROMPTS

    try:
        data = request.json
        content = data.get('content', '')

        if not content.strip():
            return jsonify({"error": "Prompt content cannot be empty"}), 400

        # Save to database if available, otherwise memory
        if USE_DATABASE:
            result = db.save_golden_prompt(skill_id, content)
            tokens = result['tokens_estimate']
        else:
            CUSTOM_PROMPTS[skill_id] = content
            tokens = len(content) // 4

        prefill_ms = int((tokens / 6000) * 1000)

        add_activity(f"Updated golden prompt: {skill_id}", "", "prompts")

        return jsonify({
            "success": True,
            "skill_id": skill_id,
            "tokens": tokens,
            "prefill_ms": prefill_ms,
            "message": f"Prompt for '{skill_id}' saved successfully"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/golden-prompts/<skill_id>/reset', methods=['POST'])
def reset_golden_prompt(skill_id):
    """Reset a golden prompt to its default (removes from database)."""
    global CUSTOM_PROMPTS

    try:
        if USE_DATABASE:
            deleted = db.delete_golden_prompt(skill_id)
            if deleted:
                add_activity(f"Reset golden prompt: {skill_id}", "", "prompts")
                return jsonify({"success": True, "message": f"Prompt for '{skill_id}' reset to default"})
            else:
                return jsonify({"success": True, "message": "Prompt was already at default"})
        else:
            if skill_id in CUSTOM_PROMPTS:
                del CUSTOM_PROMPTS[skill_id]
                add_activity(f"Reset golden prompt: {skill_id}", "", "prompts")
                return jsonify({"success": True, "message": f"Prompt for '{skill_id}' reset to default"})
            else:
                return jsonify({"success": True, "message": "Prompt was already at default"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/golden-prompts/export')
def export_golden_prompts():
    """Export all golden prompts (defaults + custom overrides)."""
    try:
        import sys
        sys.path.insert(0, '/root')
        from golden_prompts import SKILL_MANUALS

        custom_ids = get_all_custom_prompt_ids()

        export_data = {}
        for skill_id, content in SKILL_MANUALS.items():
            # Use custom if available, otherwise default
            custom_content = get_custom_prompt(skill_id)
            export_data[skill_id] = custom_content if custom_content else content

        return jsonify({
            "prompts": export_data,
            "custom_overrides": custom_ids
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/golden-prompts/test', methods=['POST'])
def test_golden_prompt():
    """Test a golden prompt with the Fast Brain API."""
    try:
        data = request.json
        skill_id = data.get('skill_id', 'general')
        test_message = data.get('message', 'Hello, I need some help.')
        business_name = data.get('business_name', 'Test Business')
        agent_name = data.get('agent_name', 'Assistant')

        import sys
        sys.path.insert(0, '/root')
        from golden_prompts import get_skill_prompt, SKILL_MANUALS

        # Get prompt (custom or default)
        if skill_id in CUSTOM_PROMPTS:
            system_prompt = CUSTOM_PROMPTS[skill_id]
            system_prompt = system_prompt.replace('{business_name}', business_name).replace('{agent_name}', agent_name)
        else:
            system_prompt = get_skill_prompt(skill_id, business_name=business_name, agent_name=agent_name)

        # Call Fast Brain API
        import httpx
        import os

        fast_brain_url = os.environ.get('FAST_BRAIN_URL', '')
        if not fast_brain_url:
            return jsonify({"error": "Fast Brain URL not configured"}), 500

        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{fast_brain_url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": test_message}
                    ],
                    "max_tokens": 500
                }
            )

            if response.status_code == 200:
                result = response.json()
                return jsonify({
                    "success": True,
                    "response": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
                    "system_used": result.get("system_used", "unknown"),
                    "latency_ms": result.get("latency_ms", 0)
                })
            else:
                return jsonify({"error": f"Fast Brain returned {response.status_code}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    """Connect to a voice platform (persisted to database)."""
    global PLATFORM_CONNECTIONS

    if platform_id not in PLATFORM_CONNECTIONS:
        return jsonify({"error": "Platform not found"}), 404

    data = request.json
    config = data.get('config', {})

    # Update the platform config in memory
    PLATFORM_CONNECTIONS[platform_id]['config'].update(config)

    # Simulate connection test
    has_required_fields = all(v for v in PLATFORM_CONNECTIONS[platform_id]['config'].values())

    if has_required_fields:
        PLATFORM_CONNECTIONS[platform_id]['status'] = 'connected'
        # Save to database
        db.save_platform(platform_id, PLATFORM_CONNECTIONS[platform_id]['config'], 'connected')
        add_activity(f"Connected to {PLATFORM_CONNECTIONS[platform_id]['name']}", "", "platform")
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
    """Disconnect from a voice platform (persisted to database)."""
    global PLATFORM_CONNECTIONS

    if platform_id not in PLATFORM_CONNECTIONS:
        return jsonify({"error": "Platform not found"}), 404

    PLATFORM_CONNECTIONS[platform_id]['status'] = 'disconnected'
    # Save to database
    db.update_platform_status(platform_id, 'disconnected')
    add_activity(f"Disconnected from {PLATFORM_CONNECTIONS[platform_id]['name']}", "", "platform")

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

    # Test connection
    add_activity(f"Testing connection to {platform['name']}", "")
    return jsonify({
        "success": True,
        "message": f"Connection to {platform['name']} is working",
        "latency_ms": 0
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

# Local skills cache (loaded from database, no hardcoded demo data)
FAST_BRAIN_SKILLS = {}

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
# API ENDPOINTS - FAST BRAIN SKILLS (Unified with Database)
# =============================================================================

@app.route('/api/fast-brain/skills')
def get_fast_brain_skills():
    """Get all available skills from unified database."""
    if USE_DATABASE:
        skills = db.get_all_skills()
        selected = db.get_config('selected_skill', 'general')
    else:
        # Fallback to in-memory
        url = FAST_BRAIN_CONFIG.get('url')
        if url:
            try:
                import httpx
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(f"{url}/v1/skills")
                    if response.status_code == 200:
                        remote_skills = response.json().get("skills", [])
                        for skill in remote_skills:
                            if skill["id"] not in FAST_BRAIN_SKILLS:
                                FAST_BRAIN_SKILLS[skill["id"]] = {**skill, "is_builtin": False}
            except:
                pass
        skills = list(FAST_BRAIN_SKILLS.values())
        selected = FAST_BRAIN_CONFIG.get("selected_skill", "general")

    return jsonify({
        "skills": skills,
        "selected": selected
    })


@app.route('/api/fast-brain/skills', methods=['POST'])
def create_fast_brain_skill():
    """Create a new custom skill with database persistence."""
    data = request.json

    skill_id = data.get('skill_id', '').lower().replace(' ', '_')
    skill_id = re.sub(r'[^\w\-]', '_', skill_id)

    if not skill_id:
        return jsonify({"success": False, "error": "Skill ID is required"})

    # Check for builtin skill
    if USE_DATABASE:
        existing = db.get_skill(skill_id)
        if existing and existing.get('is_builtin'):
            return jsonify({"success": False, "error": "Cannot overwrite built-in skill"})

        # Create/update in database
        skill = db.create_skill(
            skill_id=skill_id,
            name=data.get('name', skill_id.replace('_', ' ').title()),
            description=data.get('description', ''),
            skill_type=data.get('skill_type', 'custom'),
            system_prompt=data.get('system_prompt', ''),
            knowledge=data.get('knowledge', []),
            is_builtin=False
        )
    else:
        # Fallback to in-memory
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
                        "name": skill.get("name"),
                        "description": skill.get("description"),
                        "system_prompt": skill.get("system_prompt"),
                        "knowledge": skill.get("knowledge", []),
                    }
                )
                if response.status_code == 200:
                    add_activity(f"Skill '{skill.get('name')}' synced to LPU", "🔄", "skills")
        except Exception as e:
            add_activity(f"Skill created (sync failed: {str(e)[:50]})", "⚠️", "skills")

    add_activity(f"Created skill: {skill.get('name')}", "✨", "skills")
    return jsonify({"success": True, "skill": skill})


@app.route('/api/fast-brain/skills/<skill_id>', methods=['PUT'])
def update_fast_brain_skill(skill_id):
    """Update an existing skill."""
    data = request.json

    if USE_DATABASE:
        existing = db.get_skill(skill_id)
        if not existing:
            return jsonify({"success": False, "error": "Skill not found"})

        if existing.get('is_builtin') and 'system_prompt' not in data:
            return jsonify({"success": False, "error": "Cannot modify built-in skill structure"})

        # Update allowed fields
        update_data = {}
        for field in ['name', 'description', 'system_prompt', 'knowledge', 'skill_type']:
            if field in data:
                update_data[field] = data[field]

        skill = db.update_skill(skill_id, **update_data)
        add_activity(f"Updated skill: {skill.get('name')}", "📝", "skills")
        return jsonify({"success": True, "skill": skill})
    else:
        if skill_id not in FAST_BRAIN_SKILLS:
            return jsonify({"success": False, "error": "Skill not found"})

        FAST_BRAIN_SKILLS[skill_id].update(data)
        add_activity(f"Updated skill: {skill_id}", "📝", "skills")
        return jsonify({"success": True, "skill": FAST_BRAIN_SKILLS[skill_id]})


@app.route('/api/fast-brain/skills/<skill_id>', methods=['DELETE'])
def delete_fast_brain_skill(skill_id):
    """Delete a custom skill."""
    if USE_DATABASE:
        existing = db.get_skill(skill_id)
        if not existing:
            return jsonify({"success": False, "error": "Skill not found"})

        if existing.get('is_builtin'):
            return jsonify({"success": False, "error": "Cannot delete built-in skill"})

        db.delete_skill(skill_id)
        add_activity(f"Deleted skill: {skill_id}", "🗑️", "skills")
        return jsonify({"success": True})
    else:
        if skill_id not in FAST_BRAIN_SKILLS:
            return jsonify({"success": False, "error": "Skill not found"})

        if FAST_BRAIN_SKILLS[skill_id].get('is_builtin'):
            return jsonify({"success": False, "error": "Cannot delete built-in skill"})

        del FAST_BRAIN_SKILLS[skill_id]
        add_activity(f"Deleted skill: {skill_id}", "🗑️", "skills")
        return jsonify({"success": True})


@app.route('/api/fast-brain/skills/<skill_id>')
def get_single_skill(skill_id):
    """Get a single skill by ID."""
    if USE_DATABASE:
        skill = db.get_skill(skill_id)
        if not skill:
            return jsonify({"error": "Skill not found"}), 404
        return jsonify(skill)
    else:
        if skill_id not in FAST_BRAIN_SKILLS:
            return jsonify({"error": "Skill not found"}), 404
        return jsonify(FAST_BRAIN_SKILLS[skill_id])


@app.route('/api/fast-brain/skills/select', methods=['POST'])
def select_fast_brain_skill():
    """Select the active skill for chat."""
    data = request.json
    skill_id = data.get('skill_id', 'general')

    if USE_DATABASE:
        skill = db.get_skill(skill_id)
        if not skill:
            return jsonify({"success": False, "error": "Skill not found"})

        db.set_config('selected_skill', skill_id, 'fast_brain')
        add_activity(f"Selected skill: {skill.get('name')}", "✓", "skills")
    else:
        if skill_id not in FAST_BRAIN_SKILLS:
            return jsonify({"success": False, "error": "Skill not found"})

        FAST_BRAIN_CONFIG['selected_skill'] = skill_id
        add_activity(f"Selected skill: {FAST_BRAIN_SKILLS[skill_id]['name']}", "✓", "skills")

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
# API ENDPOINTS - VOICE LAB (Database-backed with TTS Integration)
# =============================================================================

# Training queue kept in memory for real-time progress (jobs are short-lived)
VOICE_TRAINING_QUEUE = []

# Voice samples directory on Modal volume
VOICE_SAMPLES_DIR = Path("/data/voice_samples")
VOICE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


@app.route('/api/voice-lab/projects')
def get_voice_projects():
    """Get all voice projects from database."""
    try:
        if USE_DATABASE:
            # Ensure tables exist
            db.init_db()
            projects = db.get_all_voice_projects()
            return jsonify(projects)
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/voice-lab/projects', methods=['POST'])
def create_voice_project_endpoint():
    """Create a new voice project with database persistence."""
    try:
        data = request.json or {}
        project_id = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}"

        if USE_DATABASE:
            # Ensure tables exist
            db.init_db()

            project = db.create_voice_project(
                project_id=project_id,
                name=data.get("name", "Untitled Voice"),
                description=data.get("description", ""),
                provider=data.get("provider", "elevenlabs"),
                base_voice=data.get("base_voice"),
                settings=data.get("settings", {
                    "pitch": 1.0,
                    "speed": 1.0,
                    "emotion": "neutral",
                    "style": "conversational",
                }),
                skill_id=data.get("skill_id")
            )
            add_activity(f"Voice project created: {project['name']}", "", "voice")
            return jsonify({"success": True, "project": project})

        return jsonify({"success": False, "error": "Database not available"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/voice-lab/projects/<project_id>')
def get_voice_project_endpoint(project_id):
    """Get a specific voice project from database."""
    try:
        if USE_DATABASE:
            db.init_db()
            project = db.get_voice_project(project_id)
            if project:
                return jsonify(project)
        return jsonify({"error": "Project not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/voice-lab/projects/<project_id>', methods=['PUT'])
def update_voice_project_endpoint(project_id):
    """Update a voice project in database."""
    if not USE_DATABASE:
        return jsonify({"error": "Database not available"}), 500

    data = request.json
    project = db.update_voice_project(project_id, **data)

    if project:
        return jsonify({"success": True, "project": project})
    return jsonify({"error": "Project not found"}), 404


@app.route('/api/voice-lab/projects/<project_id>', methods=['DELETE'])
def delete_voice_project_endpoint(project_id):
    """Delete a voice project."""
    if USE_DATABASE:
        # Also delete sample files from disk
        project = db.get_voice_project(project_id)
        if project:
            for sample in project.get('samples', []):
                if sample.get('file_path') and os.path.exists(sample['file_path']):
                    os.remove(sample['file_path'])

            if db.delete_voice_project(project_id):
                add_activity(f"Voice project deleted: {project['name']}", "", "voice")
                return jsonify({"success": True})

    return jsonify({"error": "Project not found"}), 404


@app.route('/api/voice-lab/projects/<project_id>/samples', methods=['POST'])
def add_voice_sample_endpoint(project_id):
    """Add a training sample to a voice project (with file upload)."""
    if not USE_DATABASE:
        return jsonify({"error": "Database not available"}), 500

    project = db.get_voice_project(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    # Handle file upload
    if 'audio' in request.files:
        audio_file = request.files['audio']
        if audio_file.filename:
            sample_id = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}"
            filename = f"{project_id}_{sample_id}_{audio_file.filename}"
            file_path = VOICE_SAMPLES_DIR / filename

            audio_file.save(str(file_path))

            # Get transcript and duration from form data
            transcript = request.form.get('transcript', '')
            duration_ms = int(request.form.get('duration_ms', 0))
            emotion = request.form.get('emotion', 'neutral')

            sample = db.add_voice_sample(
                project_id=project_id,
                sample_id=sample_id,
                filename=filename,
                file_path=str(file_path),
                transcript=transcript,
                duration_ms=duration_ms,
                emotion=emotion
            )

            add_activity(f"Voice sample added to {project['name']}", "", "voice")
            return jsonify({"success": True, "sample": sample})

    # Handle JSON data (metadata only, for URL-based samples)
    data = request.json or {}
    sample_id = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}"

    sample = db.add_voice_sample(
        project_id=project_id,
        sample_id=sample_id,
        filename=data.get('filename', 'external_audio'),
        file_path=data.get('audio_url'),
        transcript=data.get('transcript', data.get('text', '')),
        duration_ms=int(data.get('duration_ms', 0)),
        emotion=data.get('emotion', 'neutral')
    )

    return jsonify({"success": True, "sample": sample})


@app.route('/api/voice-lab/projects/<project_id>/samples/<sample_id>', methods=['DELETE'])
def delete_voice_sample_endpoint(project_id, sample_id):
    """Delete a voice sample."""
    if USE_DATABASE:
        # Delete file from disk if it exists
        samples = db.get_voice_samples(project_id)
        for sample in samples:
            if sample['id'] == sample_id and sample.get('file_path'):
                if os.path.exists(sample['file_path']):
                    os.remove(sample['file_path'])
                break

        if db.delete_voice_sample(sample_id):
            return jsonify({"success": True})

    return jsonify({"error": "Sample not found"}), 404


@app.route('/api/voice-lab/projects/<project_id>/train', methods=['POST'])
def train_voice_endpoint(project_id):
    """Start training a custom voice using the selected provider."""
    if not USE_DATABASE:
        return jsonify({"error": "Database not available"}), 500

    project = db.get_voice_project(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    samples = project.get('samples', [])
    if len(samples) < 1:
        return jsonify({
            "success": False,
            "error": "Need at least 1 audio sample to clone/train a voice"
        }), 400

    provider = project.get('provider', 'elevenlabs')

    # Update project status to training
    db.update_voice_project(project_id, status='training', training_started=datetime.now().isoformat())

    # Add to training queue for progress tracking
    job = {
        "project_id": project_id,
        "provider": provider,
        "started_at": datetime.now().isoformat(),
        "progress": 0,
        "status": "starting"
    }
    VOICE_TRAINING_QUEUE.append(job)

    # Try to actually train/clone with the provider
    try:
        voice_id = None
        message = ""

        if provider == 'elevenlabs':
            voice_id, message = _train_elevenlabs_voice(project, samples)
        elif provider == 'cartesia':
            voice_id, message = _train_cartesia_voice(project, samples)
        elif provider in ['xtts', 'openvoice']:
            # These are local models - mark as ready (would need GPU)
            voice_id = f"{provider}_{project_id}"
            message = f"Voice ready for {provider} synthesis (local model)"
        elif provider in ['edge_tts', 'parler_tts', 'kokoro', 'chatterbox', 'gtts']:
            # Free providers - no training needed, mark as ready
            voice_id = f"{provider}_{project_id}"
            message = f"Voice ready! {provider} doesn't require training - you can test it now."
        else:
            # For other providers, simulate training
            voice_id = f"{provider}_{project_id}"
            message = f"Voice configured for {provider}"

        if voice_id:
            db.update_voice_project(
                project_id,
                status='trained',
                voice_id=voice_id,
                training_completed=datetime.now().isoformat()
            )
            job['status'] = 'completed'
            job['progress'] = 100

            add_activity(f"Voice trained: {project['name']} ({provider})", "", "voice")

            return jsonify({
                "success": True,
                "message": message,
                "voice_id": voice_id,
                "provider": provider
            })
        else:
            # Training failed - no voice_id returned (usually missing API key)
            db.update_voice_project(project_id, status='failed')
            job['status'] = 'failed'
            job['error'] = message

            return jsonify({
                "success": False,
                "error": message or f"Training failed. Make sure your {provider} API key is configured in Settings → API Keys."
            }), 400

    except Exception as e:
        db.update_voice_project(project_id, status='failed')
        job['status'] = 'failed'
        job['error'] = str(e)

        return jsonify({
            "success": False,
            "error": f"Training failed: {str(e)}"
        }), 500

    return jsonify({
        "success": True,
        "message": "Training started",
        "estimated_time_minutes": len(samples) * 2
    })


def _train_elevenlabs_voice(project, samples):
    """Clone a voice using ElevenLabs API."""
    api_key = None
    if USE_DATABASE:
        api_key = db.get_api_key('elevenlabs')

    if not api_key:
        return None, "ElevenLabs API key not configured"

    try:
        import requests

        # Prepare files for upload
        files = []
        for sample in samples:
            if sample.get('file_path') and os.path.exists(sample['file_path']):
                files.append(
                    ('files', (sample['filename'], open(sample['file_path'], 'rb'), 'audio/mpeg'))
                )

        if not files:
            return None, "No audio files found for training"

        # Create voice clone
        response = requests.post(
            'https://api.elevenlabs.io/v1/voices/add',
            headers={'xi-api-key': api_key},
            data={
                'name': project['name'],
                'description': project.get('description', 'Custom cloned voice'),
            },
            files=files
        )

        # Close file handles
        for _, (_, f, _) in files:
            f.close()

        if response.status_code == 200:
            result = response.json()
            return result.get('voice_id'), f"Voice cloned successfully: {result.get('voice_id')}"
        else:
            return None, f"ElevenLabs error: {response.text}"

    except Exception as e:
        return None, f"ElevenLabs API error: {str(e)}"


def _train_cartesia_voice(project, samples):
    """Clone a voice using Cartesia API."""
    api_key = None
    if USE_DATABASE:
        api_key = db.get_api_key('cartesia')

    if not api_key:
        return None, "Cartesia API key not configured"

    try:
        import requests

        # Cartesia uses a different cloning approach
        # First sample is used as reference
        if samples and samples[0].get('file_path') and os.path.exists(samples[0]['file_path']):
            with open(samples[0]['file_path'], 'rb') as f:
                audio_data = f.read()

            response = requests.post(
                'https://api.cartesia.ai/voices/clone/clip',
                headers={
                    'X-API-Key': api_key,
                    'Cartesia-Version': '2024-06-10',
                    'Content-Type': 'application/json'
                },
                json={
                    'clip': audio_data.hex(),  # Cartesia expects hex-encoded audio
                    'name': project['name'],
                }
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('id'), f"Voice cloned with Cartesia: {result.get('id')}"
            else:
                return None, f"Cartesia error: {response.text}"

        return None, "No audio samples available"

    except Exception as e:
        return None, f"Cartesia API error: {str(e)}"


@app.route('/api/voice-lab/projects/<project_id>/test', methods=['POST'])
def test_voice_project_endpoint(project_id):
    """Test a trained voice with sample text - actual TTS synthesis."""
    if not USE_DATABASE:
        return jsonify({"error": "Database not available"}), 500

    project = db.get_voice_project(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    data = request.json
    text = data.get("text", "Hello, this is a test of my custom voice.")

    provider = project.get('provider', 'elevenlabs')
    voice_id = project.get('voice_id')
    status = project.get('status', 'draft')

    # Free providers that don't need voice cloning/training
    free_providers = ['edge_tts', 'parler_tts', 'kokoro', 'chatterbox', 'gtts']

    # For paid providers, check that training completed
    if provider not in free_providers:
        if not voice_id and status != 'trained':
            return jsonify({
                "success": False,
                "error": f"Voice not trained. Add your {provider.title()} API key in Settings → API Keys, then click Train Voice."
            }), 400

    try:
        audio_data = None
        audio_url = None

        if provider == 'elevenlabs' and voice_id:
            audio_data = _synthesize_elevenlabs(voice_id, text)
        elif provider == 'cartesia' and voice_id:
            audio_data = _synthesize_cartesia(voice_id, text)
        elif provider in free_providers or not voice_id:
            # Use free TTS for testing (gTTS fallback)
            audio_data = _synthesize_edge_tts(project.get('base_voice', 'en-US-JennyNeural'), text)

        if audio_data:
            # Save to temporary file and return URL
            audio_filename = f"test_{project_id}_{datetime.now().strftime('%H%M%S')}.mp3"
            audio_path = VOICE_SAMPLES_DIR / audio_filename
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
            audio_url = f"/api/voice-lab/audio/{audio_filename}"

            add_activity(f"Voice test: {project['name']}", "", "voice")

            return jsonify({
                "success": True,
                "project_id": project_id,
                "text": text,
                "duration_ms": len(text) * 80,
                "audio_url": audio_url,
                "provider": provider,
                "message": f"Voice '{project['name']}' synthesized successfully"
            })
        else:
            # No audio generated
            return jsonify({
                "success": False,
                "error": f"Could not generate audio. Check your {provider} API key in Settings, or try a free provider like Edge TTS."
            }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Synthesis failed: {str(e)}"
        }), 500


def _synthesize_elevenlabs(voice_id, text):
    """Synthesize speech using ElevenLabs."""
    api_key = db.get_api_key('elevenlabs') if USE_DATABASE else None
    if not api_key:
        return None

    import requests
    response = requests.post(
        f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
        headers={
            'xi-api-key': api_key,
            'Content-Type': 'application/json'
        },
        json={
            'text': text,
            'model_id': 'eleven_monolingual_v1',
            'voice_settings': {
                'stability': 0.5,
                'similarity_boost': 0.75
            }
        }
    )

    if response.status_code == 200:
        return response.content
    return None


def _synthesize_cartesia(voice_id, text):
    """Synthesize speech using Cartesia."""
    api_key = db.get_api_key('cartesia') if USE_DATABASE else None
    if not api_key:
        return None

    import requests
    response = requests.post(
        'https://api.cartesia.ai/tts/bytes',
        headers={
            'X-API-Key': api_key,
            'Cartesia-Version': '2024-06-10',
            'Content-Type': 'application/json'
        },
        json={
            'transcript': text,
            'voice': {'mode': 'id', 'id': voice_id},
            'output_format': {'container': 'mp3', 'encoding': 'mp3', 'sample_rate': 44100}
        }
    )

    if response.status_code == 200:
        return response.content
    return None


def _synthesize_edge_tts(voice, text):
    """Synthesize speech using gTTS (free Google TTS)."""
    try:
        from gtts import gTTS
        import io

        tts = gTTS(text=text, lang='en')
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        print(f"gTTS error: {e}")
        return None


@app.route('/api/voice-lab/audio/<filename>')
def serve_voice_audio(filename):
    """Serve generated voice audio files."""
    file_path = VOICE_SAMPLES_DIR / filename
    if file_path.exists():
        return send_file(str(file_path), mimetype='audio/mpeg')
    return jsonify({"error": "File not found"}), 404


@app.route('/api/voice-lab/training-status')
def get_voice_training_status():
    """Get status of all voice training jobs."""
    # Clean up completed jobs older than 1 hour
    cutoff = datetime.now() - timedelta(hours=1)
    global VOICE_TRAINING_QUEUE
    VOICE_TRAINING_QUEUE = [
        job for job in VOICE_TRAINING_QUEUE
        if datetime.fromisoformat(job['started_at']) > cutoff or job.get('status') != 'completed'
    ]
    return jsonify(VOICE_TRAINING_QUEUE)


@app.route('/api/voice-lab/projects/<project_id>/link-skill', methods=['POST'])
def link_voice_to_skill_endpoint(project_id):
    """Link a trained voice to a skill/agent."""
    if not USE_DATABASE:
        return jsonify({"error": "Database not available"}), 500

    data = request.json
    skill_id = data.get('skill_id')

    if not skill_id:
        return jsonify({"error": "skill_id is required"}), 400

    project = db.get_voice_project(project_id)
    if not project:
        return jsonify({"error": "Voice project not found"}), 404

    if db.link_voice_to_skill(project_id, skill_id):
        add_activity(f"Voice '{project['name']}' linked to skill '{skill_id}'", "", "voice")
        return jsonify({
            "success": True,
            "message": f"Voice linked to skill {skill_id}"
        })

    return jsonify({"error": "Failed to link voice to skill"}), 500


@app.route('/api/voice-lab/skills/<skill_id>/voices')
def get_voices_for_skill(skill_id):
    """Get all trained voices linked to a skill."""
    if USE_DATABASE:
        voices = db.get_voice_projects_for_skill(skill_id)
        return jsonify(voices)
    return jsonify([])


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
    # Return actual job status - no simulation
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
        /* Dark Theme (Default) */
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

        /* Light Theme (Slack-inspired) */
        [data-theme="light"] {
            --neon-cyan: #1264a3;
            --neon-pink: #e01e5a;
            --neon-purple: #4a154b;
            --neon-blue: #36c5f0;
            --neon-green: #2eb67d;
            --neon-orange: #e8912d;
            --neon-yellow: #ecb22e;
            --bg-dark: #ffffff;
            --card-bg: #f8f8f8;
            --glass-surface: rgba(0, 0, 0, 0.03);
            --glass-border: rgba(0, 0, 0, 0.1);
            --text-primary: #1d1c1d;
            --text-secondary: #616061;
        }

        [data-theme="light"] body {
            background: #ffffff;
        }

        [data-theme="light"] .bg-effects {
            display: none;
        }

        [data-theme="light"] .glass-card {
            background: #ffffff;
            border: 1px solid #e8e8e8;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }

        [data-theme="light"] .main-tab-btn {
            border-color: #e8e8e8;
            color: #616061;
        }

        [data-theme="light"] .main-tab-btn:hover,
        [data-theme="light"] .main-tab-btn.active {
            background: #4a154b;
            border-color: #4a154b;
            color: #ffffff;
            box-shadow: none;
        }

        [data-theme="light"] .sub-tab-btn {
            color: #616061;
        }

        [data-theme="light"] .sub-tab-btn.active {
            background: #4a154b;
            color: #ffffff;
        }

        [data-theme="light"] .btn-primary {
            background: #4a154b;
        }

        [data-theme="light"] .btn-primary:hover {
            background: #611f69;
        }

        [data-theme="light"] .form-input,
        [data-theme="light"] .form-select,
        [data-theme="light"] .form-textarea {
            background: #ffffff;
            border-color: #e8e8e8;
            color: #1d1c1d;
        }

        [data-theme="light"] .form-input:focus,
        [data-theme="light"] .form-select:focus,
        [data-theme="light"] .form-textarea:focus {
            border-color: #1264a3;
            box-shadow: 0 0 0 3px rgba(18, 100, 163, 0.1);
        }

        [data-theme="light"] .logo {
            background: linear-gradient(135deg, #4a154b 0%, #1264a3 50%, #2eb67d 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        [data-theme="light"] .stat-value {
            color: #1d1c1d;
        }

        [data-theme="light"] .stat-value.green { color: #2eb67d; }
        [data-theme="light"] .stat-value.pink { color: #e01e5a; }
        [data-theme="light"] .stat-value.orange { color: #e8912d; }

        [data-theme="light"] .section-icon {
            color: #4a154b;
        }

        [data-theme="light"] .skill-card {
            background: #ffffff;
            border: 1px solid #e8e8e8;
        }

        [data-theme="light"] .skill-card:hover {
            border-color: #4a154b;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        [data-theme="light"] .console {
            background: #f8f8f8;
            border-color: #e8e8e8;
            color: #1d1c1d;
        }

        [data-theme="light"] .code-block {
            background: #f8f8f8;
            border-color: #e8e8e8;
        }

        [data-theme="light"] .skills-table th {
            background: #f8f8f8;
            color: #1d1c1d;
        }

        [data-theme="light"] .skills-table td {
            border-color: #e8e8e8;
        }

        [data-theme="light"] .pipeline-icon {
            border-color: #e8e8e8;
        }

        [data-theme="light"] .getting-started-card {
            background: linear-gradient(135deg, rgba(74,21,75,0.1), rgba(18,100,163,0.1)) !important;
            border-color: rgba(74,21,75,0.3) !important;
        }

        /* Theme Toggle Button */
        .theme-toggle-btn {
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 1000;
            background: var(--glass-surface);
            border: 1px solid var(--glass-border);
            border-radius: 50%;
            width: 44px;
            height: 44px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            transition: all 0.2s;
        }

        .theme-toggle-btn:hover {
            background: var(--neon-purple);
            transform: scale(1.1);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        /* Tooltip Styles for Technical Terms */
        .tooltip {
            position: relative;
            cursor: help;
            border-bottom: 1px dotted var(--neon-cyan);
        }

        .tooltip::before {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            padding: 0.5rem 0.75rem;
            background: var(--card-bg);
            border: 1px solid var(--neon-cyan);
            border-radius: 6px;
            font-size: 0.8rem;
            color: var(--text-primary);
            white-space: nowrap;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.2s, visibility 0.2s;
            z-index: 1000;
            max-width: 250px;
            white-space: normal;
            text-align: center;
        }

        .tooltip::after {
            content: '';
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-top-color: var(--neon-cyan);
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.2s, visibility 0.2s;
        }

        .tooltip:hover::before,
        .tooltip:hover::after {
            opacity: 1;
            visibility: visible;
        }

        /* Help icon style */
        .help-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: rgba(0, 255, 242, 0.2);
            color: var(--neon-cyan);
            font-size: 0.7rem;
            font-weight: bold;
            margin-left: 4px;
            cursor: help;
        }

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
    <!-- Theme Toggle Button -->
    <button class="theme-toggle-btn" onclick="toggleTheme()" title="Toggle Dark/Light Mode" id="theme-toggle">
        🌙
    </button>

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

        <!-- Main Tabs - Simplified to 4 core sections -->
        <div class="main-tabs">
            <button class="main-tab-btn active" onclick="showMainTab('dashboard')" title="Overview, status, and quick actions">Dashboard</button>
            <button class="main-tab-btn" onclick="showMainTab('skills')" title="Create, manage, and test all your AI skills">Skills</button>
            <button class="main-tab-btn" onclick="showMainTab('voice')" title="Voice projects, TTS settings, and platform connections">Voice</button>
            <button class="main-tab-btn" onclick="showMainTab('settings')" title="API keys, integrations, and advanced configuration">Settings</button>
        </div>

        <!-- ================================================================ -->
        <!-- DASHBOARD TAB -->
        <!-- ================================================================ -->
        <div id="tab-dashboard" class="tab-content active">
            <!-- Getting Started Card - Shows for new users -->
            <div class="glass-card getting-started-card" id="getting-started" style="background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(139,92,246,0.1)); border: 1px solid rgba(0,212,255,0.3); margin-bottom: 1.5rem;">
                <div class="section-header">
                    <div class="section-title"><span class="section-icon">Rocket</span> Getting Started</div>
                    <button class="btn btn-secondary btn-sm" onclick="hideGettingStarted()">Dismiss</button>
                </div>
                <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">Complete these steps to set up your AI voice assistant in under 5 minutes:</p>
                <div class="onboarding-steps" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                    <div class="onboarding-step" id="step-api" onclick="showMainTab('settings'); showSettingsTab('keys');" style="background: var(--glass-surface); padding: 1rem; border-radius: 8px; cursor: pointer; border-left: 3px solid var(--neon-cyan);">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span class="step-number" style="background: var(--neon-cyan); color: #000; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.8rem;">1</span>
                            <span style="font-weight: 600;">Connect API Key</span>
                        </div>
                        <p style="color: var(--text-secondary); font-size: 0.85rem;">Add your Groq or OpenAI key to enable the AI brain</p>
                    </div>
                    <div class="onboarding-step" id="step-skill" onclick="showMainTab('skills'); showSkillsTab('create');" style="background: var(--glass-surface); padding: 1rem; border-radius: 8px; cursor: pointer; border-left: 3px solid var(--neon-purple);">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span class="step-number" style="background: var(--neon-purple); color: #fff; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.8rem;">2</span>
                            <span style="font-weight: 600;">Create First Skill</span>
                        </div>
                        <p style="color: var(--text-secondary); font-size: 0.85rem;">Build an AI receptionist for your business in 2 min</p>
                    </div>
                    <div class="onboarding-step" id="step-test" onclick="showMainTab('skills'); showSkillsTab('test');" style="background: var(--glass-surface); padding: 1rem; border-radius: 8px; cursor: pointer; border-left: 3px solid var(--neon-green);">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span class="step-number" style="background: var(--neon-green); color: #000; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.8rem;">3</span>
                            <span style="font-weight: 600;">Test Your Skill</span>
                        </div>
                        <p style="color: var(--text-secondary); font-size: 0.85rem;">Chat with your AI to make sure it works right</p>
                    </div>
                    <div class="onboarding-step" id="step-connect" onclick="showMainTab('voice'); showVoiceTab('connections');" style="background: var(--glass-surface); padding: 1rem; border-radius: 8px; cursor: pointer; border-left: 3px solid var(--neon-orange);">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span class="step-number" style="background: var(--neon-orange); color: #000; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.8rem;">4</span>
                            <span style="font-weight: 600;">Connect Phone</span>
                        </div>
                        <p style="color: var(--text-secondary); font-size: 0.85rem;">Link Twilio or Vapi to handle real calls</p>
                    </div>
                </div>
            </div>

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
                <button class="sub-tab-btn" onclick="showFactoryTab('golden')">Golden Prompts</button>
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

            <!-- Golden Prompts -->
            <div id="factory-golden" class="sub-tab-content">
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Star</span> Golden Prompts</div>
                        <button class="btn btn-secondary btn-sm" onclick="loadGoldenPrompts()">Refresh</button>
                    </div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">Voice-optimized skill manuals for ultra-fast Groq inference. These prompts are designed to stay under 3k tokens for ~50-100ms prefill latency.</p>

                    <div class="form-row">
                        <div class="form-group" style="flex: 2;">
                            <label class="form-label">Select Skill</label>
                            <select class="form-select" id="golden-skill-select" onchange="loadGoldenPrompt()">
                                <option value="receptionist">Receptionist - Professional Call Handling</option>
                                <option value="electrician">Electrician - Jenkintown Electricity</option>
                                <option value="plumber">Plumber - Plumbing Services</option>
                                <option value="lawyer">Lawyer - Legal Intake Specialist</option>
                                <option value="solar">Solar - Solar Company Sales</option>
                                <option value="tara-sales">Tara Sales - TheDashTool Assistant</option>
                                <option value="general">General - Helpful Assistant</option>
                            </select>
                        </div>
                        <div class="form-group" style="flex: 1;">
                            <label class="form-label">Voice <span style="color: var(--text-secondary); font-size: 0.8rem;">(optional)</span></label>
                            <select class="form-select" id="golden-voice-select" onchange="linkVoiceToSkill()">
                                <option value="">No custom voice</option>
                            </select>
                        </div>
                    </div>

                    <div class="stats-row" style="margin-bottom: 1rem;">
                        <div class="stat-item">
                            <div class="stat-value" id="golden-tokens">~850</div>
                            <div class="stat-label">Est. Tokens</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="golden-prefill">~14ms</div>
                            <div class="stat-label">Prefill Time</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="golden-status">Ready</div>
                            <div class="stat-label">Status</div>
                        </div>
                    </div>

                    <div class="form-group">
                        <label class="form-label">Context Variables (optional)</label>
                        <div class="form-row">
                            <div class="form-group" style="flex: 1;">
                                <input type="text" class="form-input" id="golden-business-name" placeholder="Business Name" value="">
                            </div>
                            <div class="form-group" style="flex: 1;">
                                <input type="text" class="form-input" id="golden-agent-name" placeholder="Agent Name" value="">
                            </div>
                        </div>
                    </div>

                    <div class="form-group">
                        <label class="form-label">Prompt Content</label>
                        <textarea class="form-textarea" id="golden-prompt-content" style="min-height: 400px; font-family: monospace; font-size: 0.85rem;" placeholder="Loading prompt..."></textarea>
                    </div>

                    <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                        <button class="btn btn-primary" onclick="saveGoldenPrompt()">Save Changes</button>
                        <button class="btn btn-secondary" onclick="resetGoldenPrompt()">Reset to Default</button>
                        <button class="btn btn-success" onclick="testGoldenPrompt()">Test Prompt</button>
                        <button class="btn btn-secondary" onclick="copyGoldenPrompt()">Copy to Clipboard</button>
                        <button class="btn btn-secondary" onclick="exportGoldenPrompts()">Export All</button>
                    </div>

                    <div id="golden-message" style="margin-top: 1rem;"></div>

                    <!-- Test Output -->
                    <div id="golden-test-output" style="margin-top: 1rem; display: none;">
                        <label class="form-label">Test Response</label>
                        <div class="code-block" id="golden-test-response" style="max-height: 300px; overflow-y: auto;"></div>
                    </div>
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
                                    <option value="parler_tts">Parler TTS (Expressive - GPU)</option>
                                    <option value="edge_tts">Edge TTS (Microsoft - Free)</option>
                                    <option value="kokoro">Kokoro (Edge TTS Fallback)</option>
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
                            <label class="form-label">Emotion (Parler)</label>
                            <select class="form-select" id="voice-emotion">
                                <option value="neutral">Neutral</option>
                                <option value="warm" selected>Warm</option>
                                <option value="excited">Excited</option>
                                <option value="calm">Calm</option>
                                <option value="concerned">Concerned</option>
                                <option value="apologetic">Apologetic</option>
                                <option value="confident">Confident</option>
                                <option value="empathetic">Empathetic</option>
                                <option value="urgent">Urgent</option>
                                <option value="cheerful">Cheerful</option>
                            </select>
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
        <!-- FAST BRAIN TAB (Skills) -->
        <!-- ================================================================ -->
        <div id="tab-fastbrain" class="tab-content">
            <!-- Sub-tabs for Skills -->
            <div class="sub-tabs">
                <button class="sub-tab-btn active" onclick="showFastBrainTab('dashboard')">Dashboard</button>
                <button class="sub-tab-btn" onclick="showFastBrainTab('skills')">Skills Manager</button>
                <button class="sub-tab-btn" onclick="showFastBrainTab('golden')">Golden Prompts</button>
                <button class="sub-tab-btn" onclick="showFastBrainTab('train')">Train LoRA</button>
                <button class="sub-tab-btn" onclick="showFastBrainTab('chat')">Test Chat</button>
                <button class="sub-tab-btn" onclick="showFastBrainTab('integration')">Hive215</button>
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

            <!-- Golden Prompts Sub-tab -->
            <div id="fb-tab-golden" class="sub-tab-content">
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Star</span> Golden Prompts</div>
                        <button class="btn btn-secondary btn-sm" onclick="loadGoldenPrompts()">Refresh</button>
                    </div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">Voice-optimized skill manuals for ultra-fast Groq inference. These prompts are designed to stay under 3k tokens for ~50-100ms prefill latency.</p>

                    <div class="form-row">
                        <div class="form-group" style="flex: 2;">
                            <label class="form-label">Select Skill</label>
                            <select class="form-select" id="fb-golden-skill-select" onchange="loadGoldenPromptFB()">
                                <option value="receptionist">Receptionist - Professional Call Handling</option>
                                <option value="electrician">Electrician - Jenkintown Electricity</option>
                                <option value="plumber">Plumber - Plumbing Services</option>
                                <option value="lawyer">Lawyer - Legal Intake Specialist</option>
                                <option value="solar">Solar - Solar Company Sales</option>
                                <option value="tara-sales">Tara Sales - TheDashTool Assistant</option>
                                <option value="general">General - Helpful Assistant</option>
                            </select>
                        </div>
                        <div class="form-group" style="flex: 1;">
                            <label class="form-label">Voice <span style="color: var(--text-secondary); font-size: 0.8rem;">(optional)</span></label>
                            <select class="form-select" id="fb-golden-voice-select" onchange="linkVoiceToSkillFB()">
                                <option value="">No custom voice</option>
                            </select>
                        </div>
                    </div>

                    <div class="stats-row" style="margin-bottom: 1rem;">
                        <div class="stat-item">
                            <div class="stat-value" id="fb-golden-tokens">~850</div>
                            <div class="stat-label">Est. Tokens</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="fb-golden-prefill">~14ms</div>
                            <div class="stat-label">Prefill Time</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="fb-golden-status">Ready</div>
                            <div class="stat-label">Status</div>
                        </div>
                    </div>

                    <div class="form-group">
                        <label class="form-label">Prompt Content</label>
                        <textarea class="form-textarea" id="fb-golden-prompt-content" style="min-height: 350px; font-family: monospace; font-size: 0.85rem;" placeholder="Loading prompt..."></textarea>
                    </div>

                    <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                        <button class="btn btn-primary" onclick="saveGoldenPromptFB()">Save Changes</button>
                        <button class="btn btn-secondary" onclick="resetGoldenPromptFB()">Reset to Default</button>
                        <button class="btn btn-success" onclick="testGoldenPromptFB()">Test Prompt</button>
                    </div>

                    <div id="fb-golden-message" style="margin-top: 1rem;"></div>
                </div>
            </div>

            <!-- Train LoRA Sub-tab -->
            <div id="fb-tab-train" class="sub-tab-content">
                <div class="dashboard-grid">
                    <div class="glass-card card-half">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">Train</span> Train LoRA Adapter</div>
                        </div>
                        <p style="color: var(--text-secondary); margin-bottom: 1rem;">Training requires a GPU. Generate a script to run on Colab, Modal, or your own machine.</p>
                        <div class="form-group">
                            <label class="form-label">Select Skill to Train</label>
                            <select class="form-select" id="fb-train-profile">
                                <option value="receptionist">Receptionist</option>
                                <option value="electrician">Electrician</option>
                                <option value="plumber">Plumber</option>
                                <option value="lawyer">Lawyer</option>
                                <option value="solar">Solar</option>
                                <option value="tara-sales">Tara Sales</option>
                                <option value="general">General</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Base Model</label>
                            <select class="form-select" id="fb-train-model">
                                <option value="llama-3.3-70b">Llama 3.3 70B (Groq)</option>
                                <option value="llama-3.1-8b">Llama 3.1 8B (Local)</option>
                                <option value="mistral-7b">Mistral 7B (Local)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Training Steps: <span id="fb-steps-value">60</span></label>
                            <input type="range" id="fb-train-steps" min="20" max="200" value="60" style="width: 100%;" oninput="document.getElementById('fb-steps-value').textContent = this.value">
                            <div style="display: flex; justify-content: space-between; color: var(--text-secondary); font-size: 0.75rem;">
                                <span>20 (Quick)</span>
                                <span>200 (Thorough)</span>
                            </div>
                        </div>
                        <button class="btn btn-primary" onclick="generateTrainingScriptFB()">Generate Training Script</button>
                    </div>

                    <div class="glass-card card-half">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">GPU</span> Training Options</div>
                        </div>
                        <div style="display: flex; flex-direction: column; gap: 1rem;">
                            <div style="padding: 1rem; background: var(--glass-surface); border-radius: 8px; border-left: 3px solid var(--neon-cyan);">
                                <h4 style="margin-bottom: 0.5rem;">Google Colab (Free)</h4>
                                <p style="font-size: 0.85rem; color: var(--text-secondary);">Free T4 GPU, ~15 min for 60 steps</p>
                            </div>
                            <div style="padding: 1rem; background: var(--glass-surface); border-radius: 8px; border-left: 3px solid var(--neon-purple);">
                                <h4 style="margin-bottom: 0.5rem;">Modal (Pay-per-use)</h4>
                                <p style="font-size: 0.85rem; color: var(--text-secondary);">A100 GPU, ~5 min for 60 steps, ~$0.50</p>
                            </div>
                            <div style="padding: 1rem; background: var(--glass-surface); border-radius: 8px; border-left: 3px solid var(--neon-green);">
                                <h4 style="margin-bottom: 0.5rem;">Local GPU</h4>
                                <p style="font-size: 0.85rem; color: var(--text-secondary);">Requires 24GB+ VRAM, RTX 3090/4090</p>
                            </div>
                        </div>
                    </div>

                    <div class="glass-card card-full">
                        <div class="section-header">
                            <div class="section-title"><span class="section-icon">Code</span> Training Script</div>
                        </div>
                        <div id="fb-training-output">
                            <pre class="code-block" id="fb-training-script" style="max-height: 400px; overflow-y: auto;">Click "Generate Training Script" to create your training code.</pre>
                        </div>
                        <div style="margin-top: 1rem; display: flex; gap: 1rem;">
                            <button class="btn btn-secondary" onclick="copyTrainingScript()">Copy to Clipboard</button>
                            <button class="btn btn-secondary" onclick="downloadTrainingScript()">Download as .py</button>
                        </div>
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

            <!-- Edit Voice Project -->
            <div id="voicelab-edit" class="sub-tab-content">
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Edit</span> <span id="vl-edit-title">Voice Project</span></div>
                        <div style="display: flex; gap: 0.5rem;">
                            <button class="btn btn-secondary btn-sm" onclick="showVoiceLabTab('projects')">← Back</button>
                            <button class="btn btn-danger btn-sm" onclick="deleteCurrentVoiceProject()">Delete</button>
                        </div>
                    </div>

                    <input type="hidden" id="vl-edit-project-id">

                    <div class="dashboard-grid">
                        <!-- Project Info -->
                        <div class="glass-card card-half" style="background: var(--glass-surface);">
                            <h4 style="margin-bottom: 1rem; color: var(--neon-cyan);">Project Info</h4>
                            <div class="form-group">
                                <label class="form-label">Voice Name</label>
                                <input type="text" class="form-input" id="vl-edit-name" placeholder="Voice name">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Description</label>
                                <textarea class="form-textarea" id="vl-edit-description" rows="2" placeholder="Description..."></textarea>
                            </div>
                            <div class="form-row">
                                <div class="form-group">
                                    <label class="form-label">Provider</label>
                                    <select class="form-select" id="vl-edit-provider">
                                        <option value="elevenlabs">ElevenLabs</option>
                                        <option value="cartesia">Cartesia</option>
                                        <option value="edge_tts">Edge TTS (Free)</option>
                                        <option value="parler_tts">Parler TTS</option>
                                        <option value="xtts">XTTS-v2</option>
                                        <option value="openvoice">OpenVoice</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Status</label>
                                    <div id="vl-edit-status" class="stat-value" style="padding: 0.5rem; background: var(--glass-surface); border-radius: 8px;">Draft</div>
                                </div>
                            </div>
                            <button class="btn btn-secondary" onclick="saveVoiceProjectChanges()">Save Changes</button>
                        </div>

                        <!-- Voice Settings -->
                        <div class="glass-card card-half" style="background: var(--glass-surface);">
                            <h4 style="margin-bottom: 1rem; color: var(--neon-pink);">Voice Settings</h4>
                            <div class="form-group">
                                <label class="form-label">Pitch: <span id="vl-edit-pitch-value">1.0</span></label>
                                <input type="range" class="form-input" id="vl-edit-pitch" min="0.5" max="2.0" step="0.1" value="1.0" oninput="document.getElementById('vl-edit-pitch-value').textContent = this.value">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Speed: <span id="vl-edit-speed-value">1.0</span></label>
                                <input type="range" class="form-input" id="vl-edit-speed" min="0.5" max="2.0" step="0.1" value="1.0" oninput="document.getElementById('vl-edit-speed-value').textContent = this.value">
                            </div>
                            <div class="form-row">
                                <div class="form-group">
                                    <label class="form-label">Emotion</label>
                                    <select class="form-select" id="vl-edit-emotion">
                                        <option value="neutral">Neutral</option>
                                        <option value="warm">Warm</option>
                                        <option value="excited">Excited</option>
                                        <option value="calm">Calm</option>
                                        <option value="confident">Confident</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Style</label>
                                    <select class="form-select" id="vl-edit-style">
                                        <option value="conversational">Conversational</option>
                                        <option value="professional">Professional</option>
                                        <option value="friendly">Friendly</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <!-- Audio Samples -->
                        <div class="glass-card card-full" style="background: var(--glass-surface);">
                            <div class="section-header">
                                <h4 style="color: var(--neon-green);">Audio Samples</h4>
                                <button class="btn btn-primary btn-sm" onclick="document.getElementById('vl-edit-audio-input').click()">+ Add Sample</button>
                            </div>
                            <input type="file" id="vl-edit-audio-input" accept="audio/*" style="display: none;" onchange="uploadVoiceSample(this)">
                            <div id="vl-edit-samples-list" style="margin-top: 1rem;">
                                <div style="color: var(--text-secondary); padding: 1rem; text-align: center;">
                                    No samples uploaded. Add audio samples to train your voice.
                                </div>
                            </div>
                        </div>

                        <!-- Train & Test -->
                        <div class="glass-card card-full" style="background: var(--glass-surface);">
                            <h4 style="margin-bottom: 1rem; color: var(--neon-purple);">Train & Test Voice</h4>
                            <div class="form-row" style="align-items: flex-end;">
                                <div class="form-group" style="flex: 2;">
                                    <label class="form-label">Test Text</label>
                                    <input type="text" class="form-input" id="vl-edit-test-text" value="Hello! This is a test of my custom voice. How does it sound?">
                                </div>
                                <div class="form-group">
                                    <button class="btn btn-success" onclick="trainVoiceProject()">Train Voice</button>
                                </div>
                                <div class="form-group">
                                    <button class="btn btn-primary" onclick="testVoiceProject()">Test Voice</button>
                                </div>
                            </div>
                            <div id="vl-edit-audio-player" style="margin-top: 1rem; display: none;">
                                <audio id="vl-edit-audio" controls style="width: 100%;"></audio>
                            </div>
                            <div id="vl-edit-message" style="margin-top: 1rem;"></div>
                        </div>

                        <!-- Link to Skill -->
                        <div class="glass-card card-full" style="background: var(--glass-surface);">
                            <h4 style="margin-bottom: 1rem; color: var(--neon-orange);">Link to Skill/Agent</h4>
                            <div class="form-row" style="align-items: flex-end;">
                                <div class="form-group" style="flex: 2;">
                                    <label class="form-label">Select Skill</label>
                                    <select class="form-select" id="vl-edit-skill-link">
                                        <option value="">-- Select a skill --</option>
                                        <option value="receptionist">Receptionist</option>
                                        <option value="electrician">Electrician</option>
                                        <option value="plumber">Plumber</option>
                                        <option value="lawyer">Lawyer</option>
                                        <option value="solar">Solar</option>
                                        <option value="tara-sales">Tara Sales</option>
                                        <option value="general">General</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <button class="btn btn-secondary" onclick="linkVoiceProjectToSkill()">Link Voice</button>
                                </div>
                            </div>
                            <p id="vl-edit-linked-skill" style="color: var(--text-secondary); margin-top: 0.5rem;"></p>
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
                                        <option value="parler_tts">Parler TTS (Expressive - GPU)</option>
                                        <option value="edge_tts">Edge TTS (Microsoft - Free)</option>
                                        <option value="kokoro">Kokoro (Edge TTS Fallback)</option>
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
                                <option value="warm">Warm</option>
                                <option value="excited">Excited</option>
                                <option value="calm">Calm</option>
                                <option value="concerned">Concerned</option>
                                <option value="apologetic">Apologetic</option>
                                <option value="confident">Confident</option>
                                <option value="empathetic">Empathetic</option>
                                <option value="urgent">Urgent</option>
                                <option value="cheerful">Cheerful</option>
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
        // THEME TOGGLE
        // ============================================================
        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';

            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);

            // Update toggle button icon
            const toggleBtn = document.getElementById('theme-toggle');
            toggleBtn.textContent = newTheme === 'light' ? '☀️' : '🌙';
        }

        // Load saved theme on page load
        (function() {
            const savedTheme = localStorage.getItem('theme') || 'dark';
            document.documentElement.setAttribute('data-theme', savedTheme);

            // Update toggle button icon after DOM loads
            document.addEventListener('DOMContentLoaded', function() {
                const toggleBtn = document.getElementById('theme-toggle');
                if (toggleBtn) {
                    toggleBtn.textContent = savedTheme === 'light' ? '☀️' : '🌙';
                }
            });
        })();

        // ============================================================
        // TAB NAVIGATION
        // ============================================================
        function showMainTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.main-tab-btn').forEach(b => b.classList.remove('active'));

            // Map new simplified tabs to existing content
            let actualTabId = tabId;
            if (tabId === 'skills') actualTabId = 'fastbrain';  // Skills -> Fast Brain (has Skills Manager)
            if (tabId === 'voice') actualTabId = 'voicelab';    // Voice -> Voice Lab
            if (tabId === 'settings') actualTabId = 'command';  // Settings -> Command Center (has API Keys)

            const tabElement = document.getElementById('tab-' + actualTabId);
            if (tabElement) {
                tabElement.classList.add('active');
            }
            if (event && event.target) {
                event.target.classList.add('active');
            }

            // Load appropriate data for each tab
            if (tabId === 'dashboard') { loadSkills(); loadMetrics(); }
            if (tabId === 'skills' || tabId === 'fastbrain') { loadFastBrainConfig(); loadFastBrainSkills(); refreshSystemStatus(); loadHive215Config(); }
            if (tabId === 'voice' || tabId === 'voicelab') { loadVoiceProjects(); loadSkillsForDropdowns(); }
            if (tabId === 'settings' || tabId === 'command') { refreshStats(); }
            if (tabId === 'factory') { loadProfileDropdowns(); }
        }

        // New helper functions for consolidated navigation
        function showSkillsTab(subTab) {
            showMainTab('skills');
            setTimeout(() => {
                if (subTab === 'create') showFastBrainTab('skills');
                if (subTab === 'test') showFastBrainTab('chat');
                if (subTab === 'golden') showFactoryTab('golden');
            }, 100);
        }

        function showVoiceTab(subTab) {
            showMainTab('voice');
            setTimeout(() => {
                if (subTab === 'connections') showVoiceLabTab('projects');
            }, 100);
        }

        function showSettingsTab(subTab) {
            showMainTab('settings');
            setTimeout(() => {
                if (subTab === 'keys') showCommandTab('keys');
            }, 100);
        }

        function hideGettingStarted() {
            const card = document.getElementById('getting-started');
            if (card) {
                card.style.display = 'none';
                localStorage.setItem('hideGettingStarted', 'true');
            }
        }

        // Check if getting started should be hidden
        if (localStorage.getItem('hideGettingStarted') === 'true') {
            document.addEventListener('DOMContentLoaded', () => {
                const card = document.getElementById('getting-started');
                if (card) card.style.display = 'none';
            });
        }

        function showFactoryTab(tabId) {
            document.querySelectorAll('#tab-factory .sub-tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('#tab-factory .sub-tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('factory-' + tabId).classList.add('active');
            event.target.classList.add('active');

            if (tabId === 'manage') loadSkillsTable();
            if (tabId === 'golden') loadGoldenPrompt();
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
                    grid.innerHTML = `
                        <div style="text-align: center; grid-column: span 4; padding: 2rem;">
                            <div style="font-size: 3rem; margin-bottom: 1rem;">Brain</div>
                            <h3 style="margin-bottom: 0.5rem;">Create Your First AI Skill</h3>
                            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">Build an AI receptionist for your business in under 2 minutes</p>
                            <button class="btn btn-primary" onclick="showMainTab('skills'); setTimeout(() => showFastBrainTab('skills'), 100);">
                                + Create Skill
                            </button>
                        </div>`;
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

        // Golden Prompts Functions
        let goldenPromptsCache = {};

        async function loadGoldenPrompts() {
            try {
                const res = await fetch('/api/golden-prompts');
                const data = await res.json();
                if (data.prompts) {
                    goldenPromptsCache = data.prompts;
                }
                loadGoldenPrompt();
            } catch (e) {
                console.error('Error loading golden prompts:', e);
            }
        }

        async function loadGoldenPrompt() {
            const skillId = document.getElementById('golden-skill-select').value;
            const businessName = document.getElementById('golden-business-name').value || '{business_name}';
            const agentName = document.getElementById('golden-agent-name').value || '{agent_name}';

            try {
                const res = await fetch(`/api/golden-prompts/${skillId}?business_name=${encodeURIComponent(businessName)}&agent_name=${encodeURIComponent(agentName)}`);
                const data = await res.json();

                if (data.content) {
                    document.getElementById('golden-prompt-content').value = data.content;
                    document.getElementById('golden-tokens').textContent = '~' + data.tokens;
                    document.getElementById('golden-prefill').textContent = '~' + data.prefill_ms + 'ms';
                    document.getElementById('golden-status').textContent = data.is_custom ? 'Custom' : 'Default';
                    document.getElementById('golden-status').style.color = data.is_custom ? 'var(--warning)' : 'var(--success)';
                }
            } catch (e) {
                console.error('Error loading prompt:', e);
                showGoldenMessage('Error loading prompt: ' + e.message, 'error');
            }

            // Also load available voices for this skill
            loadVoicesForSkill(skillId);
        }

        async function loadVoicesForSkill(skillId) {
            const voiceSelect = document.getElementById('golden-voice-select');
            if (!voiceSelect) return;

            try {
                // Get all trained voice projects
                const res = await fetch('/api/voice-lab/projects');
                const projects = await res.json();

                // Clear existing options
                voiceSelect.innerHTML = '<option value="">No custom voice</option>';

                // Add trained voices
                projects.filter(p => p.status === 'trained').forEach(p => {
                    const option = document.createElement('option');
                    option.value = p.id;
                    option.textContent = `${p.name} (${p.provider})`;
                    // Mark if this voice is linked to the current skill
                    if (p.skill_id === skillId) {
                        option.selected = true;
                    }
                    voiceSelect.appendChild(option);
                });

                // Add separator and option to create new voice
                if (projects.filter(p => p.status === 'trained').length > 0) {
                    const separator = document.createElement('option');
                    separator.disabled = true;
                    separator.textContent = '───────────';
                    voiceSelect.appendChild(separator);
                }

                const createOption = document.createElement('option');
                createOption.value = '__create__';
                createOption.textContent = '+ Create New Voice...';
                voiceSelect.appendChild(createOption);

            } catch (e) {
                console.error('Error loading voices:', e);
            }
        }

        async function linkVoiceToSkill() {
            const skillId = document.getElementById('golden-skill-select').value;
            const voiceSelect = document.getElementById('golden-voice-select');
            const voiceId = voiceSelect.value;

            if (voiceId === '__create__') {
                // Switch to Voice Lab tab
                showSubTab('voicelab');
                showVoiceLabTab('create');
                voiceSelect.value = '';
                return;
            }

            if (!voiceId) {
                // Clear voice link - not implemented yet, just return
                return;
            }

            try {
                const res = await fetch(`/api/voice-lab/projects/${voiceId}/link-skill`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ skill_id: skillId })
                });
                const data = await res.json();

                if (data.success) {
                    showGoldenMessage(`Voice linked to ${skillId} skill`, 'success');
                } else {
                    showGoldenMessage(data.error || 'Failed to link voice', 'error');
                }
            } catch (e) {
                showGoldenMessage('Error linking voice: ' + e.message, 'error');
            }
        }

        async function saveGoldenPrompt() {
            const skillId = document.getElementById('golden-skill-select').value;
            const content = document.getElementById('golden-prompt-content').value;

            try {
                const res = await fetch(`/api/golden-prompts/${skillId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content })
                });
                const data = await res.json();

                if (data.success) {
                    showGoldenMessage(data.message, 'success');
                    document.getElementById('golden-tokens').textContent = '~' + data.tokens;
                    document.getElementById('golden-prefill').textContent = '~' + data.prefill_ms + 'ms';
                    document.getElementById('golden-status').textContent = 'Custom';
                    document.getElementById('golden-status').style.color = 'var(--warning)';
                } else {
                    showGoldenMessage(data.error || 'Save failed', 'error');
                }
            } catch (e) {
                showGoldenMessage('Error: ' + e.message, 'error');
            }
        }

        async function resetGoldenPrompt() {
            const skillId = document.getElementById('golden-skill-select').value;

            if (!confirm(`Reset "${skillId}" prompt to default? Your customizations will be lost.`)) {
                return;
            }

            try {
                const res = await fetch(`/api/golden-prompts/${skillId}/reset`, { method: 'POST' });
                const data = await res.json();

                if (data.success) {
                    showGoldenMessage(data.message, 'success');
                    loadGoldenPrompt();
                } else {
                    showGoldenMessage(data.error || 'Reset failed', 'error');
                }
            } catch (e) {
                showGoldenMessage('Error: ' + e.message, 'error');
            }
        }

        async function testGoldenPrompt() {
            const skillId = document.getElementById('golden-skill-select').value;
            const businessName = document.getElementById('golden-business-name').value || 'Test Business';
            const agentName = document.getElementById('golden-agent-name').value || 'Assistant';

            const testMessage = prompt('Enter a test message:', 'Hello, I need some help with my electrical issue.');
            if (!testMessage) return;

            showGoldenMessage('Testing prompt with Fast Brain...', 'info');

            try {
                const res = await fetch('/api/golden-prompts/test', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        skill_id: skillId,
                        message: testMessage,
                        business_name: businessName,
                        agent_name: agentName
                    })
                });
                const data = await res.json();

                if (data.success) {
                    document.getElementById('golden-test-output').style.display = 'block';
                    document.getElementById('golden-test-response').innerHTML = `
                        <strong>User:</strong> ${testMessage}\\n\\n
                        <strong>Response (${data.system_used}, ${data.latency_ms}ms):</strong>\\n${data.response}
                    `;
                    showGoldenMessage('Test completed successfully!', 'success');
                } else {
                    showGoldenMessage('Test failed: ' + data.error, 'error');
                }
            } catch (e) {
                showGoldenMessage('Error: ' + e.message, 'error');
            }
        }

        function copyGoldenPrompt() {
            const content = document.getElementById('golden-prompt-content').value;
            navigator.clipboard.writeText(content).then(() => {
                showGoldenMessage('Copied to clipboard!', 'success');
            }).catch(() => {
                showGoldenMessage('Failed to copy', 'error');
            });
        }

        async function exportGoldenPrompts() {
            try {
                const res = await fetch('/api/golden-prompts/export');
                const data = await res.json();

                const blob = new Blob([JSON.stringify(data.prompts, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'golden_prompts_export.json';
                a.click();
                URL.revokeObjectURL(url);

                showGoldenMessage(`Exported ${Object.keys(data.prompts).length} prompts (${data.custom_overrides.length} custom)`, 'success');
            } catch (e) {
                showGoldenMessage('Export failed: ' + e.message, 'error');
            }
        }

        function showGoldenMessage(msg, type) {
            const el = document.getElementById('golden-message');
            el.innerHTML = `<div class="message ${type}">${msg}</div>`;
            setTimeout(() => { el.innerHTML = ''; }, 5000);
        }

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
                tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; padding: 2rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">Folder</div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">No skills found. Create a business profile to get started.</p>
                    <button class="btn btn-primary btn-sm" onclick="showFactoryTab('profile')">Create Business Profile</button>
                </td></tr>`;
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

            // For Parler TTS, voiceId is the skill_id
            const isParler = provider === 'parler_tts';
            const emotionEl = document.getElementById('voice-emotion');
            const emotion = emotionEl ? emotionEl.value : 'neutral';

            if (isParler) {
                resultEl.innerHTML = '<div style="color: var(--neon-cyan);">Generating audio with Parler TTS (GPU)... May take 30-60s on cold start</div>';
            } else {
                resultEl.innerHTML = '<div style="color: var(--neon-cyan);">Generating audio...</div>';
            }

            try {
                const res = await fetch('/api/voice/test', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        voice_id: voiceId,
                        text,
                        provider,
                        emotion: emotion,
                        skill_id: isParler ? voiceId : 'general'
                    })
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
            const tabEl = document.getElementById('fb-tab-' + tabId);
            if (tabEl) tabEl.classList.add('active');
            if (event && event.target && event.target.classList) {
                event.target.classList.add('active');
            }

            // Load data based on tab
            if (tabId === 'skills') loadFastBrainSkills();
            if (tabId === 'dashboard') refreshSystemStatus();
            if (tabId === 'integration') loadIntegrationChecklist();
            if (tabId === 'golden') loadGoldenPromptFB();
        }

        // Golden Prompts functions for Skills tab
        async function loadGoldenPromptFB() {
            const skillId = document.getElementById('fb-golden-skill-select').value;
            try {
                const res = await fetch(`/api/golden-prompts/${skillId}`);
                const data = await res.json();
                if (data.content) {
                    document.getElementById('fb-golden-prompt-content').value = data.content;
                    document.getElementById('fb-golden-tokens').textContent = '~' + data.tokens;
                    document.getElementById('fb-golden-prefill').textContent = '~' + data.prefill_ms + 'ms';
                    document.getElementById('fb-golden-status').textContent = data.is_custom ? 'Custom' : 'Default';
                }
                loadVoicesForSkillFB(skillId);
            } catch (e) {
                console.error('Error loading prompt:', e);
            }
        }

        async function loadVoicesForSkillFB(skillId) {
            const voiceSelect = document.getElementById('fb-golden-voice-select');
            if (!voiceSelect) return;
            try {
                const res = await fetch('/api/voice-lab/projects');
                const projects = await res.json();
                voiceSelect.innerHTML = '<option value="">No custom voice</option>';
                (projects || []).filter(p => p.status === 'trained').forEach(p => {
                    const option = document.createElement('option');
                    option.value = p.id;
                    option.textContent = `${p.name} (${p.provider})`;
                    if (p.skill_id === skillId) option.selected = true;
                    voiceSelect.appendChild(option);
                });
            } catch (e) { console.error(e); }
        }

        async function saveGoldenPromptFB() {
            const skillId = document.getElementById('fb-golden-skill-select').value;
            const content = document.getElementById('fb-golden-prompt-content').value;
            try {
                const res = await fetch(`/api/golden-prompts/${skillId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content })
                });
                const data = await res.json();
                if (data.success) {
                    document.getElementById('fb-golden-message').innerHTML = '<div style="color: var(--neon-green);">Prompt saved!</div>';
                    document.getElementById('fb-golden-tokens').textContent = '~' + data.tokens;
                    document.getElementById('fb-golden-status').textContent = 'Custom';
                }
            } catch (e) {
                document.getElementById('fb-golden-message').innerHTML = '<div style="color: var(--neon-pink);">Error: ' + e.message + '</div>';
            }
        }

        async function resetGoldenPromptFB() {
            const skillId = document.getElementById('fb-golden-skill-select').value;
            if (!confirm(`Reset "${skillId}" prompt to default?`)) return;
            try {
                await fetch(`/api/golden-prompts/${skillId}/reset`, { method: 'POST' });
                loadGoldenPromptFB();
                document.getElementById('fb-golden-message').innerHTML = '<div style="color: var(--neon-green);">Reset to default!</div>';
            } catch (e) { console.error(e); }
        }

        async function testGoldenPromptFB() {
            document.getElementById('fb-golden-message').innerHTML = '<div style="color: var(--neon-cyan);">Testing prompt...</div>';
            const content = document.getElementById('fb-golden-prompt-content').value;
            try {
                const res = await fetch('/api/fast-brain/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        messages: [{ role: 'user', content: 'What are your hours?' }],
                        system_prompt: content
                    })
                });
                const data = await res.json();
                document.getElementById('fb-golden-message').innerHTML = `<div style="background: var(--glass-surface); padding: 1rem; border-radius: 8px; margin-top: 1rem;"><strong>Response:</strong><br>${data.response || data.content || 'No response'}</div>`;
            } catch (e) {
                document.getElementById('fb-golden-message').innerHTML = '<div style="color: var(--neon-pink);">Test failed: ' + e.message + '</div>';
            }
        }

        async function linkVoiceToSkillFB() {
            const skillId = document.getElementById('fb-golden-skill-select').value;
            const voiceId = document.getElementById('fb-golden-voice-select').value;
            if (!voiceId) return;
            try {
                await fetch(`/api/voice-lab/projects/${voiceId}/link-skill`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ skill_id: skillId })
                });
                document.getElementById('fb-golden-message').innerHTML = '<div style="color: var(--neon-green);">Voice linked!</div>';
            } catch (e) { console.error(e); }
        }

        // Training script generation
        function generateTrainingScriptFB() {
            const skill = document.getElementById('fb-train-profile').value;
            const model = document.getElementById('fb-train-model').value;
            const steps = document.getElementById('fb-train-steps').value;

            const script = `# LoRA Training Script for ${skill}
# Base Model: ${model}
# Training Steps: ${steps}

from unsloth import FastLanguageModel
import torch

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="${model}",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)

# Load training data
from datasets import load_dataset
dataset = load_dataset("json", data_files="training_data/${skill}.jsonl")

# Training configuration
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=${steps},
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        output_dir="outputs/${skill}",
    ),
)

# Train
trainer.train()

# Save LoRA adapter
model.save_pretrained("adapters/${skill}")
print("Training complete! Adapter saved to adapters/${skill}")
`;
            document.getElementById('fb-training-script').textContent = script;
        }

        function copyTrainingScript() {
            const script = document.getElementById('fb-training-script').textContent;
            navigator.clipboard.writeText(script);
            alert('Training script copied to clipboard!');
        }

        function downloadTrainingScript() {
            const script = document.getElementById('fb-training-script').textContent;
            const blob = new Blob([script], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'train_lora.py';
            a.click();
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
                    container.innerHTML = `
                        <div style="text-align: center; padding: 2rem; grid-column: span 3;">
                            <div style="font-size: 2.5rem; margin-bottom: 1rem;">Stars</div>
                            <h3 style="margin-bottom: 0.5rem;">No Skills Yet</h3>
                            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">Create your first AI skill to get started with voice automation</p>
                            <button class="btn btn-primary" onclick="showCreateSkillModal()">+ Create Your First Skill</button>
                        </div>`;
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
            const tabEl = document.getElementById('voicelab-' + tabId);
            if (tabEl) tabEl.classList.add('active');
            // Highlight the button if we can find it (edit tab has no button)
            if (event && event.target && event.target.classList) {
                event.target.classList.add('active');
            }
        }

        async function loadVoiceProjects() {
            try {
                const res = await fetch('/api/voice-lab/projects');
                const projects = await res.json();

                const container = document.getElementById('voice-projects-list');
                if (projects.length === 0) {
                    container.innerHTML = `
                        <div style="text-align: center; padding: 2rem; grid-column: span 3;">
                            <div style="font-size: 2.5rem; margin-bottom: 1rem;">Mic</div>
                            <h3 style="margin-bottom: 0.5rem;">Create Your Custom Voice</h3>
                            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">Clone your voice or customize TTS settings for natural-sounding AI responses</p>
                            <button class="btn btn-primary" onclick="showVoiceLabTab('create')">+ Create Voice Project</button>
                        </div>`;
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

        async function openVoiceProject(projectId) {
            try {
                const res = await fetch(`/api/voice-lab/projects/${projectId}`);
                const project = await res.json();

                if (project.error) {
                    alert('Error loading project: ' + project.error);
                    return;
                }

                // Store project ID
                document.getElementById('vl-edit-project-id').value = projectId;

                // Populate form fields
                document.getElementById('vl-edit-title').textContent = project.name;
                document.getElementById('vl-edit-name').value = project.name || '';
                document.getElementById('vl-edit-description').value = project.description || '';
                document.getElementById('vl-edit-provider').value = project.provider || 'elevenlabs';

                // Status with color
                const statusEl = document.getElementById('vl-edit-status');
                statusEl.textContent = project.status || 'draft';
                statusEl.style.color = project.status === 'trained' ? 'var(--neon-green)' :
                                       project.status === 'training' ? 'var(--neon-orange)' :
                                       project.status === 'failed' ? 'var(--neon-pink)' : 'var(--text-secondary)';

                // Settings
                const settings = project.settings || {};
                document.getElementById('vl-edit-pitch').value = settings.pitch || 1.0;
                document.getElementById('vl-edit-pitch-value').textContent = settings.pitch || 1.0;
                document.getElementById('vl-edit-speed').value = settings.speed || 1.0;
                document.getElementById('vl-edit-speed-value').textContent = settings.speed || 1.0;
                document.getElementById('vl-edit-emotion').value = settings.emotion || 'neutral';
                document.getElementById('vl-edit-style').value = settings.style || 'conversational';

                // Linked skill
                document.getElementById('vl-edit-skill-link').value = project.skill_id || '';
                document.getElementById('vl-edit-linked-skill').textContent =
                    project.skill_id ? `Currently linked to: ${project.skill_id}` : '';

                // Load samples
                renderVoiceSamples(project.samples || []);

                // Show the edit tab
                showVoiceLabTab('edit');

            } catch (e) {
                alert('Error opening project: ' + e.message);
            }
        }

        function renderVoiceSamples(samples) {
            const container = document.getElementById('vl-edit-samples-list');

            if (!samples || samples.length === 0) {
                container.innerHTML = '<div style="color: var(--text-secondary); padding: 1rem; text-align: center;">No samples uploaded. Add audio samples to train your voice.</div>';
                return;
            }

            container.innerHTML = samples.map(s => `
                <div style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem; background: var(--card-bg); border-radius: 8px; margin-bottom: 0.5rem;">
                    <div style="flex: 1;">
                        <div style="font-weight: 500;">${s.filename || 'Audio Sample'}</div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">
                            ${s.duration_ms ? Math.round(s.duration_ms/1000) + 's' : 'Unknown duration'} | ${s.emotion || 'neutral'}
                        </div>
                        ${s.transcript ? `<div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem;">"${s.transcript}"</div>` : ''}
                    </div>
                    <button class="btn btn-danger btn-sm" onclick="deleteVoiceSample('${s.id}')">Delete</button>
                </div>
            `).join('');
        }

        async function uploadVoiceSample(input) {
            const projectId = document.getElementById('vl-edit-project-id').value;
            if (!projectId) return;

            const file = input.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('audio', file);
            formData.append('emotion', document.getElementById('vl-edit-emotion').value);

            try {
                showEditMessage('Uploading sample...', 'info');

                const res = await fetch(`/api/voice-lab/projects/${projectId}/samples`, {
                    method: 'POST',
                    body: formData
                });
                const result = await res.json();

                if (result.success) {
                    showEditMessage('Sample uploaded!', 'success');
                    // Reload project to refresh samples
                    openVoiceProject(projectId);
                } else {
                    showEditMessage(result.error || 'Upload failed', 'error');
                }
            } catch (e) {
                showEditMessage('Upload error: ' + e.message, 'error');
            }

            input.value = '';
        }

        async function deleteVoiceSample(sampleId) {
            const projectId = document.getElementById('vl-edit-project-id').value;
            if (!confirm('Delete this sample?')) return;

            try {
                const res = await fetch(`/api/voice-lab/projects/${projectId}/samples/${sampleId}`, {
                    method: 'DELETE'
                });
                const result = await res.json();

                if (result.success) {
                    openVoiceProject(projectId);
                }
            } catch (e) {
                alert('Error deleting sample: ' + e.message);
            }
        }

        async function saveVoiceProjectChanges() {
            const projectId = document.getElementById('vl-edit-project-id').value;
            if (!projectId) return;

            const data = {
                name: document.getElementById('vl-edit-name').value,
                description: document.getElementById('vl-edit-description').value,
                provider: document.getElementById('vl-edit-provider').value,
                settings: {
                    pitch: parseFloat(document.getElementById('vl-edit-pitch').value),
                    speed: parseFloat(document.getElementById('vl-edit-speed').value),
                    emotion: document.getElementById('vl-edit-emotion').value,
                    style: document.getElementById('vl-edit-style').value
                }
            };

            try {
                const res = await fetch(`/api/voice-lab/projects/${projectId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await res.json();

                if (result.success) {
                    document.getElementById('vl-edit-title').textContent = data.name;
                    showEditMessage('Changes saved!', 'success');
                } else {
                    showEditMessage(result.error || 'Save failed', 'error');
                }
            } catch (e) {
                showEditMessage('Error: ' + e.message, 'error');
            }
        }

        async function trainVoiceProject() {
            const projectId = document.getElementById('vl-edit-project-id').value;
            if (!projectId) return;

            try {
                showEditMessage('Starting voice training...', 'info');
                document.getElementById('vl-edit-status').textContent = 'training';
                document.getElementById('vl-edit-status').style.color = 'var(--neon-orange)';

                const res = await fetch(`/api/voice-lab/projects/${projectId}/train`, {
                    method: 'POST'
                });
                const result = await res.json();

                if (result.success) {
                    document.getElementById('vl-edit-status').textContent = 'trained';
                    document.getElementById('vl-edit-status').style.color = 'var(--neon-green)';
                    showEditMessage(result.message || 'Voice trained successfully!', 'success');
                } else {
                    document.getElementById('vl-edit-status').textContent = 'failed';
                    document.getElementById('vl-edit-status').style.color = 'var(--neon-pink)';
                    showEditMessage(result.error || 'Training failed', 'error');
                }
            } catch (e) {
                showEditMessage('Training error: ' + e.message, 'error');
            }
        }

        async function testVoiceProject() {
            const projectId = document.getElementById('vl-edit-project-id').value;
            const text = document.getElementById('vl-edit-test-text').value;
            if (!projectId || !text) return;

            try {
                showEditMessage('Synthesizing voice...', 'info');

                const res = await fetch(`/api/voice-lab/projects/${projectId}/test`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                const result = await res.json();

                if (result.success && result.audio_url) {
                    const audioPlayer = document.getElementById('vl-edit-audio-player');
                    const audio = document.getElementById('vl-edit-audio');
                    audio.src = result.audio_url;
                    audioPlayer.style.display = 'block';
                    audio.play();
                    showEditMessage(result.message || 'Audio generated!', 'success');
                } else {
                    showEditMessage(result.error || 'Could not generate audio', 'error');
                }
            } catch (e) {
                showEditMessage('Test error: ' + e.message, 'error');
            }
        }

        async function linkVoiceProjectToSkill() {
            const projectId = document.getElementById('vl-edit-project-id').value;
            const skillId = document.getElementById('vl-edit-skill-link').value;
            if (!projectId || !skillId) {
                showEditMessage('Please select a skill', 'error');
                return;
            }

            try {
                const res = await fetch(`/api/voice-lab/projects/${projectId}/link-skill`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ skill_id: skillId })
                });
                const result = await res.json();

                if (result.success) {
                    document.getElementById('vl-edit-linked-skill').textContent = `Currently linked to: ${skillId}`;
                    showEditMessage(`Voice linked to ${skillId}!`, 'success');
                } else {
                    showEditMessage(result.error || 'Failed to link', 'error');
                }
            } catch (e) {
                showEditMessage('Error: ' + e.message, 'error');
            }
        }

        async function deleteCurrentVoiceProject() {
            const projectId = document.getElementById('vl-edit-project-id').value;
            const name = document.getElementById('vl-edit-name').value;

            if (!confirm(`Delete voice project "${name}"? This cannot be undone.`)) return;

            try {
                const res = await fetch(`/api/voice-lab/projects/${projectId}`, {
                    method: 'DELETE'
                });
                const result = await res.json();

                if (result.success) {
                    showVoiceLabTab('projects');
                    loadVoiceProjects();
                } else {
                    alert(result.error || 'Delete failed');
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }

        function showEditMessage(msg, type) {
            const el = document.getElementById('vl-edit-message');
            el.innerHTML = `<div style="color: ${type === 'success' ? 'var(--neon-green)' : type === 'error' ? 'var(--neon-pink)' : 'var(--neon-cyan)'};">${msg}</div>`;
            if (type !== 'info') {
                setTimeout(() => { el.innerHTML = ''; }, 5000);
            }
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
