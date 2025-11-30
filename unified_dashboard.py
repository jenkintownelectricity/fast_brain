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
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Voices</span> Free TTS Voice Library</div>
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
                <div class="glass-card">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">Voice</span> Voice Platform Integration</div>
                    </div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">Connect the Skill Command Center to your voice platform.</p>
                    <h4 style="color: var(--neon-cyan); margin: 1rem 0 0.5rem;">Supported Platforms</h4>
                    <ul style="color: var(--text-secondary); margin-bottom: 1rem;">
                        <li>LiveKit</li>
                        <li>Vapi</li>
                        <li>Twilio</li>
                        <li>Custom WebSocket</li>
                    </ul>
                    <h4 style="color: var(--neon-cyan); margin-bottom: 0.5rem;">Basic Integration</h4>
                    <pre class="code-block">from skill_command_center import SkillCommandCenter

center = SkillCommandCenter()

# In your voice handler:
async def on_user_speech(text: str):
    async for chunk in center.process_query(text, use_latency_masking=True):
        await tts.speak(chunk)</pre>
                    <h4 style="color: var(--neon-cyan); margin: 1rem 0 0.5rem;">LiveKit Integration</h4>
                    <pre class="code-block">from livekit.agents import llm

class HybridLLM(llm.LLM):
    def __init__(self):
        self.center = SkillCommandCenter()

    async def chat(self, messages):
        prompt = messages[-1].content
        async for chunk in self.center.process_query(prompt):
            yield llm.ChatChunk(content=chunk)</pre>
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
        // INITIALIZATION
        // ============================================================
        document.addEventListener('DOMContentLoaded', () => {
            loadSkills();
            refreshServerStatus();
            loadMetrics();
            loadActivity();

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
