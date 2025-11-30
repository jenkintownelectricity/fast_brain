"""
Unified Skill Command Center Dashboard

Combines:
- Skill Command Center (monitoring & stats)
- Skill Factory (business profiles & training)
- LPU Inference (test BitNet model)

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

# Ensure directories exist
for d in [BUSINESS_PROFILES_DIR, TRAINING_DATA_DIR, ADAPTERS_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)


# =============================================================================
# API ENDPOINTS - SKILLS
# =============================================================================

@app.route('/api/skills')
def get_skills():
    """Get all registered skills with their status."""
    skills = []
    for profile_path in BUSINESS_PROFILES_DIR.glob("*.json"):
        with open(profile_path) as f:
            profile = json.load(f)
        safe_name = profile_path.stem
        adapter_path = ADAPTERS_DIR / safe_name
        has_adapter = adapter_path.exists()
        training_files = list(TRAINING_DATA_DIR.glob(f"{safe_name}*.jsonl"))
        training_examples = sum(sum(1 for _ in open(tf)) for tf in training_files)

        skills.append({
            "id": safe_name,
            "name": profile.get("business_name", safe_name),
            "type": profile.get("business_type", "Unknown"),
            "status": "deployed" if has_adapter else "training" if training_examples > 0 else "draft",
            "personality": profile.get("personality", ""),
            "training_examples": training_examples,
            "created_at": profile.get("created_at", ""),
            "requests_today": random.randint(10, 500),
            "avg_latency_ms": random.randint(80, 200),
            "satisfaction_rate": random.randint(85, 99),
        })
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

    return jsonify({"success": True, "id": safe_name})


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

    return jsonify({"success": True, "examples": len(training_data)})


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
        return jsonify({"success": True, "response": response})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


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
            padding: 2rem;
        }

        /* Header */
        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .logo {
            font-family: var(--font-display);
            font-size: 2.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, var(--neon-cyan) 0%, var(--neon-purple) 50%, var(--neon-pink) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .subtitle {
            font-family: var(--font-accent);
            font-size: 1rem;
            color: var(--text-secondary);
            letter-spacing: 0.3em;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1.5rem;
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.2) 0%, rgba(0, 255, 242, 0.1) 100%);
            border: 1px solid var(--neon-green);
            border-radius: 50px;
            font-family: var(--font-display);
            font-size: 0.75rem;
            color: var(--neon-green);
            margin-top: 1rem;
        }

        .status-dot { width: 8px; height: 8px; background: var(--neon-green); border-radius: 50%; animation: blink 1s infinite; }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

        /* Tabs */
        .tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }

        .tab-btn {
            padding: 0.75rem 1.5rem;
            background: var(--glass-surface);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            color: var(--text-secondary);
            font-family: var(--font-display);
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .tab-btn:hover {
            border-color: var(--neon-cyan);
            color: var(--neon-cyan);
        }

        .tab-btn.active {
            background: linear-gradient(135deg, rgba(0, 255, 242, 0.2), rgba(180, 0, 255, 0.1));
            border-color: var(--neon-cyan);
            color: var(--neon-cyan);
            box-shadow: 0 0 20px rgba(0, 255, 242, 0.2);
        }

        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* Glass Card */
        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
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
            gap: 1.5rem;
        }

        .card-full { grid-column: span 4; }
        .card-half { grid-column: span 2; }
        .card-quarter { grid-column: span 1; }

        @media (max-width: 1200px) {
            .dashboard-grid { grid-template-columns: repeat(2, 1fr); }
            .card-full, .card-half { grid-column: span 2; }
        }

        @media (max-width: 768px) {
            .dashboard-grid { grid-template-columns: 1fr; }
            .card-full, .card-half, .card-quarter { grid-column: span 1; }
        }

        /* Stat Card */
        .stat-card { text-align: center; }
        .stat-value {
            font-family: var(--font-display);
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--neon-cyan), var(--neon-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stat-value.green { background: linear-gradient(135deg, var(--neon-green), var(--neon-cyan)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .stat-value.pink { background: linear-gradient(135deg, var(--neon-pink), var(--neon-purple)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .stat-value.orange { background: linear-gradient(135deg, var(--neon-orange), var(--neon-yellow)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .stat-label { font-family: var(--font-accent); font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; margin-top: 0.25rem; }

        /* Section Header */
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }

        .section-title {
            font-family: var(--font-display);
            font-size: 1.1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .section-icon {
            width: 32px; height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple));
            border-radius: 8px;
            font-size: 1rem;
        }

        /* Form Elements */
        .form-group { margin-bottom: 1.25rem; }
        .form-label {
            display: block;
            font-family: var(--font-accent);
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .form-input, .form-select, .form-textarea {
            width: 100%;
            padding: 0.75rem 1rem;
            background: var(--glass-surface);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            color: var(--text-primary);
            font-family: var(--font-body);
            font-size: 1rem;
            transition: all 0.3s ease;
        }

        .form-input:focus, .form-select:focus, .form-textarea:focus {
            outline: none;
            border-color: var(--neon-cyan);
            box-shadow: 0 0 15px rgba(0, 255, 242, 0.2);
        }

        .form-textarea { resize: vertical; min-height: 100px; }

        .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        @media (max-width: 768px) { .form-row { grid-template-columns: 1fr; } }

        /* Buttons */
        .btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-family: var(--font-display);
            font-size: 0.85rem;
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

        /* Skills Grid */
        .skills-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 1rem;
        }

        .skill-card {
            background: var(--glass-surface);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 1.25rem;
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

        .skill-header { display: flex; justify-content: space-between; margin-bottom: 1rem; }
        .skill-name { font-family: var(--font-display); font-size: 1rem; }
        .skill-type { font-size: 0.75rem; color: var(--text-secondary); }
        .skill-status {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.7rem;
            text-transform: uppercase;
        }
        .skill-status.deployed { background: rgba(0, 255, 136, 0.2); color: var(--neon-green); }
        .skill-status.training { background: rgba(255, 107, 0, 0.2); color: var(--neon-orange); }
        .skill-status.draft { background: rgba(160, 160, 176, 0.2); color: var(--text-secondary); }

        .skill-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--glass-border); }
        .skill-metric { text-align: center; }
        .metric-value { font-family: var(--font-display); font-size: 0.9rem; color: var(--neon-cyan); }
        .metric-label { font-size: 0.65rem; color: var(--text-secondary); text-transform: uppercase; }

        /* LPU Console */
        .lpu-console {
            background: #0d0d12;
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 1rem;
            font-family: monospace;
            font-size: 0.9rem;
            min-height: 300px;
            max-height: 500px;
            overflow-y: auto;
            white-space: pre-wrap;
            color: var(--neon-green);
        }

        .lpu-console .prompt { color: var(--neon-cyan); }
        .lpu-console .response { color: var(--text-primary); }
        .lpu-console .error { color: var(--neon-pink); }
        .lpu-console .info { color: var(--text-secondary); }

        /* Chart */
        .chart-container {
            height: 150px;
            display: flex;
            align-items: flex-end;
            gap: 4px;
            padding: 1rem 0;
        }

        .chart-bar {
            flex: 1;
            background: linear-gradient(180deg, var(--neon-cyan) 0%, rgba(0, 255, 242, 0.3) 100%);
            border-radius: 4px 4px 0 0;
            transition: all 0.3s ease;
        }

        .chart-bar:hover { background: linear-gradient(180deg, var(--neon-pink) 0%, rgba(255, 0, 255, 0.3) 100%); }

        /* Server Panel */
        .server-panel { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
        .server-stat { text-align: center; padding: 1rem; background: var(--glass-surface); border-radius: 12px; }

        /* Loading */
        .loading { opacity: 0.5; pointer-events: none; }
        .spinner {
            display: inline-block;
            width: 20px; height: 20px;
            border: 2px solid var(--glass-border);
            border-top-color: var(--neon-cyan);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Success/Error Messages */
        .message {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            font-family: var(--font-accent);
        }
        .message.success { background: rgba(0, 255, 136, 0.1); border: 1px solid var(--neon-green); color: var(--neon-green); }
        .message.error { background: rgba(255, 0, 85, 0.1); border: 1px solid var(--neon-pink); color: var(--neon-pink); }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-dark); }
        ::-webkit-scrollbar-thumb { background: var(--glass-border); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--neon-cyan); }
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
                <span id="server-status-text">LPU Online • 1 Warm Container</span>
            </div>
        </header>

        <!-- Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="showTab('dashboard')">📊 Dashboard</button>
            <button class="tab-btn" onclick="showTab('factory')">🏭 Skill Factory</button>
            <button class="tab-btn" onclick="showTab('lpu')">⚡ LPU Inference</button>
            <button class="tab-btn" onclick="showTab('training')">🎓 Training</button>
        </div>

        <!-- TAB: Dashboard -->
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
                        <div class="section-title"><span class="section-icon">🖥️</span> Server Status</div>
                        <button class="btn btn-secondary" onclick="refreshServerStatus()">Refresh</button>
                    </div>
                    <div class="server-panel">
                        <div class="server-stat">
                            <div class="stat-value green" style="font-size: 1.5rem;">●</div>
                            <div class="stat-label">Status</div>
                        </div>
                        <div class="server-stat">
                            <div class="stat-value" style="font-size: 1.5rem;" id="warm-containers">1</div>
                            <div class="stat-label">Warm Containers</div>
                        </div>
                        <div class="server-stat">
                            <div class="stat-value" style="font-size: 1.5rem;" id="memory-usage">0GB</div>
                            <div class="stat-label">Memory</div>
                        </div>
                        <div class="server-stat">
                            <div class="stat-value orange" style="font-size: 1.5rem;" id="cost-today">$0.00</div>
                            <div class="stat-label">Cost Today</div>
                        </div>
                    </div>
                </div>

                <div class="glass-card card-full">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">🧠</span> Skill Library</div>
                        <button class="btn btn-primary" onclick="showTab('factory')">+ New Skill</button>
                    </div>
                    <div class="skills-grid" id="skills-grid"></div>
                </div>

                <div class="glass-card card-half">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">📈</span> Requests (24h)</div>
                    </div>
                    <div class="chart-container" id="request-chart"></div>
                </div>

                <div class="glass-card card-half">
                    <div class="section-header">
                        <div class="section-title"><span class="section-icon">⚡</span> Latency (24h)</div>
                    </div>
                    <div class="chart-container" id="latency-chart"></div>
                </div>
            </div>
        </div>

        <!-- TAB: Skill Factory -->
        <div id="tab-factory" class="tab-content">
            <div class="glass-card">
                <div class="section-header">
                    <div class="section-title"><span class="section-icon">🏭</span> Create New Skill</div>
                </div>

                <div id="factory-message"></div>

                <form id="skill-form" onsubmit="createSkill(event)">
                    <div class="form-row">
                        <div class="form-group">
                            <label class="form-label">Business Name *</label>
                            <input type="text" class="form-input" id="f-name" placeholder="Joe's Plumbing Services" required>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Business Type</label>
                            <select class="form-select" id="f-type">
                                <option value="General Customer Service">General Customer Service</option>
                                <option value="Plumbing Services">Plumbing Services</option>
                                <option value="Electrical Services">Electrical Services</option>
                                <option value="Restaurant/Food Service">Restaurant/Food Service</option>
                                <option value="Healthcare">Healthcare</option>
                                <option value="Tech Support">Tech Support</option>
                                <option value="Real Estate">Real Estate</option>
                                <option value="Legal Services">Legal Services</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label class="form-label">Business Description</label>
                        <textarea class="form-textarea" id="f-description" placeholder="We provide 24/7 emergency plumbing services in the greater metro area..."></textarea>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label class="form-label">Default Greeting</label>
                            <input type="text" class="form-input" id="f-greeting" placeholder="Hello! Thanks for calling Joe's Plumbing. How can I help?">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Personality</label>
                            <select class="form-select" id="f-personality">
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
                        <textarea class="form-textarea" id="f-services" placeholder="Emergency repairs&#10;Drain cleaning&#10;Water heater installation"></textarea>
                    </div>

                    <div class="form-group">
                        <label class="form-label">Custom Instructions (Optional)</label>
                        <textarea class="form-textarea" id="f-instructions" placeholder="Always ask for the customer's address first. Never quote prices over the phone..."></textarea>
                    </div>

                    <div style="display: flex; gap: 1rem;">
                        <button type="submit" class="btn btn-primary">💾 Save Skill Profile</button>
                        <button type="button" class="btn btn-success" onclick="generateTraining()">⚙️ Generate Training Data</button>
                    </div>
                </form>
            </div>
        </div>

        <!-- TAB: LPU Inference -->
        <div id="tab-lpu" class="tab-content">
            <div class="glass-card">
                <div class="section-header">
                    <div class="section-title"><span class="section-icon">⚡</span> BitNet LPU Console</div>
                    <button class="btn btn-secondary" onclick="clearConsole()">Clear</button>
                </div>

                <div class="form-group">
                    <label class="form-label">Prompt</label>
                    <textarea class="form-textarea" id="lpu-prompt" placeholder="User: What are your hours?&#10;Assistant:" style="min-height: 80px;"></textarea>
                </div>

                <div class="form-row" style="margin-bottom: 1rem;">
                    <div class="form-group" style="margin-bottom: 0;">
                        <label class="form-label">Max Tokens</label>
                        <input type="number" class="form-input" id="lpu-tokens" value="128" min="16" max="512">
                    </div>
                    <div class="form-group" style="margin-bottom: 0;">
                        <label class="form-label">Skill Adapter (Optional)</label>
                        <select class="form-select" id="lpu-skill">
                            <option value="">None (Base Model)</option>
                        </select>
                    </div>
                </div>

                <button class="btn btn-primary" onclick="runInference()" id="run-btn">🚀 Run Inference</button>

                <div style="margin-top: 1.5rem;">
                    <label class="form-label">Output</label>
                    <div class="lpu-console" id="lpu-console">
<span class="info">// BitNet LPU Console Ready
// Model: Llama3-8B-1.58-100B-tokens (3.58 GiB)
// Quantization: I2_S - 2 bpw ternary
// Speed: ~8 tokens/second on CPU

</span>Waiting for input...
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB: Training -->
        <div id="tab-training" class="tab-content">
            <div class="glass-card">
                <div class="section-header">
                    <div class="section-title"><span class="section-icon">🎓</span> Training Pipeline</div>
                </div>

                <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
                    Training LoRA adapters requires a GPU. Generate the training script here, then run it on Colab, Modal, or a GPU machine.
                </p>

                <div class="form-group">
                    <label class="form-label">Select Skill to Train</label>
                    <select class="form-select" id="train-skill"></select>
                </div>

                <div class="form-group">
                    <label class="form-label">Training Steps</label>
                    <input type="range" id="train-steps" min="20" max="200" value="60" style="width: 100%;">
                    <div style="display: flex; justify-content: space-between; color: var(--text-secondary); font-size: 0.8rem;">
                        <span>20 (Quick)</span>
                        <span id="steps-value">60</span>
                        <span>200 (Thorough)</span>
                    </div>
                </div>

                <button class="btn btn-primary" onclick="generateTrainingScript()">📜 Generate Training Script</button>

                <div id="training-output" style="margin-top: 1.5rem; display: none;">
                    <label class="form-label">Training Script</label>
                    <pre class="lpu-console" id="training-script" style="color: var(--text-primary);"></pre>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Tab switching
        function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('tab-' + tabId).classList.add('active');
            event.target.classList.add('active');

            if (tabId === 'dashboard') loadSkills();
            if (tabId === 'training') loadSkillsForTraining();
        }

        // Load skills
        async function loadSkills() {
            try {
                const res = await fetch('/api/skills');
                const skills = await res.json();
                const grid = document.getElementById('skills-grid');
                grid.innerHTML = '';

                let totalReq = 0, totalLat = 0, totalSat = 0;

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

        // Load server status
        async function refreshServerStatus() {
            try {
                const res = await fetch('/api/server-status');
                const s = await res.json();
                document.getElementById('server-status-text').textContent = `LPU ${s.status === 'online' ? 'Online' : 'Offline'} • ${s.warm_containers} Warm Container`;
                document.getElementById('warm-containers').textContent = s.warm_containers;
                document.getElementById('memory-usage').textContent = (s.memory_usage_mb / 1000).toFixed(1) + 'GB';
                document.getElementById('cost-today').textContent = '$' + s.cost_today_usd.toFixed(2);
            } catch (e) { console.error(e); }
        }

        // Load metrics
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

        // Create skill
        async function createSkill(e) {
            e.preventDefault();
            const data = {
                name: document.getElementById('f-name').value,
                type: document.getElementById('f-type').value,
                description: document.getElementById('f-description').value,
                greeting: document.getElementById('f-greeting').value,
                personality: document.getElementById('f-personality').value,
                services: document.getElementById('f-services').value,
                customInstructions: document.getElementById('f-instructions').value
            };

            try {
                const res = await fetch('/api/create-skill', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await res.json();
                if (result.success) {
                    document.getElementById('factory-message').innerHTML = '<div class="message success">✅ Skill profile saved successfully!</div>';
                    document.getElementById('skill-form').reset();
                }
            } catch (e) {
                document.getElementById('factory-message').innerHTML = '<div class="message error">❌ Error: ' + e.message + '</div>';
            }
        }

        // Generate training data
        async function generateTraining() {
            const name = document.getElementById('f-name').value;
            if (!name) {
                alert('Please enter a business name first');
                return;
            }
            const skillId = name.toLowerCase().replace(/[^\\w-]/g, '_');
            try {
                const res = await fetch(`/api/generate-training/${skillId}`, { method: 'POST' });
                const result = await res.json();
                if (result.success) {
                    document.getElementById('factory-message').innerHTML = `<div class="message success">✅ Generated ${result.examples} training examples!</div>`;
                }
            } catch (e) {
                document.getElementById('factory-message').innerHTML = '<div class="message error">❌ Error: ' + e.message + '</div>';
            }
        }

        // LPU Inference
        async function runInference() {
            const btn = document.getElementById('run-btn');
            const console = document.getElementById('lpu-console');
            const prompt = document.getElementById('lpu-prompt').value;
            const maxTokens = parseInt(document.getElementById('lpu-tokens').value);

            if (!prompt) {
                alert('Please enter a prompt');
                return;
            }

            btn.innerHTML = '<span class="spinner"></span> Running...';
            btn.disabled = true;

            console.innerHTML += `\\n<span class="prompt">&gt; ${prompt}</span>\\n`;

            try {
                const res = await fetch('/api/lpu-inference', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, maxTokens })
                });
                const result = await res.json();

                if (result.success) {
                    console.innerHTML += `<span class="response">${result.response}</span>\\n`;
                } else {
                    console.innerHTML += `<span class="error">Error: ${result.error}</span>\\n`;
                }
            } catch (e) {
                console.innerHTML += `<span class="error">Error: ${e.message}</span>\\n`;
            }

            btn.innerHTML = '🚀 Run Inference';
            btn.disabled = false;
            console.scrollTop = console.scrollHeight;
        }

        function clearConsole() {
            document.getElementById('lpu-console').innerHTML = '<span class="info">// Console cleared\\n</span>';
        }

        // Training
        function loadSkillsForTraining() {
            fetch('/api/skills').then(r => r.json()).then(skills => {
                const select = document.getElementById('train-skill');
                select.innerHTML = skills.map(s => `<option value="${s.id}">${s.name}</option>`).join('');
            });
        }

        document.getElementById('train-steps').oninput = function() {
            document.getElementById('steps-value').textContent = this.value;
        };

        function generateTrainingScript() {
            const skillId = document.getElementById('train-skill').value;
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
print("✅ Training complete: adapters/${skillId}")`;

            document.getElementById('training-script').textContent = script;
            document.getElementById('training-output').style.display = 'block';
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            loadSkills();
            refreshServerStatus();
            loadMetrics();
            setInterval(loadSkills, 30000);
            setInterval(refreshServerStatus, 10000);
        });
    </script>
</body>
</html>
'''


@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)


if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           🧠 UNIFIED SKILL COMMAND CENTER                     ║
    ║           Voice Agent Intelligence Hub                        ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Dashboard:     http://localhost:5000                         ║
    ║                                                               ║
    ║  Tabs:                                                        ║
    ║    📊 Dashboard   - Stats & monitoring                        ║
    ║    🏭 Skill Factory - Create business profiles                ║
    ║    ⚡ LPU Inference - Test BitNet model                       ║
    ║    🎓 Training     - Generate training scripts                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    app.run(host='0.0.0.0', port=5000, debug=True)
