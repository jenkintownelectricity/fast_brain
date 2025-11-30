"""
Skill Command Center - The Ultimate Skill Management Dashboard

A cyberpunk-themed dashboard for managing voice agent skills:
- Real-time skill monitoring
- Warm server status tracking
- Training pipeline visualization
- Performance analytics
- Integration with skill_factory.py and continuous_learner.py

Usage:
    pip install flask flask-cors
    python skill_dashboard.py
    # Opens at http://localhost:5000
"""

from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import random  # For demo data

app = Flask(__name__)
CORS(app)

# Data paths (shared with skill_factory.py)
BUSINESS_PROFILES_DIR = Path("business_profiles")
TRAINING_DATA_DIR = Path("training_data")
ADAPTERS_DIR = Path("adapters")
LOGS_DIR = Path("logs")

# Ensure directories exist
for d in [BUSINESS_PROFILES_DIR, TRAINING_DATA_DIR, ADAPTERS_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/skills')
def get_skills():
    """Get all registered skills with their status."""
    skills = []

    for profile_path in BUSINESS_PROFILES_DIR.glob("*.json"):
        with open(profile_path) as f:
            profile = json.load(f)

        # Check adapter status
        safe_name = profile_path.stem
        adapter_path = ADAPTERS_DIR / safe_name
        has_adapter = adapter_path.exists()

        # Check training data
        training_files = list(TRAINING_DATA_DIR.glob(f"{safe_name}*.jsonl"))
        training_examples = 0
        for tf in training_files:
            with open(tf) as f:
                training_examples += sum(1 for _ in f)

        skills.append({
            "id": safe_name,
            "name": profile.get("business_name", safe_name),
            "type": profile.get("business_type", "Unknown"),
            "status": "deployed" if has_adapter else "training" if training_examples > 0 else "draft",
            "personality": profile.get("personality", ""),
            "training_examples": training_examples,
            "created_at": profile.get("created_at", ""),
            "last_used": datetime.now().isoformat(),  # Would come from logs
            "requests_today": random.randint(10, 500),  # Demo data
            "avg_latency_ms": random.randint(80, 200),  # Demo data
            "satisfaction_rate": random.randint(85, 99),  # Demo data
        })

    return jsonify(skills)


@app.route('/api/skills/<skill_id>')
def get_skill_detail(skill_id):
    """Get detailed info for a specific skill."""
    profile_path = BUSINESS_PROFILES_DIR / f"{skill_id}.json"

    if not profile_path.exists():
        return jsonify({"error": "Skill not found"}), 404

    with open(profile_path) as f:
        profile = json.load(f)

    return jsonify(profile)


@app.route('/api/server-status')
def get_server_status():
    """Get warm server status from Modal."""
    # In production, this would query Modal's API
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
    # Generate demo time series data
    now = datetime.now()
    metrics = []

    for i in range(24):
        timestamp = now - timedelta(hours=23-i)
        metrics.append({
            "timestamp": timestamp.isoformat(),
            "requests": random.randint(50, 300),
            "latency_p50": random.randint(70, 120),
            "latency_p99": random.randint(150, 300),
            "errors": random.randint(0, 5),
            "satisfaction": random.randint(88, 99),
        })

    return jsonify(metrics)


@app.route('/api/training-pipeline')
def get_training_pipeline():
    """Get training pipeline status."""
    pipeline_jobs = []

    for skill_dir in TRAINING_DATA_DIR.glob("*_merged.jsonl"):
        skill_name = skill_dir.stem.replace("_merged", "")

        # Count examples
        with open(skill_dir) as f:
            example_count = sum(1 for _ in f)

        pipeline_jobs.append({
            "skill": skill_name,
            "stage": random.choice(["queued", "preprocessing", "training", "validating", "complete"]),
            "progress": random.randint(0, 100),
            "examples": example_count,
            "started_at": (datetime.now() - timedelta(minutes=random.randint(5, 60))).isoformat(),
            "eta_minutes": random.randint(5, 30),
        })

    return jsonify(pipeline_jobs)


@app.route('/api/feedback-queue')
def get_feedback_queue():
    """Get pending feedback for review."""
    feedback_path = LOGS_DIR / "feedback.jsonl"
    feedback = []

    if feedback_path.exists():
        with open(feedback_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("rating", 5) <= 3:  # Low-rated
                    feedback.append(entry)

    return jsonify(feedback[-20:])  # Last 20


@app.route('/api/create-skill', methods=['POST'])
def create_skill():
    """Quick-create a new skill."""
    data = request.json

    safe_name = data['name'].lower().replace(' ', '_')
    profile_path = BUSINESS_PROFILES_DIR / f"{safe_name}.json"

    profile = {
        "business_name": data['name'],
        "business_type": data.get('type', 'General'),
        "description": data.get('description', ''),
        "greeting": data.get('greeting', f"Hello! How can I help you with {data['name']}?"),
        "personality": data.get('personality', 'Friendly and helpful'),
        "key_services": data.get('services', []),
        "faq": [],
        "created_at": datetime.now().isoformat(),
    }

    with open(profile_path, 'w') as f:
        json.dump(profile, f, indent=2)

    return jsonify({"success": True, "id": safe_name})


# =============================================================================
# MAIN DASHBOARD HTML
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

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: var(--font-body);
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Animated Background */
        .bg-effects {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 0;
        }

        .grid-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
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

        .blob-1 {
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(0, 255, 242, 0.15) 0%, transparent 70%);
            top: -200px;
            right: -200px;
        }

        .blob-2 {
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(180, 0, 255, 0.1) 0%, transparent 70%);
            bottom: -150px;
            left: -150px;
            animation-delay: -4s;
        }

        .blob-3 {
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(255, 0, 255, 0.08) 0%, transparent 70%);
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            animation-delay: -2s;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.2); opacity: 0.7; }
        }

        /* Floating Particles */
        .particles {
            position: absolute;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }

        .particle {
            position: absolute;
            width: 4px;
            height: 4px;
            background: var(--neon-cyan);
            border-radius: 50%;
            animation: float 10s linear infinite;
            opacity: 0.5;
        }

        @keyframes float {
            0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
            10% { opacity: 0.5; }
            90% { opacity: 0.5; }
            100% { transform: translateY(-100px) rotate(720deg); opacity: 0; }
        }

        /* Main Container */
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
            margin-bottom: 3rem;
        }

        .logo {
            font-family: var(--font-display);
            font-size: 3rem;
            font-weight: 900;
            background: linear-gradient(135deg, var(--neon-cyan) 0%, var(--neon-purple) 50%, var(--neon-pink) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            text-shadow: 0 0 30px rgba(0, 255, 242, 0.5);
            animation: glow 2s ease-in-out infinite alternate;
        }

        @keyframes glow {
            from { filter: drop-shadow(0 0 20px rgba(0, 255, 242, 0.5)); }
            to { filter: drop-shadow(0 0 40px rgba(180, 0, 255, 0.5)); }
        }

        .subtitle {
            font-family: var(--font-accent);
            font-size: 1.2rem;
            color: var(--text-secondary);
            letter-spacing: 0.3em;
            margin-top: 0.5rem;
        }

        /* Status Badge */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1.5rem;
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.2) 0%, rgba(0, 255, 242, 0.1) 100%);
            border: 1px solid var(--neon-green);
            border-radius: 50px;
            font-family: var(--font-display);
            font-size: 0.8rem;
            color: var(--neon-green);
            margin-top: 1rem;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--neon-green);
            border-radius: 50%;
            animation: blink 1s ease-in-out infinite;
        }

        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        /* Grid Layout */
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
        }

        /* Glassmorphism Card */
        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 1.5rem;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }

        .glass-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        }

        .glass-card:hover {
            transform: translateY(-5px);
            border-color: var(--neon-cyan);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 30px rgba(0, 255, 242, 0.1);
        }

        /* Card Variants */
        .card-full { grid-column: span 4; }
        .card-half { grid-column: span 2; }
        .card-third { grid-column: span 1; }

        /* Stat Card */
        .stat-card {
            text-align: center;
        }

        .stat-value {
            font-family: var(--font-display);
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--neon-cyan) 0%, var(--neon-blue) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .stat-value.green {
            background: linear-gradient(135deg, var(--neon-green) 0%, var(--neon-cyan) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .stat-value.pink {
            background: linear-gradient(135deg, var(--neon-pink) 0%, var(--neon-purple) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .stat-value.orange {
            background: linear-gradient(135deg, var(--neon-orange) 0%, var(--neon-yellow) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .stat-label {
            font-family: var(--font-accent);
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 0.5rem;
        }

        /* Section Header */
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }

        .section-title {
            font-family: var(--font-display);
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .section-icon {
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple));
            border-radius: 8px;
            font-size: 1rem;
        }

        /* Skill Cards */
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
            position: relative;
            overflow: hidden;
        }

        .skill-card::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0, 255, 242, 0.1), transparent);
            transition: left 0.5s ease;
        }

        .skill-card:hover::after {
            left: 100%;
        }

        .skill-card:hover {
            border-color: var(--neon-cyan);
            transform: scale(1.02);
        }

        .skill-card.deployed {
            border-left: 3px solid var(--neon-green);
        }

        .skill-card.training {
            border-left: 3px solid var(--neon-orange);
        }

        .skill-card.draft {
            border-left: 3px solid var(--text-secondary);
        }

        .skill-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1rem;
        }

        .skill-name {
            font-family: var(--font-display);
            font-size: 1rem;
            font-weight: 600;
        }

        .skill-type {
            font-family: var(--font-accent);
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }

        .skill-status {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-family: var(--font-accent);
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .skill-status.deployed {
            background: rgba(0, 255, 136, 0.2);
            color: var(--neon-green);
        }

        .skill-status.training {
            background: rgba(255, 107, 0, 0.2);
            color: var(--neon-orange);
        }

        .skill-status.draft {
            background: rgba(160, 160, 176, 0.2);
            color: var(--text-secondary);
        }

        .skill-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid var(--glass-border);
        }

        .skill-metric {
            text-align: center;
        }

        .metric-value {
            font-family: var(--font-display);
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--neon-cyan);
        }

        .metric-label {
            font-size: 0.65rem;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        /* Progress Bar */
        .progress-container {
            margin-top: 1rem;
        }

        .progress-bar {
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
            position: relative;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple));
            border-radius: 3px;
            position: relative;
            transition: width 0.5s ease;
        }

        .progress-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%);
            animation: shimmer 2s infinite;
        }

        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        /* Server Status Panel */
        .server-panel {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
        }

        .server-stat {
            text-align: center;
            padding: 1rem;
            background: var(--glass-surface);
            border-radius: 12px;
        }

        /* Pipeline Visualization */
        .pipeline-container {
            display: flex;
            align-items: center;
            gap: 1rem;
            overflow-x: auto;
            padding: 1rem 0;
        }

        .pipeline-node {
            flex-shrink: 0;
            width: 120px;
            padding: 1rem;
            background: var(--glass-surface);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            text-align: center;
            position: relative;
        }

        .pipeline-node.active {
            border-color: var(--neon-cyan);
            box-shadow: 0 0 20px rgba(0, 255, 242, 0.3);
        }

        .pipeline-node.complete {
            border-color: var(--neon-green);
        }

        .pipeline-arrow {
            flex-shrink: 0;
            width: 30px;
            height: 2px;
            background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple));
            position: relative;
        }

        .pipeline-arrow::after {
            content: '▶';
            position: absolute;
            right: -8px;
            top: -8px;
            color: var(--neon-purple);
            font-size: 12px;
        }

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
            letter-spacing: 0.05em;
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

        /* Quick Create Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }

        .modal-overlay.active {
            display: flex;
        }

        .modal {
            background: var(--card-bg);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2rem;
            width: 100%;
            max-width: 500px;
            position: relative;
        }

        .modal-header {
            font-family: var(--font-display);
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            background: linear-gradient(135deg, var(--neon-cyan), var(--neon-pink));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .form-group {
            margin-bottom: 1rem;
        }

        .form-label {
            display: block;
            font-family: var(--font-accent);
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .form-input {
            width: 100%;
            padding: 0.75rem 1rem;
            background: var(--glass-surface);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            color: var(--text-primary);
            font-family: var(--font-body);
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }

        .form-input:focus {
            outline: none;
            border-color: var(--neon-cyan);
            box-shadow: 0 0 10px rgba(0, 255, 242, 0.2);
        }

        .form-select {
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2300fff2' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 1rem center;
            padding-right: 2.5rem;
        }

        /* Charts Area */
        .chart-container {
            height: 200px;
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
            position: relative;
        }

        .chart-bar:hover {
            background: linear-gradient(180deg, var(--neon-pink) 0%, rgba(255, 0, 255, 0.3) 100%);
        }

        /* Activity Feed */
        .activity-feed {
            max-height: 300px;
            overflow-y: auto;
        }

        .activity-item {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            padding: 1rem;
            border-bottom: 1px solid var(--glass-border);
        }

        .activity-icon {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            flex-shrink: 0;
        }

        .activity-icon.success { background: rgba(0, 255, 136, 0.2); }
        .activity-icon.warning { background: rgba(255, 107, 0, 0.2); }
        .activity-icon.info { background: rgba(0, 162, 255, 0.2); }

        .activity-content {
            flex: 1;
        }

        .activity-title {
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }

        .activity-time {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        /* Responsive */
        @media (max-width: 1200px) {
            .dashboard-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .card-full { grid-column: span 2; }
            .card-half { grid-column: span 2; }
        }

        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            .card-full, .card-half, .card-third { grid-column: span 1; }
            .server-panel { grid-template-columns: repeat(2, 1fr); }
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-dark);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--glass-border);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--neon-cyan);
        }
    </style>
</head>
<body>
    <!-- Background Effects -->
    <div class="bg-effects">
        <div class="grid-overlay"></div>
        <div class="gradient-blob blob-1"></div>
        <div class="gradient-blob blob-2"></div>
        <div class="gradient-blob blob-3"></div>
        <div class="particles" id="particles"></div>
    </div>

    <div class="container">
        <!-- Header -->
        <header>
            <h1 class="logo">Skill Command Center</h1>
            <p class="subtitle">Voice Agent Intelligence Hub</p>
            <div class="status-badge">
                <span class="status-dot"></span>
                <span id="server-status-text">LPU Online • 1 Warm Container</span>
            </div>
        </header>

        <!-- Dashboard Grid -->
        <div class="dashboard-grid">
            <!-- Stats Row -->
            <div class="glass-card stat-card card-third">
                <div class="stat-value" id="total-skills">0</div>
                <div class="stat-label">Active Skills</div>
            </div>
            <div class="glass-card stat-card card-third">
                <div class="stat-value green" id="requests-today">0</div>
                <div class="stat-label">Requests Today</div>
            </div>
            <div class="glass-card stat-card card-third">
                <div class="stat-value pink" id="avg-latency">0ms</div>
                <div class="stat-label">Avg Latency</div>
            </div>
            <div class="glass-card stat-card card-third">
                <div class="stat-value orange" id="satisfaction">0%</div>
                <div class="stat-label">Satisfaction</div>
            </div>

            <!-- Server Status -->
            <div class="glass-card card-full">
                <div class="section-header">
                    <div class="section-title">
                        <span class="section-icon">🖥️</span>
                        Server Status
                    </div>
                    <button class="btn btn-secondary" onclick="refreshServerStatus()">Refresh</button>
                </div>
                <div class="server-panel" id="server-panel">
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

            <!-- Skills Panel -->
            <div class="glass-card card-full">
                <div class="section-header">
                    <div class="section-title">
                        <span class="section-icon">🧠</span>
                        Skill Library
                    </div>
                    <div style="display: flex; gap: 0.5rem;">
                        <button class="btn btn-secondary" onclick="openSkillFactory()">Open Factory</button>
                        <button class="btn btn-primary" onclick="openModal()">+ New Skill</button>
                    </div>
                </div>
                <div class="skills-grid" id="skills-grid">
                    <!-- Skills will be populated here -->
                </div>
            </div>

            <!-- Training Pipeline -->
            <div class="glass-card card-half">
                <div class="section-header">
                    <div class="section-title">
                        <span class="section-icon">⚙️</span>
                        Training Pipeline
                    </div>
                </div>
                <div class="pipeline-container">
                    <div class="pipeline-node complete">
                        <div>📄</div>
                        <div style="font-size: 0.75rem; margin-top: 0.5rem;">Ingest</div>
                    </div>
                    <div class="pipeline-arrow"></div>
                    <div class="pipeline-node complete">
                        <div>🔄</div>
                        <div style="font-size: 0.75rem; margin-top: 0.5rem;">Process</div>
                    </div>
                    <div class="pipeline-arrow"></div>
                    <div class="pipeline-node active">
                        <div>🎓</div>
                        <div style="font-size: 0.75rem; margin-top: 0.5rem;">Train</div>
                    </div>
                    <div class="pipeline-arrow"></div>
                    <div class="pipeline-node">
                        <div>✅</div>
                        <div style="font-size: 0.75rem; margin-top: 0.5rem;">Deploy</div>
                    </div>
                </div>
                <div class="progress-container">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="font-size: 0.8rem; color: var(--text-secondary);">Training: plumber_expert</span>
                        <span style="font-size: 0.8rem; color: var(--neon-cyan);">67%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 67%;"></div>
                    </div>
                </div>
            </div>

            <!-- Activity Feed -->
            <div class="glass-card card-half">
                <div class="section-header">
                    <div class="section-title">
                        <span class="section-icon">📊</span>
                        Activity Feed
                    </div>
                </div>
                <div class="activity-feed" id="activity-feed">
                    <div class="activity-item">
                        <div class="activity-icon success">✅</div>
                        <div class="activity-content">
                            <div class="activity-title">plumber_expert deployed successfully</div>
                            <div class="activity-time">2 minutes ago</div>
                        </div>
                    </div>
                    <div class="activity-item">
                        <div class="activity-icon info">🎓</div>
                        <div class="activity-content">
                            <div class="activity-title">Training started for restaurant_host</div>
                            <div class="activity-time">15 minutes ago</div>
                        </div>
                    </div>
                    <div class="activity-item">
                        <div class="activity-icon warning">⚠️</div>
                        <div class="activity-content">
                            <div class="activity-title">Low satisfaction detected on tech_support</div>
                            <div class="activity-time">1 hour ago</div>
                        </div>
                    </div>
                    <div class="activity-item">
                        <div class="activity-icon success">📄</div>
                        <div class="activity-content">
                            <div class="activity-title">New documents processed: 47 examples</div>
                            <div class="activity-time">2 hours ago</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Performance Chart -->
            <div class="glass-card card-half">
                <div class="section-header">
                    <div class="section-title">
                        <span class="section-icon">📈</span>
                        Request Volume (24h)
                    </div>
                </div>
                <div class="chart-container" id="request-chart">
                    <!-- Chart bars will be populated here -->
                </div>
            </div>

            <!-- Latency Chart -->
            <div class="glass-card card-half">
                <div class="section-header">
                    <div class="section-title">
                        <span class="section-icon">⚡</span>
                        Latency (24h)
                    </div>
                </div>
                <div class="chart-container" id="latency-chart">
                    <!-- Chart bars will be populated here -->
                </div>
            </div>
        </div>
    </div>

    <!-- Quick Create Modal -->
    <div class="modal-overlay" id="modal-overlay" onclick="closeModal(event)">
        <div class="modal" onclick="event.stopPropagation()">
            <h2 class="modal-header">Create New Skill</h2>
            <form id="create-skill-form" onsubmit="createSkill(event)">
                <div class="form-group">
                    <label class="form-label">Skill Name</label>
                    <input type="text" class="form-input" id="skill-name" placeholder="Joe's Plumbing" required>
                </div>
                <div class="form-group">
                    <label class="form-label">Business Type</label>
                    <select class="form-input form-select" id="skill-type">
                        <option value="Plumbing Services">Plumbing Services</option>
                        <option value="Electrical Services">Electrical Services</option>
                        <option value="Restaurant/Food Service">Restaurant/Food Service</option>
                        <option value="Tech Support">Tech Support</option>
                        <option value="Healthcare">Healthcare</option>
                        <option value="General Customer Service">General Customer Service</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="form-label">Description</label>
                    <input type="text" class="form-input" id="skill-description" placeholder="24/7 emergency plumbing services...">
                </div>
                <div class="form-group">
                    <label class="form-label">Greeting</label>
                    <input type="text" class="form-input" id="skill-greeting" placeholder="Hello! Thanks for calling...">
                </div>
                <div style="display: flex; gap: 1rem; margin-top: 1.5rem;">
                    <button type="button" class="btn btn-secondary" onclick="closeModal()" style="flex: 1;">Cancel</button>
                    <button type="submit" class="btn btn-primary" style="flex: 1;">Create Skill</button>
                </div>
            </form>
        </div>
    </div>

    <script>
        // Initialize particles
        function createParticles() {
            const container = document.getElementById('particles');
            for (let i = 0; i < 30; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 10 + 's';
                particle.style.animationDuration = (8 + Math.random() * 4) + 's';
                container.appendChild(particle);
            }
        }

        // Fetch and display skills
        async function loadSkills() {
            try {
                const response = await fetch('/api/skills');
                const skills = await response.json();

                const grid = document.getElementById('skills-grid');
                grid.innerHTML = '';

                let totalRequests = 0;
                let totalLatency = 0;
                let totalSatisfaction = 0;

                skills.forEach(skill => {
                    totalRequests += skill.requests_today;
                    totalLatency += skill.avg_latency_ms;
                    totalSatisfaction += skill.satisfaction_rate;

                    const card = document.createElement('div');
                    card.className = `skill-card ${skill.status}`;
                    card.innerHTML = `
                        <div class="skill-header">
                            <div>
                                <div class="skill-name">${skill.name}</div>
                                <div class="skill-type">${skill.type}</div>
                            </div>
                            <span class="skill-status ${skill.status}">${skill.status}</span>
                        </div>
                        <div class="skill-metrics">
                            <div class="skill-metric">
                                <div class="metric-value">${skill.requests_today}</div>
                                <div class="metric-label">Requests</div>
                            </div>
                            <div class="skill-metric">
                                <div class="metric-value">${skill.avg_latency_ms}ms</div>
                                <div class="metric-label">Latency</div>
                            </div>
                            <div class="skill-metric">
                                <div class="metric-value">${skill.satisfaction_rate}%</div>
                                <div class="metric-label">Satisfaction</div>
                            </div>
                        </div>
                    `;
                    card.onclick = () => openSkillDetail(skill.id);
                    grid.appendChild(card);
                });

                // Update stats
                document.getElementById('total-skills').textContent = skills.length;
                document.getElementById('requests-today').textContent = totalRequests.toLocaleString();
                document.getElementById('avg-latency').textContent = Math.round(totalLatency / skills.length) + 'ms';
                document.getElementById('satisfaction').textContent = Math.round(totalSatisfaction / skills.length) + '%';

            } catch (error) {
                console.error('Failed to load skills:', error);
            }
        }

        // Load server status
        async function refreshServerStatus() {
            try {
                const response = await fetch('/api/server-status');
                const status = await response.json();

                document.getElementById('server-status-text').textContent =
                    `LPU ${status.status === 'online' ? 'Online' : 'Offline'} • ${status.warm_containers} Warm Container${status.warm_containers !== 1 ? 's' : ''}`;
                document.getElementById('warm-containers').textContent = status.warm_containers;
                document.getElementById('memory-usage').textContent = (status.memory_usage_mb / 1000).toFixed(1) + 'GB';
                document.getElementById('cost-today').textContent = '$' + status.cost_today_usd.toFixed(2);

            } catch (error) {
                console.error('Failed to load server status:', error);
            }
        }

        // Load metrics charts
        async function loadMetrics() {
            try {
                const response = await fetch('/api/metrics');
                const metrics = await response.json();

                // Request chart
                const requestChart = document.getElementById('request-chart');
                requestChart.innerHTML = '';
                const maxRequests = Math.max(...metrics.map(m => m.requests));

                metrics.forEach(m => {
                    const bar = document.createElement('div');
                    bar.className = 'chart-bar';
                    bar.style.height = (m.requests / maxRequests * 100) + '%';
                    bar.title = `${m.requests} requests`;
                    requestChart.appendChild(bar);
                });

                // Latency chart
                const latencyChart = document.getElementById('latency-chart');
                latencyChart.innerHTML = '';
                const maxLatency = Math.max(...metrics.map(m => m.latency_p99));

                metrics.forEach(m => {
                    const bar = document.createElement('div');
                    bar.className = 'chart-bar';
                    bar.style.height = (m.latency_p50 / maxLatency * 100) + '%';
                    bar.style.background = 'linear-gradient(180deg, var(--neon-green) 0%, rgba(0, 255, 136, 0.3) 100%)';
                    bar.title = `P50: ${m.latency_p50}ms, P99: ${m.latency_p99}ms`;
                    latencyChart.appendChild(bar);
                });

            } catch (error) {
                console.error('Failed to load metrics:', error);
            }
        }

        // Modal functions
        function openModal() {
            document.getElementById('modal-overlay').classList.add('active');
        }

        function closeModal(event) {
            if (!event || event.target === document.getElementById('modal-overlay')) {
                document.getElementById('modal-overlay').classList.remove('active');
            }
        }

        // Create skill
        async function createSkill(event) {
            event.preventDefault();

            const data = {
                name: document.getElementById('skill-name').value,
                type: document.getElementById('skill-type').value,
                description: document.getElementById('skill-description').value,
                greeting: document.getElementById('skill-greeting').value,
            };

            try {
                const response = await fetch('/api/create-skill', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                if (response.ok) {
                    closeModal();
                    loadSkills();
                    document.getElementById('create-skill-form').reset();
                }
            } catch (error) {
                console.error('Failed to create skill:', error);
            }
        }

        // Open skill factory
        function openSkillFactory() {
            window.open('http://localhost:7860', '_blank');
        }

        // Open skill detail
        function openSkillDetail(skillId) {
            console.log('Opening skill:', skillId);
            // Could open a detail panel or navigate to skill factory
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            createParticles();
            loadSkills();
            refreshServerStatus();
            loadMetrics();

            // Refresh periodically
            setInterval(loadSkills, 30000);
            setInterval(refreshServerStatus, 10000);
            setInterval(loadMetrics, 60000);
        });
    </script>
</body>
</html>
'''


@app.route('/')
def dashboard():
    """Serve the main dashboard."""
    return render_template_string(DASHBOARD_HTML)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║           🧠 SKILL COMMAND CENTER                         ║
    ║           Voice Agent Intelligence Hub                    ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Dashboard:    http://localhost:5000                      ║
    ║  Skill Factory: http://localhost:7860 (run separately)    ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    app.run(host='0.0.0.0', port=5000, debug=True)
