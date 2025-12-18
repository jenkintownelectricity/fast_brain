"""
HIVE215 Unified Dashboard - Modal Deployment
=============================================

Deploys the Unified Skill Command Center to Modal with persistent storage.

Usage:
    modal deploy deploy_dashboard.py

URL will be:
    https://your-username--hive215-dashboard-flask-app.modal.run

Features:
    - SQLite database for persistent skill storage
    - Golden prompts management
    - Fast Brain API integration
    - Voice configuration
    - Platform connections
"""

import modal

app = modal.App("hive215-dashboard")

# Create a persistent volume for database storage
volume = modal.Volume.from_name("hive215-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "flask>=2.0.0",
        "flask-cors>=4.0.0",
        "httpx>=0.25.0",  # For Fast Brain API calls
        "requests>=2.31.0",  # HTTP requests
        "gTTS>=2.3.0",  # Google TTS (HTTP-based, more reliable)
    )
    .add_local_file("unified_dashboard.py", "/root/unified_dashboard.py")
    .add_local_file("golden_prompts.py", "/root/golden_prompts.py")
    .add_local_file("database.py", "/root/database.py")
)


@app.function(
    image=image,
    volumes={"/data": volume},  # Mount persistent volume
    scaledown_window=300,  # 5 min idle before shutdown
    min_containers=0,  # Scale to zero when not in use
)
@modal.wsgi_app()
def flask_app():
    """Serve the unified dashboard with persistent storage."""
    import sys
    import os
    sys.path.insert(0, "/root")

    # Set environment variables
    os.environ['FAST_BRAIN_URL'] = 'https://jenkintownelectricity--fast-brain-lpu-fastapi-app.modal.run'
    os.environ['HIVE215_DB_PATH'] = '/data/hive215.db'

    # Initialize database on startup
    try:
        import database as db
        if not os.path.exists('/data/hive215.db'):
            db.initialize_database()
            print("Database initialized successfully")
        else:
            # Ensure tables exist (migrations)
            db.init_db()
            print("Database connected successfully")
    except Exception as e:
        print(f"Database initialization warning: {e}")

    from unified_dashboard import app as flask_application
    return flask_application


@app.local_entrypoint()
def main():
    """Test locally."""
    print("""
    ===============================================
    HIVE215 Unified Dashboard
    ===============================================

    To deploy:
        modal deploy deploy_dashboard.py

    Dashboard URL:
        https://jenkintownelectricity--hive215-dashboard-flask-app.modal.run

    Features:
        ✓ SQLite database (persistent on Modal volume)
        ✓ Skills management (create, edit, delete)
        ✓ Golden prompts (view, customize, test)
        ✓ Fast Brain API integration
        ✓ Voice configuration
        ✓ Platform connections
        ✓ Activity logging

    Database Location:
        /data/hive215.db (on Modal volume: hive215-data)

    ===============================================
    """)
