"""
HIVE215 Unified Dashboard - Modal Deployment
=============================================

Deploys the Unified Skill Command Center to Modal.

Usage:
    modal deploy deploy_dashboard.py

URL will be:
    https://your-username--hive215-dashboard-flask-app.modal.run
"""

import modal

app = modal.App("hive215-dashboard")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "flask>=2.0.0",
        "flask-cors>=4.0.0",
    )
    .add_local_file("unified_dashboard.py", "/root/unified_dashboard.py")
    .add_local_dir("business_profiles", "/root/business_profiles", condition=lambda pth: pth.exists())
)


@app.function(
    image=image,
    scaledown_window=300,  # 5 min idle before shutdown
    min_containers=0,  # Scale to zero when not in use
)
@modal.wsgi_app()
def flask_app():
    """Serve the unified dashboard."""
    import sys
    sys.path.insert(0, "/root")

    from unified_dashboard import app as flask_application
    return flask_application


@app.local_entrypoint()
def main():
    """Test locally."""
    print("Dashboard ready to deploy!")
    print("Run: modal deploy deploy_dashboard.py")
