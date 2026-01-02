"""
Modal Deployment for HIVE215 // THE ARCHITECT
Shop Drawing Automation System

Deploys the shop drawing generator with connections to:
- fast_brain adapters for AI inference
- Shared volumes for file storage
"""

import modal

app = modal.App("hive215-shop-drawings")

# Volumes
data_volume = modal.Volume.from_name("hive215-data", create_if_missing=True)
adapters_volume = modal.Volume.from_name("hive215-adapters", create_if_missing=True)
shop_drawings_volume = modal.Volume.from_name("hive215-shop-drawings", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Web framework
        "flask>=2.0.0",
        "flask-cors>=4.0.0",
        # Document parsing
        "PyPDF2>=3.0.0",
        "python-docx>=1.0.0",
        "openpyxl>=3.1.0",
        # Utilities
        "werkzeug>=2.0.0",
    )
    .add_local_file("shop_drawing_generator.py", "/root/shop_drawing_generator.py")
)


@app.function(
    image=image,
    volumes={
        "/data": data_volume,
        "/adapters": adapters_volume,
        "/shop_drawings": shop_drawings_volume,
    },
    scaledown_window=300,
    min_containers=0,
)
@modal.wsgi_app()
def flask_app():
    """Serve The Architect Shop Drawing Dashboard."""
    import sys
    import os
    sys.path.insert(0, "/root")

    # Override paths for Modal environment
    os.environ['SHOP_DRAWINGS_UPLOAD'] = '/data/shop_drawings/uploads'
    os.environ['SHOP_DRAWINGS_OUTPUT'] = '/data/shop_drawings/outputs'

    # Ensure directories exist
    os.makedirs('/data/shop_drawings/uploads', exist_ok=True)
    os.makedirs('/data/shop_drawings/outputs', exist_ok=True)

    from shop_drawing_generator import app as flask_application
    return flask_application


@app.local_entrypoint()
def main():
    """Print deployment info."""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         HIVE215 // THE ARCHITECT - Shop Drawing System        ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║  Deployment Command:                                          ║
    ║    py -3.11 -m modal deploy deploy_shop_drawings.py           ║
    ║                                                               ║
    ║  Features:                                                    ║
    ║    • 10-Step Shop Drawing Workflow                            ║
    ║    • AI-Powered Document Analysis                             ║
    ║    • Connected to fast_brain Adapters                         ║
    ║    • Automatic File Classification                            ║
    ║    • JSON Output Generation                                   ║
    ║                                                               ║
    ║  Volumes:                                                     ║
    ║    • hive215-data (shared with dashboard)                     ║
    ║    • hive215-adapters (trained AI models)                     ║
    ║    • hive215-shop-drawings (project files)                    ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
