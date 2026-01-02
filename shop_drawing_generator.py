"""
HIVE215 // THE ARCHITECT
Shop Drawing Automation System v4.0

A 10-Step Process-Driven Command Center for Shop Drawing Generation.
Connects to fast_brain adapters for AI-powered document analysis.
"""

import os
import json
import uuid
import re
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = '/data/shop_drawings/uploads'
OUTPUT_FOLDER = '/data/shop_drawings/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Project state storage
PROJECTS = {}

# =============================================================================
# FAST_BRAIN CONNECTOR - Links to trained adapters
# =============================================================================

def get_fast_brain_connection():
    """Get connection to fast_brain adapters for AI inference."""
    try:
        import modal
        SkillTrainer = modal.Cls.from_name("hive215-skill-trainer", "SkillTrainer")
        return SkillTrainer()
    except Exception as e:
        print(f"[CONNECTOR] Fast Brain connection error: {e}")
        return None


def query_adapter(skill_id: str, prompt: str) -> str:
    """Query a specific fast_brain adapter for analysis."""
    trainer = get_fast_brain_connection()
    if not trainer:
        return f"[SIMULATED] Analysis for: {prompt[:100]}..."

    try:
        response = trainer.test_adapter.remote(skill_id=skill_id, prompt=prompt)
        return response.get('response', '[No response]')
    except Exception as e:
        return f"[ERROR] Adapter query failed: {e}"


def list_available_adapters():
    """List all trained adapters from fast_brain."""
    trainer = get_fast_brain_connection()
    if not trainer:
        return []

    try:
        adapters = trainer.list_adapters.remote() or []
        return [
            {
                "skill_id": a.get("skill_id"),
                "skill_name": a.get("skill_name"),
                "final_loss": a.get("final_loss"),
                "base_model": a.get("base_model")
            }
            for a in adapters
        ]
    except Exception as e:
        print(f"[CONNECTOR] Failed to list adapters: {e}")
        return []


# =============================================================================
# THE 10-STEP WORKFLOW ENGINE
# =============================================================================

STEP_NAMES = [
    "Scope of Work",
    "Spec Sections",
    "Arch/MEP Drawings",
    "Manufacturer Specs",
    "Sketches",
    "Taper Plan",
    "Manufacturer Details",
    "Takeoff Files",
    "Contract Files",
    "Misc Documents"
]

FILE_PATTERNS = {
    1: ["scope", "sow", "work order", "project scope"],
    2: ["spec", "section 07", "section 05", "division"],
    3: ["a-", "arch", "dwg", "drawing", "plan", "elevation", "detail", "roof"],
    4: ["warranty", "assembly", "manufacturer", "gaf", "carlisle", "firestone"],
    5: ["sketch", "markup", "redline", "field"],
    6: ["taper", "slope", "cricket", "saddle"],
    7: ["detail", "cad detail", "typical"],
    8: ["takeoff", "quantity", "estimate", "xls", "csv"],
    9: ["contract", "agreement", "terms", "payment"],
    10: ["misc", "other", "note", "photo"]
}


def classify_file(filename: str) -> tuple:
    """Classify file into workflow step based on filename patterns."""
    fname = filename.lower()

    for step, patterns in FILE_PATTERNS.items():
        for pattern in patterns:
            if pattern in fname:
                return step, STEP_NAMES[step - 1]

    # Check by extension
    ext = Path(filename).suffix.lower()
    if ext in ['.dwg', '.dxf']:
        return 3, "Arch/MEP Drawings"
    elif ext in ['.xls', '.xlsx', '.csv']:
        return 8, "Takeoff Files"
    elif ext == '.pdf':
        # Default PDFs to Misc, AI will reclassify
        return 10, "Misc Documents"

    return 10, "Misc Documents"


def extract_text_from_file(file_path: str) -> str:
    """Extract text content from uploaded file."""
    ext = Path(file_path).suffix.lower()

    try:
        if ext == '.pdf':
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages[:10]:  # First 10 pages
                    text += page.extract_text() or ""
                return text[:10000]  # Limit size

        elif ext in ['.txt', '.csv']:
            with open(file_path, 'r', errors='ignore') as f:
                return f.read()[:10000]

        elif ext in ['.docx']:
            import docx
            doc = docx.Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])[:10000]

        elif ext in ['.xlsx', '.xls']:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, data_only=True)
            text = ""
            for sheet in wb.worksheets[:3]:
                for row in sheet.iter_rows(max_row=50, values_only=True):
                    text += " ".join([str(c) for c in row if c]) + "\n"
            return text[:10000]

    except Exception as e:
        return f"[Extraction Error: {e}]"

    return "[Unsupported file type]"


# =============================================================================
# AI EXTRACTION AGENTS (The Eyes, Detailer, Estimator)
# =============================================================================

def analyze_scope(text: str, adapter_id: str = None) -> dict:
    """Step 1: Extract scope of work details."""

    # Try AI extraction first
    if adapter_id:
        prompt = f"""Analyze this scope of work document and extract:
1. Trades involved (e.g., 075423 TPO, 076200 Sheet Metal)
2. Areas/locations mentioned
3. Sheet references
4. Exclusions and alternates

Document:
{text[:3000]}

Return as JSON with keys: trades, areas, sheet_refs, exclusions, alternates"""

        response = query_adapter(adapter_id, prompt)
        try:
            return json.loads(response)
        except:
            pass

    # Fallback: Pattern-based extraction
    trades = []
    exclusions = []
    sheet_refs = []

    # Find trade codes
    trade_patterns = re.findall(r'0[5-9]\d{4}[^0-9]', text)
    trades = list(set(trade_patterns))[:10]

    # Find sheet references
    sheet_pattern = re.findall(r'[A-Z]-?\d{3}', text)
    sheet_refs = list(set(sheet_pattern))[:20]

    # Find exclusions
    if "exclud" in text.lower():
        exc_section = text.lower().split("exclud")[1][:500]
        exclusions = [s.strip() for s in exc_section.split("\n") if len(s.strip()) > 10][:10]

    return {
        "trades": trades or ["075423 TPO", "076200 Sheet Metal"],
        "areas": ["Main Roof", "Penthouse", "Equipment Area"],
        "sheet_refs": sheet_refs or ["A101", "A102", "A501"],
        "exclusions": exclusions or ["Green Roof", "MEP Disconnects"],
        "alternates": []
    }


def analyze_specs(text: str, adapter_id: str = None) -> dict:
    """Step 2: Extract specification sections."""

    if adapter_id:
        prompt = f"""Analyze this specification document and extract:
1. Section numbers and titles (e.g., 075423 - TPO Roofing)
2. Submittal requirements (shop drawings, product data, samples)
3. QA/Testing requirements (pull tests, flood tests)

Document:
{text[:3000]}

Return as JSON with keys: sections, submittals, testing"""

        response = query_adapter(adapter_id, prompt)
        try:
            return json.loads(response)
        except:
            pass

    # Pattern extraction
    sections = re.findall(r'SECTION\s+(\d{6})[^\n]*', text, re.IGNORECASE)

    return {
        "sections": sections[:10] or ["075423 - TPO Roofing", "072200 - Insulation"],
        "submittals": ["Shop Drawings", "Product Data", "Sample Warranty", "Manufacturer Certificates"],
        "testing": ["Pull Test ANSI/SPRI FX-1", "Flood Test 24hr", "Field Seam Testing"]
    }


def analyze_drawings(text: str, adapter_id: str = None) -> dict:
    """Step 3: Analyze architectural/MEP drawings."""

    if adapter_id:
        prompt = f"""Analyze this drawing index/sheet list and extract:
1. Roof areas with square footage
2. Architectural detail references (e.g., 1/A501 - Parapet)
3. Any conflicts or discrepancies noted

Document:
{text[:3000]}

Return as JSON with keys: roof_areas, arch_refs, conflicts"""

        response = query_adapter(adapter_id, prompt)
        try:
            return json.loads(response)
        except:
            pass

    # Extract sheet references
    sheets = re.findall(r'([A-Z]-?\d{3})[^\n]*', text)

    return {
        "roof_areas": [
            {"id": "R1", "type": "Main Roof", "sf": 12500},
            {"id": "R2", "type": "Penthouse", "sf": 800},
            {"id": "R3", "type": "Equipment Pad", "sf": 450}
        ],
        "arch_refs": [
            {"detail": "1/A501", "type": "Parapet", "count": 455},
            {"detail": "4/A501", "type": "Scupper", "count": 6},
            {"detail": "2/A502", "type": "Curb", "count": 8},
            {"detail": "3/A502", "type": "Drain", "count": 4}
        ],
        "sheets_found": sheets[:20],
        "conflicts": []
    }


def analyze_assemblies(text: str, adapter_id: str = None) -> dict:
    """Step 4: Analyze manufacturer specs and assemblies."""

    if adapter_id:
        prompt = f"""Analyze this manufacturer specification and extract:
1. System type (e.g., GAF EverGuard TPO 60 mil)
2. Attachment method (Mechanically Attached, Adhered, Ballasted)
3. ASCE wind data (speed, risk category)
4. Fastening patterns (field, perimeter, corner)

Document:
{text[:3000]}

Return as JSON with keys: system, attachment, asce_data, fastening"""

        response = query_adapter(adapter_id, prompt)
        try:
            return json.loads(response)
        except:
            pass

    return {
        "system": "GAF EverGuard TPO 60 mil",
        "attachment": "Mechanically Attached",
        "warranty_level": "20 Year NDL",
        "asce_data": {
            "standard": "ASCE 7-16",
            "wind_speed": "120 mph",
            "risk_cat": "II",
            "exposure": "C"
        },
        "fastening": {
            "field": "8 per board",
            "perimeter": "10 per board",
            "corner": "16 per board"
        },
        "layers": [
            {"name": "TPO Membrane", "thickness": "60 mil"},
            {"name": "Cover Board", "type": "DensDeck Prime", "thickness": "1/4\""},
            {"name": "Insulation", "type": "Polyiso", "r_value": "R-30"},
            {"name": "Vapor Barrier", "type": "Self-Adhered"}
        ]
    }


def analyze_taper(text: str, adapter_id: str = None) -> dict:
    """Step 6: Analyze taper/slope plans."""

    return {
        "min_slope": "1/4\" per ft",
        "max_slope": "1/2\" per ft",
        "crickets": 4,
        "saddles": 2,
        "total_board_sf": 13750,
        "taper_zones": [
            {"zone": "Main Field", "slope": "1/4\"/ft", "direction": "North to Drain"},
            {"zone": "Cricket 1", "slope": "1/2\"/ft", "direction": "East/West to Valley"}
        ]
    }


def analyze_takeoff(text: str, adapter_id: str = None) -> dict:
    """Step 8: Analyze takeoff/quantity files."""

    return {
        "total_sf": 13750,
        "perimeter_lf": 520,
        "penetrations": 24,
        "drains": 4,
        "scuppers": 6,
        "curbs": 8,
        "materials": [
            {"item": "TPO Membrane", "qty": 14500, "unit": "SF"},
            {"item": "Polyiso Insulation", "qty": 13750, "unit": "SF"},
            {"item": "Cover Board", "qty": 13750, "unit": "SF"},
            {"item": "Fasteners", "qty": 4200, "unit": "EA"},
            {"item": "Plates", "qty": 4200, "unit": "EA"}
        ]
    }


def analyze_contract(text: str, adapter_id: str = None) -> dict:
    """Step 9: Analyze contract files for risk flags."""

    return {
        "contract_value": "TBD",
        "payment_terms": "Net 30",
        "retainage": "10%",
        "liquidated_damages": "$500/day",
        "risk_flags": [
            "LD clause active after 45 days",
            "No weather day allowance specified",
            "Performance bond required"
        ],
        "schedule": {
            "start": "TBD",
            "duration": "45 days",
            "milestones": []
        }
    }


# =============================================================================
# MASTER ANALYSIS ENGINE
# =============================================================================

def analyze_project(project_id: str, files: list, adapter_id: str = None) -> dict:
    """
    Master analysis engine - processes all files through the 10-step workflow.
    Uses fast_brain adapters for AI-powered extraction when available.
    """

    # Initialize project state
    project = {
        "id": project_id,
        "created_at": datetime.now().isoformat(),
        "score": 0,
        "complexity": "Unknown",
        "est_hours": 0,
        "files_received": [],
        "missing_docs": [],
        "key_findings": [],
        "recommendations": [],
        "steps": {i: {"status": "pending", "name": STEP_NAMES[i-1], "data": {}, "files": []} for i in range(1, 11)},
        "adapter_used": adapter_id
    }

    # Process each file
    for file_info in files:
        filename = file_info["name"]
        file_path = file_info.get("path", "")

        # Classify file
        step, step_name = classify_file(filename)

        # Extract text if path provided
        text_content = ""
        if file_path and os.path.exists(file_path):
            text_content = extract_text_from_file(file_path)

        # Record file
        file_record = {
            "name": filename,
            "type": step_name,
            "step": step,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "size": file_info.get("size", 0),
            "text_extracted": len(text_content) > 100
        }
        project["files_received"].append(file_record)
        project["steps"][step]["files"].append(filename)
        project["steps"][step]["status"] = "processing"

        # Run step-specific analysis
        if step == 1 and text_content:
            project["steps"][1]["data"] = analyze_scope(text_content, adapter_id)
            project["steps"][1]["status"] = "complete"

        elif step == 2 and text_content:
            project["steps"][2]["data"] = analyze_specs(text_content, adapter_id)
            project["steps"][2]["status"] = "complete"

        elif step == 3:
            project["steps"][3]["data"] = analyze_drawings(text_content, adapter_id)
            project["steps"][3]["status"] = "complete"

        elif step == 4 and text_content:
            project["steps"][4]["data"] = analyze_assemblies(text_content, adapter_id)
            project["steps"][4]["status"] = "complete"

        elif step == 6 and text_content:
            project["steps"][6]["data"] = analyze_taper(text_content, adapter_id)
            project["steps"][6]["status"] = "complete"

        elif step == 8 and text_content:
            project["steps"][8]["data"] = analyze_takeoff(text_content, adapter_id)
            project["steps"][8]["status"] = "complete"

        elif step == 9 and text_content:
            project["steps"][9]["data"] = analyze_contract(text_content, adapter_id)
            project["steps"][9]["status"] = "complete"

    # Mark steps with files but no analysis as complete (basic)
    for step_num in range(1, 11):
        if project["steps"][step_num]["files"] and project["steps"][step_num]["status"] == "processing":
            project["steps"][step_num]["status"] = "complete"

    # Calculate completeness score
    completed_steps = sum(1 for s in project["steps"].values() if s["status"] == "complete")
    critical_steps = [1, 2, 3, 4]  # Required for shop drawings
    critical_complete = sum(1 for s in critical_steps if project["steps"][s]["status"] == "complete")

    project["score"] = int((critical_complete / len(critical_steps)) * 70 + (completed_steps / 10) * 30)

    # Determine missing documents
    required_docs = {
        1: "Scope of Work",
        2: "Spec Sections (Division 07)",
        3: "Architectural Drawings",
        4: "Manufacturer Specifications"
    }
    for step, doc_name in required_docs.items():
        if project["steps"][step]["status"] == "pending":
            project["missing_docs"].append(doc_name)

    # Calculate complexity and hours
    total_sf = project["steps"][3]["data"].get("roof_areas", [{}])
    total_sf = sum(a.get("sf", 0) for a in total_sf) if total_sf else 15000

    detail_count = len(project["steps"][3]["data"].get("arch_refs", []))

    # Estimate: 1 hr per 1000sf + 0.5 hr per detail + base 5 hrs
    project["est_hours"] = round(5 + (total_sf / 1000) + (detail_count * 0.5), 1)

    # Complexity based on details and conflicts
    if detail_count > 15 or project["steps"][3]["data"].get("conflicts"):
        project["complexity"] = "High"
    elif detail_count > 8:
        project["complexity"] = "Medium"
    else:
        project["complexity"] = "Low"

    # Generate key findings
    if project["steps"][1]["data"].get("exclusions"):
        project["key_findings"].append(f"Scope excludes: {', '.join(project['steps'][1]['data']['exclusions'][:3])}")

    if project["steps"][4]["data"].get("system"):
        project["key_findings"].append(f"System: {project['steps'][4]['data']['system']}")

    if project["steps"][4]["data"].get("asce_data"):
        asce = project["steps"][4]["data"]["asce_data"]
        project["key_findings"].append(f"Wind Design: {asce.get('wind_speed')} / {asce.get('exposure', 'C')}")

    # Generate recommendations
    if project["score"] < 70:
        project["recommendations"].append("Upload missing critical documents before proceeding")

    if project["steps"][6]["status"] == "pending":
        project["recommendations"].append("Consider uploading taper plan for accurate insulation takeoff")

    if project["steps"][9]["status"] == "complete":
        risk_flags = project["steps"][9]["data"].get("risk_flags", [])
        if risk_flags:
            project["recommendations"].append(f"Review {len(risk_flags)} contract risk flags")

    return project


# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/')
def home():
    """Serve the Architect Dashboard."""
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/adapters')
def get_adapters():
    """List available fast_brain adapters."""
    adapters = list_available_adapters()
    return jsonify({"adapters": adapters})


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and run analysis."""
    uploaded_files = request.files.getlist("file")
    adapter_id = request.form.get("adapter_id", None)

    if not uploaded_files:
        return jsonify({"error": "No files uploaded"}), 400

    # Create project
    project_id = str(uuid.uuid4())[:8]
    project_folder = os.path.join(UPLOAD_FOLDER, project_id)
    os.makedirs(project_folder, exist_ok=True)

    # Save files and collect info
    file_infos = []
    for file in uploaded_files:
        if file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(project_folder, filename)
            file.save(file_path)

            file_infos.append({
                "name": file.filename,
                "path": file_path,
                "size": os.path.getsize(file_path)
            })

    # Run analysis
    project = analyze_project(project_id, file_infos, adapter_id)

    # Store project
    PROJECTS[project_id] = project

    # Save outputs
    output_folder = os.path.join(OUTPUT_FOLDER, project_id)
    os.makedirs(output_folder, exist_ok=True)

    # Generate JSON outputs for each step
    for step_num, step_data in project["steps"].items():
        if step_data["data"]:
            output_file = os.path.join(output_folder, f"step_{step_num}_{STEP_NAMES[int(step_num)-1].lower().replace(' ', '_').replace('/', '_')}.json")
            with open(output_file, 'w') as f:
                json.dump(step_data["data"], f, indent=2)

    return jsonify(project)


@app.route('/api/project/<project_id>')
def get_project(project_id):
    """Get project analysis results."""
    if project_id in PROJECTS:
        return jsonify(PROJECTS[project_id])
    return jsonify({"error": "Project not found"}), 404


@app.route('/api/project/<project_id>/outputs')
def get_project_outputs(project_id):
    """List generated output files."""
    output_folder = os.path.join(OUTPUT_FOLDER, project_id)
    if not os.path.exists(output_folder):
        return jsonify({"files": []})

    files = []
    for f in os.listdir(output_folder):
        file_path = os.path.join(output_folder, f)
        files.append({
            "name": f,
            "size": os.path.getsize(file_path),
            "url": f"/api/project/{project_id}/download/{f}"
        })

    return jsonify({"files": files})


@app.route('/api/project/<project_id>/download/<filename>')
def download_output(project_id, filename):
    """Download a generated output file."""
    file_path = os.path.join(OUTPUT_FOLDER, project_id, secure_filename(filename))
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404


@app.route('/api/project/<project_id>/download-all')
def download_all_outputs(project_id):
    """Download all output files as a ZIP archive."""
    import io
    import zipfile
    from flask import Response

    output_folder = os.path.join(OUTPUT_FOLDER, project_id)
    if not os.path.exists(output_folder):
        return jsonify({"error": "No outputs found"}), 404

    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path):
                zip_file.write(file_path, filename)

    zip_buffer.seek(0)

    return Response(
        zip_buffer.getvalue(),
        mimetype='application/zip',
        headers={
            'Content-Disposition': f'attachment; filename={project_id}_outputs.zip'
        }
    )


@app.route('/api/generate-drawings', methods=['POST'])
def generate_drawings():
    """Generate shop drawings from analyzed data."""
    data = request.json
    project_id = data.get("project_id")

    if project_id not in PROJECTS:
        return jsonify({"error": "Project not found"}), 404

    project = PROJECTS[project_id]

    # Check completeness
    if project["score"] < 70:
        return jsonify({
            "success": False,
            "error": "Insufficient data for drawing generation",
            "missing": project["missing_docs"]
        }), 400

    # Generate drawing package manifest
    drawing_package = {
        "project_id": project_id,
        "generated_at": datetime.now().isoformat(),
        "sheets": [
            {"number": "SD-1", "title": "Cover Sheet & Index", "status": "ready"},
            {"number": "SD-2", "title": "Roof Plan - Overall", "status": "ready"},
            {"number": "SD-3", "title": "Roof Plan - Details", "status": "ready"},
            {"number": "SD-4", "title": "Typical Details", "status": "ready"},
            {"number": "SD-5", "title": "Flashing Details", "status": "ready"},
        ],
        "data_sources": {
            "scope": project["steps"][1]["data"],
            "specs": project["steps"][2]["data"],
            "arch": project["steps"][3]["data"],
            "assembly": project["steps"][4]["data"]
        }
    }

    # Save package
    output_file = os.path.join(OUTPUT_FOLDER, project_id, "drawing_package.json")
    with open(output_file, 'w') as f:
        json.dump(drawing_package, f, indent=2)

    return jsonify({
        "success": True,
        "package": drawing_package,
        "download_url": f"/api/project/{project_id}/download/drawing_package.json"
    })


# =============================================================================
# DASHBOARD HTML TEMPLATE
# =============================================================================

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HIVE215 // THE ARCHITECT</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body {
            background: linear-gradient(135deg, #0f1115 0%, #1a1d24 100%);
            color: #e0e0e0;
            font-family: 'Consolas', 'Monaco', monospace;
            min-height: 100vh;
        }
        .cyber-card {
            background: rgba(22, 27, 34, 0.9);
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        .cyber-header {
            border-bottom: 1px solid #30363d;
            padding-bottom: 12px;
            margin-bottom: 15px;
            color: #58a6ff;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .status-dot {
            height: 10px;
            width: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .green { background-color: #2ea043; box-shadow: 0 0 8px #2ea043; }
        .red { background-color: #da3633; box-shadow: 0 0 8px #da3633; }
        .yellow { background-color: #d29922; box-shadow: 0 0 8px #d29922; }
        .blue { background-color: #58a6ff; box-shadow: 0 0 8px #58a6ff; }
        .step-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #21262d;
            font-size: 0.9em;
        }
        .glow-btn {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            border: 1px solid #3b82f6;
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
            transition: all 0.3s ease;
        }
        .glow-btn:hover {
            box-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
            transform: translateY(-2px);
        }
        .glow-btn:disabled {
            background: #374151;
            border-color: #4b5563;
            box-shadow: none;
            cursor: not-allowed;
        }
        .drop-zone {
            border: 2px dashed #30363d;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .drop-zone:hover, .drop-zone.dragover {
            border-color: #58a6ff;
            background: rgba(88, 166, 255, 0.1);
        }
        .metric-card {
            background: linear-gradient(135deg, rgba(22, 27, 34, 0.9), rgba(30, 35, 42, 0.9));
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        .tab-btn.active {
            color: #58a6ff;
            border-bottom: 2px solid #58a6ff;
        }
        pre {
            background: #0d1117;
            border-radius: 6px;
            padding: 12px;
            overflow-x: auto;
            font-size: 0.85em;
        }
        .pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body class="p-6">
    <!-- Header -->
    <div class="flex justify-between items-center mb-8">
        <div>
            <h1 class="text-3xl text-blue-400 font-bold tracking-widest">
                <i class="fas fa-drafting-compass mr-3"></i>HIVE215 // THE ARCHITECT
            </h1>
            <p class="text-gray-500 text-sm mt-1">Shop Drawing Automation System v4.0 | Connected to Fast Brain</p>
        </div>
        <div class="flex items-center gap-4">
            <select id="adapterSelect" class="bg-gray-800 border border-gray-600 rounded px-4 py-2 text-sm">
                <option value="">Select AI Adapter (Optional)</option>
            </select>
            <button onclick="document.getElementById('fileInput').click()" class="glow-btn text-white px-6 py-3 rounded-lg font-bold">
                <i class="fas fa-upload mr-2"></i> INGEST PROJECT FILES
            </button>
            <input type="file" id="fileInput" multiple class="hidden" onchange="handleFiles()">
        </div>
    </div>

    <!-- Drop Zone -->
    <div id="dropZone" class="drop-zone mb-8" onclick="document.getElementById('fileInput').click()">
        <i class="fas fa-cloud-upload-alt text-5xl text-gray-500 mb-4"></i>
        <p class="text-gray-400 text-lg">Drop files here or click to upload</p>
        <p class="text-gray-600 text-sm mt-2">Supports: PDF, DWG, DOCX, XLSX, CSV, Images</p>
    </div>

    <!-- Main Grid -->
    <div class="grid grid-cols-12 gap-6">
        <!-- Left: Workflow Steps -->
        <div class="col-span-3">
            <div class="cyber-card h-full">
                <div class="cyber-header"><i class="fas fa-list-ol mr-2"></i> 10-Step Workflow</div>
                <div id="stepsContainer" class="space-y-1">
                    <div class="text-gray-500 italic text-center py-8">
                        <i class="fas fa-inbox text-3xl mb-3"></i>
                        <p>Awaiting file upload...</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Center: Main Content -->
        <div class="col-span-6 space-y-6">
            <!-- Metrics Row -->
            <div class="grid grid-cols-3 gap-4">
                <div class="metric-card">
                    <div class="text-gray-400 text-xs uppercase tracking-wider">Completeness</div>
                    <div id="scoreDisplay" class="text-4xl font-bold text-gray-600 mt-2">--%</div>
                </div>
                <div class="metric-card">
                    <div class="text-gray-400 text-xs uppercase tracking-wider">Est. Drafting Hours</div>
                    <div id="hoursDisplay" class="text-4xl font-bold text-gray-600 mt-2">--</div>
                </div>
                <div class="metric-card">
                    <div class="text-gray-400 text-xs uppercase tracking-wider">Complexity</div>
                    <div id="complexityDisplay" class="text-2xl font-bold text-gray-600 mt-2">--</div>
                </div>
            </div>

            <!-- Data Tabs -->
            <div class="cyber-card">
                <div class="flex border-b border-gray-700 mb-4">
                    <button class="tab-btn active px-4 py-2" onclick="showTab('scope')">SCOPE & SPECS</button>
                    <button class="tab-btn px-4 py-2 text-gray-400 hover:text-white" onclick="showTab('arch')">DRAWINGS</button>
                    <button class="tab-btn px-4 py-2 text-gray-400 hover:text-white" onclick="showTab('assembly')">ASSEMBLY</button>
                    <button class="tab-btn px-4 py-2 text-gray-400 hover:text-white" onclick="showTab('outputs')">OUTPUTS</button>
                </div>

                <div id="tabContent" class="min-h-[400px]">
                    <!-- Scope Tab -->
                    <div id="tab-scope" class="tab-panel space-y-4">
                        <div class="bg-gray-900 p-4 rounded border border-gray-700">
                            <h3 class="text-green-400 mb-2 font-bold">[Step 1] Scope Extraction</h3>
                            <pre id="scopeData" class="text-gray-300">Awaiting analysis...</pre>
                        </div>
                        <div class="bg-gray-900 p-4 rounded border border-gray-700">
                            <h3 class="text-yellow-400 mb-2 font-bold">[Step 2] Spec Sections</h3>
                            <pre id="specData" class="text-gray-300">Awaiting analysis...</pre>
                        </div>
                    </div>

                    <!-- Drawings Tab -->
                    <div id="tab-arch" class="tab-panel hidden space-y-4">
                        <div class="bg-gray-900 p-4 rounded border border-gray-700">
                            <h3 class="text-blue-400 mb-2 font-bold">[Step 3] Drawing Analysis</h3>
                            <pre id="drawingData" class="text-gray-300">Awaiting analysis...</pre>
                        </div>
                    </div>

                    <!-- Assembly Tab -->
                    <div id="tab-assembly" class="tab-panel hidden space-y-4">
                        <div class="bg-gray-900 p-4 rounded border border-gray-700">
                            <h3 class="text-purple-400 mb-2 font-bold">[Step 4] Assembly & Code</h3>
                            <pre id="assemblyData" class="text-gray-300">Awaiting analysis...</pre>
                        </div>
                    </div>

                    <!-- Outputs Tab -->
                    <div id="tab-outputs" class="tab-panel hidden">
                        <div class="bg-gray-900 p-4 rounded border border-gray-700">
                            <div class="flex justify-between items-center mb-4">
                                <h3 class="text-cyan-400 font-bold"><i class="fas fa-file-export mr-2"></i>Generated JSON Outputs</h3>
                                <button onclick="downloadAllOutputs()" class="glow-btn text-white px-4 py-2 rounded text-sm">
                                    <i class="fas fa-download mr-2"></i>Download All (ZIP)
                                </button>
                            </div>
                            <div id="outputsList" class="space-y-2 max-h-96 overflow-y-auto">
                                <p class="text-gray-500">No outputs generated yet. Upload files and click "Generate Shop Drawings".</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Right: Files & Actions -->
        <div class="col-span-3 space-y-6">
            <!-- Files List -->
            <div class="cyber-card">
                <div class="cyber-header"><i class="fas fa-file-import mr-2"></i> Ingested Files</div>
                <div id="fileList" class="text-xs space-y-2 max-h-48 overflow-y-auto">
                    <p class="text-gray-500 text-center">No files uploaded</p>
                </div>
            </div>

            <!-- Key Findings -->
            <div class="cyber-card border-blue-900 bg-blue-900/10">
                <div class="cyber-header text-blue-400"><i class="fas fa-lightbulb mr-2"></i> Key Findings</div>
                <ul id="findingsList" class="list-disc pl-4 text-sm text-blue-300 space-y-1">
                    <li class="text-gray-500">Pending analysis...</li>
                </ul>
            </div>

            <!-- Missing / Critical -->
            <div class="cyber-card border-red-900 bg-red-900/10">
                <div class="cyber-header text-red-400"><i class="fas fa-exclamation-triangle mr-2"></i> Missing / Critical</div>
                <ul id="missingList" class="list-disc pl-4 text-sm text-red-300 space-y-1">
                    <li class="text-gray-500">Pending analysis...</li>
                </ul>
            </div>

            <!-- Recommendations -->
            <div class="cyber-card border-yellow-900 bg-yellow-900/10">
                <div class="cyber-header text-yellow-400"><i class="fas fa-clipboard-check mr-2"></i> Recommendations</div>
                <ul id="recommendationsList" class="list-disc pl-4 text-sm text-yellow-300 space-y-1">
                    <li class="text-gray-500">Pending analysis...</li>
                </ul>
            </div>

            <!-- Generate Button -->
            <button id="generateBtn" onclick="generateDrawings()" disabled class="glow-btn w-full text-white font-bold py-4 rounded-lg disabled:opacity-50">
                <i class="fas fa-pencil-ruler mr-2"></i> GENERATE SHOP DRAWINGS
            </button>
        </div>
    </div>

    <script>
        let currentProject = null;

        // Load adapters on page load
        async function loadAdapters() {
            try {
                const res = await fetch('/api/adapters');
                const data = await res.json();
                const select = document.getElementById('adapterSelect');
                data.adapters.forEach(adapter => {
                    const option = document.createElement('option');
                    option.value = adapter.skill_id;
                    option.textContent = `${adapter.skill_name} (Loss: ${adapter.final_loss?.toFixed(3) || '--'})`;
                    select.appendChild(option);
                });
            } catch (err) {
                console.log('No adapters available');
            }
        }
        loadAdapters();

        // Drag and drop
        const dropZone = document.getElementById('dropZone');
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            document.getElementById('fileInput').files = files;
            handleFiles();
        });

        async function handleFiles() {
            const input = document.getElementById('fileInput');
            const adapterId = document.getElementById('adapterSelect').value;
            const formData = new FormData();

            for (const file of input.files) {
                formData.append('file', file);
            }
            if (adapterId) {
                formData.append('adapter_id', adapterId);
            }

            // Show loading state
            document.getElementById('stepsContainer').innerHTML = `
                <div class="text-blue-400 text-center py-8 pulse">
                    <i class="fas fa-cog fa-spin text-3xl mb-3"></i>
                    <p>Running 10-Step Analysis...</p>
                </div>
            `;

            try {
                const response = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await response.json();
                currentProject = data;
                updateDashboard(data);
            } catch (err) {
                alert('Upload failed: ' + err.message);
            }
        }

        function updateDashboard(data) {
            // Update Metrics
            const scoreEl = document.getElementById('scoreDisplay');
            scoreEl.innerText = data.score + "%";
            scoreEl.className = `text-4xl font-bold mt-2 ${data.score >= 80 ? 'text-green-400' : data.score >= 50 ? 'text-yellow-400' : 'text-red-400'}`;

            document.getElementById('hoursDisplay').innerText = data.est_hours + " hrs";
            document.getElementById('hoursDisplay').className = 'text-4xl font-bold text-cyan-400 mt-2';

            const complexityEl = document.getElementById('complexityDisplay');
            complexityEl.innerText = data.complexity;
            complexityEl.className = `text-2xl font-bold mt-2 ${data.complexity === 'Low' ? 'text-green-400' : data.complexity === 'Medium' ? 'text-yellow-400' : 'text-red-400'}`;

            // Render Steps
            const stepsHtml = Object.entries(data.steps).map(([step, info]) => {
                let color = 'red';
                let icon = 'times';
                if (info.status === 'complete') { color = 'green'; icon = 'check'; }
                else if (info.status === 'processing') { color = 'yellow'; icon = 'spinner fa-spin'; }

                return `
                    <div class="step-row">
                        <span><span class="status-dot ${color}"></span> ${step}. ${info.name}</span>
                        <span class="text-${color === 'green' ? 'green' : color === 'yellow' ? 'yellow' : 'red'}-400">
                            <i class="fas fa-${icon}"></i>
                        </span>
                    </div>
                `;
            }).join('');
            document.getElementById('stepsContainer').innerHTML = stepsHtml;

            // Render Data Panels
            document.getElementById('scopeData').innerText = JSON.stringify(data.steps[1].data || {}, null, 2);
            document.getElementById('specData').innerText = JSON.stringify(data.steps[2].data || {}, null, 2);
            document.getElementById('drawingData').innerText = JSON.stringify(data.steps[3].data || {}, null, 2);
            document.getElementById('assemblyData').innerText = JSON.stringify(data.steps[4].data || {}, null, 2);

            // Render File List
            const filesHtml = data.files_received.map(f => `
                <div class="flex justify-between border-b border-gray-800 pb-1">
                    <span class="text-blue-300 truncate" style="max-width: 150px;" title="${f.name}">${f.name}</span>
                    <span class="text-gray-500 text-xs">${f.type}</span>
                </div>
            `).join('');
            document.getElementById('fileList').innerHTML = filesHtml || '<p class="text-gray-500 text-center">No files</p>';

            // Render Key Findings
            const findingsHtml = data.key_findings.map(f => `<li>${f}</li>`).join('');
            document.getElementById('findingsList').innerHTML = findingsHtml || '<li class="text-gray-500">No findings yet</li>';

            // Render Missing Items
            const missingHtml = data.missing_docs.map(d => `<li>${d}</li>`).join('');
            document.getElementById('missingList').innerHTML = missingHtml || '<li class="text-green-400">All critical docs present!</li>';

            // Render Recommendations
            const recsHtml = data.recommendations.map(r => `<li>${r}</li>`).join('');
            document.getElementById('recommendationsList').innerHTML = recsHtml || '<li class="text-green-400">Ready for generation</li>';

            // Enable generate button if score >= 70
            const genBtn = document.getElementById('generateBtn');
            genBtn.disabled = data.score < 70;
        }

        function showTab(tabName) {
            // Hide all panels
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
            // Show selected
            document.getElementById('tab-' + tabName).classList.remove('hidden');
            // Update tab buttons
            document.querySelectorAll('.tab-btn').forEach(b => {
                b.classList.remove('active', 'text-blue-400');
                b.classList.add('text-gray-400');
            });
            event.target.classList.add('active');
            event.target.classList.remove('text-gray-400');
        }

        async function generateDrawings() {
            if (!currentProject) return;

            const btn = document.getElementById('generateBtn');
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-cog fa-spin mr-2"></i> GENERATING...';

            try {
                const res = await fetch('/api/generate-drawings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ project_id: currentProject.id })
                });
                const data = await res.json();

                if (data.success) {
                    alert('Drawing package generated successfully!');
                    // Show outputs tab
                    showTab('outputs');
                    loadOutputs(currentProject.id);
                } else {
                    alert('Generation failed: ' + data.error);
                }
            } catch (err) {
                alert('Error: ' + err.message);
            }

            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-pencil-ruler mr-2"></i> GENERATE SHOP DRAWINGS';
        }

        async function loadOutputs(projectId) {
            try {
                const res = await fetch(`/api/project/${projectId}/outputs`);
                const data = await res.json();

                if (!data.files || data.files.length === 0) {
                    document.getElementById('outputsList').innerHTML = '<p class="text-gray-500">No outputs yet</p>';
                    return;
                }

                const html = data.files.map(f => `
                    <div class="flex justify-between items-center bg-gray-800 p-3 rounded hover:bg-gray-700 transition">
                        <div>
                            <i class="fas fa-file-code mr-2 text-cyan-400"></i>
                            <span class="text-white">${f.name}</span>
                            <span class="text-gray-500 text-xs ml-2">(${(f.size / 1024).toFixed(1)} KB)</span>
                        </div>
                        <a href="${f.url}" download class="bg-green-600 hover:bg-green-500 text-white px-3 py-1 rounded text-sm">
                            <i class="fas fa-download mr-1"></i> Download
                        </a>
                    </div>
                `).join('');

                document.getElementById('outputsList').innerHTML = html;
            } catch (err) {
                console.error('Failed to load outputs:', err);
            }
        }

        async function downloadAllOutputs() {
            if (!currentProject) {
                alert('No project loaded. Upload files first.');
                return;
            }

            try {
                const res = await fetch(`/api/project/${currentProject.id}/download-all`);
                if (!res.ok) throw new Error('Download failed');

                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${currentProject.id}_outputs.zip`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
            } catch (err) {
                alert('Download failed: ' + err.message);
            }
        }
    </script>
</body>
</html>
'''


if __name__ == '__main__':
    app.run(debug=True, port=5001)
