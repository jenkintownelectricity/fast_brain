"""
Dashboard Database Module
=========================
SQLite-based persistent storage for the Unified Dashboard.

Provides scalable, persistent storage for:
- Skills (business profiles, custom skills)
- Golden Prompts (custom overrides)
- API Keys (encrypted)
- Platform Connections
- Activity Log
- Training Data metadata
- Voice Projects

Schema designed for scalability with proper indexes and foreign keys.
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from contextlib import contextmanager
import hashlib
import base64

# Database file location
DB_PATH = Path(__file__).parent / "dashboard.db"

# Simple encryption key (in production, use proper secrets management)
ENCRYPTION_KEY = os.environ.get("DASHBOARD_ENCRYPTION_KEY", "fast_brain_default_key_2024")


def _encrypt(value: str) -> str:
    """Simple obfuscation for API keys (use proper encryption in production)."""
    if not value:
        return ""
    key_bytes = hashlib.sha256(ENCRYPTION_KEY.encode()).digest()
    encrypted = []
    for i, char in enumerate(value):
        encrypted.append(chr(ord(char) ^ key_bytes[i % len(key_bytes)]))
    return base64.b64encode("".join(encrypted).encode("utf-8")).decode("utf-8")


def _decrypt(value: str) -> str:
    """Decrypt obfuscated API keys."""
    if not value:
        return ""
    try:
        decoded = base64.b64decode(value.encode("utf-8")).decode("utf-8")
        key_bytes = hashlib.sha256(ENCRYPTION_KEY.encode()).digest()
        decrypted = []
        for i, char in enumerate(decoded):
            decrypted.append(chr(ord(char) ^ key_bytes[i % len(key_bytes)]))
        return "".join(decrypted)
    except Exception:
        return ""


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_db():
    """Initialize the database with all required tables."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Skills table - unified storage for all skills
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                business_type TEXT DEFAULT 'General',
                system_prompt TEXT,
                greeting TEXT,
                personality TEXT,
                voice_description TEXT,
                knowledge TEXT,  -- JSON array
                is_builtin INTEGER DEFAULT 0,
                status TEXT DEFAULT 'draft',  -- draft, training, deployed
                training_examples INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Custom prompts table - overrides for golden prompts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS custom_prompts (
                skill_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tokens_estimate INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # API keys table (encrypted)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                provider TEXT PRIMARY KEY,
                api_key_encrypted TEXT NOT NULL,
                is_valid INTEGER DEFAULT 1,
                last_verified TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Platform connections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS platform_connections (
                platform_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'disconnected',
                config TEXT,  -- JSON object
                last_connected TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Activity log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                icon TEXT,
                category TEXT,  -- skill, api, platform, training, etc.
                metadata TEXT,  -- JSON object for additional data
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Training data metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_id TEXT NOT NULL,
                file_path TEXT,
                examples_count INTEGER DEFAULT 0,
                source TEXT,  -- upload, generated, feedback
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE
            )
        """)

        # Voice projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voice_projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                base_provider TEXT,
                base_voice TEXT,
                settings TEXT,  -- JSON object (pitch, speed, emotion, etc.)
                training_status TEXT DEFAULT 'pending',  -- pending, training, ready
                samples_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Dashboard config table (for misc settings)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dashboard_config (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_skills_status ON skills(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_skills_type ON skills(business_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_created ON activity_log(created_at DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_skill ON training_data(skill_id)")

        conn.commit()

        # Initialize default platform connections if not exist
        _init_default_platforms(cursor)
        conn.commit()


def _init_default_platforms(cursor):
    """Initialize default platform connections."""
    platforms = [
        ("livekit", "LiveKit", "Real-time voice and video with agents SDK"),
        ("vapi", "Vapi", "Voice AI platform for building phone agents"),
        ("twilio", "Twilio", "Cloud communications platform for voice calls"),
        ("retell", "Retell AI", "Conversational voice AI for customer interactions"),
        ("bland", "Bland AI", "AI phone agents for enterprises"),
        ("vocode", "Vocode", "Open-source voice agent framework"),
        ("daily", "Daily.co", "Real-time video/audio platform with Pipecat integration"),
        ("websocket", "Custom WebSocket", "Connect to any WebSocket-based voice service"),
    ]

    for platform_id, name, description in platforms:
        cursor.execute("""
            INSERT OR IGNORE INTO platform_connections (platform_id, name, description, config)
            VALUES (?, ?, ?, '{}')
        """, (platform_id, name, description))


# =============================================================================
# SKILLS CRUD Operations
# =============================================================================

def get_all_skills() -> List[Dict]:
    """Get all skills from database."""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT * FROM skills ORDER BY is_builtin DESC, name ASC
        """)
        skills = []
        for row in cursor.fetchall():
            skill = dict(row)
            if skill.get("knowledge"):
                skill["knowledge"] = json.loads(skill["knowledge"])
            skills.append(skill)
        return skills


def get_skill(skill_id: str) -> Optional[Dict]:
    """Get a single skill by ID."""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,))
        row = cursor.fetchone()
        if row:
            skill = dict(row)
            if skill.get("knowledge"):
                skill["knowledge"] = json.loads(skill["knowledge"])
            return skill
        return None


def save_skill(skill: Dict) -> bool:
    """Save or update a skill."""
    with get_db() as conn:
        knowledge = skill.get("knowledge", [])
        if isinstance(knowledge, list):
            knowledge = json.dumps(knowledge)

        conn.execute("""
            INSERT INTO skills (id, name, description, business_type, system_prompt,
                               greeting, personality, voice_description, knowledge,
                               is_builtin, status, training_examples, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                business_type = excluded.business_type,
                system_prompt = excluded.system_prompt,
                greeting = excluded.greeting,
                personality = excluded.personality,
                voice_description = excluded.voice_description,
                knowledge = excluded.knowledge,
                status = excluded.status,
                training_examples = excluded.training_examples,
                updated_at = CURRENT_TIMESTAMP
        """, (
            skill.get("id"),
            skill.get("name"),
            skill.get("description"),
            skill.get("business_type", "General"),
            skill.get("system_prompt"),
            skill.get("greeting"),
            skill.get("personality"),
            skill.get("voice_description"),
            knowledge,
            skill.get("is_builtin", 0),
            skill.get("status", "draft"),
            skill.get("training_examples", 0),
        ))
        return True


def delete_skill(skill_id: str) -> bool:
    """Delete a skill."""
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM skills WHERE id = ? AND is_builtin = 0", (skill_id,))
        return cursor.rowcount > 0


def update_skill_status(skill_id: str, status: str) -> bool:
    """Update skill status (draft, training, deployed)."""
    with get_db() as conn:
        conn.execute("""
            UPDATE skills SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
        """, (status, skill_id))
        return True


# =============================================================================
# CUSTOM PROMPTS CRUD Operations
# =============================================================================

def get_custom_prompts() -> Dict[str, str]:
    """Get all custom prompt overrides."""
    with get_db() as conn:
        cursor = conn.execute("SELECT skill_id, content FROM custom_prompts")
        return {row["skill_id"]: row["content"] for row in cursor.fetchall()}


def get_custom_prompt(skill_id: str) -> Optional[str]:
    """Get a custom prompt override for a skill."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT content FROM custom_prompts WHERE skill_id = ?", (skill_id,)
        )
        row = cursor.fetchone()
        return row["content"] if row else None


def save_custom_prompt(skill_id: str, content: str) -> bool:
    """Save a custom prompt override."""
    tokens_estimate = len(content) // 4  # Rough estimate
    with get_db() as conn:
        conn.execute("""
            INSERT INTO custom_prompts (skill_id, content, tokens_estimate, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(skill_id) DO UPDATE SET
                content = excluded.content,
                tokens_estimate = excluded.tokens_estimate,
                updated_at = CURRENT_TIMESTAMP
        """, (skill_id, content, tokens_estimate))
        return True


def delete_custom_prompt(skill_id: str) -> bool:
    """Delete a custom prompt override (reset to default)."""
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM custom_prompts WHERE skill_id = ?", (skill_id,))
        return cursor.rowcount > 0


# =============================================================================
# API KEYS CRUD Operations
# =============================================================================

def get_api_keys() -> Dict[str, str]:
    """Get all API keys (decrypted)."""
    with get_db() as conn:
        cursor = conn.execute("SELECT provider, api_key_encrypted FROM api_keys")
        return {row["provider"]: _decrypt(row["api_key_encrypted"]) for row in cursor.fetchall()}


def get_api_key(provider: str) -> Optional[str]:
    """Get a specific API key (decrypted)."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT api_key_encrypted FROM api_keys WHERE provider = ?", (provider,)
        )
        row = cursor.fetchone()
        return _decrypt(row["api_key_encrypted"]) if row else None


def save_api_key(provider: str, api_key: str) -> bool:
    """Save an API key (encrypted)."""
    encrypted = _encrypt(api_key)
    with get_db() as conn:
        conn.execute("""
            INSERT INTO api_keys (provider, api_key_encrypted, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(provider) DO UPDATE SET
                api_key_encrypted = excluded.api_key_encrypted,
                updated_at = CURRENT_TIMESTAMP
        """, (provider, encrypted))
        return True


def delete_api_key(provider: str) -> bool:
    """Delete an API key."""
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM api_keys WHERE provider = ?", (provider,))
        return cursor.rowcount > 0


# =============================================================================
# PLATFORM CONNECTIONS CRUD Operations
# =============================================================================

def get_all_platforms() -> Dict[str, Dict]:
    """Get all platform connections."""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM platform_connections")
        platforms = {}
        for row in cursor.fetchall():
            platform = dict(row)
            platform["config"] = json.loads(platform.get("config") or "{}")
            platforms[platform["platform_id"]] = platform
        return platforms


def get_platform(platform_id: str) -> Optional[Dict]:
    """Get a specific platform connection."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT * FROM platform_connections WHERE platform_id = ?", (platform_id,)
        )
        row = cursor.fetchone()
        if row:
            platform = dict(row)
            platform["config"] = json.loads(platform.get("config") or "{}")
            return platform
        return None


def save_platform(platform_id: str, config: Dict, status: str = None) -> bool:
    """Update platform connection config and status."""
    with get_db() as conn:
        if status:
            conn.execute("""
                UPDATE platform_connections
                SET config = ?, status = ?, updated_at = CURRENT_TIMESTAMP,
                    last_connected = CASE WHEN ? = 'connected' THEN CURRENT_TIMESTAMP ELSE last_connected END
                WHERE platform_id = ?
            """, (json.dumps(config), status, status, platform_id))
        else:
            conn.execute("""
                UPDATE platform_connections
                SET config = ?, updated_at = CURRENT_TIMESTAMP
                WHERE platform_id = ?
            """, (json.dumps(config), platform_id))
        return True


def update_platform_status(platform_id: str, status: str) -> bool:
    """Update platform connection status."""
    with get_db() as conn:
        conn.execute("""
            UPDATE platform_connections
            SET status = ?, updated_at = CURRENT_TIMESTAMP,
                last_connected = CASE WHEN ? = 'connected' THEN CURRENT_TIMESTAMP ELSE last_connected END
            WHERE platform_id = ?
        """, (status, status, platform_id))
        return True


# =============================================================================
# ACTIVITY LOG Operations
# =============================================================================

def add_activity(message: str, icon: str = "", category: str = None, metadata: Dict = None) -> bool:
    """Add an activity to the log."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO activity_log (message, icon, category, metadata)
            VALUES (?, ?, ?, ?)
        """, (message, icon, category, json.dumps(metadata) if metadata else None))
        return True


def get_recent_activity(limit: int = 20) -> List[Dict]:
    """Get recent activity log entries."""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT * FROM activity_log ORDER BY created_at DESC LIMIT ?
        """, (limit,))
        activities = []
        for row in cursor.fetchall():
            activity = dict(row)
            if activity.get("metadata"):
                activity["metadata"] = json.loads(activity["metadata"])
            # Calculate time ago
            created = datetime.fromisoformat(activity["created_at"])
            delta = datetime.now() - created
            if delta.seconds < 60:
                activity["ago"] = "just now"
            elif delta.seconds < 3600:
                activity["ago"] = f"{delta.seconds // 60} minutes ago"
            elif delta.seconds < 86400:
                activity["ago"] = f"{delta.seconds // 3600} hours ago"
            else:
                activity["ago"] = f"{delta.days} days ago"
            activities.append(activity)
        return activities


def clear_old_activity(days: int = 30) -> int:
    """Clear activity log entries older than specified days."""
    with get_db() as conn:
        cursor = conn.execute("""
            DELETE FROM activity_log
            WHERE created_at < datetime('now', '-' || ? || ' days')
        """, (days,))
        return cursor.rowcount


# =============================================================================
# DASHBOARD CONFIG Operations
# =============================================================================

def get_config(key: str, default: Any = None) -> Any:
    """Get a dashboard config value."""
    with get_db() as conn:
        cursor = conn.execute("SELECT value FROM dashboard_config WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            try:
                return json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                return row["value"]
        return default


def set_config(key: str, value: Any) -> bool:
    """Set a dashboard config value."""
    with get_db() as conn:
        value_str = json.dumps(value) if not isinstance(value, str) else value
        conn.execute("""
            INSERT INTO dashboard_config (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
        """, (key, value_str))
        return True


# =============================================================================
# VOICE PROJECTS Operations
# =============================================================================

def get_all_voice_projects() -> List[Dict]:
    """Get all voice projects."""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM voice_projects ORDER BY created_at DESC")
        projects = []
        for row in cursor.fetchall():
            project = dict(row)
            if project.get("settings"):
                project["settings"] = json.loads(project["settings"])
            projects.append(project)
        return projects


def save_voice_project(project: Dict) -> bool:
    """Save or update a voice project."""
    settings = project.get("settings", {})
    if isinstance(settings, dict):
        settings = json.dumps(settings)

    with get_db() as conn:
        conn.execute("""
            INSERT INTO voice_projects (id, name, description, base_provider, base_voice,
                                        settings, training_status, samples_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                base_provider = excluded.base_provider,
                base_voice = excluded.base_voice,
                settings = excluded.settings,
                training_status = excluded.training_status,
                samples_count = excluded.samples_count,
                updated_at = CURRENT_TIMESTAMP
        """, (
            project.get("id"),
            project.get("name"),
            project.get("description"),
            project.get("base_provider"),
            project.get("base_voice"),
            settings,
            project.get("training_status", "pending"),
            project.get("samples_count", 0),
        ))
        return True


def delete_voice_project(project_id: str) -> bool:
    """Delete a voice project."""
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM voice_projects WHERE id = ?", (project_id,))
        return cursor.rowcount > 0


# =============================================================================
# TRAINING DATA Operations
# =============================================================================

def add_training_data(skill_id: str, file_path: str, examples_count: int, source: str = "upload") -> bool:
    """Add training data metadata."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO training_data (skill_id, file_path, examples_count, source)
            VALUES (?, ?, ?, ?)
        """, (skill_id, file_path, examples_count, source))
        # Update skill training examples count
        conn.execute("""
            UPDATE skills SET training_examples = (
                SELECT COALESCE(SUM(examples_count), 0) FROM training_data WHERE skill_id = ?
            ) WHERE id = ?
        """, (skill_id, skill_id))
        return True


def get_training_data(skill_id: str) -> List[Dict]:
    """Get training data for a skill."""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT * FROM training_data WHERE skill_id = ? ORDER BY created_at DESC
        """, (skill_id,))
        return [dict(row) for row in cursor.fetchall()]


# =============================================================================
# Initialization
# =============================================================================

# Initialize the database on import
init_db()


if __name__ == "__main__":
    # Test the database
    print("Testing dashboard database...")

    # Test skill operations
    test_skill = {
        "id": "test-skill",
        "name": "Test Skill",
        "description": "A test skill",
        "system_prompt": "You are a test assistant.",
        "knowledge": ["fact1", "fact2"],
    }
    save_skill(test_skill)
    skill = get_skill("test-skill")
    print(f"Saved and retrieved skill: {skill['name']}")

    # Test custom prompt
    save_custom_prompt("test-skill", "Custom prompt content here")
    prompt = get_custom_prompt("test-skill")
    print(f"Saved and retrieved custom prompt: {prompt[:30]}...")

    # Test API key
    save_api_key("test-provider", "sk-test-12345")
    key = get_api_key("test-provider")
    print(f"Saved and retrieved API key: {key}")

    # Test activity
    add_activity("Test activity message", "✓", "test")
    activities = get_recent_activity(5)
    print(f"Recent activities: {len(activities)}")

    # Cleanup test data
    delete_skill("test-skill")
    delete_custom_prompt("test-skill")
    delete_api_key("test-provider")

    print("Database tests completed successfully!")
