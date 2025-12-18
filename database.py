"""
HIVE215 Skills Database
=======================
Unified SQLite database for persistent skill storage, configurations, and activity.

This replaces the fragmented storage systems:
- In-memory dicts (lost on restart)
- JSON files (scattered)
- Python module constants

Schema designed for scalability and easy API integration.
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

# Database location - use Modal volume for persistence
DB_PATH = os.environ.get('HIVE215_DB_PATH', '/data/hive215.db')

# Ensure directory exists
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


def get_connection():
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize the database schema."""
    with get_db() as conn:
        cursor = conn.cursor()

        # =================================================================
        # SKILLS TABLE - Unified skill storage
        # =================================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                skill_type TEXT DEFAULT 'custom',
                system_prompt TEXT,
                knowledge TEXT,  -- JSON array
                voice_config TEXT,  -- JSON object
                is_builtin INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,

                -- Metrics (updated periodically)
                total_requests INTEGER DEFAULT 0,
                avg_latency_ms REAL DEFAULT 0,
                satisfaction_rate REAL DEFAULT 0
            )
        ''')

        # =================================================================
        # GOLDEN PROMPTS TABLE - Custom prompt overrides
        # =================================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS golden_prompts (
                skill_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tokens_estimate INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (skill_id) REFERENCES skills(id)
            )
        ''')

        # =================================================================
        # CONFIGURATIONS TABLE - Key-value settings
        # =================================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS configurations (
                key TEXT PRIMARY KEY,
                value TEXT,  -- JSON for complex values
                category TEXT DEFAULT 'general',
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # =================================================================
        # PLATFORM CONNECTIONS TABLE
        # =================================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS platforms (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT DEFAULT 'disconnected',
                config TEXT,  -- JSON object
                description TEXT,
                last_connected TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # =================================================================
        # ACTIVITY LOG TABLE
        # =================================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                icon TEXT,
                category TEXT DEFAULT 'general',
                metadata TEXT,  -- JSON for extra data
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # =================================================================
        # TRAINING DATA TABLE - For fine-tuning
        # =================================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                rating INTEGER,  -- 1-5 quality rating
                corrected_response TEXT,  -- If user provided correction
                metadata TEXT,  -- JSON for latency, system used, etc.
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (skill_id) REFERENCES skills(id)
            )
        ''')

        # =================================================================
        # API KEYS TABLE - Encrypted storage
        # =================================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                provider TEXT PRIMARY KEY,
                api_key TEXT NOT NULL,
                is_valid INTEGER DEFAULT 1,
                last_validated TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # =================================================================
        # VOICE PROJECTS TABLE - Custom voice cloning/training
        # =================================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS voice_projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                provider TEXT DEFAULT 'elevenlabs',
                base_voice TEXT,
                status TEXT DEFAULT 'draft',
                settings TEXT,  -- JSON: pitch, speed, emotion, style
                samples TEXT,   -- JSON array of sample metadata
                voice_id TEXT,  -- External voice ID from provider (after training)
                skill_id TEXT,  -- Linked skill/agent
                training_started TEXT,
                training_completed TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (skill_id) REFERENCES skills(id)
            )
        ''')

        # =================================================================
        # VOICE SAMPLES TABLE - Audio samples for training
        # =================================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS voice_samples (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT,  -- Path on Modal volume
                transcript TEXT,
                duration_ms INTEGER,
                emotion TEXT DEFAULT 'neutral',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES voice_projects(id)
            )
        ''')

        # =================================================================
        # API CONNECTIONS TABLE - Outgoing API integrations
        # =================================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_connections (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                api_key TEXT,
                headers TEXT,  -- JSON object for custom headers
                auth_type TEXT DEFAULT 'bearer',  -- bearer, api_key, basic, none
                status TEXT DEFAULT 'disconnected',
                last_tested TEXT,
                last_error TEXT,
                webhook_url TEXT,  -- For receiving callbacks
                webhook_secret TEXT,
                settings TEXT,  -- JSON for additional settings
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indexes for common queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_skills_type ON skills(skill_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_skills_active ON skills(is_active)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_activity_created ON activity_log(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_skill ON training_data(skill_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_voice_projects_status ON voice_projects(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_voice_projects_skill ON voice_projects(skill_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_voice_samples_project ON voice_samples(project_id)')

        conn.commit()


# =============================================================================
# SKILLS CRUD OPERATIONS
# =============================================================================

def create_skill(
    skill_id: str,
    name: str,
    description: str = "",
    skill_type: str = "custom",
    system_prompt: str = "",
    knowledge: List[str] = None,
    voice_config: Dict = None,
    is_builtin: bool = False
) -> Dict:
    """Create a new skill."""
    with get_db() as conn:
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute('''
            INSERT OR REPLACE INTO skills
            (id, name, description, skill_type, system_prompt, knowledge, voice_config, is_builtin, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            skill_id,
            name,
            description,
            skill_type,
            system_prompt,
            json.dumps(knowledge or []),
            json.dumps(voice_config or {}),
            1 if is_builtin else 0,
            now,
            now
        ))

        return get_skill(skill_id)


def get_skill(skill_id: str) -> Optional[Dict]:
    """Get a skill by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM skills WHERE id = ?', (skill_id,))
        row = cursor.fetchone()

        if row:
            return {
                **dict(row),
                'knowledge': json.loads(row['knowledge'] or '[]'),
                'voice_config': json.loads(row['voice_config'] or '{}'),
                'is_builtin': bool(row['is_builtin']),
                'is_active': bool(row['is_active'])
            }
        return None


def get_all_skills(include_inactive: bool = False) -> List[Dict]:
    """Get all skills."""
    with get_db() as conn:
        cursor = conn.cursor()

        if include_inactive:
            cursor.execute('SELECT * FROM skills ORDER BY name')
        else:
            cursor.execute('SELECT * FROM skills WHERE is_active = 1 ORDER BY name')

        skills = []
        for row in cursor.fetchall():
            skills.append({
                **dict(row),
                'knowledge': json.loads(row['knowledge'] or '[]'),
                'voice_config': json.loads(row['voice_config'] or '{}'),
                'is_builtin': bool(row['is_builtin']),
                'is_active': bool(row['is_active'])
            })

        return skills


def update_skill(skill_id: str, **kwargs) -> Optional[Dict]:
    """Update a skill's properties."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Build dynamic update query
        updates = []
        values = []

        for key, value in kwargs.items():
            if key in ('knowledge', 'voice_config'):
                value = json.dumps(value)
            updates.append(f'{key} = ?')
            values.append(value)

        updates.append('updated_at = ?')
        values.append(datetime.now().isoformat())
        values.append(skill_id)

        cursor.execute(f'''
            UPDATE skills SET {', '.join(updates)} WHERE id = ?
        ''', values)

        return get_skill(skill_id)


def delete_skill(skill_id: str) -> bool:
    """Delete a skill (soft delete by setting inactive)."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Check if builtin
        cursor.execute('SELECT is_builtin FROM skills WHERE id = ?', (skill_id,))
        row = cursor.fetchone()

        if row and row['is_builtin']:
            return False  # Can't delete builtin skills

        cursor.execute('UPDATE skills SET is_active = 0, updated_at = ? WHERE id = ?',
                      (datetime.now().isoformat(), skill_id))

        return cursor.rowcount > 0


def hard_delete_skill(skill_id: str) -> bool:
    """Permanently delete a skill."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM skills WHERE id = ? AND is_builtin = 0', (skill_id,))
        return cursor.rowcount > 0


# =============================================================================
# GOLDEN PROMPTS OPERATIONS
# =============================================================================

def save_golden_prompt(skill_id: str, content: str) -> Dict:
    """Save a custom golden prompt."""
    with get_db() as conn:
        cursor = conn.cursor()
        tokens = len(content) // 4  # Rough estimate
        now = datetime.now().isoformat()

        cursor.execute('''
            INSERT OR REPLACE INTO golden_prompts (skill_id, content, tokens_estimate, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (skill_id, content, tokens, now, now))

        return {
            'skill_id': skill_id,
            'content': content,
            'tokens_estimate': tokens,
            'updated_at': now
        }


def get_golden_prompt(skill_id: str) -> Optional[Dict]:
    """Get a custom golden prompt."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM golden_prompts WHERE skill_id = ?', (skill_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_golden_prompt(skill_id: str) -> bool:
    """Delete a custom golden prompt (revert to default)."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM golden_prompts WHERE skill_id = ?', (skill_id,))
        return cursor.rowcount > 0


def get_all_golden_prompts() -> List[Dict]:
    """Get all custom golden prompts."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM golden_prompts')
        return [dict(row) for row in cursor.fetchall()]


# =============================================================================
# CONFIGURATION OPERATIONS
# =============================================================================

def set_config(key: str, value: Any, category: str = 'general') -> None:
    """Set a configuration value."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO configurations (key, value, category, updated_at)
            VALUES (?, ?, ?, ?)
        ''', (key, json.dumps(value), category, datetime.now().isoformat()))


def get_config(key: str, default: Any = None) -> Any:
    """Get a configuration value."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT value FROM configurations WHERE key = ?', (key,))
        row = cursor.fetchone()

        if row:
            return json.loads(row['value'])
        return default


def get_configs_by_category(category: str) -> Dict[str, Any]:
    """Get all configurations in a category."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT key, value FROM configurations WHERE category = ?', (category,))
        return {row['key']: json.loads(row['value']) for row in cursor.fetchall()}


# =============================================================================
# ACTIVITY LOG OPERATIONS
# =============================================================================

def add_activity(message: str, icon: str = "", category: str = "general", metadata: Dict = None) -> int:
    """Add an activity log entry."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO activity_log (message, icon, category, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (message, icon, category, json.dumps(metadata or {}), datetime.now().isoformat()))
        return cursor.lastrowid


def get_recent_activity(limit: int = 20, category: str = None) -> List[Dict]:
    """Get recent activity log entries."""
    with get_db() as conn:
        cursor = conn.cursor()

        if category:
            cursor.execute('''
                SELECT * FROM activity_log WHERE category = ?
                ORDER BY created_at DESC LIMIT ?
            ''', (category, limit))
        else:
            cursor.execute('''
                SELECT * FROM activity_log ORDER BY created_at DESC LIMIT ?
            ''', (limit,))

        activities = []
        now = datetime.now()

        for row in cursor.fetchall():
            activity = dict(row)
            activity['metadata'] = json.loads(row['metadata'] or '{}')

            # Calculate "ago" time
            try:
                created = datetime.fromisoformat(row['created_at'])
                delta = now - created
                if delta.seconds < 60:
                    activity['ago'] = 'just now'
                elif delta.seconds < 3600:
                    activity['ago'] = f'{delta.seconds // 60} minutes ago'
                elif delta.seconds < 86400:
                    activity['ago'] = f'{delta.seconds // 3600} hours ago'
                else:
                    activity['ago'] = f'{delta.days} days ago'
            except:
                activity['ago'] = 'unknown'

            activities.append(activity)

        return activities


# =============================================================================
# TRAINING DATA OPERATIONS
# =============================================================================

def add_training_example(
    skill_id: str,
    user_message: str,
    assistant_response: str,
    rating: int = None,
    corrected_response: str = None,
    metadata: Dict = None
) -> int:
    """Add a training data example."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO training_data
            (skill_id, user_message, assistant_response, rating, corrected_response, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            skill_id,
            user_message,
            assistant_response,
            rating,
            corrected_response,
            json.dumps(metadata or {}),
            datetime.now().isoformat()
        ))
        return cursor.lastrowid


def get_training_data(skill_id: str = None, min_rating: int = None) -> List[Dict]:
    """Get training data, optionally filtered."""
    with get_db() as conn:
        cursor = conn.cursor()

        query = 'SELECT * FROM training_data WHERE 1=1'
        params = []

        if skill_id:
            query += ' AND skill_id = ?'
            params.append(skill_id)

        if min_rating:
            query += ' AND rating >= ?'
            params.append(min_rating)

        query += ' ORDER BY created_at DESC'

        cursor.execute(query, params)

        return [{
            **dict(row),
            'metadata': json.loads(row['metadata'] or '{}')
        } for row in cursor.fetchall()]


def export_training_jsonl(skill_id: str = None, min_rating: int = 4) -> str:
    """Export training data as JSONL for fine-tuning."""
    data = get_training_data(skill_id, min_rating)

    lines = []
    for item in data:
        # Use corrected response if available
        response = item.get('corrected_response') or item['assistant_response']

        lines.append(json.dumps({
            'messages': [
                {'role': 'user', 'content': item['user_message']},
                {'role': 'assistant', 'content': response}
            ]
        }))

    return '\n'.join(lines)


# =============================================================================
# PLATFORM OPERATIONS
# =============================================================================

def save_platform(platform_id: str, name: str, config: Dict, description: str = "", status: str = "disconnected") -> Dict:
    """Save platform configuration."""
    with get_db() as conn:
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute('''
            INSERT OR REPLACE INTO platforms (id, name, status, config, description, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (platform_id, name, status, json.dumps(config), description, now))

        return get_platform(platform_id)


def get_platform(platform_id: str) -> Optional[Dict]:
    """Get a platform by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM platforms WHERE id = ?', (platform_id,))
        row = cursor.fetchone()

        if row:
            return {
                **dict(row),
                'config': json.loads(row['config'] or '{}')
            }
        return None


def get_all_platforms() -> List[Dict]:
    """Get all platforms."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM platforms ORDER BY name')

        return [{
            **dict(row),
            'config': json.loads(row['config'] or '{}')
        } for row in cursor.fetchall()]


def update_platform_status(platform_id: str, status: str) -> bool:
    """Update platform connection status."""
    with get_db() as conn:
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute('''
            UPDATE platforms SET status = ?, last_connected = ?, updated_at = ? WHERE id = ?
        ''', (status, now if status == 'connected' else None, now, platform_id))

        return cursor.rowcount > 0


# =============================================================================
# API KEY OPERATIONS
# =============================================================================

def save_api_key(provider: str, api_key: str) -> None:
    """Save an API key."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO api_keys (provider, api_key, updated_at)
            VALUES (?, ?, ?)
        ''', (provider, api_key, datetime.now().isoformat()))


def get_api_key(provider: str) -> Optional[str]:
    """Get an API key."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT api_key FROM api_keys WHERE provider = ?', (provider,))
        row = cursor.fetchone()
        return row['api_key'] if row else None


def get_all_api_keys() -> Dict[str, str]:
    """Get all API keys (masked)."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT provider, api_key FROM api_keys')

        keys = {}
        for row in cursor.fetchall():
            key = row['api_key']
            # Mask the key for display
            if len(key) > 8:
                keys[row['provider']] = f"{key[:4]}...{key[-4:]}"
            else:
                keys[row['provider']] = "***"

        return keys


# =============================================================================
# SEED DATA - Initialize with built-in skills
# =============================================================================

def seed_builtin_skills():
    """Seed the database with built-in skills from golden_prompts."""
    try:
        from golden_prompts import SKILL_MANUALS, TOKEN_ESTIMATES

        builtin_skills = {
            'general': {
                'name': 'General Assistant',
                'description': 'Helpful general-purpose assistant',
                'skill_type': 'general'
            },
            'receptionist': {
                'name': 'Professional Receptionist',
                'description': 'Expert phone answering and call handling',
                'skill_type': 'service'
            },
            'electrician': {
                'name': 'Electrician Assistant',
                'description': 'Expert in electrical services and scheduling',
                'skill_type': 'trade'
            },
            'plumber': {
                'name': 'Plumber Assistant',
                'description': 'Expert in plumbing services',
                'skill_type': 'trade'
            },
            'lawyer': {
                'name': 'Legal Intake Assistant',
                'description': 'Professional legal intake and scheduling',
                'skill_type': 'professional'
            },
            'solar': {
                'name': 'Solar Sales Assistant',
                'description': 'Solar installation qualification and scheduling',
                'skill_type': 'sales'
            },
            'tara-sales': {
                'name': "Tara's Sales Assistant",
                'description': 'Sales assistant for TheDashTool demos',
                'skill_type': 'sales'
            }
        }

        for skill_id, meta in builtin_skills.items():
            system_prompt = SKILL_MANUALS.get(skill_id, '')

            create_skill(
                skill_id=skill_id,
                name=meta['name'],
                description=meta['description'],
                skill_type=meta['skill_type'],
                system_prompt=system_prompt,
                is_builtin=True
            )

        print(f"Seeded {len(builtin_skills)} built-in skills")

    except ImportError as e:
        print(f"Could not import golden_prompts: {e}")


def seed_default_platforms():
    """Seed default platform configurations."""
    platforms = {
        'livekit': {
            'name': 'LiveKit',
            'description': 'Real-time voice and video with agents SDK',
            'config': {'url': '', 'api_key': '', 'api_secret': ''}
        },
        'vapi': {
            'name': 'Vapi',
            'description': 'Voice AI platform for building phone agents',
            'config': {'api_key': '', 'assistant_id': ''}
        },
        'twilio': {
            'name': 'Twilio',
            'description': 'Cloud communications platform for voice calls',
            'config': {'account_sid': '', 'auth_token': '', 'phone_number': ''}
        },
        'retell': {
            'name': 'Retell AI',
            'description': 'Conversational voice AI for customer interactions',
            'config': {'api_key': '', 'agent_id': ''}
        },
        'daily': {
            'name': 'Daily.co',
            'description': 'Real-time video/audio platform with Pipecat integration',
            'config': {'api_key': '', 'room_url': ''}
        }
    }

    for platform_id, data in platforms.items():
        if not get_platform(platform_id):
            save_platform(
                platform_id,
                data['name'],
                data['config'],
                data['description']
            )

    print(f"Seeded {len(platforms)} default platforms")


# =============================================================================
# VOICE PROJECTS CRUD
# =============================================================================

def create_voice_project(
    project_id: str,
    name: str,
    description: str = "",
    provider: str = "elevenlabs",
    base_voice: str = None,
    settings: Dict = None,
    skill_id: str = None
) -> Dict:
    """Create a new voice project."""
    with get_db() as conn:
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute('''
            INSERT INTO voice_projects
            (id, name, description, provider, base_voice, settings, skill_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            project_id,
            name,
            description,
            provider,
            base_voice,
            json.dumps(settings or {'pitch': 1.0, 'speed': 1.0, 'emotion': 'neutral', 'style': 'conversational'}),
            skill_id,
            now,
            now
        ))

        return get_voice_project(project_id)


def get_voice_project(project_id: str) -> Optional[Dict]:
    """Get a voice project by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM voice_projects WHERE id = ?', (project_id,))
        row = cursor.fetchone()

        if row:
            project = dict(row)
            project['settings'] = json.loads(project['settings'] or '{}')
            project['samples'] = json.loads(project['samples'] or '[]')
            # Get samples from voice_samples table
            cursor.execute('SELECT * FROM voice_samples WHERE project_id = ? ORDER BY created_at', (project_id,))
            project['samples'] = [dict(s) for s in cursor.fetchall()]
            return project
        return None


def get_all_voice_projects(status: str = None) -> List[Dict]:
    """Get all voice projects, optionally filtered by status."""
    with get_db() as conn:
        cursor = conn.cursor()

        if status:
            cursor.execute('SELECT * FROM voice_projects WHERE status = ? ORDER BY updated_at DESC', (status,))
        else:
            cursor.execute('SELECT * FROM voice_projects ORDER BY updated_at DESC')

        projects = []
        for row in cursor.fetchall():
            project = dict(row)
            project['settings'] = json.loads(project['settings'] or '{}')
            project['samples'] = json.loads(project['samples'] or '[]')
            projects.append(project)

        return projects


def update_voice_project(project_id: str, **kwargs) -> Optional[Dict]:
    """Update a voice project."""
    with get_db() as conn:
        cursor = conn.cursor()

        allowed_fields = ['name', 'description', 'provider', 'base_voice', 'status',
                         'settings', 'voice_id', 'skill_id', 'training_started', 'training_completed']

        updates = []
        values = []

        for key, value in kwargs.items():
            if key in allowed_fields:
                if key == 'settings' and isinstance(value, dict):
                    value = json.dumps(value)
                updates.append(f"{key} = ?")
                values.append(value)

        if updates:
            updates.append("updated_at = ?")
            values.append(datetime.now().isoformat())
            values.append(project_id)

            cursor.execute(f'''
                UPDATE voice_projects SET {', '.join(updates)} WHERE id = ?
            ''', values)

        return get_voice_project(project_id)


def delete_voice_project(project_id: str) -> bool:
    """Delete a voice project and its samples."""
    with get_db() as conn:
        cursor = conn.cursor()
        # Delete samples first
        cursor.execute('DELETE FROM voice_samples WHERE project_id = ?', (project_id,))
        # Delete project
        cursor.execute('DELETE FROM voice_projects WHERE id = ?', (project_id,))
        return cursor.rowcount > 0


def add_voice_sample(
    project_id: str,
    sample_id: str,
    filename: str,
    file_path: str = None,
    transcript: str = "",
    duration_ms: int = 0,
    emotion: str = "neutral"
) -> Dict:
    """Add a voice sample to a project."""
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO voice_samples
            (id, project_id, filename, file_path, transcript, duration_ms, emotion, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sample_id,
            project_id,
            filename,
            file_path,
            transcript,
            duration_ms,
            emotion,
            datetime.now().isoformat()
        ))

        # Update project updated_at
        cursor.execute('UPDATE voice_projects SET updated_at = ? WHERE id = ?',
                      (datetime.now().isoformat(), project_id))

        cursor.execute('SELECT * FROM voice_samples WHERE id = ?', (sample_id,))
        return dict(cursor.fetchone())


def get_voice_samples(project_id: str) -> List[Dict]:
    """Get all samples for a voice project."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM voice_samples WHERE project_id = ? ORDER BY created_at', (project_id,))
        return [dict(row) for row in cursor.fetchall()]


def delete_voice_sample(sample_id: str) -> bool:
    """Delete a voice sample."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM voice_samples WHERE id = ?', (sample_id,))
        return cursor.rowcount > 0


def get_voice_projects_for_skill(skill_id: str) -> List[Dict]:
    """Get all trained voice projects linked to a skill."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM voice_projects
            WHERE skill_id = ? AND status = 'trained'
            ORDER BY updated_at DESC
        ''', (skill_id,))

        projects = []
        for row in cursor.fetchall():
            project = dict(row)
            project['settings'] = json.loads(project['settings'] or '{}')
            projects.append(project)

        return projects


def link_voice_to_skill(project_id: str, skill_id: str) -> bool:
    """Link a trained voice project to a skill."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE voice_projects SET skill_id = ?, updated_at = ? WHERE id = ?
        ''', (skill_id, datetime.now().isoformat(), project_id))

        # Also update the skill's voice_config
        project = get_voice_project(project_id)
        if project:
            cursor.execute('''
                UPDATE skills SET voice_config = ?, updated_at = ? WHERE id = ?
            ''', (
                json.dumps({
                    'voice_project_id': project_id,
                    'voice_id': project.get('voice_id'),
                    'provider': project.get('provider'),
                    'settings': project.get('settings')
                }),
                datetime.now().isoformat(),
                skill_id
            ))

        return cursor.rowcount > 0


# =============================================================================
# API CONNECTIONS - Outgoing integrations
# =============================================================================

def get_all_api_connections() -> List[Dict]:
    """Get all API connections."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM api_connections ORDER BY name')
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_api_connection(connection_id: str) -> Optional[Dict]:
    """Get a single API connection by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM api_connections WHERE id = ?', (connection_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def create_api_connection(
    connection_id: str,
    name: str,
    url: str,
    api_key: str = None,
    headers: dict = None,
    auth_type: str = 'bearer',
    webhook_url: str = None,
    webhook_secret: str = None,
    settings: dict = None
) -> bool:
    """Create a new API connection."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO api_connections
            (id, name, url, api_key, headers, auth_type, webhook_url, webhook_secret, settings, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            connection_id,
            name,
            url,
            api_key,
            json.dumps(headers) if headers else None,
            auth_type,
            webhook_url,
            webhook_secret,
            json.dumps(settings) if settings else None,
            datetime.now().isoformat()
        ))
        return cursor.rowcount > 0


def update_api_connection(connection_id: str, **kwargs) -> bool:
    """Update an API connection."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Build dynamic update query
        updates = []
        values = []
        for key, value in kwargs.items():
            if key in ['name', 'url', 'api_key', 'auth_type', 'status', 'last_tested',
                       'last_error', 'webhook_url', 'webhook_secret']:
                updates.append(f"{key} = ?")
                values.append(value)
            elif key in ['headers', 'settings']:
                updates.append(f"{key} = ?")
                values.append(json.dumps(value) if value else None)

        if not updates:
            return False

        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(connection_id)

        query = f"UPDATE api_connections SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, values)
        return cursor.rowcount > 0


def delete_api_connection(connection_id: str) -> bool:
    """Delete an API connection."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM api_connections WHERE id = ?', (connection_id,))
        return cursor.rowcount > 0


def update_api_connection_status(connection_id: str, status: str, error: str = None) -> bool:
    """Update API connection status after testing."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE api_connections
            SET status = ?, last_tested = ?, last_error = ?, updated_at = ?
            WHERE id = ?
        ''', (status, datetime.now().isoformat(), error, datetime.now().isoformat(), connection_id))
        return cursor.rowcount > 0


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_database(seed_data=False):
    """Initialize database with schema. Optionally seed demo data."""
    init_db()
    if seed_data:
        seed_builtin_skills()
        seed_default_platforms()
    else:
        # Just seed platform configs (empty), no demo skills
        seed_default_platforms()
    add_activity("Database initialized", "🗄️", "system")
    print(f"Database initialized at: {DB_PATH}")


# Auto-initialize on import if database doesn't exist
if not Path(DB_PATH).exists():
    initialize_database()
