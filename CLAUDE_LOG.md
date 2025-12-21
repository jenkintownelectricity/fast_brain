# Claude Development Log

## 2024-12-19: Settings & Voice Test Environment Updates

### Session: `claude/unified-dashboard-AJdAZ`

#### Commits Made:
1. **feat: Add API Endpoints card to Settings page**
   - Added "API Endpoints" card in Command Center → API Keys section
   - Displays Fast Brain API URL with copy button
   - Displays Dashboard URL with copy button
   - Health check button with status indicators (green/orange/red)
   - Quick reference table for common API endpoints

2. **feat: Add Voice Test section to Test Chat**
   - Added "Test with Voice (Optional)" section at bottom of Test Chat tab
   - Voice Provider dropdown (ElevenLabs, Cartesia, Deepgram, OpenAI, Edge TTS, Parler)
   - Voice ID dropdown (dynamically loads voices from selected provider)
   - Text input for custom test phrases
   - Test Voice button with audio playback
   - Similar styling to Golden Prompts optional feature

#### Files Modified:
- `unified_dashboard.py` - Settings API card, Voice Test UI in Test Chat
- `CLAUDE_LOG.md` - Session notes
- `TODO.md` - Updated status

---

## 2024-12-19: Voice Provider Integration & Caching

### Session: `claude/add-voice-providers-g9dDC`

#### Commits Made:
1. **feat: Add dynamic voice fetching from providers (ElevenLabs, Cartesia, Deepgram, OpenAI)**
   - Added `/api/voice-lab/provider-voices/<provider>` endpoint
   - Added `/api/voice-lab/all-provider-voices` endpoint
   - Implemented `fetch_elevenlabs_voices()` - GET /v1/voices API
   - Implemented `fetch_cartesia_voices()` - GET /voices with pagination
   - Implemented `fetch_deepgram_voices()` - GET /v1/models (TTS models)
   - Implemented `get_openai_voices()` - static list of 11 voices
   - Added "Browse Voices" tab to Voice Lab with search/filter
   - Updated all provider dropdowns to include premium + free providers

2. **fix: Resolve voice code conflicts and add missing database methods**
   - Added 6 missing database methods:
     - `add_voice_sample()`, `get_voice_samples()`, `get_voice_sample()`
     - `delete_voice_sample()`, `delete_voice_project()`
     - `link_voice_to_skill()`, `get_voice_projects_for_skill()`
   - Removed duplicate 'kokoro' definition in VOICE_CHOICES
   - Renamed conflicting `loadProviderVoices()` to `loadVoiceSettingsVoices()`
   - Fixed function conflict between Voice Settings and Voice Lab tabs

3. **feat: Add TTS caching and consolidate voice provider paths**
   - Added `TTSCache` class with LRU eviction and TTL expiration
   - Cache: 500 items, 1-hour TTL, SHA256 key hashing
   - Added `/api/voice/cache-stats` and `/api/voice/cache-clear` endpoints
   - Cache hit latency: <5ms vs 500-2000ms for fresh synthesis
   - Client-side `voiceCache` with 5-minute TTL for voice lists
   - Updated `selectVoice()` to work with dynamic voice data
   - Added preview button to voice details panel
   - UI shows "CACHED" vs "generated" status

#### Research Summary:
- **Best Practices**: Edge caching, WebRTC streaming, <100ms target latency
- **Competitor Analysis**: ElevenLabs (quality leader), Cartesia (speed leader), Deepgram (enterprise)
- **Customer Pain Points**: Billing surprises, credit systems, quality degradation
- **2025 Breakthroughs**: Fish Speech V1.5, IndexTTS-2 (emotion/identity separation), Bland AI

4. **feat: Prioritize custom Cartesia voices in Browse Voices**
   - Updated `fetch_cartesia_voices()` to identify owned voices via `is_owner` field
   - Custom voices appear first in list with green border and "CUSTOM" badge
   - Shows `custom_count` in API response

5. **feat: Add voice preview and skill editing features**
   - Added `testBrowseVoice()` function with "Test" button in Browse Voices
   - Added ElevenLabs, Cartesia, Deepgram, OpenAI TTS support to `/api/voice/test`
   - Added Edit button to skill cards
   - Implemented `editSkill()`, `hideEditSkillModal()`, `saveEditedSkill()` functions
   - Added Edit Skill modal form with pre-populated data

6. **docs: Create HIVE215 integration guide**
   - Created comprehensive `docs/HIVE215_INTEGRATION_GUIDE.md`
   - Covers all API endpoints, integration patterns, best practices
   - Includes sample code for LiveKit voice agents
   - Documents filler phrase strategy and skill-based routing

#### Research Summary (Skills/Training Competitors):
- **Vapi AI**: Squads (multi-agent), Flow Studio (visual builder), A/B testing
- **Retell AI**: Node-based flows, Node KB (skill-specific knowledge), 700-800ms latency
- **Bland AI**: Custom fine-tuning, proprietary models, $150K+ enterprise
- **Key Trends**: Multi-agent orchestration, LoRA hotswapping (73% latency reduction), RAG knowledge bases
- **Opportunity Gaps**: Knowledge base integration, visual flow builder, multi-skill routing

#### Files Modified:
- `unified_dashboard.py` - Voice Lab, caching, browse UI, skill editing (+800 lines)
- `database.py` - Voice sample CRUD methods (+107 lines)
- `README.md` - Updated with new features
- `docs/HIVE215_INTEGRATION_GUIDE.md` - NEW: Full integration documentation
- `docs/visuals/voice-lab-guide.html` - Updated with Browse Voices

---

## Development Notes

### Voice Provider Architecture
```
Single Path (Consolidated):
/api/voice-lab/provider-voices/<provider>
  ├── Dynamic: ElevenLabs, Cartesia, Deepgram (live API calls)
  ├── Static: OpenAI (fixed list)
  └── Fallback: VOICE_CHOICES dictionary

Client-Side Cache (5 min TTL):
voiceCache.get(provider) → cached voices or null
voiceCache.set(provider, voices) → stores with timestamp
```

### TTS Cache Architecture
```
TTSCache (Server-Side):
├── Key: SHA256(text|voice_id|provider|emotion)
├── Value: {audio_bytes, format, timestamp, metadata}
├── Eviction: LRU when at capacity (500 items)
└── TTL: 1 hour expiration

Endpoints:
├── GET /api/voice/cache-stats → {hits, misses, hit_rate, size}
└── POST /api/voice/cache-clear → clears cache
```

### Provider API Reference
| Provider | Endpoint | Auth Header |
|----------|----------|-------------|
| ElevenLabs | GET /v1/voices | xi-api-key |
| Cartesia | GET /voices | X-API-Key + Cartesia-Version |
| Deepgram | GET /v1/models | Authorization: Token |
| OpenAI | N/A (static) | N/A |

---

## TODO for Next Session
- [ ] Pre-cache common phrases on startup
- [ ] Add WebSocket streaming for real-time TTS
- [ ] Implement edge caching with Modal regions
- [ ] Add voice A/B testing feature
- [ ] Integrate Fish Speech V1.5 (open source leader)

---

## 2024-12-20: Fix Skills Not Appearing in HIVE215 UI

### Session: `claude/qwen3-vllm-exploration-5LDqg`

#### Problem
Skills created in the Unified Dashboard (e.g., "The Molasses Alchemist") were not appearing in the HIVE215 UI dropdown, despite being visible in the Dashboard.

#### Root Cause Analysis

**Architecture Disconnect:**
| Component | Database Access | Skills Source |
|-----------|-----------------|---------------|
| `deploy_dashboard.py` (Modal) | ✅ hive215-data volume at `/data` | SQLite database |
| `deploy_groq.py` (Modal) | ❌ No volume mounted | BUILT_IN_SKILLS dict only |

**Flow of the Bug:**
```
User creates skill in Dashboard
    ↓
unified_dashboard.py stores in SQLite (/data/hive215.db)
    ↓
Dashboard shows skill ✓
    ↓
HIVE215 calls Fast Brain /v1/skills API
    ↓
deploy_groq.py queries BUILT_IN_SKILLS only (ignores database)
    ↓
Skill NOT returned ✗
    ↓
HIVE215 dropdown missing the skill
```

#### Solution
Modified `deploy_groq.py` to share the same SQLite database volume as the dashboard:

1. **Added shared volume declaration:**
   ```python
   skills_volume = modal.Volume.from_name("hive215-data", create_if_missing=True)
   ```

2. **Added database.py to Modal image:**
   ```python
   .add_local_file("database.py", "/root/database.py")
   ```

3. **Mounted volume in fastapi_app:**
   ```python
   @app.function(
       image=image,
       region=MODAL_REGION,
       volumes={"/data": skills_volume},
   )
   ```

4. **Updated all skill-related functions to query database:**
   - `get_all_skills()` - Now queries database first
   - `get_skill_detail()` - Checks database before built-in
   - `get_skill()` - Helper function now includes database
   - `list_skills()` - Returns database + built-in + runtime skills
   - `get_all_skills_dict()` - Combines all sources for chat
   - `/health` endpoint - Shows all skills including database

5. **Added helper functions:**
   - `_get_database_skills()` - Query all skills from SQLite
   - `_get_database_skill(skill_id)` - Query single skill
   - `_load_database_skill()` - Load with format conversion
   - `_load_all_database_skill_ids()` - Get all IDs

#### Priority Order (skills lookup):
1. **Runtime skills** - Created via API, temporary (highest priority)
2. **Database skills** - Created in Dashboard, persistent
3. **Built-in skills** - Hardcoded defaults (lowest priority)

#### Files Modified:
- `fast_brain/deploy_groq.py` - Added volume, database integration, updated all skill endpoints

#### Verification Steps:
After deploying, verify with:
```bash
# Check if molasses-master-expert appears
curl https://jenkintownelectricity--fast-brain-lpu-fastapi-app.modal.run/v1/skills | jq '.skills[] | select(.id | contains("molasses"))'

# Check individual skill
curl https://jenkintownelectricity--fast-brain-lpu-fastapi-app.modal.run/v1/skills/molasses-master-expert

# Check health endpoint shows all skills
curl https://jenkintownelectricity--fast-brain-lpu-fastapi-app.modal.run/health | jq '.skills_available'
```

#### Deployment:
```bash
cd /path/to/fast_brain
modal deploy fast_brain/deploy_groq.py
```

---

## 2025-12-21: Unified Skills & Training Tab - Complete Overhaul

### Session: `claude/unified-skills-training-5LDqg`

#### Overview
Combined the separate "Skills" and "Training" tabs into a single unified "🧠 Skills & Training" section with full feature integration, fixing multiple critical bugs along the way.

#### Major Features Added

1. **Unified Skills & Training Tab**
   - Combined 2 separate tabs into 1 unified section
   - 7 sub-tabs: Skills, Golden Prompts, Training, Data Manager, Test Chat, Adapters, API
   - Skill cards with training status badges (untrained, has_data, trained)
   - Click-to-manage skill detail modal

2. **Skill Cards View**
   - Visual grid of all skills with status indicators
   - Search and filter (All, Untrained, Has Data, Trained)
   - Sort by name/status
   - Inline "Create Skill" form
   - "Sync from LPU" and "Seed Defaults" buttons

3. **Skill Detail Modal**
   - 4-tab modal: Overview, Training Data, Train, Adapters
   - Edit skill name, description, system prompt
   - View/manage training examples
   - Manual entry, bulk import, AI generate options
   - Training configuration with intensity slider
   - Adapter management

4. **Training Data Management**
   - Manual Q&A entry modal
   - Bulk document import (70+ file types)
   - AI-powered training data generation
   - Approve/delete data items
   - Token counting and stats

#### Critical Bugs Fixed

1. **Missing API Endpoints** (Root cause of "Failed to load skills")
   ```
   JavaScript was calling:
   - /api/training/adapters    ❌ DID NOT EXIST
   - /api/parser/stats         ❌ DID NOT EXIST
   - /api/training/start       ❌ DID NOT EXIST
   - /api/training/status      ❌ DID NOT EXIST

   Promise.all() failed → showed "Failed to load skills"
   ```

2. **Duplicate Function Name**
   - Two functions named `get_training_status()` caused Flask crash
   - Renamed to `get_skill_training_status()`

3. **Database Table Missing**
   - Added `trained_adapters` table for tracking LoRA adapters
   - Added `extracted_data` table functions for parser stats

#### New API Endpoints Added

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/training/adapters` | GET | List all trained adapters |
| `/api/training/adapters/<skill_id>` | GET | Get adapters for skill |
| `/api/training/start` | POST | Start training job |
| `/api/training/status/<skill_id>` | GET | Get training status |
| `/api/parser/stats` | GET | Get training data statistics |
| `/api/parser/data` | GET | Get training data by skill |
| `/api/parser/data` | POST | Add manual training data |
| `/api/parser/data/<id>` | DELETE | Delete training data |
| `/api/parser/data/<id>/approve` | POST | Approve training data |
| `/api/fast-brain/sync-skills` | POST | Sync skills from LPU API |
| `/api/fast-brain/seed-skills` | POST | Seed default skills |

#### Database Schema Changes

```sql
-- New table for tracking trained adapters
CREATE TABLE trained_adapters (
    id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL,
    skill_name TEXT,
    adapter_name TEXT,
    base_model TEXT DEFAULT 'unsloth/Qwen2.5-1.5B-Instruct',
    epochs INTEGER DEFAULT 10,
    lora_r INTEGER DEFAULT 16,
    final_loss REAL,
    training_time_seconds INTEGER,
    adapter_path TEXT,
    status TEXT DEFAULT 'completed',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
);
```

#### New Database Functions

```python
# Adapters
db.get_all_adapters()
db.get_adapters_by_skill(skill_id)
db.create_adapter(adapter_id, skill_id, **kwargs)

# Parser Stats
db.get_extracted_data_stats()
db.get_extracted_data_by_skill(skill_id)
```

#### Files Modified

| File | Changes |
|------|---------|
| `unified_dashboard.py` | +800 lines: Unified tab, modals, 11 new endpoints, JS functions |
| `database.py` | +150 lines: trained_adapters table, adapter/parser functions |
| `deploy_dashboard.py` | Updated Modal deployment |

#### Commits Made

1. `feat: Unify Skills & Training into single tab with skill cards`
2. `feat: Add complete sub-tabs to unified Skills & Training section`
3. `feat: Add skill sync and seed functions`
4. `fix: Add missing API endpoints and proper error handling`
5. `fix: Add /api/training/start and /api/training/status endpoints`
6. `fix: Rename duplicate get_training_status function`

#### Modal Deployments

```powershell
# All 4 services deployed successfully
py -3.11 -m modal deploy deploy_dashboard.py       # hive215-dashboard
py -3.11 -m modal deploy fast_brain/deploy_groq.py # fast-brain-lpu
py -3.11 -m modal deploy train_skill_modal.py      # hive215-skill-trainer
py -3.11 -m modal deploy parler_integration.py     # hive215-parler-tts
```

#### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    🧠 Skills & Training Tab                  │
├─────────────────────────────────────────────────────────────┤
│  Sub-tabs:                                                   │
│  ┌────────┬─────────┬──────────┬──────┬──────┬────────┬───┐ │
│  │ Skills │ Golden  │ Training │ Data │ Chat │Adapters│API│ │
│  │        │ Prompts │          │ Mgr  │      │        │   │ │
│  └────────┴─────────┴──────────┴──────┴──────┴────────┴───┘ │
├─────────────────────────────────────────────────────────────┤
│  Skills Tab:                                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │ 🎯 Skill A  │ │ 🎯 Skill B  │ │ 🎯 Skill C  │            │
│  │ ● Trained   │ │ ◐ Has Data  │ │ ○ Untrained │            │
│  │ 50 examples │ │ 20 examples │ │ 0 examples  │            │
│  │ 2 adapters  │ │ 0 adapters  │ │ 0 adapters  │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
│                                                              │
│  [+ Create Skill] [🔄 Sync from LPU] [📦 Seed Defaults]     │
└─────────────────────────────────────────────────────────────┘

Click Skill Card → Opens Detail Modal:
┌─────────────────────────────────────────────────────────────┐
│  🎯 Skill Name                           [● Trained] [✕]   │
├─────────────────────────────────────────────────────────────┤
│  [Overview] [Training Data] [Train] [Adapters]              │
├─────────────────────────────────────────────────────────────┤
│  Overview Tab:                                               │
│  ├── Skill ID, Name, Description                            │
│  ├── System Prompt editor                                   │
│  └── Stats: Examples, Tokens, Adapters, Quality             │
│                                                              │
│  Training Data Tab:                                          │
│  ├── [✏️ Manual] [📤 Bulk Import] [🤖 AI Generate]         │
│  ├── Stats: Total | Pending | Approved | Tokens             │
│  └── Data table with approve/delete actions                 │
│                                                              │
│  Train Tab:                                                  │
│  ├── Readiness checks (examples, system prompt)             │
│  ├── Intensity slider (Quick/Standard/Deep)                 │
│  ├── Cost/time estimates                                    │
│  └── [🚀 Start Training] button                             │
│                                                              │
│  Adapters Tab:                                               │
│  └── List of trained adapters with Test/Deploy buttons      │
└─────────────────────────────────────────────────────────────┘
```

#### Testing Notes

After deployment, click "🔄 Sync from LPU" to import existing skills from the Fast Brain LPU API into the dashboard's local database.

Training is currently a "queue" action - actual GPU training runs with:
```powershell
py -3.11 -m modal run train_skill_modal.py --skill-id <skill_id>
```

---
