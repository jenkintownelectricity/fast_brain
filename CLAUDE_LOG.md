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
