# Claude Development Log

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

#### Files Modified:
- `unified_dashboard.py` - Voice Lab, caching, browse UI (+700 lines)
- `database.py` - Voice sample CRUD methods (+107 lines)
- `README.md` - Updated with new features
- `docs/visuals/voice-lab-guide.html` - (pending update)

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
