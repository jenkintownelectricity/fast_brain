# Fast Brain / HIVE215 - Project TODO

## Completed (December 2024)

### Voice Lab
- [x] Dynamic voice fetching from ElevenLabs, Cartesia, Deepgram, OpenAI
- [x] Browse Voices tab with search and filter
- [x] Voice preview playback (provider sample URLs)
- [x] TTS audio caching (500 items, 1hr TTL, <5ms hit latency)
- [x] Database CRUD for voice samples
- [x] Consolidated voice provider paths
- [x] Fixed duplicate kokoro definition
- [x] Fixed JS function naming conflicts

### Dashboard
- [x] Four Pillars navigation
- [x] Dark/Light mode toggle
- [x] Getting Started onboarding
- [x] Activity logging
- [x] 7 API key providers

---

## In Progress

### Voice Features
- [ ] Pre-cache common phrases on server startup
- [ ] Voice A/B testing (compare providers side-by-side)
- [ ] Batch voice testing (test same text across all voices)

---

## Backlog

### High Priority
- [ ] WebSocket streaming for real-time TTS
- [ ] Edge caching with Modal regions (US, EU, Asia)
- [ ] Voice quality metrics and analytics
- [ ] Automatic fallback chain (ElevenLabs → Cartesia → gTTS)

### Medium Priority
- [ ] Fish Speech V1.5 integration (open source leader, ELO 1339)
- [ ] IndexTTS-2 integration (emotion/identity separation)
- [ ] Voice emotion detection from text context
- [ ] Custom voice fine-tuning UI
- [ ] Voice usage billing tracking

### Low Priority
- [ ] Voice marketplace (share voices between projects)
- [ ] Voice versioning (track changes over time)
- [ ] Multi-language voice cloning
- [ ] SSML support for advanced synthesis control

---

## Technical Debt

### Code Quality
- [ ] Remove debug logging from production (`[DEBUG]`, `[TRAIN DEBUG]`)
- [ ] Consolidate dashboard_db.py with database.py
- [ ] Add unit tests for voice caching
- [ ] Add integration tests for provider APIs

### Documentation
- [x] Update README with voice features
- [x] Create CLAUDE_LOG.md
- [x] Create TODO.md
- [ ] Update voice-lab-guide.html with Browse Voices
- [ ] Add API documentation for new endpoints
- [ ] Create voice provider comparison guide

---

## Feature Requests (User Feedback)

*Add user-requested features here*

---

## Research & Exploration

### 2025 Voice AI Trends
- Emotion-driven TTS mainstream
- Sub-100ms latency standard
- 3-second voice cloning
- LLM-native speech generation (Bland AI)

### Providers to Evaluate
- [ ] Smallest.ai Lightning (100ms, studio quality)
- [ ] Speechmatics (150ms streaming)
- [ ] CosyVoice2 (open source, 150ms)
- [ ] Bland AI (crosses "uncanny valley")

---

## Deployment Checklist

### Before Production
- [ ] Remove all debug prints
- [ ] Test all provider API keys
- [ ] Verify TTS cache performance
- [ ] Check Modal cold start times
- [ ] Test voice playback cross-browser

### Modal Apps to Deploy
```bash
# All 3 apps (PowerShell)
modal deploy deploy_dashboard.py; modal deploy fast_brain/deploy_groq.py; modal deploy parler_integration.py
```
