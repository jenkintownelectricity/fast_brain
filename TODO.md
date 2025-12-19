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
- [x] Custom Cartesia voices prioritized (green border, CUSTOM badge)
- [x] Voice test button in Browse Voices (ElevenLabs, Cartesia, Deepgram, OpenAI TTS)

### Dashboard
- [x] Four Pillars navigation
- [x] Dark/Light mode toggle
- [x] Getting Started onboarding
- [x] Activity logging
- [x] 7 API key providers
- [x] Edit skill functionality (edit modal, pre-populated form, save to API)
- [x] API Endpoints card in Settings (Fast Brain API URL, Dashboard URL, copy buttons, health check)

### Test Chat
- [x] Voice Test section (optional, similar to Golden Prompts)
- [x] Voice Provider dropdown with all providers
- [x] Dynamic voice loading from provider APIs
- [x] Text-to-speech test with audio playback

### Documentation
- [x] HIVE215 Integration Guide (`docs/HIVE215_INTEGRATION_GUIDE.md`)

---

## In Progress

### Voice Features
- [ ] Pre-cache common phrases on server startup
- [ ] Voice A/B testing (compare providers side-by-side)
- [ ] Batch voice testing (test same text across all voices)

---

## Backlog

### High Priority (P0 - Must Have for 2025)
- [ ] RAG/Knowledge Base Integration (Supabase pgvector)
- [ ] LoRA hotswapping in production (73% latency reduction, Groq native support)
- [ ] Analytics Dashboard (System 1/2 split, training opportunities)
- [ ] WebSocket streaming for real-time TTS
- [ ] Edge caching with Modal regions (US, EU, Asia)
- [ ] Voice quality metrics and analytics
- [ ] Automatic fallback chain (ElevenLabs → Cartesia → gTTS)

### Medium Priority (P1 - Differentiators)
- [ ] Node-based conversation flows (reduce hallucinations, like Retell)
- [ ] Skill-specific knowledge bases (Node KB pattern)
- [ ] Multi-skill routing / Squads (multi-agent orchestration)
- [ ] Visual flow builder (reduce technical barriers)
- [ ] Fish Speech V1.5 integration (open source leader, ELO 1339)
- [ ] IndexTTS-2 integration (emotion/identity separation)
- [ ] Voice emotion detection from text context
- [ ] Custom voice fine-tuning UI
- [ ] Voice usage billing tracking

### Low Priority (P2 - Nice to Have)
- [ ] Voice marketplace (share voices between projects)
- [ ] Voice versioning (track changes over time)
- [ ] Multi-language voice cloning
- [ ] SSML support for advanced synthesis control
- [ ] Automated retraining pipeline
- [ ] Real-time call analytics (sentiment, coaching)

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
- [x] Update voice-lab-guide.html with Browse Voices
- [x] Create HIVE215 Integration Guide (`docs/HIVE215_INTEGRATION_GUIDE.md`)
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
- Multi-agent orchestration (Squads, context-preserving transfers)
- LoRA hotswapping for dynamic skill switching

### Competitor Analysis (December 2024)
| Platform | Strengths | Weaknesses | Latency |
|----------|-----------|------------|---------|
| **Vapi AI** | Squads, Flow Studio, A/B testing | Complex pricing, steep learning curve | ~800ms |
| **Retell AI** | Node-based flows, Node KB, analytics | 700-800ms latency | 700-800ms |
| **Bland AI** | Custom fine-tuning, proprietary models | $150K+ budget, English-only | 950ms |
| **PlayAI** | Knowledge base uploads, on-prem | Limited flow control | N/A |
| **Air AI** | Agentic workflows, CRM integration | Poor call quality, hidden fees | N/A |

### Fast Brain Competitive Advantage
- **Unique**: System 1/System 2 hybrid architecture (no competitor has this)
- **Best latency potential**: ~80ms System 1 + LoRA hotswapping
- **Best cost efficiency**: 70B local model + occasional Claude
- **Self-improving**: Automatic training data collection from System 2 responses

### Providers to Evaluate
- [ ] Smallest.ai Lightning (100ms, studio quality)
- [ ] Speechmatics (150ms streaming)
- [ ] CosyVoice2 (open source, 150ms)
- [ ] Bland AI (crosses "uncanny valley")
- [ ] Groq LoRA Fine-tune (native support announced)

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
