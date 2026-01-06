# Fast Brain Golden Package

Core logic files extracted from fast_brain repository.

## Structure

```
fast_brain_golden/
├── fast_brain/           # Core Brain Logic
│   ├── deploy_groq.py    # Main Fast Brain API (System 1 + System 2 hybrid)
│   ├── skills.py         # Skills definitions
│   ├── config.py         # Configuration dataclasses
│   ├── client.py         # Python client library
│   └── model.py          # Model definitions
│
├── _TRANSFER_TO_HIVE215/ # Voice Transfer Files
│   ├── skill_command_center.py  # LatencyMasker, SmartRouter
│   └── turn_taking.py           # Silence detection, backchanneling
│
├── worker/               # Worker Components
│   ├── voice_agent.py    # Voice agent implementation
│   └── test_fast_brain.py # Test suite
│
├── docs/                 # Documentation
│   └── HIVE215_INTEGRATION_GUIDE.md
│
├── deploy_dashboard.py   # Dashboard Modal deployment
├── unified_dashboard.py  # Full management UI (Flask + HTML/JS, ~834KB)
├── parler_integration.py # Parler TTS Modal deployment
├── golden_prompts.py     # Voice-optimized skill prompts
├── train_skill_modal.py  # LoRA training infrastructure
├── training_collector.py # Supabase metrics + LoRA export
├── continuous_learner.py # Feedback collection + DPO training
├── database.py           # SQLite database with CRUD operations
│
├── CLAUDE.md             # Claude Code instructions
├── README.md             # Project documentation
└── CHANGELOG.md          # Version history
```

## File Sizes

| File | Size | Purpose |
|------|------|---------|
| unified_dashboard.py | 834 KB | Full Flask + HTML/JS dashboard |
| database.py | 57 KB | SQLite CRUD operations |
| deploy_groq.py | 59 KB | Main Fast Brain API |
| training_collector.py | 24 KB | Metrics + LoRA export |
| train_skill_modal.py | 22 KB | LoRA training infrastructure |
| parler_integration.py | 21 KB | Parler TTS deployment |
| turn_taking.py | 20 KB | Voice silence detection |
| golden_prompts.py | 18 KB | Voice-optimized prompts |
| voice_agent.py | 18 KB | Voice agent implementation |
| skill_command_center.py | 15 KB | LatencyMasker, SmartRouter |
| continuous_learner.py | 14 KB | Feedback + DPO training |
| HIVE215_INTEGRATION_GUIDE.md | 13 KB | Integration documentation |
| client.py | 12 KB | Python client library |
| skills.py | 12 KB | Skills definitions |
| model.py | 11 KB | Model definitions |
| config.py | 9 KB | Configuration dataclasses |
| CLAUDE.md | 7 KB | Claude Code instructions |
| test_fast_brain.py | 5 KB | Test suite |
| deploy_dashboard.py | 3 KB | Modal dashboard deployment |
| CHANGELOG.md | 3 KB | Version history |

**Total: ~1.1 MB uncompressed, 243 KB compressed**

## Notes

- `modal_fish_speech.py` does not exist in the repository (was listed as optional)
- All files extracted on 2026-01-06
