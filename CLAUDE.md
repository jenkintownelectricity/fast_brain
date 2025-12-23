# Claude Code Instructions for Fast Brain

## Environment

- **Claude Code**: Running in web browser (not CLI)
- **Terminal**: PowerShell in VS Code (Windows)
- **Python**: Always use `py -3.11` for all Python commands
- **Git**: Standard git commands work in PowerShell

## Current State (2024-12-23)

### Deployment URLs
- **Dashboard**: https://jenkintownelectricity--hive215-dashboard-flask-app.modal.run
- **Trainer**: Modal app `hive215-skill-trainer` (A10G GPU)
- **TTS**: Modal app `hive215-parler-tts`

### Recent Training Success
- **Skill**: monday_com_expert_skills
- **Examples**: 126
- **Final Loss**: 0.2660
- **Training Time**: ~10 minutes

### Key Architecture Decision
**Modal volume is the single source of truth for adapters.**
- Training saves to `/adapters/<skill_id>/training_metadata.json`
- Dashboard reads via `trainer.list_adapters.remote()`
- No database sync needed - always fresh data

## Development Philosophy

### No Quick Fixes or Workarounds

We are building world-class applications. Every fix must be:

1. **Root Cause Resolution** - Find and fix the actual problem, not symptoms
2. **Production Quality** - Code must be robust, maintainable, and scalable
3. **Properly Tested** - Verify fixes work before considering them complete
4. **Well Documented** - Complex logic should be clear to future developers

### What This Means in Practice

- **DO NOT** add defensive code to mask underlying bugs
- **DO NOT** use temporary patches that "work for now"
- **DO NOT** skip error handling or validation
- **DO NOT** leave TODO comments for critical functionality
- **DO** trace issues to their source
- **DO** fix problems at the architectural level when needed
- **DO** refactor if the current design is fundamentally flawed
- **DO** write clean, readable, maintainable code

## Command Reference

### Python Commands (PowerShell)
```powershell
py -3.11 -m pip install <package>
py -3.11 -m modal deploy deploy_dashboard.py
py -3.11 -m modal deploy train_skill_modal.py
py -3.11 -m modal deploy parler_integration.py
py -3.11 script.py
```

### Modal Deployments
| Command | App |
|---------|-----|
| `py -3.11 -m modal deploy deploy_dashboard.py` | Dashboard |
| `py -3.11 -m modal deploy train_skill_modal.py` | Trainer |
| `py -3.11 -m modal deploy parler_integration.py` | TTS |

**Note**: Skip `modal_lpu.py` - BitNet build is currently broken.

## Key Project Files

| File | Purpose |
|------|---------|
| `unified_dashboard.py` | Main Flask app with HTML/JS frontend |
| `database.py` | SQLite database operations |
| `deploy_dashboard.py` | Modal deployment wrapper |
| `train_skill_modal.py` | Skill training infrastructure |
| `parler_integration.py` | Text-to-speech integration |

## Important: Adapter Architecture

### How Adapters Work Now

```
Training:
  Dashboard → spawn(skill_id, training_data) → Modal Trainer
                                                    ↓
                                          Save to /adapters/<skill_id>/
                                          - adapter_model.safetensors
                                          - training_metadata.json
                                                    ↓
                                          volume.commit()

Reading:
  Dashboard → trainer.list_adapters.remote()
                     ↓
           Read training_metadata.json from each folder
                     ↓
           Return fresh data to UI
```

### What NOT to Do
- Don't try to sync adapters to database (removed)
- Don't add cleanup endpoints for duplicates (removed)
- Don't store adapter info in `trained_adapters` table (deprecated)

## Code Standards

### API Endpoints
- Always validate input data thoroughly
- Return consistent JSON response format: `{"success": bool, "error"?: string, "data"?: any}`
- Handle both expected field names for compatibility (e.g., `id` and `skill_id`)
- Include meaningful error messages

### Error Handling
- Catch specific exceptions, not bare `except:`
- Log errors with context for debugging
- Return user-friendly error messages
- Never silently swallow errors

### Database Operations
- Use parameterized queries (no SQL injection)
- Handle connection failures gracefully
- Validate data before insertion
- Use transactions for multi-step operations

## Workflow Reminders

1. **Before fixing a bug**: Understand the root cause completely
2. **Before deploying**: Verify the fix locally if possible
3. **After deploying**: Test the fix on the live dashboard
4. **Always**: Commit with clear, descriptive messages

## Debug Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/api/debug/database` | Database state, table counts, skill_ids |
| `/api/trained-adapters` | Raw adapter data from Modal volume |

## Documentation

| File | Content |
|------|---------|
| `docs/ARCHITECTURE.md` | System architecture and data flow |
| `docs/TROUBLESHOOTING.md` | Common issues and solutions |
| `docs/API_REFERENCE.md` | All API endpoints |
| `docs/SESSION_LOG_2024-12-23.md` | Today's changes |
| `CHANGELOG.md` | All changes by date |

## Session Logging

Document significant changes in: `docs/SESSION_LOG_YYYY-MM-DD.md`
