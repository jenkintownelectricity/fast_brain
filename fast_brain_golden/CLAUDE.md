# Claude Code Instructions for Fast Brain

## SESSION RULES (NON-NEGOTIABLE)

### 1. ONE FIX AT A TIME
- Complete one bug fix before starting another
- No "cleanup" or "refactoring" during bug fixes

### 2. VERIFY BEFORE CHANGE
- grep to find code BEFORE proposing changes
- NEVER change JS to call endpoints without verifying they exist:
```bash
grep -n "@app.route.*endpoint-name" unified_dashboard.py
```

### 3. SHOW BEFORE COMMIT
- Display the exact change before making it
- Wait for user approval on significant changes

### 4. NO "LATER" FIXES
- Never say "we can fix it later"
- Fix completely or document explicitly in TODO

### 5. VERIFY AFTER CHANGE
- grep after every change to confirm it worked
- Test endpoints with curl/PowerShell before declaring fixed

## STANDARD WORKFLOW

For every bug:
1. **GREP** → Find the code: `grep -n "pattern" file.py`
2. **READ** → Show context around the issue
3. **EXPLAIN** → "Bug is X because Y"
4. **PROPOSE** → "Change line N from A to B"
5. **APPROVE** → Wait for user "YES"
6. **CHANGE** → Make ONE edit
7. **VERIFY** → grep to confirm
8. **COMMIT** → Descriptive message

## Environment

- **Claude Code**: Running in web browser (not CLI)
- **Terminal**: PowerShell in VS Code (Windows)
- **Python**: Always use `py -3.11` for all Python commands
- **Git**: Standard git commands work in PowerShell

## Current State (2025-12-23)

### Deployment URLs
- **Dashboard**: https://jenkintownelectricity--hive215-dashboard-flask-app.modal.run
- **Trainer**: Modal app `hive215-skill-trainer` (A10G GPU)
- **TTS**: Modal app `hive215-parler-tts`

### Production Adapters
| Adapter | Examples | Loss | Status |
|---------|----------|------|--------|
| monday_com_expert_skills | 179 | 0.201 | Ready |
| molasses-master-expert | 107 | 0.292 | Ready |
| plumbing_receptionist_expert | 106 | 0.204 | Ready |
| electrician | 51 | 0.423 | Ready |

### Key Architecture Decision
**Modal volume is the single source of truth for adapters.**
- Training saves to `/adapters/<skill_id>/training_metadata.json`
- Dashboard reads via `trainer.list_adapters.remote()`
- No database sync needed - always fresh data

## FIXED (Do Not Touch)

1. **Double-start guard** - Returns 409 if training already running
2. **Polling endpoint** - Uses /api/training/status/ (not /api/training-job/)
3. **Double-click prevention** - Button disabled check at function start
4. **Button state management** - Disabled during training, re-enabled on complete/fail
5. **Enhanced training UI** - Stats grid, progress bar, Chart.js, educational facts, confetti
6. **Modal volume persistence** - commit_volume() after all writes
7. **Adapter listing** - Reads from Modal volume directly (single source of truth)
8. **API key reading** - Uses db.get_api_key()

## TODO (Priority Order)

1. **Groq Rate Limit Fallback** - P1
   - Issue: Free tier hits 100k tokens/day
   - Solution: Add fallback to Claude API when Groq returns 429

2. **Phase 5-7 Training Enhancements** - P2
   - Reference: /docs/training_experience_spec.md
   - Skill Avatar Customization
   - Sample Response Previews
   - Test Scenario Queue

3. **Consolidate Training UIs** - P3
   - 3 separate buttons call same endpoint
   - Consider single training component

## MODAL-SPECIFIC RULES

### API Keys (Stateless Containers)
```python
# WRONG - stale global variable
api_key = API_KEYS.get('groq')

# RIGHT - fresh from database
api_key = db.get_api_key('groq') if USE_DATABASE else None
```

### Async Jobs (No Daemon Threads)
```python
# WRONG - thread dies with container
thread = threading.Thread(daemon=True)
thread.start()

# RIGHT - survives container shutdown
fn = modal.Function.lookup("hive215-skill-trainer", "SkillTrainer.train")
call = fn.spawn(skill_id=skill_id, config=config)
```

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
| `unified_dashboard.py` | Main Flask app (~11,500 lines) |
| `database.py` | SQLite database operations |
| `deploy_dashboard.py` | Modal deployment wrapper |
| `train_skill_modal.py` | Skill training infrastructure |
| `parler_integration.py` | Text-to-speech integration |
| `/docs/training_experience_spec.md` | Feature specs |

## Adapter Architecture

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
| `docs/SESSION_LOG_2025-12-23_Training_UI.md` | Latest session log |
| `CHANGELOG.md` | All changes by date |

## Session Logging

Document significant changes in: `docs/SESSION_LOG_YYYY-MM-DD_Description.md`

## Development Philosophy

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
