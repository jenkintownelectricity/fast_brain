# Claude Code Instructions for Fast Brain

## Environment

- **Claude Code**: Running in web browser (not CLI)
- **Terminal**: PowerShell in VS Code (Windows)
- **Python**: Always use `py -3.11` for all Python commands
- **Git**: Standard git commands work in PowerShell

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

### Dashboard URL
https://jenkintownelectricity--hive215-dashboard-flask-app.modal.run

## Key Project Files

| File | Purpose |
|------|---------|
| `unified_dashboard.py` | Main Flask app with HTML/JS frontend |
| `database.py` | SQLite database operations |
| `deploy_dashboard.py` | Modal deployment wrapper |
| `train_skill_modal.py` | Skill training infrastructure |
| `parler_integration.py` | Text-to-speech integration |

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

## Session Logging

Document significant changes in: `docs/CLAUDE_CODE_SESSION_YYYY-MM-DD.md`
