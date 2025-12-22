# Next Session Instructions - Fast Brain Dashboard

## Quick Context for Claude Code

**Project**: Fast Brain - Hybrid Voice AI Engine with training dashboard
**Last Session**: December 22, 2025
**Branch**: `main` (all changes merged)
**Dashboard URL**: https://jenkintownelectricity--hive215-dashboard-flask-app.modal.run

---

## What Was Done Last Session

### 1. Major UX Restructure
Converted chaotic 7-sub-tab Skills & Training interface into clean 3-step workflow:

```
Step 1: Select Skill → Step 2: Add Training Data → Step 3: Test & Train
```

### 2. New Features Added
- Persistent Skill Context Bar (shows selected skill info)
- Step Indicator with clickable navigation
- Toast notification system
- Side-by-side layout for Step 2 (manual entry + file upload/AI generate)
- Side-by-side layout for Step 3 (test chat + training controls)

### 3. Bug Fixes Applied
| Bug | Fix Applied |
|-----|-------------|
| AI Generate "topic is required" | Changed `context` → `topic` parameter |
| Manual entry "Failed to save" | Changed `/api/parser/save` → `/api/parser/data` |
| Old tabs showing | Redirected `skills` → `skills-training` |

---

## CRITICAL BUGS STILL PRESENT

### Bug 1: Manual Entry Still Failing
**Error**: "Failed to save entry"
**What to check**:
1. Open browser console (F12) and look at network tab
2. Find the POST to `/api/parser/data` and check response
3. The endpoint is at line ~802 in `unified_dashboard.py`
4. Possible issues:
   - Database connection
   - Missing table columns
   - Validation errors

### Bug 2: File Upload Extracting 0 Q&A Pairs
**Error**: "✓ Uploaded 3 file(s), extracted 0 Q&A pairs"
**User's document format**:
```
"How do I notify a team lead when a status changes to 'Stuck'?" -> Setup: When Status changes to Stuck, notify Team Lead.
```
**What to fix**:
- The parser expects `Q: question A: answer` format
- User's format is `"question" -> answer`
- Need to add regex pattern for arrow format
- Check `/api/parser/upload` endpoint (~line 1007)

### Bug 3: AI Generate Still Showing "topic is required"
**Status**: Fix was pushed but may not be deployed yet
**What happened**:
- User entered topic: "Monday.com automations, CRM, dashboards..."
- Got error: "✗ topic is required"
**Possible causes**:
- Fix not deployed to Modal yet
- JavaScript variable issue
- Need to verify the workflow function sends `topic` not `context`

---

## Test Data User Provided

Copy this into the "Topic" field for AI Generate testing:

```
Monday.com Workflow Expert training data covering:

Pillar 1: Automations & Logic
"How do I notify a team lead when a status changes to 'Stuck'?" -> Setup: When Status changes to Stuck, notify Team Lead.
"Can I move an item to another board based on a date?" -> Yes, use the 'When date arrives, move item to board' automation.
"How do I create a dependency between two tasks?" -> Use the Dependency Column and ensure 'Ensure dates don't overlap' is toggled.
"How do I automate recurring tasks every Monday?" -> Use the 'Every time period, create an item' recipe.

Pillar 2: Sales CRM & Lead Management
"How do I calculate commission based on deal size?" -> Use a Formula Column: ({Deal Value} * 0.10).
"How do I track lead aging?" -> Use the 'Creation Log' and a formula to subtract today's date.
"Can I send an automated email to a lead from Monday?" -> Yes, integrate Gmail/Outlook and use the 'When status changes, send email' recipe.
"How do I visualize my sales pipeline?" -> Use the Kanban View grouped by 'Deal Stage'.

Pillar 3: Dashboard & Reporting (Widgets)
"How do I see total revenue across 5 different boards?" -> Use a 'Battery' or 'Numbers' widget in a Multi-Board Dashboard.
"Can I track time spent on tasks per employee?" -> Use the Time Tracking Column and the 'Workload' widget.
"How do I create a Pivot Table for project categories?" -> Use the Pivot Table widget and group by 'Type' and 'Status'.

Pillar 4: Technical Integrations (API/Webhooks)
"How do I find my API V2 Token?" -> Admin > Developers > API.
"What is the rate limit for Monday's API?" -> 5,000 complexity points per minute.
"How do I update a column value via GraphQL?" -> Use the change_simple_column_value mutation.
```

---

## Key Files to Check

| File | Purpose | Key Lines |
|------|---------|-----------|
| `unified_dashboard.py` | Main Flask app | All endpoints and frontend |
| Line ~802 | `/api/parser/data` POST | Manual entry save |
| Line ~1007 | `/api/parser/upload` | File upload & parsing |
| Line ~1554 | `/api/parser/generate` | AI generation |
| Line ~10459 | `generateWorkflowAiData()` | JS function for AI generate |
| Line ~10347 | `saveWorkflowEntry()` | JS function for manual save |

---

## Commands to Deploy

After fixing bugs, deploy to Modal:

```powershell
py -3.11 -m modal deploy deploy_dashboard.py
py -3.11 -m modal deploy train_skill_modal.py
py -3.11 -m modal deploy parler_integration.py
```

---

## Architecture Reference

```
┌─────────────────────────────────────────────────────────────────────┐
│                    🧠 Skills & Training Tab                          │
├─────────────────────────────────────────────────────────────────────┤
│  CONTEXT BAR: 🎯 Skill Name | ● Status | 📊 Examples | 🔤 Tokens    │
├─────────────────────────────────────────────────────────────────────┤
│  STEPS: (1) Select Skill ─── (2) Add Data ─── (3) Test & Train     │
├─────────────────────────────────────────────────────────────────────┤
│  Step 2 Layout (Side-by-Side):                                      │
│  ┌─────────────────────┐ ┌─────────────────────┐                   │
│  │ ✏️ Manual Entry      │ │ 📤 Bulk Upload      │                   │
│  │                     │ │ 🤖 AI Generate      │                   │
│  └─────────────────────┘ └─────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Session Goals

1. [ ] Fix manual entry save (debug API response)
2. [ ] Fix file upload parsing (add arrow format regex)
3. [ ] Verify AI generate works after deploy
4. [ ] Test full workflow end-to-end
5. [ ] Deploy to Modal and verify live

---

## Notes

- Python version: 3.11 (`py -3.11`)
- Environment: Windows PowerShell in VS Code
- Branch convention: `claude/[description]-[sessionId]`
- Dashboard is deployed on Modal with SQLite on shared volume
