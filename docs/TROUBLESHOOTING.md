# Troubleshooting Guide

## Common Issues and Solutions

---

## Training Issues

### Problem: Modal trainer loads 5 examples but dashboard shows 126

**Symptoms:**
- Dashboard shows 126 training examples
- Modal logs show "Loaded 5 examples" or "Using 5 examples passed from dashboard"
- Training completes with high loss (~3.5)

**Root Causes & Solutions:**

1. **is_approved filter blocking data**
   - Check if `extracted_data` has `is_approved = 0`
   - Solution: Remove `is_approved` filter from `get_all_training_examples()`

2. **Database sync issue**
   - Modal container has separate SQLite, not connected to dashboard
   - Solution: Pass training data directly in spawn() call

**Debug Steps:**
```bash
# Check database state
curl https://[your-url]/api/debug/database

# Look for:
# - extracted_data count (should match dashboard)
# - extracted_data_skill_ids (should include your skill)
```

---

### Problem: UI shows wrong loss value after training

**Symptoms:**
- Training completes with loss 0.266
- UI shows old loss value (e.g., 3.527)

**Root Cause:**
UI was reading from database, but database wasn't updated after training.

**Solution:**
As of 2024-12-23, `/api/training/adapters` reads directly from Modal volume.
This ensures UI always shows current data. Just refresh the page.

**If still seeing old data:**
1. Clear browser cache (Ctrl+Shift+R)
2. Check Modal logs for `list_adapters` call
3. Verify adapter exists: `curl https://[your-url]/api/trained-adapters`

---

### Problem: trained_adapters table doesn't exist

**Symptoms:**
```
ERROR: no such table: trained_adapters
```

**Root Cause:**
`init_db()` only ran when database file was missing, not when tables were missing.

**Solution:**
As of commit `d918a91`, `init_db()` runs on every import.
Just redeploy the dashboard:
```powershell
py -3.11 -m modal deploy deploy_dashboard.py
```

---

## API Issues

### Problem: TEST button doesn't work

**Symptoms:**
- Click TEST on adapter → nothing happens
- Console error about missing function

**Solution:**
Fixed in commits `5446dba`, `a937ad0`, `c8c7656`.
Redeploy dashboard to get the fix.

---

### Problem: /api/chat returns error about httpx

**Symptoms:**
```
ModuleNotFoundError: No module named 'httpx'
```

**Solution:**
Fixed in commit `eb54612`. Chat endpoint now uses `requests` instead.
Redeploy dashboard.

---

## Debug Endpoints

### /api/debug/database

Returns database state including:
- `db_path`: Path to SQLite file
- `db_exists`: Whether file exists
- `db_size_bytes`: File size
- `tables`: Row count for each table
- `extracted_data_skill_ids`: Skills with training data

**Example Response:**
```json
{
  "db_path": "/data/hive215.db",
  "db_exists": true,
  "db_size_bytes": 524288,
  "tables": {
    "skills": 5,
    "training_data": 0,
    "extracted_data": 262,
    "trained_adapters": 0
  },
  "extracted_data_skill_ids": [
    "monday_com_expert_skills",
    "test_skill"
  ]
}
```

---

## Modal-Specific Issues

### Problem: Container restarts lose data

**Cause:** Modal volumes have eventual consistency.

**Solution:**
- Always call `volume.commit()` after writes
- Training already does this at line 457 of `train_skill_modal.py`

### Problem: list_adapters returns empty

**Cause:** Adapter volume not mounted or empty.

**Debug:**
```python
# In Modal, check volume contents
import os
print(os.listdir("/adapters"))
```

---

## Quick Diagnostic Commands

```bash
# Check what adapters exist on Modal volume
curl https://[your-url]/api/trained-adapters

# Check database state
curl https://[your-url]/api/debug/database

# Check training status
curl https://[your-url]/api/training-job/[skill_id]

# List all training jobs
curl https://[your-url]/api/training-jobs
```
