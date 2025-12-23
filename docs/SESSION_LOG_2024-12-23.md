# Session Log: 2024-12-23

## Training Pipeline Debug & Architectural Fix

### Session Overview
- **Duration**: ~3 hours
- **Focus**: Fixing critical training data sync issues
- **Outcome**: Successful training with 126 examples, loss 0.266

---

## Bugs Fixed

### 1. Modal Trainer Loading Wrong Number of Examples
**Symptom**: Modal showed 5 examples, dashboard showed 126
**Root Cause**: Modal container had separate SQLite database, not connected to dashboard's database
**Fix**: Pass training data directly in spawn() call

```python
# Before (broken)
call = trainer.train.spawn(skill_id=skill_id, config=config)

# After (fixed)
training_examples = db.get_all_training_examples(skill_id)
call = trainer.train.spawn(
    skill_id=skill_id,
    config=config,
    training_data=training_examples,
    skill_metadata=skill_metadata
)
```

**Commit**: `c397fb1`

---

### 2. get_all_training_examples Returning 0
**Symptom**: Database has 262 rows, but query returns 0
**Root Cause**: `is_approved = 1` filter blocked all data (all rows had `is_approved = 0`)
**Fix**: Removed `is_approved` filter from training query

```sql
-- Before
WHERE skill_id = ? AND is_approved = 1 AND is_archived = 0

-- After
WHERE skill_id = ? AND is_archived = 0
```

**Commit**: `86d9983`

---

### 3. trained_adapters Table Doesn't Exist
**Symptom**: `ERROR: no such table: trained_adapters`
**Root Cause**: `init_db()` only ran when database file was missing
**Fix**: `init_db()` now runs on every import

```python
# Before
if not Path(DB_PATH).exists():
    initialize_database()

# After
init_db()  # Always run, CREATE TABLE IF NOT EXISTS is safe
```

**Commit**: `d918a91`

---

### 4. UI Shows Wrong Loss Value
**Symptom**: Training completed with loss 0.266, UI showed 3.527
**Root Cause**: UI read from database, but Modal trainer couldn't write to it
**Architectural Fix**: Make Modal volume the single source of truth

**Commit**: `0206594`

---

## Architectural Change

### Before: Database as Source of Truth
```
Training → Save to volume → Try to sync to database → Often fails
UI → Read from database → Shows stale data
```

### After: Modal Volume as Source of Truth
```
Training → Save to volume (always works)
UI → Read from volume via list_adapters.remote() → Always fresh
```

---

## Commits Made

| Commit | Type | Description |
|--------|------|-------------|
| `0206594` | refactor | Make Modal volume single source of truth for adapters |
| `d06d5ce` | feat | Add adapter sync endpoint (later removed) |
| `d918a91` | fix | Ensure trained_adapters table exists on startup |
| `3c280b0` | feat | Add adapter cleanup functions (later removed) |
| `86d9983` | fix | Remove is_approved filter from training data query |
| `faf3d93` | debug | Add /api/debug/database endpoint |
| `ffcad39` | debug | Add skill_id logging |
| `c3b2bf9` | debug | Add logging to diagnose data passing |
| `c397fb1` | fix | Pass training data directly to Modal |

---

## Training Results

### monday_com_expert_skills

| Metric | Value |
|--------|-------|
| Examples | 126 |
| Final Loss | 0.2660 |
| Epochs | 3 |
| LoRA Rank | 16 |
| Training Time | 10.1 minutes |
| Base Model | unsloth/Qwen2.5-1.5B-Instruct |
| GPU | A10G (24GB VRAM) |
| Estimated Cost | ~$0.65 |

---

## Debug Process

### Step 1: Identify the 5 vs 126 mismatch
- Added logging to spawn() call
- Discovered Modal had separate SQLite

### Step 2: Pass data directly
- Modified spawn() to pass training_data parameter
- Modified train() to accept and use passed data

### Step 3: Still getting 0 examples
- Added /api/debug/database endpoint
- Discovered extracted_data had 262 rows
- Found is_approved filter was blocking all data

### Step 4: UI showing wrong loss
- Discovered UI read from database
- Database never got updated after training
- Architectural decision: Use Modal volume as source of truth

---

## Files Modified

| File | Changes |
|------|---------|
| `unified_dashboard.py` | Spawn with training data, adapter endpoints read from volume |
| `train_skill_modal.py` | Accept training_data parameter |
| `database.py` | Remove is_approved filter, init_db on every import |

---

## Lessons Learned

1. **Modal containers are isolated** - They can't write to dashboard's database
2. **Single source of truth** - Don't try to sync between systems; pick one
3. **Debug endpoints are invaluable** - /api/debug/database saved hours
4. **Check filters carefully** - is_approved=1 filter was non-obvious blocker
5. **Volume commit is critical** - Always call volume.commit() after writes

---

## Next Steps

1. Monitor training jobs for any remaining issues
2. Consider adding more debug logging for production
3. Test with other skills to verify fix works universally
4. Clean up debug code once stable
