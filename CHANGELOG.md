# Changelog

All notable changes to the Fast Brain / HIVE215 project.

## [2024-12-23] - Training Pipeline Overhaul

### Major Architectural Change
- **Modal Volume as Single Source of Truth for Adapters**
  - `/api/training/adapters` now reads directly from Modal volume via `list_adapters.remote()`
  - Removed database dependency for adapter listing
  - Training saves to `/adapters/<skill_id>/training_metadata.json`
  - UI always shows current, correct data without sync

### Bug Fixes

#### Training Data Sync Issue (CRITICAL)
- **Problem**: Modal trainer loaded only 5 fallback examples while dashboard showed 126
- **Root Cause**: `is_approved = 1` filter in `get_all_training_examples()` blocked all data (data had `is_approved = 0`)
- **Fix**: Removed `is_approved` filter - all non-archived extracted data now included in training
- **Commit**: `86d9983`

#### Database Table Creation
- **Problem**: `trained_adapters` table didn't exist in database
- **Root Cause**: `init_db()` only ran when database file was missing, not when tables were missing
- **Fix**: `init_db()` now runs on every import (uses `CREATE TABLE IF NOT EXISTS`)
- **Commit**: `d918a91`

#### Training Data Passing to Modal
- **Problem**: Modal container had separate SQLite database, not connected to dashboard's database
- **Fix**: Pass training data directly in `spawn()` call instead of having Modal fetch from its own DB
- **Commit**: `c397fb1`

#### Test Adapter Buttons
- **Problem**: TEST buttons passed wrong adapter ID format
- **Fix**: Now passes `skill_id` instead of `adapter.id` with timestamp
- **Commits**: `5446dba`, `a937ad0`, `c8c7656`

#### Chat Endpoint
- **Problem**: `/api/chat` used `httpx` which wasn't installed
- **Fix**: Replaced with `requests` library
- **Commit**: `eb54612`

### Removed
- `/api/training/adapters/sync` endpoint (no longer needed with Modal volume as source of truth)
- `/api/training/adapters/cleanup` endpoint (duplicates impossible with single source of truth)
- Database adapter sync logic

### Debug Endpoints Added
- `/api/debug/database` - Shows database state, table counts, skill_ids

## [2024-12-22] - Dashboard Improvements

### Added
- `/api/chat` endpoint for pre-training skill testing
- Adapter gallery with test functionality
- Training status polling

### Fixed
- Stats query now excludes archived items
- Database logging improvements

---

## Training Pipeline Architecture

```
Dashboard (Supabase/SQLite)     Modal Trainer (GPU)
         │                              │
         │  1. User clicks "Train"      │
         ├─────────────────────────────►│
         │     spawn(skill_id,          │
         │            training_data,    │
         │            skill_metadata)   │
         │                              │
         │                              ▼
         │                    2. Train with 126 examples
         │                              │
         │                              ▼
         │                    3. Save to /adapters/skill_id/
         │                       - adapter_model.safetensors
         │                       - training_metadata.json
         │                              │
         │  4. UI requests adapters     │
         ├─────────────────────────────►│
         │     list_adapters.remote()   │
         │                              │
         │◄─────────────────────────────┤
         │     Returns fresh metadata   │
         ▼
   5. Display correct loss: 0.266
```
