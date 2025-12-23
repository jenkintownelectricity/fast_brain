# API Reference

Base URL: `https://jenkintownelectricity--hive215-dashboard-flask-app.modal.run`

---

## Training Endpoints

### Start Training
```
POST /api/training/start/<skill_id>
```

Starts a LoRA training job on Modal GPU.

**Request Body:**
```json
{
  "epochs": 3,
  "learning_rate": 0.0002,
  "lora_r": 16
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "monday_com_expert_skills_20241223_021500",
  "modal_call_id": "fc-01KD4FM95QSH5MG4SA3ZKH5ZWP",
  "message": "Training started for skill 'monday_com_expert_skills'",
  "status_url": "/api/training-job/monday_com_expert_skills"
}
```

---

### Get Training Status
```
GET /api/training-job/<skill_id>
```

**Response:**
```json
{
  "job_id": "monday_com_expert_skills_20241223_021500",
  "skill_id": "monday_com_expert_skills",
  "status": "completed",
  "started_at": "2024-12-23T02:15:00.000Z",
  "completed_at": "2024-12-23T02:25:00.000Z",
  "progress": 100,
  "logs": [
    "Training job spawned on Modal GPU...",
    "Training completed! Adapter: /adapters/monday_com_expert_skills"
  ],
  "result": {
    "success": true,
    "final_loss": 0.266,
    "training_examples": 126,
    "training_time_seconds": 606.5
  }
}
```

---

### List Training Jobs
```
GET /api/training-jobs
```

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "monday_com_expert_skills_20241223_021500",
      "skill_id": "monday_com_expert_skills",
      "status": "completed",
      "started_at": "2024-12-23T02:15:00.000Z",
      "completed_at": "2024-12-23T02:25:00.000Z"
    }
  ]
}
```

---

## Adapter Endpoints

### List All Adapters
```
GET /api/training/adapters
```

**Source:** Reads directly from Modal volume (single source of truth)

**Response:**
```json
{
  "adapters": [
    {
      "id": "monday_com_expert_skills",
      "skill_id": "monday_com_expert_skills",
      "skill_name": "Monday.com Expert",
      "final_loss": 0.266,
      "training_examples": 126,
      "epochs": 3,
      "lora_r": 16,
      "base_model": "unsloth/Qwen2.5-1.5B-Instruct",
      "adapter_path": "/adapters/monday_com_expert_skills",
      "created_at": "2024-12-23T02:25:00.000Z",
      "training_time_seconds": 606.5,
      "status": "ready"
    }
  ],
  "success": true
}
```

---

### Get Adapter for Skill
```
GET /api/training/adapters/<skill_id>
```

**Response:** Same format as above, filtered to specific skill.

---

### Legacy: List Trained Adapters
```
GET /api/trained-adapters
```

Older endpoint, returns same data from Modal volume.

---

## Skill Endpoints

### List All Skills
```
GET /api/fast-brain/skills
```

**Response:**
```json
{
  "skills": [
    {
      "id": "monday_com_expert_skills",
      "name": "Monday.com Expert",
      "description": "Expert on Monday.com platform",
      "system_prompt": "You are an expert on Monday.com...",
      "knowledge": ["Monday.com is a work OS..."],
      "skill_type": "custom"
    }
  ]
}
```

---

### Get Single Skill
```
GET /api/fast-brain/skills/<skill_id>
```

---

### Create Skill
```
POST /api/fast-brain/skills
```

**Request Body:**
```json
{
  "id": "my_skill",
  "name": "My Skill",
  "description": "Description here",
  "system_prompt": "You are...",
  "knowledge": ["Fact 1", "Fact 2"]
}
```

---

## Testing Endpoints

### Test Adapter
```
POST /api/test-adapter/<skill_id>
```

**Request Body:**
```json
{
  "prompt": "Hello, how can you help me?"
}
```

**Response:**
```json
{
  "success": true,
  "response": "Hello! I'm the Monday.com Expert...",
  "skill_id": "monday_com_expert_skills"
}
```

---

### Chat (Pre-Training Test)
```
POST /api/chat
```

Test a skill WITHOUT an adapter (uses base model + system prompt).

**Request Body:**
```json
{
  "skill_id": "monday_com_expert_skills",
  "message": "What is Monday.com?"
}
```

---

## Parser/Data Endpoints

### Get Parser Stats
```
GET /api/parser/stats
```

**Response:**
```json
{
  "total": 262,
  "total_tokens": 45000,
  "approved": 126,
  "pending": 136,
  "by_skill": [
    {"skill_id": "monday_com_expert_skills", "total": 126, "tokens": 22000}
  ]
}
```

---

### Get Parser Data
```
GET /api/parser/data?skill_id=<skill_id>
```

Returns all extracted training examples for a skill.

---

## Debug Endpoints

### Database Debug
```
GET /api/debug/database
```

**Response:**
```json
{
  "db_path": "/data/hive215.db",
  "db_exists": true,
  "db_size_bytes": 524288,
  "data_dir_contents": ["/data/hive215.db"],
  "tables": {
    "skills": 5,
    "training_data": 0,
    "extracted_data": 262,
    "trained_adapters": 0
  },
  "training_data_skill_ids": [],
  "extracted_data_skill_ids": ["monday_com_expert_skills", "test_skill"]
}
```

---

## Deprecated/Removed Endpoints

| Endpoint | Status | Reason |
|----------|--------|--------|
| `POST /api/training/adapters/sync` | REMOVED | Not needed with Modal volume as source of truth |
| `POST /api/training/adapters/cleanup` | REMOVED | Duplicates impossible with single source of truth |

---

## Error Response Format

All endpoints return errors in this format:

```json
{
  "success": false,
  "error": "Human-readable error message"
}
```

HTTP status codes:
- `200` - Success
- `400` - Bad request (validation error)
- `404` - Not found
- `500` - Server error
