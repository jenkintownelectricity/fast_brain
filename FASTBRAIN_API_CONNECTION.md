# Fast Brain API Connection Guide

## Dashboard URL
```
https://jenkintownelectricity--hive215-dashboard-flask-app.modal.run
```

## API Endpoints

### Adapters (Trained AI Models)

**List All Adapters**
```
GET /api/trained-adapters
```
Response:
```json
{
  "adapters": [
    {
      "skill_id": "monday_com_expert_skills",
      "skill_name": "Monday.com Expert",
      "final_loss": 0.201,
      "base_model": "unsloth/Llama-3.2-1B-Instruct",
      "created_at": "2025-12-20T..."
    }
  ]
}
```

**Get Adapters for Specific Skill**
```
GET /api/training/adapters/<skill_id>
```

**Test an Adapter**
```
POST /api/test-adapter/<skill_id>
Content-Type: application/json

{
  "prompt": "Your test question here"
}
```
Response:
```json
{
  "response": "AI generated response...",
  "skill_id": "skill_id",
  "model": "unsloth/Llama-3.2-1B-Instruct"
}
```

### Training

**Start Training**
```
POST /api/train-skill/<skill_id>
Content-Type: application/json

{
  "epochs": 10,
  "learning_rate": 0.0002,
  "lora_r": 16
}
```

**Check Training Status**
```
GET /api/training/status/<skill_id>
```

### Skills

**List All Skills**
```
GET /api/skills
```

**Get Skill Details**
```
GET /api/skill/<skill_id>
```

### Training Data

**Get Training Data for Skill**
```
GET /api/parser/data?skill_id=<skill_id>
```

**Add Training Example**
```
POST /api/training-data
Content-Type: application/json

{
  "skill_id": "skill_id",
  "user_message": "User input",
  "assistant_response": "Expected response"
}
```

## Modal Python SDK Connection

```python
import modal

# Connect to deployed trainer
SkillTrainer = modal.Cls.from_name("hive215-skill-trainer", "SkillTrainer")
trainer = SkillTrainer()

# List adapters
adapters = trainer.list_adapters.remote()

# Test adapter
response = trainer.test_adapter.remote(
    skill_id="your_skill_id",
    prompt="Your question here"
)

# Start training (async)
call = trainer.train.spawn(
    skill_id="skill_id",
    config={"epochs": 10, "learning_rate": 2e-4, "lora_r": 16},
    training_data=[
        {"instruction": "system prompt", "input": "user msg", "output": "response"}
    ],
    skill_metadata={"skill_id": "id", "skill_name": "Name"}
)
```

## Modal Apps Reference

| App Name | Purpose |
|----------|---------|
| `hive215-dashboard` | Main web dashboard |
| `hive215-skill-trainer` | GPU training on A10G |
| `hive215-shop-drawings` | Shop drawing generator |
| `hive215-parler-tts` | Text-to-speech |

## Volumes (Shared Storage)

| Volume | Mount Point | Contents |
|--------|-------------|----------|
| `hive215-data` | `/data` | SQLite database, uploads |
| `hive215-adapters` | `/adapters` | Trained LoRA models |
| `hive215-shop-drawings` | `/shop_drawings` | Project files |

## Production Adapters Available

| Adapter | Loss | Examples |
|---------|------|----------|
| `monday_com_expert_skills` | 0.201 | 179 |
| `molasses-master-expert` | 0.292 | 107 |
| `plumbing_receptionist_expert` | 0.204 | 106 |
| `electrician` | 0.423 | 51 |
