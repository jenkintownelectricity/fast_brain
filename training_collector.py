"""
HIVE215 Training Data Collector
================================
Captures high-quality System 2 (Claude) responses for fine-tuning System 1 (Llama).

This module:
1. Logs all voice agent interactions to Supabase
2. Captures Claude responses when System 2 is triggered
3. Collects user feedback for quality scoring
4. Exports training pairs in LoRA-compatible format

Setup:
    1. Create Supabase tables (see SQL below)
    2. Add Modal secrets: modal secret create supabase-credentials SUPABASE_URL=... SUPABASE_KEY=...
    3. Import and use MetricsCollector in your deploy_groq.py
"""

import modal
import os
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import asyncio

# =============================================================================
# SUPABASE SCHEMA - Run this in Supabase SQL Editor
# =============================================================================
SCHEMA_SQL = """
-- Voice agent metrics and interactions
CREATE TABLE IF NOT EXISTS voice_interactions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Session info
    session_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    phone_number TEXT,
    
    -- The interaction
    user_input TEXT NOT NULL,
    agent_response TEXT NOT NULL,
    
    -- System routing
    system_used TEXT CHECK (system_used IN ('fast', 'deep')),
    routing_reason TEXT,  -- Why was System 2 triggered?
    
    -- Latency metrics
    fast_latency_ms INTEGER,
    deep_latency_ms INTEGER,
    total_latency_ms INTEGER,
    stt_latency_ms INTEGER,
    tts_latency_ms INTEGER,
    
    -- Quality signals
    user_feedback INTEGER CHECK (user_feedback BETWEEN 1 AND 5),
    call_completed BOOLEAN DEFAULT false,
    error_occurred BOOLEAN DEFAULT false,
    error_message TEXT,
    
    -- Cost tracking  
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    estimated_cost_usd DECIMAL(10,6),
    
    -- For training data extraction
    is_training_candidate BOOLEAN DEFAULT false,
    exported_for_training BOOLEAN DEFAULT false
);

-- Indexes for fast queries
CREATE INDEX idx_voice_created_at ON voice_interactions(created_at DESC);
CREATE INDEX idx_voice_skill ON voice_interactions(skill_id);
CREATE INDEX idx_voice_system ON voice_interactions(system_used);
CREATE INDEX idx_voice_training ON voice_interactions(is_training_candidate, exported_for_training);

-- Training pairs view (for export)
CREATE OR REPLACE VIEW training_candidates AS
SELECT 
    id,
    skill_id,
    user_input,
    agent_response,
    user_feedback,
    created_at
FROM voice_interactions
WHERE 
    system_used = 'deep'  -- Only Claude responses
    AND user_feedback >= 4  -- Only high-quality
    AND error_occurred = false
    AND exported_for_training = false
ORDER BY user_feedback DESC, created_at DESC;

-- Connection health tracking
CREATE TABLE IF NOT EXISTS service_health (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    checked_at TIMESTAMPTZ DEFAULT NOW(),
    groq_healthy BOOLEAN,
    anthropic_healthy BOOLEAN,
    deepgram_healthy BOOLEAN,
    cartesia_healthy BOOLEAN,
    livekit_healthy BOOLEAN,
    groq_latency_ms INTEGER,
    anthropic_latency_ms INTEGER
);

-- Hourly aggregates for dashboard
CREATE TABLE IF NOT EXISTS hourly_metrics (
    hour_bucket TIMESTAMPTZ PRIMARY KEY,
    total_calls INTEGER DEFAULT 0,
    system1_calls INTEGER DEFAULT 0,
    system2_calls INTEGER DEFAULT 0,
    avg_latency_ms INTEGER,
    error_count INTEGER DEFAULT 0,
    total_cost_usd DECIMAL(10,4)
);

-- Function to aggregate hourly metrics
CREATE OR REPLACE FUNCTION aggregate_hourly_metrics()
RETURNS void AS $$
BEGIN
    INSERT INTO hourly_metrics (hour_bucket, total_calls, system1_calls, system2_calls, avg_latency_ms, error_count, total_cost_usd)
    SELECT 
        date_trunc('hour', created_at) as hour_bucket,
        COUNT(*) as total_calls,
        COUNT(*) FILTER (WHERE system_used = 'fast') as system1_calls,
        COUNT(*) FILTER (WHERE system_used = 'deep') as system2_calls,
        AVG(total_latency_ms)::INTEGER as avg_latency_ms,
        COUNT(*) FILTER (WHERE error_occurred = true) as error_count,
        SUM(estimated_cost_usd) as total_cost_usd
    FROM voice_interactions
    WHERE created_at >= NOW() - INTERVAL '2 hours'
    GROUP BY date_trunc('hour', created_at)
    ON CONFLICT (hour_bucket) DO UPDATE SET
        total_calls = EXCLUDED.total_calls,
        system1_calls = EXCLUDED.system1_calls,
        system2_calls = EXCLUDED.system2_calls,
        avg_latency_ms = EXCLUDED.avg_latency_ms,
        error_count = EXCLUDED.error_count,
        total_cost_usd = EXCLUDED.total_cost_usd;
END;
$$ LANGUAGE plpgsql;
"""

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class InteractionRecord:
    """A single voice agent interaction."""
    session_id: str
    skill_id: str
    user_input: str
    agent_response: str
    system_used: str  # 'fast' or 'deep'
    fast_latency_ms: int
    deep_latency_ms: Optional[int] = None
    total_latency_ms: Optional[int] = None
    stt_latency_ms: Optional[int] = None
    tts_latency_ms: Optional[int] = None
    routing_reason: Optional[str] = None
    phone_number: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    error_occurred: bool = False
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.total_latency_ms is None:
            self.total_latency_ms = self.fast_latency_ms + (self.deep_latency_ms or 0)
        
        # Mark as training candidate if it used System 2
        self.is_training_candidate = (self.system_used == 'deep' and not self.error_occurred)
    
    def estimate_cost(self) -> float:
        """Estimate USD cost based on tokens and system used."""
        if self.system_used == 'fast':
            # Groq Llama 3.3 70B pricing (very cheap)
            return (self.input_tokens * 0.00000059 + self.output_tokens * 0.00000079)
        else:
            # Claude 3.5 Sonnet pricing
            return (self.input_tokens * 0.000003 + self.output_tokens * 0.000015)


# =============================================================================
# METRICS COLLECTOR (Main Class)
# =============================================================================

class MetricsCollector:
    """
    Collects voice agent metrics and training data.
    
    Usage in deploy_groq.py:
        from training_collector import MetricsCollector
        
        collector = MetricsCollector()
        
        # After each interaction
        await collector.record(InteractionRecord(
            session_id="...",
            skill_id="electrician",
            user_input="What's your service area?",
            agent_response="We serve all of Philadelphia...",
            system_used="fast",
            fast_latency_ms=85
        ))
        
        # When user provides feedback
        await collector.record_feedback(session_id, rating=5)
    """
    
    def __init__(self):
        self._client = None
        self._initialized = False
    
    async def _ensure_client(self):
        """Lazy initialization of Supabase client."""
        if not self._initialized:
            from supabase import create_client
            
            url = os.environ.get("SUPABASE_URL")
            key = os.environ.get("SUPABASE_KEY")
            
            if not url or not key:
                print("WARNING: Supabase credentials not found. Metrics will not be collected.")
                return False
            
            self._client = create_client(url, key)
            self._initialized = True
        return self._client is not None
    
    async def record(self, interaction: InteractionRecord) -> bool:
        """
        Record an interaction to Supabase.
        
        Returns True if successful, False otherwise.
        """
        if not await self._ensure_client():
            return False
        
        try:
            data = asdict(interaction)
            data['estimated_cost_usd'] = interaction.estimate_cost()
            
            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}
            
            result = self._client.table("voice_interactions").insert(data).execute()
            return len(result.data) > 0
            
        except Exception as e:
            print(f"Error recording interaction: {e}")
            return False
    
    async def record_feedback(self, session_id: str, rating: int) -> bool:
        """
        Record user feedback for a session (1-5 stars).
        Updates all interactions in that session.
        """
        if not await self._ensure_client():
            return False
        
        if not 1 <= rating <= 5:
            print(f"Invalid rating: {rating}. Must be 1-5.")
            return False
        
        try:
            result = self._client.table("voice_interactions").update({
                "user_feedback": rating,
                "call_completed": True
            }).eq("session_id", session_id).execute()
            
            return len(result.data) > 0
            
        except Exception as e:
            print(f"Error recording feedback: {e}")
            return False
    
    async def record_health_check(
        self,
        groq_healthy: bool,
        anthropic_healthy: bool,
        deepgram_healthy: bool = True,
        cartesia_healthy: bool = True,
        livekit_healthy: bool = True,
        groq_latency_ms: Optional[int] = None,
        anthropic_latency_ms: Optional[int] = None
    ) -> bool:
        """Record service health status."""
        if not await self._ensure_client():
            return False
        
        try:
            self._client.table("service_health").insert({
                "groq_healthy": groq_healthy,
                "anthropic_healthy": anthropic_healthy,
                "deepgram_healthy": deepgram_healthy,
                "cartesia_healthy": cartesia_healthy,
                "livekit_healthy": livekit_healthy,
                "groq_latency_ms": groq_latency_ms,
                "anthropic_latency_ms": anthropic_latency_ms
            }).execute()
            return True
        except Exception as e:
            print(f"Error recording health check: {e}")
            return False
    
    async def get_dashboard_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get aggregated metrics for dashboard display.
        
        Returns:
            {
                "total_calls": 150,
                "system1_percentage": 87.5,
                "avg_latency_ms": 245,
                "error_rate": 1.2,
                "cost_usd": 0.45,
                "calls_by_skill": {"electrician": 50, ...},
                "recent_errors": [...]
            }
        """
        if not await self._ensure_client():
            return {}
        
        try:
            from_time = datetime.now(timezone.utc).isoformat()
            
            # Get recent interactions
            result = self._client.table("voice_interactions").select("*").gte(
                "created_at", 
                f"now() - interval '{hours} hours'"
            ).execute()
            
            data = result.data
            if not data:
                return {"total_calls": 0}
            
            total = len(data)
            system1 = sum(1 for d in data if d.get('system_used') == 'fast')
            errors = [d for d in data if d.get('error_occurred')]
            
            # Aggregate by skill
            skills = {}
            for d in data:
                skill = d.get('skill_id', 'unknown')
                skills[skill] = skills.get(skill, 0) + 1
            
            return {
                "total_calls": total,
                "system1_percentage": round((system1 / total) * 100, 1) if total > 0 else 0,
                "system1_count": system1,
                "system2_count": total - system1,
                "avg_latency_ms": round(sum(d.get('total_latency_ms', 0) for d in data) / total),
                "error_rate": round((len(errors) / total) * 100, 2) if total > 0 else 0,
                "cost_usd": sum(float(d.get('estimated_cost_usd', 0)) for d in data),
                "calls_by_skill": skills,
                "recent_errors": errors[:5]
            }
            
        except Exception as e:
            print(f"Error getting dashboard metrics: {e}")
            return {}


# =============================================================================
# TRAINING DATA EXPORTER
# =============================================================================

class TrainingExporter:
    """
    Exports high-quality interactions for LoRA fine-tuning.
    
    Usage:
        exporter = TrainingExporter()
        
        # Export for a specific skill
        pairs = await exporter.export_skill("electrician", min_rating=4)
        
        # Save in LoRA-compatible format
        exporter.save_jsonl(pairs, "electrician_training.jsonl")
    """
    
    def __init__(self):
        self._client = None
    
    async def _ensure_client(self):
        from supabase import create_client
        if self._client is None:
            self._client = create_client(
                os.environ["SUPABASE_URL"],
                os.environ["SUPABASE_KEY"]
            )
        return self._client
    
    async def export_skill(
        self, 
        skill_id: str, 
        min_rating: int = 4,
        limit: int = 1000
    ) -> List[Dict[str, str]]:
        """
        Export training pairs for a specific skill.
        
        Returns list of {"prompt": ..., "completion": ...} pairs.
        """
        client = await self._ensure_client()
        
        result = client.table("voice_interactions").select(
            "user_input, agent_response, user_feedback"
        ).eq(
            "skill_id", skill_id
        ).eq(
            "system_used", "deep"  # Only Claude responses
        ).gte(
            "user_feedback", min_rating
        ).eq(
            "error_occurred", False
        ).eq(
            "exported_for_training", False
        ).limit(limit).execute()
        
        pairs = []
        for row in result.data:
            pairs.append({
                "prompt": row["user_input"],
                "completion": row["agent_response"]
            })
        
        return pairs
    
    async def mark_exported(self, skill_id: str, ids: List[str]) -> None:
        """Mark interactions as exported to prevent duplicates."""
        client = await self._ensure_client()
        
        for id in ids:
            client.table("voice_interactions").update({
                "exported_for_training": True
            }).eq("id", id).execute()
    
    def save_jsonl(self, pairs: List[Dict], filepath: str) -> None:
        """
        Save training pairs in JSONL format for torchtune.
        
        Format compatible with Modal's "Fine-tune Llama 3.1" template.
        """
        with open(filepath, 'w') as f:
            for pair in pairs:
                # Format for instruction fine-tuning
                record = {
                    "instruction": f"You are a helpful voice assistant. Respond naturally and conversationally.",
                    "input": pair["prompt"],
                    "output": pair["completion"]
                }
                f.write(json.dumps(record) + "\n")
        
        print(f"Saved {len(pairs)} training pairs to {filepath}")
    
    def save_conversation_format(self, pairs: List[Dict], filepath: str) -> None:
        """
        Save in conversation format for chat fine-tuning.
        """
        with open(filepath, 'w') as f:
            for pair in pairs:
                record = {
                    "messages": [
                        {"role": "user", "content": pair["prompt"]},
                        {"role": "assistant", "content": pair["completion"]}
                    ]
                }
                f.write(json.dumps(record) + "\n")
        
        print(f"Saved {len(pairs)} conversations to {filepath}")


# =============================================================================
# ALERT SYSTEM
# =============================================================================

class AlertManager:
    """
    Sends alerts for critical issues.
    
    Usage:
        alerts = AlertManager(slack_webhook="https://hooks.slack.com/...")
        await alerts.send("Groq API down!", severity="critical")
    """
    
    def __init__(self, slack_webhook: Optional[str] = None, discord_webhook: Optional[str] = None):
        self.slack_webhook = slack_webhook or os.environ.get("SLACK_WEBHOOK_URL")
        self.discord_webhook = discord_webhook or os.environ.get("DISCORD_WEBHOOK_URL")
    
    async def send(self, message: str, severity: str = "warning") -> bool:
        """Send alert to configured webhooks."""
        import httpx
        
        emoji = {"info": "ℹ️", "warning": "⚠️", "error": "🚨", "critical": "🔥"}.get(severity, "📢")
        formatted = f"{emoji} [{severity.upper()}] HIVE215: {message}"
        
        success = True
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            if self.slack_webhook:
                try:
                    await client.post(self.slack_webhook, json={"text": formatted})
                except Exception as e:
                    print(f"Slack alert failed: {e}")
                    success = False
            
            if self.discord_webhook:
                try:
                    await client.post(self.discord_webhook, json={"content": formatted})
                except Exception as e:
                    print(f"Discord alert failed: {e}")
                    success = False
        
        return success
    
    async def alert_on_error_rate(self, collector: MetricsCollector, threshold: float = 5.0):
        """Alert if error rate exceeds threshold."""
        metrics = await collector.get_dashboard_metrics(hours=1)
        error_rate = metrics.get("error_rate", 0)
        
        if error_rate > threshold:
            await self.send(
                f"Error rate is {error_rate}% (threshold: {threshold}%)",
                severity="error"
            )
    
    async def alert_on_latency(self, collector: MetricsCollector, threshold_ms: int = 500):
        """Alert if average latency exceeds threshold."""
        metrics = await collector.get_dashboard_metrics(hours=1)
        avg_latency = metrics.get("avg_latency_ms", 0)
        
        if avg_latency > threshold_ms:
            await self.send(
                f"Avg latency is {avg_latency}ms (threshold: {threshold_ms}ms)",
                severity="warning"
            )


# =============================================================================
# MODAL INTEGRATION EXAMPLE
# =============================================================================

MODAL_INTEGRATION_EXAMPLE = '''
# Add this to your deploy_groq.py

from training_collector import MetricsCollector, InteractionRecord, AlertManager
import modal

supabase_secret = modal.Secret.from_name("supabase-credentials")
slack_secret = modal.Secret.from_name("slack-webhook")

# Initialize collector
collector = MetricsCollector()
alerts = AlertManager()

@app.function(secrets=[groq_secret, anthropic_secret, supabase_secret])
@modal.web_endpoint(method="POST")
async def chat_hybrid(request: dict):
    """Enhanced hybrid chat with metrics collection."""
    import time
    
    session_id = request.get("session_id", str(uuid.uuid4()))
    skill_id = request.get("skill", "general")
    user_input = request["messages"][-1]["content"]
    
    start = time.perf_counter()
    
    try:
        # Your existing Fast Brain logic here
        response = await fast_brain_inference(request)
        
        fast_time = int((time.perf_counter() - start) * 1000)
        
        # Record the interaction
        await collector.record(InteractionRecord(
            session_id=session_id,
            skill_id=skill_id,
            user_input=user_input,
            agent_response=response["content"],
            system_used=response["system_used"],
            fast_latency_ms=response.get("fast_latency_ms", fast_time),
            deep_latency_ms=response.get("deep_latency_ms"),
            routing_reason=response.get("filler"),  # Filler indicates System 2 was triggered
            input_tokens=response.get("input_tokens", 0),
            output_tokens=response.get("output_tokens", 0)
        ))
        
        return response
        
    except Exception as e:
        # Record error
        await collector.record(InteractionRecord(
            session_id=session_id,
            skill_id=skill_id,
            user_input=user_input,
            agent_response="",
            system_used="fast",
            fast_latency_ms=0,
            error_occurred=True,
            error_message=str(e)
        ))
        
        await alerts.send(f"Inference error: {str(e)[:100]}", severity="error")
        raise


@app.function(secrets=[supabase_secret], schedule=modal.Cron("*/30 * * * *"))
async def health_check_cron():
    """Check service health every 30 minutes."""
    import httpx
    
    groq_ok = False
    anthropic_ok = False
    groq_latency = None
    anthropic_latency = None
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Check Groq
        try:
            start = time.perf_counter()
            resp = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
            )
            groq_latency = int((time.perf_counter() - start) * 1000)
            groq_ok = resp.status_code == 200
        except:
            pass
        
        # Check Anthropic (just validate API key)
        try:
            start = time.perf_counter()
            resp = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                    "anthropic-version": "2023-06-01"
                }
            )
            anthropic_latency = int((time.perf_counter() - start) * 1000)
            anthropic_ok = resp.status_code in [200, 401]  # 401 means key is valid format
        except:
            pass
    
    await collector.record_health_check(
        groq_healthy=groq_ok,
        anthropic_healthy=anthropic_ok,
        groq_latency_ms=groq_latency,
        anthropic_latency_ms=anthropic_latency
    )
    
    # Alert on failures
    if not groq_ok:
        await alerts.send("Groq API is not responding!", severity="critical")
    if not anthropic_ok:
        await alerts.send("Anthropic API is not responding!", severity="critical")


@app.function(secrets=[supabase_secret])
@modal.web_endpoint(method="GET")
async def dashboard_metrics():
    """Get dashboard metrics for the UI."""
    return await collector.get_dashboard_metrics(hours=24)


@app.function(secrets=[supabase_secret])
@modal.web_endpoint(method="POST")
async def record_user_feedback(request: dict):
    """Record user feedback after a call."""
    session_id = request["session_id"]
    rating = request["rating"]  # 1-5 stars
    
    success = await collector.record_feedback(session_id, rating)
    return {"success": success}
'''

if __name__ == "__main__":
    print("=== HIVE215 Training Data Collector ===")
    print("\n1. Run the SQL schema in Supabase SQL Editor:")
    print(SCHEMA_SQL[:500] + "...\n")
    print("\n2. Add Modal secrets:")
    print("   modal secret create supabase-credentials SUPABASE_URL=https://xxx.supabase.co SUPABASE_KEY=your_key")
    print("\n3. Add to your deploy_groq.py:")
    print(MODAL_INTEGRATION_EXAMPLE[:500] + "...")
