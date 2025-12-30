# ============================================================================
# Fast Brain Voice Features Extraction Script
# ============================================================================
# Purpose: Extract runtime voice features for re-implementation in HIVE215
# Created: 2025-12-30
# ============================================================================

$DestFolder = "D:\FastBrain_HIVE215_Transfer"
$SourceFolder = Get-Location

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Fast Brain Voice Features Extraction" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Create destination folder
if (-not (Test-Path $DestFolder)) {
    New-Item -ItemType Directory -Path $DestFolder -Force | Out-Null
    Write-Host "[+] Created folder: $DestFolder" -ForegroundColor Green
} else {
    Write-Host "[i] Folder exists: $DestFolder" -ForegroundColor Yellow
}

# Files to extract
$Files = @(
    "skill_factory.py",
    "skill_dashboard.py",
    "skill_command_center.py",
    "dashboard.py"
)

# Copy each file
Write-Host ""
Write-Host "Copying files..." -ForegroundColor Cyan
foreach ($File in $Files) {
    $SourcePath = Join-Path $SourceFolder $File
    $DestPath = Join-Path $DestFolder $File

    if (Test-Path $SourcePath) {
        Copy-Item -Path $SourcePath -Destination $DestPath -Force
        Write-Host "[+] Copied: $File" -ForegroundColor Green
    } else {
        Write-Host "[!] Not found: $File" -ForegroundColor Red
    }
}

# Create the README
Write-Host ""
Write-Host "Generating README.md..." -ForegroundColor Cyan

$ReadmeContent = @"
# Fast Brain Voice Features - Awaiting HIVE215 Re-implementation

## Overview

These files were extracted from the Fast Brain repository on $(Get-Date -Format "yyyy-MM-dd").
They contain **runtime voice features** that were built during Fast Brain development but
belong in **HIVE215** (the production voice platform where LiveKit calls happen).

Fast Brain is a **skill factory** - it builds and trains voice agent skills.
HIVE215 is the **voice runtime** - it handles live calls with real-time latency requirements.

These features were identified during a codebase cleanup as "superseded" by unified_dashboard.py,
but they contain unique functionality that was **never migrated** to the production system.

---

## Why These Files Are Waiting

| Feature | Why It's Here | Why It Belongs in HIVE215 |
|---------|---------------|---------------------------|
| LatencyMasker | Built in Fast Brain during prototyping | Runtime feature - masks LLM wait time during live calls |
| SmartRouter | Built for query routing experiments | Runtime feature - decides fast vs smart LLM per query |
| Cached Responses | Pattern-matched instant replies | Runtime feature - zero-latency responses for common queries |
| Skill Fillers | Context-aware filler sounds | Runtime feature - "customer_service" vs "technical" fillers |

---

## File 1: skill_command_center.py (431 lines)

### Purpose
The **brain of the voice AI platform**. Contains core logic for:
1. Routing queries to the best LLM (fast vs smart)
2. Masking latency with natural filler sounds
3. Retrieving skill-specific knowledge
4. Integrating with voice platforms (LiveKit, Vapi)

### Key Classes

#### LatencyMasker (Lines 22-121)
Generates natural filler sounds while waiting for LLM response.

``````python
class LatencyMasker:
    # Filler sounds - short, natural thinking sounds (<100ms to speak)
    FILLER_SOUNDS = [
        "Hmm...",    # Universal thinking sound
        "Mmm...",    # Acknowledgment
        "Umm...",    # Processing
        "Ah...",     # Realization
        "Well...",   # Transition
    ]

    # Thinking phrases - for longer waits (>300ms)
    THINKING_PHRASES = [
        "Let me think about that...",   # Shows active processing
        "That's a good question...",    # Validates the user
        "Let me check...",              # Implies looking up info
        "One moment...",                # Explicit wait request
        "Interesting...",               # Engagement signal
        "Let me see...",                # Visual metaphor
    ]

    # Domain-specific fillers (customize per skill type)
    SKILL_FILLERS = {
        "technical": ["Let me look that up...", "Checking the docs..."],
        "customer_service": ["I understand.", "Let me help with that..."],
        "scheduling": ["Let me check the calendar...", "One moment..."],
        "sales": ["Great question!", "Let me find the best option..."],
    }
``````

**How it works:**
- `get_instant_filler()` - Returns quick sound for immediate response
- `get_thinking_phrase()` - Returns longer phrase, avoids repetition
- `mask_latency()` - Async generator that wraps LLM response, yields fillers during waits

#### LLMProvider Enum (Lines 127-134)
Supported LLM backends:

``````python
class LLMProvider(Enum):
    BITNET = "bitnet"        # Local 1-bit model (free, slower)
    GROQ = "groq"            # Groq API (fast, free tier)
    OPENAI = "openai"        # OpenAI (smart, paid)
    ANTHROPIC = "anthropic"  # Claude (smart, paid)
    CEREBRAS = "cerebras"    # Cerebras (fast, free tier)
    TOGETHER = "together"    # Together.ai (various models)
``````

#### LLMConfig Dataclass (Lines 136-150)
Configuration for each provider:

``````python
@dataclass
class LLMConfig:
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.7

    # Performance characteristics (for routing decisions)
    avg_first_token_ms: int = 500   # Time to first token
    avg_tokens_per_sec: int = 50    # Generation speed
    cost_per_1k_tokens: float = 0.0 # Cost in dollars
``````

#### QueryAnalysis Dataclass (Lines 152-160)
Analysis result for routing:

``````python
@dataclass
class QueryAnalysis:
    complexity: str           # "simple", "medium", "complex"
    intent: str               # "greeting", "question", "command", "conversation"
    skill_match: Optional[str] = None  # Matched skill if any
    requires_reasoning: bool = False   # Needs smart LLM
    requires_current_info: bool = False
    confidence: float = 0.9
``````

#### SmartRouter (Lines 167-242)
Routes queries to optimal LLM:

``````python
class SmartRouter:
    # Simple patterns - don't need smart LLMs
    SIMPLE_PATTERNS = [
        "hello", "hi", "hey", "good morning", "good afternoon",
        "how are you", "what's up", "thanks", "thank you", "bye",
        "goodbye", "yes", "no", "okay", "ok", "sure",
    ]

    def analyze_query(self, query: str) -> QueryAnalysis:
        # Check for simple patterns first
        # Count words, detect question words (why, how, explain)
        # Detect reasoning words (because, therefore, analyze)
        # Return complexity: simple/medium/complex

    def select_llm(self, analysis: QueryAnalysis) -> LLMProvider:
        # Simple queries -> Groq (fast) or BitNet (free)
        # Complex queries -> Claude or GPT-4 (smart)
        # Medium + latency critical -> Groq
        # Medium + quality priority -> OpenAI
``````

#### Skill Dataclass (Lines 249-258)
Represents a narrow-role assistant:

``````python
@dataclass
class Skill:
    name: str                           # e.g., "dental_receptionist"
    description: str                    # What this skill does
    system_prompt: str                  # LLM system message
    knowledge_base: list = field(...)   # RAG documents
    example_responses: Dict[str, str]   # Pattern -> Cached response
    filler_type: str = "default"        # Which filler set to use
    preferred_llm: Optional[LLMProvider] = None  # Override routing
``````

#### SkillRetriever (Lines 261-291)
Matches queries to skills and retrieves cached responses:

``````python
class SkillRetriever:
    def match_skill(self, query: str) -> Optional[Skill]:
        # Keyword matching (could use embeddings)
        # Returns best matching skill

    def get_cached_response(self, skill: Skill, query: str) -> Optional[str]:
        # Check example_responses for pattern match
        # Returns instant response if found (no LLM call needed!)
``````

#### SkillCommandCenter (Lines 298-395)
Main orchestrator combining all components:

``````python
class SkillCommandCenter:
    async def process_query(self, query: str) -> AsyncGenerator[str, None]:
        # 1. Check for skill match and cached response
        #    - If cached, yield immediately (zero latency!)
        # 2. Analyze query complexity
        # 3. Select LLM (fast vs smart)
        # 4. Generate response with latency masking
        # 5. Track stats (queries, latency, cache hits)
``````

---

## File 2: skill_dashboard.py (1,394 lines)

### Purpose
Flask web dashboard with **cyberpunk theme** for skill management and monitoring.

### Key Features

#### Server Status Panel (Lines 200-250)
Real-time monitoring with P50/P99 latency:

``````python
# Stats tracked:
- Total Requests
- Active Skills
- Avg Response Time
- Cache Hit Rate
- P50 Latency (50th percentile)
- P99 Latency (99th percentile)
- Cost tracking per provider
``````

#### Training Pipeline Visualization (Lines 400-500)
Visual pipeline with 5 stages:

``````
[Queued] → [Preprocessing] → [Training] → [Validating] → [Complete]
    ↓           ↓               ↓            ↓              ↓
  Gray        Blue           Yellow       Orange         Green
``````

Each stage shows:
- Current status with animated indicator
- Time elapsed in stage
- Error state (red) if failed

#### Feedback Queue (Lines 600-700)
Collects user feedback for continuous learning:

``````python
@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    # User submits: query, response, rating, correction
    # Stored for later training data generation
    # Enables "thumbs up/down" in voice UI
``````

#### Metrics API (Lines 800-900)
Detailed analytics:

``````python
@app.route('/api/metrics')
def get_metrics():
    return {
        "by_provider": {"groq": 150, "anthropic": 45},
        "by_complexity": {"simple": 120, "medium": 50, "complex": 25},
        "latency_p50_ms": 180,
        "latency_p99_ms": 850,
        "cache_hit_rate": 0.23,
        "cost_today_usd": 0.45
    }
``````

---

## File 3: dashboard.py (611 lines)

### Purpose
Gradio dashboard for **testing LLM providers** and generating integration code.

### Key Features

#### LLM Client Implementations (Lines 50-200)
Working clients for each provider:

``````python
class BitNetClient:
    # Calls local BitNet server at http://localhost:8080
    # Free, runs on CPU, slower but no API costs

class GroqClient:
    # Calls Groq API with llama3-70b-8192
    # 100ms first token, free tier available

class OpenAIClient:
    # Calls OpenAI API with gpt-4
    # Smart, reliable, paid

class AnthropicClient:
    # Calls Anthropic API with claude-3-opus
    # Best reasoning, paid
``````

#### Compare Providers Side-by-Side (Lines 300-400)
Test same prompt across all providers:

``````python
def compare_providers(prompt: str):
    results = {}
    for name, client in clients.items():
        start = time.time()
        response = client.generate(prompt)
        elapsed = time.time() - start
        results[name] = {
            "response": response,
            "latency_ms": elapsed * 1000,
            "tokens": len(response.split())
        }
    return results
``````

#### LiveKit Voice Integration Examples (Lines 450-550)
Code snippets for LiveKit integration:

``````python
# Example code generated for user:
from livekit import rtc

async def on_track_subscribed(track, publication, participant):
    if track.kind == rtc.TrackKind.KIND_AUDIO:
        audio_stream = rtc.AudioStream(track)
        async for frame in audio_stream:
            # Send to STT
            text = await transcribe(frame)
            # Route through SmartRouter
            response = await router.process(text)
            # Send to TTS
            await synthesize_and_play(response)
``````

#### Mock Client for Testing (Lines 200-250)
Simulates LLM responses for testing:

``````python
class MockClient:
    def generate(self, prompt: str) -> str:
        # Returns canned responses for testing
        # Simulates realistic latency
        # Useful for UI development without API costs
``````

---

## File 4: skill_factory.py (636 lines)

### Purpose
Gradio UI for **creating skills from business profiles**.

### Key Features

#### Business Profile Manager (Lines 100-200)
Structured input for business info:

``````python
# Fields collected:
- Business Name
- Industry (dropdown)
- Business Description
- Services Offered
- Business Hours
- Contact Information
- FAQs (special Q:/A: format)
- Documents (PDF, TXT, MD, JSON upload)
``````

#### FAQ Parser (Lines 250-300)
Parses Q:/A: formatted FAQs:

``````python
def parse_faq(faq_text: str) -> list:
    """
    Input format:
    Q: What are your hours?
    A: We're open Monday-Friday, 9 AM to 5 PM.

    Q: Do you accept insurance?
    A: Yes, we accept most major insurance providers.
    """
    faqs = []
    current_q = None
    for line in lines:
        if line.lower().startswith('q:'):
            current_q = line.lstrip('qQ:? ').strip()
        elif line.lower().startswith('a:') and current_q:
            answer = line.lstrip('aA: ').strip()
            faqs.append({"question": current_q, "answer": answer})
            current_q = None
    return faqs
``````

#### Document Processing (Lines 300-400)
Extracts text from uploaded files:

``````python
def process_document(file) -> str:
    ext = file.name.split('.')[-1].lower()

    if ext == 'pdf':
        # PyPDF2 extraction
        reader = PdfReader(file)
        return ' '.join(page.extract_text() for page in reader.pages)

    elif ext == 'docx':
        # python-docx extraction
        doc = Document(file)
        return ' '.join(p.text for p in doc.paragraphs)

    elif ext in ['txt', 'md']:
        return file.read().decode('utf-8')

    elif ext == 'json':
        data = json.load(file)
        return json.dumps(data, indent=2)
``````

#### Training Data Generator (Lines 400-500)
Creates training examples from profile:

``````python
def generate_training_data(profile: dict) -> list:
    examples = []

    # From FAQs
    for faq in profile['faqs']:
        examples.append({
            "instruction": faq['question'],
            "response": faq['answer']
        })

    # From business info
    examples.append({
        "instruction": f"What does {profile['name']} do?",
        "response": profile['description']
    })

    # From services
    for service in profile['services']:
        examples.append({
            "instruction": f"Tell me about {service['name']}",
            "response": service['description']
        })

    return examples
``````

#### Training Steps Slider (Lines 500-550)
Configurable training duration:

``````python
training_steps = gr.Slider(
    minimum=100,
    maximum=2000,
    value=500,
    step=100,
    label="Training Steps",
    info="More steps = better quality but longer training time"
)

# Generates LoRA training script with these parameters
``````

---

## Re-implementation Priority

### HIGH PRIORITY (Core Voice Experience)
1. **LatencyMasker** - Users experience awkward silence without this
2. **Cached Responses** - Zero-latency for common queries (hours, location, etc.)
3. **SmartRouter** - Saves money by using fast LLM for simple queries

### MEDIUM PRIORITY (Quality of Life)
4. **Skill-specific fillers** - More natural per-context responses
5. **P50/P99 tracking** - Essential for production monitoring
6. **Feedback queue** - Enables continuous improvement

### LOWER PRIORITY (Nice to Have)
7. **Compare providers** - Useful for testing but not runtime
8. **LiveKit examples** - Documentation, not code
9. **Mock client** - Testing only

---

## Integration Notes for HIVE215

### Where to Add LatencyMasker
In the LiveKit voice agent, wrap LLM calls:

``````python
# Before (awkward silence during LLM wait):
response = await llm.generate(prompt)
await tts.speak(response)

# After (natural filler sounds):
masker = LatencyMasker(skill_type="customer_service")
async for chunk in masker.mask_latency(llm.generate(prompt)):
    await tts.speak(chunk)
``````

### Where to Add SmartRouter
Before calling any LLM:

``````python
router = SmartRouter(llm_configs)
analysis = router.analyze_query(user_input)
provider = router.select_llm(analysis)
# Now call the selected provider
``````

### Where to Add Cached Responses
In skill loading:

``````python
skill = load_skill("dental_receptionist")
cached = skill.get_cached_response(user_input)
if cached:
    await tts.speak(cached)  # Instant!
else:
    # Route to LLM
``````

---

## Questions for Re-implementation

1. Should fillers be audio files or TTS-generated?
2. Should SmartRouter use embeddings for complexity detection?
3. How should cached responses sync between Fast Brain and HIVE215?
4. Should feedback queue be real-time or batched?

---

*Extracted from Fast Brain repository: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")*
*Target: HIVE215 voice platform re-implementation*
"@

# Write README
$ReadmePath = Join-Path $DestFolder "README.md"
Set-Content -Path $ReadmePath -Value $ReadmeContent -Encoding UTF8
Write-Host "[+] Created: README.md" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Extraction Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Location: $DestFolder" -ForegroundColor Yellow
Write-Host ""
Write-Host "Files extracted:" -ForegroundColor White
Write-Host "  - skill_factory.py (636 lines)" -ForegroundColor Gray
Write-Host "  - skill_dashboard.py (1,394 lines)" -ForegroundColor Gray
Write-Host "  - skill_command_center.py (431 lines)" -ForegroundColor Gray
Write-Host "  - dashboard.py (611 lines)" -ForegroundColor Gray
Write-Host "  - README.md (comprehensive documentation)" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Review README.md for feature details" -ForegroundColor Gray
Write-Host "  2. Prioritize features for HIVE215" -ForegroundColor Gray
Write-Host "  3. Implement LatencyMasker first (biggest UX impact)" -ForegroundColor Gray
Write-Host ""
