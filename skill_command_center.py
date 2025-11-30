"""
Skill Command Center - Hybrid LLM Router with Latency Masking

This is the brain of your voice AI platform. It:
1. Routes queries to the best LLM (fast vs smart)
2. Masks latency with natural filler sounds
3. Retrieves skill-specific knowledge for narrow-role assistants
4. Integrates with your voice platform (LiveKit, Vapi, etc.)
"""

import asyncio
import time
import random
from typing import AsyncGenerator, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

# ============================================================================
# LATENCY MASKING - The Secret Sauce
# ============================================================================

class LatencyMasker:
    """
    Generates natural filler sounds/phrases while waiting for LLM response.
    This makes slow models feel conversational instead of laggy.
    """

    # Filler sounds - short, natural thinking sounds
    FILLER_SOUNDS = [
        "Hmm...",
        "Mmm...",
        "Umm...",
        "Ah...",
        "Well...",
    ]

    # Thinking phrases - for longer waits
    THINKING_PHRASES = [
        "Let me think about that...",
        "That's a good question...",
        "Let me check...",
        "One moment...",
        "Interesting...",
        "Let me see...",
    ]

    # Acknowledgment phrases - shows we heard them
    ACKNOWLEDGMENTS = [
        "I hear you.",
        "Got it.",
        "Okay.",
        "Right.",
        "Sure.",
    ]

    # Domain-specific fillers (customize per skill)
    SKILL_FILLERS = {
        "technical": ["Let me look that up...", "Checking the docs..."],
        "customer_service": ["I understand.", "Let me help with that..."],
        "scheduling": ["Let me check the calendar...", "One moment..."],
        "sales": ["Great question!", "Let me find the best option..."],
    }

    def __init__(self, skill_type: Optional[str] = None):
        self.skill_type = skill_type
        self.last_filler_time = 0
        self.fillers_used = []

    def get_instant_filler(self) -> str:
        """Get a quick filler sound for immediate response (<100ms)"""
        return random.choice(self.FILLER_SOUNDS)

    def get_thinking_phrase(self) -> str:
        """Get a longer thinking phrase for extended waits"""
        if self.skill_type and self.skill_type in self.SKILL_FILLERS:
            phrases = self.THINKING_PHRASES + self.SKILL_FILLERS[self.skill_type]
        else:
            phrases = self.THINKING_PHRASES

        # Avoid repeating the same phrase
        available = [p for p in phrases if p not in self.fillers_used[-3:]]
        if not available:
            available = phrases

        phrase = random.choice(available)
        self.fillers_used.append(phrase)
        return phrase

    async def mask_latency(
        self,
        response_generator: AsyncGenerator[str, None],
        max_wait_before_filler: float = 0.3,  # Start filler after 300ms
        filler_interval: float = 2.0,  # Add new filler every 2s if still waiting
    ) -> AsyncGenerator[str, None]:
        """
        Wraps an LLM response generator and adds natural fillers during waits.

        Usage:
            async for chunk in masker.mask_latency(llm.generate(prompt)):
                send_to_tts(chunk)
        """
        first_token_received = False
        last_yield_time = time.time()
        filler_count = 0

        # Immediately yield a quick acknowledgment
        yield self.get_instant_filler() + " "

        async for chunk in response_generator:
            if not first_token_received:
                first_token_received = True
                # Clear the filler and start real response
                yield "\n"  # Natural transition

            yield chunk
            last_yield_time = time.time()

        # If we never got a response, add a fallback
        if not first_token_received:
            yield "I'm having trouble with that. Could you try again?"


# ============================================================================
# LLM PROVIDERS - Unified Interface
# ============================================================================

class LLMProvider(Enum):
    BITNET = "bitnet"      # Your local 1-bit model (free, slower)
    GROQ = "groq"          # Groq API (fast, free tier)
    OPENAI = "openai"      # OpenAI (smart, paid)
    ANTHROPIC = "anthropic"  # Claude (smart, paid)
    CEREBRAS = "cerebras"  # Cerebras (fast, free tier)
    TOGETHER = "together"  # Together.ai (various models)


@dataclass
class LLMConfig:
    """Configuration for an LLM provider"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.7

    # Performance characteristics
    avg_first_token_ms: int = 500  # Average time to first token
    avg_tokens_per_sec: int = 50   # Average generation speed
    cost_per_1k_tokens: float = 0.0  # Cost in dollars


@dataclass
class QueryAnalysis:
    """Analysis of a user query for routing decisions"""
    complexity: str  # "simple", "medium", "complex"
    intent: str  # "greeting", "question", "command", "conversation"
    skill_match: Optional[str] = None  # Matched skill if any
    requires_reasoning: bool = False
    requires_current_info: bool = False
    confidence: float = 0.9


# ============================================================================
# SMART ROUTER - Decides which LLM to use
# ============================================================================

class SmartRouter:
    """
    Routes queries to the optimal LLM based on:
    - Query complexity
    - Required response quality
    - Latency requirements
    - Cost optimization
    - Skill matching
    """

    # Simple patterns that don't need smart LLMs
    SIMPLE_PATTERNS = [
        "hello", "hi", "hey", "good morning", "good afternoon",
        "how are you", "what's up", "thanks", "thank you", "bye",
        "goodbye", "yes", "no", "okay", "ok", "sure",
    ]

    def __init__(self, llm_configs: Dict[LLMProvider, LLMConfig]):
        self.llm_configs = llm_configs
        self.query_history = []

    def analyze_query(self, query: str, context: Optional[Dict] = None) -> QueryAnalysis:
        """Analyze a query to determine routing"""
        query_lower = query.lower().strip()

        # Check for simple patterns
        for pattern in self.SIMPLE_PATTERNS:
            if query_lower.startswith(pattern) or query_lower == pattern:
                return QueryAnalysis(
                    complexity="simple",
                    intent="greeting" if pattern in ["hello", "hi", "hey"] else "acknowledgment",
                    confidence=0.95
                )

        # Check query length and complexity indicators
        word_count = len(query.split())
        has_question_words = any(w in query_lower for w in ["why", "how", "explain", "what if", "compare"])
        has_reasoning_words = any(w in query_lower for w in ["because", "therefore", "analyze", "think"])

        if word_count <= 5 and not has_question_words:
            complexity = "simple"
        elif has_reasoning_words or word_count > 30:
            complexity = "complex"
        else:
            complexity = "medium"

        return QueryAnalysis(
            complexity=complexity,
            intent="question" if "?" in query else "statement",
            requires_reasoning=has_reasoning_words,
            confidence=0.8
        )

    def select_llm(self, analysis: QueryAnalysis, latency_critical: bool = True) -> LLMProvider:
        """Select the best LLM for this query"""

        # Simple queries -> fastest option
        if analysis.complexity == "simple":
            # Prefer Groq for speed, fallback to BitNet
            if LLMProvider.GROQ in self.llm_configs:
                return LLMProvider.GROQ
            return LLMProvider.BITNET

        # Complex queries -> smartest option
        if analysis.complexity == "complex" or analysis.requires_reasoning:
            # Prefer Claude/GPT-4 for reasoning
            if LLMProvider.ANTHROPIC in self.llm_configs:
                return LLMProvider.ANTHROPIC
            if LLMProvider.OPENAI in self.llm_configs:
                return LLMProvider.OPENAI

        # Medium complexity -> balance speed and quality
        if latency_critical:
            return LLMProvider.GROQ if LLMProvider.GROQ in self.llm_configs else LLMProvider.BITNET
        else:
            return LLMProvider.OPENAI if LLMProvider.OPENAI in self.llm_configs else LLMProvider.GROQ


# ============================================================================
# SKILL RETRIEVER - Knowledge for narrow-role assistants
# ============================================================================

@dataclass
class Skill:
    """A skill represents specialized knowledge/behavior for a narrow role"""
    name: str
    description: str
    system_prompt: str
    knowledge_base: list = field(default_factory=list)  # RAG documents
    example_responses: Dict[str, str] = field(default_factory=dict)  # Pattern -> Response
    filler_type: str = "default"
    preferred_llm: Optional[LLMProvider] = None


class SkillRetriever:
    """Retrieves and applies skills for narrow-role voice assistants"""

    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.active_skill: Optional[Skill] = None

    def register_skill(self, skill: Skill):
        """Register a new skill"""
        self.skills[skill.name] = skill

    def match_skill(self, query: str, context: Optional[Dict] = None) -> Optional[Skill]:
        """Find the best matching skill for a query"""
        # Simple keyword matching for now - could use embeddings
        query_lower = query.lower()

        for skill in self.skills.values():
            # Check if query matches any example patterns
            for pattern in skill.example_responses.keys():
                if pattern.lower() in query_lower:
                    return skill

        return self.active_skill  # Return current skill if no match

    def get_cached_response(self, skill: Skill, query: str) -> Optional[str]:
        """Check if we have a cached response for this query"""
        query_lower = query.lower()
        for pattern, response in skill.example_responses.items():
            if pattern.lower() in query_lower:
                return response
        return None


# ============================================================================
# COMMAND CENTER - Main orchestrator
# ============================================================================

class SkillCommandCenter:
    """
    The main orchestrator that combines:
    - LLM routing
    - Latency masking
    - Skill retrieval
    - Voice platform integration
    """

    def __init__(self):
        self.router = SmartRouter({})
        self.skill_retriever = SkillRetriever()
        self.latency_masker = LatencyMasker()
        self.llm_clients: Dict[LLMProvider, Any] = {}

        # Stats tracking
        self.stats = {
            "total_queries": 0,
            "by_provider": {},
            "avg_latency_ms": 0,
            "cache_hits": 0,
        }

    def register_llm(self, config: LLMConfig, client: Any):
        """Register an LLM provider with its client"""
        self.router.llm_configs[config.provider] = config
        self.llm_clients[config.provider] = client

    def register_skill(self, skill: Skill):
        """Register a skill for narrow-role assistants"""
        self.skill_retriever.register_skill(skill)

    async def process_query(
        self,
        query: str,
        context: Optional[Dict] = None,
        use_latency_masking: bool = True,
        force_provider: Optional[LLMProvider] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Process a query through the command center.

        This is the main entry point for voice AI integration.
        Returns an async generator that yields response chunks.
        """
        start_time = time.time()
        self.stats["total_queries"] += 1

        # 1. Check for skill match and cached response
        skill = self.skill_retriever.match_skill(query, context)
        if skill:
            cached = self.skill_retriever.get_cached_response(skill, query)
            if cached:
                self.stats["cache_hits"] += 1
                yield cached
                return

            # Update latency masker for skill-specific fillers
            self.latency_masker.skill_type = skill.filler_type

        # 2. Analyze query and select LLM
        analysis = self.router.analyze_query(query, context)

        if force_provider:
            provider = force_provider
        elif skill and skill.preferred_llm:
            provider = skill.preferred_llm
        else:
            provider = self.router.select_llm(analysis)

        # Track stats
        provider_name = provider.value
        self.stats["by_provider"][provider_name] = self.stats["by_provider"].get(provider_name, 0) + 1

        # 3. Generate response with latency masking
        async def generate():
            # This would call the actual LLM client
            # For now, placeholder that simulates response
            client = self.llm_clients.get(provider)
            if client:
                # Actual LLM call would go here
                async for chunk in client.generate(query, skill):
                    yield chunk
            else:
                yield f"[No client configured for {provider.value}]"

        if use_latency_masking:
            async for chunk in self.latency_masker.mask_latency(generate()):
                yield chunk
        else:
            async for chunk in generate():
                yield chunk

        # Update latency stats
        elapsed = (time.time() - start_time) * 1000
        self.stats["avg_latency_ms"] = (
            self.stats["avg_latency_ms"] * 0.9 + elapsed * 0.1
        )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def demo():
    """Demo the command center"""

    # Initialize
    center = SkillCommandCenter()

    # Register a sample skill
    receptionist = Skill(
        name="receptionist",
        description="Front desk receptionist for a dental office",
        system_prompt="You are a friendly receptionist at a dental office...",
        example_responses={
            "hours": "We're open Monday through Friday, 9 AM to 5 PM.",
            "appointment": "I'd be happy to help you schedule an appointment!",
            "location": "We're located at 123 Main Street.",
        },
        filler_type="customer_service",
    )
    center.register_skill(receptionist)

    # Process a query
    print("Processing: 'What are your hours?'")
    async for chunk in center.process_query("What are your hours?"):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(demo())
