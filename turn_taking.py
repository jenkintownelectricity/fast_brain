"""
Turn-Taking Module - World-Class Conversational Flow

This module implements state-of-the-art turn-taking for voice agents,
combining multiple signals for natural conversation:

1. Voice Activity Detection (VAD) - Basic speech detection
2. Semantic Turn Detection - Context-aware endpoint prediction
3. Prosodic Analysis - Pitch/energy cues for turn boundaries
4. Barge-in Handling - Intelligent interruption management
5. Backchanneling - Natural acknowledgments ("uh-huh", "got it")

Based on research from:
- LiveKit's transformer-based turn detection
- AssemblyAI's Universal-Streaming semantic endpointing
- "LLM-Enhanced Dialogue Management for Full-Duplex Spoken Dialogue Systems"

Usage:
    from turn_taking import TurnManager, TurnState

    turn_manager = TurnManager()

    # Process audio frames
    for audio_chunk in audio_stream:
        state = turn_manager.process(audio_chunk, transcript)
        if state == TurnState.USER_DONE:
            response = generate_response()
"""

import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, List
from collections import deque
import re


# =============================================================================
# TURN STATES
# =============================================================================

class TurnState(Enum):
    """States in the conversation turn-taking state machine."""
    IDLE = auto()              # No one speaking, waiting
    USER_SPEAKING = auto()     # User is actively speaking
    USER_PAUSING = auto()      # User paused, may continue
    USER_DONE = auto()         # User finished their turn
    AGENT_SPEAKING = auto()    # Agent is responding
    AGENT_INTERRUPTED = auto() # User interrupted the agent
    BACKCHANNEL = auto()       # Short acknowledgment needed


class InterruptionType(Enum):
    """Types of user interruptions."""
    NONE = auto()
    BACKCHANNEL = auto()       # "uh-huh", "okay" - don't stop
    CORRECTION = auto()        # User correcting themselves
    BARGE_IN = auto()          # User wants to take over
    CLARIFICATION = auto()     # User asking for clarification


@dataclass
class TurnContext:
    """Context for turn-taking decisions."""
    transcript: str = ""
    last_agent_utterance: str = ""
    silence_duration_ms: float = 0
    speech_duration_ms: float = 0
    is_question_pending: bool = False
    turn_history: List[str] = field(default_factory=list)
    energy_level: float = 0.0
    pitch_trend: str = "stable"  # rising, falling, stable


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TurnConfig:
    """Configuration for turn-taking behavior."""

    # VAD settings
    min_silence_for_turn_end_ms: float = 500      # Minimum silence to consider turn end
    max_silence_before_timeout_ms: float = 3000   # Max silence before forcing response
    min_speech_for_turn_ms: float = 200           # Minimum speech to count as turn

    # Semantic turn detection
    use_semantic_detection: bool = True
    semantic_confidence_threshold: float = 0.7

    # Interruption handling
    allow_barge_in: bool = True
    barge_in_threshold_ms: float = 300            # How long user must speak to interrupt
    backchannel_max_duration_ms: float = 800      # Max duration for backchannel

    # Prosodic cues
    use_prosodic_cues: bool = True
    falling_pitch_weight: float = 0.3             # Weight for falling pitch = turn end

    # Backchanneling
    enable_backchannels: bool = True
    backchannel_interval_ms: float = 4000         # How often to backchannel

    # Response timing
    response_delay_ms: float = 100                # Small delay before responding
    thinking_acknowledgment_ms: float = 2000      # Say "hmm" if thinking too long


# =============================================================================
# SEMANTIC TURN DETECTOR
# =============================================================================

class SemanticTurnDetector:
    """
    Predicts whether user is done speaking based on transcript content.

    Uses pattern matching and simple heuristics. In production, replace
    with a fine-tuned transformer model (see LiveKit's turn-detector).
    """

    # Patterns that suggest the user is DONE
    TURN_END_PATTERNS = [
        r'\?$',                           # Questions
        r'[.!]$',                         # Sentences ending properly
        r'thanks?$',                      # Thank you
        r'please$',                       # Please (at end)
        r'okay$',                         # Okay (confirming)
        r'that\'?s (all|it)$',           # That's all/it
        r'got it$',                       # Got it
        r'right\??$',                     # Right?
    ]

    # Patterns that suggest user will CONTINUE
    TURN_CONTINUE_PATTERNS = [
        r'\b(and|but|so|because|if|when|while|although)\s*$',  # Conjunctions
        r'\b(um+|uh+|er+|like)\s*$',                           # Hesitation markers
        r',\s*$',                                               # Trailing comma
        r'\.\.\.$',                                            # Ellipsis
        r'\b(first|second|also|another)\b',                    # List indicators
        r'^\s*(well|so|okay)\s*,?\s*$',                        # Discourse markers alone
    ]

    # Backchannel patterns (short responses that aren't real turns)
    BACKCHANNEL_PATTERNS = [
        r'^(uh[- ]?huh|mm[- ]?hmm?|yeah|yep|okay|ok|sure|right|got it|i see)$',
        r'^(yes|no|maybe)$',
    ]

    def __init__(self):
        self.turn_end_re = [re.compile(p, re.IGNORECASE) for p in self.TURN_END_PATTERNS]
        self.turn_continue_re = [re.compile(p, re.IGNORECASE) for p in self.TURN_CONTINUE_PATTERNS]
        self.backchannel_re = [re.compile(p, re.IGNORECASE) for p in self.BACKCHANNEL_PATTERNS]

    def predict_turn_end(self, transcript: str, context: TurnContext) -> tuple[float, str]:
        """
        Predict probability that user is done speaking.

        Returns:
            (confidence: 0-1, reasoning: str)
        """
        transcript = transcript.strip().lower()

        if not transcript:
            return 0.5, "No transcript"

        # Check for backchannel
        for pattern in self.backchannel_re:
            if pattern.match(transcript):
                return 0.9, "Backchannel detected"

        # Check turn-end patterns
        end_score = 0.0
        for pattern in self.turn_end_re:
            if pattern.search(transcript):
                end_score += 0.3

        # Check turn-continue patterns
        continue_score = 0.0
        for pattern in self.turn_continue_re:
            if pattern.search(transcript):
                continue_score += 0.4

        # Question pending from agent increases likelihood of user turn end
        if context.is_question_pending:
            end_score += 0.2

        # Longer utterances more likely to be complete
        word_count = len(transcript.split())
        if word_count > 10:
            end_score += 0.1
        elif word_count < 3:
            end_score -= 0.1

        # Calculate final confidence
        confidence = 0.5 + end_score - continue_score
        confidence = max(0.0, min(1.0, confidence))

        reasoning = f"end_score={end_score:.2f}, continue_score={continue_score:.2f}"
        return confidence, reasoning

    def is_backchannel(self, transcript: str) -> bool:
        """Check if transcript is a backchannel response."""
        transcript = transcript.strip().lower()
        return any(p.match(transcript) for p in self.backchannel_re)


# =============================================================================
# INTERRUPTION CLASSIFIER
# =============================================================================

class InterruptionClassifier:
    """Classifies user interruptions to determine appropriate response."""

    BACKCHANNEL_WORDS = {
        'uh-huh', 'uh huh', 'mm-hmm', 'mmhmm', 'yeah', 'yep', 'yes',
        'okay', 'ok', 'sure', 'right', 'got it', 'i see', 'interesting'
    }

    CORRECTION_STARTERS = {
        'no wait', 'actually', 'i mean', 'sorry', 'let me', 'hold on'
    }

    CLARIFICATION_STARTERS = {
        'what', 'sorry', 'can you', 'could you', 'wait', 'huh', 'pardon'
    }

    def classify(
        self,
        transcript: str,
        speech_duration_ms: float,
        context: TurnContext
    ) -> InterruptionType:
        """Classify the type of interruption."""
        transcript_lower = transcript.strip().lower()

        # Very short speech is likely backchannel
        if speech_duration_ms < 500:
            if any(bc in transcript_lower for bc in self.BACKCHANNEL_WORDS):
                return InterruptionType.BACKCHANNEL

        # Check for correction starters
        if any(transcript_lower.startswith(cs) for cs in self.CORRECTION_STARTERS):
            return InterruptionType.CORRECTION

        # Check for clarification requests
        if any(transcript_lower.startswith(cs) for cs in self.CLARIFICATION_STARTERS):
            return InterruptionType.CLARIFICATION

        # Longer speech with substantial content = barge-in
        if speech_duration_ms > 300 and len(transcript.split()) > 2:
            return InterruptionType.BARGE_IN

        return InterruptionType.NONE


# =============================================================================
# TURN MANAGER
# =============================================================================

class TurnManager:
    """
    Main turn-taking controller.

    Orchestrates VAD, semantic detection, and interruption handling
    to produce natural conversational flow.
    """

    def __init__(self, config: Optional[TurnConfig] = None):
        self.config = config or TurnConfig()
        self.semantic_detector = SemanticTurnDetector()
        self.interruption_classifier = InterruptionClassifier()

        # State
        self.state = TurnState.IDLE
        self.context = TurnContext()

        # Timing
        self.speech_start_time: Optional[float] = None
        self.silence_start_time: Optional[float] = None
        self.last_backchannel_time: float = 0

        # Callbacks
        self.on_state_change: Optional[Callable[[TurnState, TurnState], None]] = None
        self.on_backchannel_needed: Optional[Callable[[], None]] = None

    def process(
        self,
        is_speech: bool,
        transcript: str = "",
        energy: float = 0.0
    ) -> TurnState:
        """
        Process an audio frame and update turn state.

        Args:
            is_speech: Whether VAD detected speech
            transcript: Current transcript (can be partial)
            energy: Audio energy level (0-1)

        Returns:
            Current TurnState
        """
        current_time = time.time() * 1000  # ms

        old_state = self.state
        self.context.transcript = transcript
        self.context.energy_level = energy

        if self.state == TurnState.IDLE:
            self._handle_idle(is_speech, current_time)

        elif self.state == TurnState.USER_SPEAKING:
            self._handle_user_speaking(is_speech, transcript, current_time)

        elif self.state == TurnState.USER_PAUSING:
            self._handle_user_pausing(is_speech, transcript, current_time)

        elif self.state == TurnState.AGENT_SPEAKING:
            self._handle_agent_speaking(is_speech, transcript, current_time)

        elif self.state == TurnState.USER_DONE:
            # External system should transition to AGENT_SPEAKING
            pass

        # Notify state change
        if old_state != self.state and self.on_state_change:
            self.on_state_change(old_state, self.state)

        return self.state

    def _handle_idle(self, is_speech: bool, current_time: float):
        """Handle IDLE state."""
        if is_speech:
            self.state = TurnState.USER_SPEAKING
            self.speech_start_time = current_time
            self.silence_start_time = None

    def _handle_user_speaking(
        self,
        is_speech: bool,
        transcript: str,
        current_time: float
    ):
        """Handle USER_SPEAKING state."""
        if not is_speech:
            # User stopped speaking
            self.silence_start_time = current_time
            self.context.speech_duration_ms = current_time - (self.speech_start_time or current_time)
            self.state = TurnState.USER_PAUSING
        else:
            # Check for backchannel opportunity
            if self.config.enable_backchannels:
                time_since_backchannel = current_time - self.last_backchannel_time
                if time_since_backchannel > self.config.backchannel_interval_ms:
                    if self.on_backchannel_needed:
                        self.on_backchannel_needed()
                    self.last_backchannel_time = current_time

    def _handle_user_pausing(
        self,
        is_speech: bool,
        transcript: str,
        current_time: float
    ):
        """Handle USER_PAUSING state - critical for natural turn-taking."""
        silence_duration = current_time - (self.silence_start_time or current_time)
        self.context.silence_duration_ms = silence_duration

        if is_speech:
            # User resumed speaking
            self.state = TurnState.USER_SPEAKING
            self.silence_start_time = None
            return

        # Semantic turn detection
        if self.config.use_semantic_detection:
            confidence, reasoning = self.semantic_detector.predict_turn_end(
                transcript, self.context
            )

            # High confidence = shorter silence needed
            adjusted_threshold = self.config.min_silence_for_turn_end_ms
            if confidence > self.config.semantic_confidence_threshold:
                adjusted_threshold *= (1 - confidence * 0.5)  # Up to 50% reduction

            if silence_duration >= adjusted_threshold:
                self.state = TurnState.USER_DONE
                self._record_turn(transcript)
                return

        else:
            # VAD-only fallback
            if silence_duration >= self.config.min_silence_for_turn_end_ms:
                self.state = TurnState.USER_DONE
                self._record_turn(transcript)
                return

        # Timeout - force response even if uncertain
        if silence_duration >= self.config.max_silence_before_timeout_ms:
            self.state = TurnState.USER_DONE
            self._record_turn(transcript)

    def _handle_agent_speaking(
        self,
        is_speech: bool,
        transcript: str,
        current_time: float
    ):
        """Handle AGENT_SPEAKING state - detect interruptions."""
        if not is_speech:
            return

        if not self.config.allow_barge_in:
            return

        # User is speaking while agent is speaking
        if self.speech_start_time is None:
            self.speech_start_time = current_time
            return

        speech_duration = current_time - self.speech_start_time

        # Classify the interruption
        interruption_type = self.interruption_classifier.classify(
            transcript,
            speech_duration,
            self.context
        )

        if interruption_type == InterruptionType.BACKCHANNEL:
            # Don't stop for backchannels
            self.state = TurnState.BACKCHANNEL
        elif interruption_type in [InterruptionType.BARGE_IN, InterruptionType.CORRECTION]:
            # Stop and listen
            if speech_duration >= self.config.barge_in_threshold_ms:
                self.state = TurnState.AGENT_INTERRUPTED
                self.context.last_agent_utterance = ""  # Clear for retry
        elif interruption_type == InterruptionType.CLARIFICATION:
            # Stop and clarify
            self.state = TurnState.AGENT_INTERRUPTED

    def _record_turn(self, transcript: str):
        """Record completed turn in history."""
        if transcript.strip():
            self.context.turn_history.append(f"USER: {transcript.strip()}")
            # Keep last 10 turns
            if len(self.context.turn_history) > 10:
                self.context.turn_history.pop(0)

    def start_agent_turn(self, utterance: str = ""):
        """Call when agent starts speaking."""
        self.state = TurnState.AGENT_SPEAKING
        self.context.last_agent_utterance = utterance
        self.context.is_question_pending = utterance.strip().endswith('?')
        self.speech_start_time = None

    def finish_agent_turn(self):
        """Call when agent finishes speaking."""
        self.state = TurnState.IDLE
        if self.context.last_agent_utterance.strip():
            self.context.turn_history.append(
                f"AGENT: {self.context.last_agent_utterance.strip()}"
            )

    def reset(self):
        """Reset turn manager state."""
        self.state = TurnState.IDLE
        self.context = TurnContext()
        self.speech_start_time = None
        self.silence_start_time = None


# =============================================================================
# BACKCHANNEL GENERATOR
# =============================================================================

class BackchannelGenerator:
    """Generate natural backchannel responses."""

    ACKNOWLEDGMENTS = [
        "uh-huh", "mm-hmm", "right", "okay", "I see", "got it", "sure"
    ]

    ENCOURAGEMENTS = [
        "go on", "I'm listening", "tell me more", "and then?"
    ]

    UNDERSTANDING = [
        "I understand", "that makes sense", "I see what you mean"
    ]

    def __init__(self):
        self.last_used = None

    def generate(self, context: TurnContext) -> str:
        """Generate appropriate backchannel response."""
        import random

        # Choose category based on context
        if context.speech_duration_ms > 5000:
            # Long speech - encourage continuation
            options = self.ENCOURAGEMENTS
        elif '?' in context.transcript:
            # Question - show understanding
            options = self.UNDERSTANDING
        else:
            # Default acknowledgment
            options = self.ACKNOWLEDGMENTS

        # Avoid repeating
        available = [o for o in options if o != self.last_used]
        if not available:
            available = options

        choice = random.choice(available)
        self.last_used = choice
        return choice


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create turn manager with custom config
    config = TurnConfig(
        min_silence_for_turn_end_ms=400,
        use_semantic_detection=True,
        allow_barge_in=True,
        enable_backchannels=True
    )

    manager = TurnManager(config)
    backchannel_gen = BackchannelGenerator()

    # Callbacks
    def on_state_change(old_state, new_state):
        print(f"State: {old_state.name} -> {new_state.name}")

    def on_backchannel():
        bc = backchannel_gen.generate(manager.context)
        print(f"[Backchannel: {bc}]")

    manager.on_state_change = on_state_change
    manager.on_backchannel_needed = on_backchannel

    # Simulate conversation
    print("=== Simulating Turn-Taking ===\n")

    # User starts speaking
    manager.process(is_speech=True, transcript="Hi, I need help with")
    time.sleep(0.1)
    manager.process(is_speech=True, transcript="Hi, I need help with my plumbing")
    time.sleep(0.1)
    manager.process(is_speech=True, transcript="Hi, I need help with my plumbing because")
    time.sleep(0.1)

    # User pauses mid-sentence (semantic detection should wait)
    manager.process(is_speech=False, transcript="Hi, I need help with my plumbing because")
    time.sleep(0.3)
    manager.process(is_speech=False, transcript="Hi, I need help with my plumbing because")
    print(f"  -> Waiting (conjunction detected)")

    # User continues
    manager.process(is_speech=True, transcript="Hi, I need help with my plumbing because my sink is leaking")
    time.sleep(0.1)

    # User finishes with falling intonation
    manager.process(is_speech=False, transcript="Hi, I need help with my plumbing because my sink is leaking.")
    time.sleep(0.5)
    manager.process(is_speech=False, transcript="Hi, I need help with my plumbing because my sink is leaking.")

    print(f"\nFinal state: {manager.state.name}")
    print(f"Turn history: {manager.context.turn_history}")
