"""
HIVE215 Parler-TTS Integration
==============================
Dynamic voice expression using text-described speech synthesis.

Parler-TTS lets you control voice characteristics through natural language:
- "A young woman speaking quickly with excitement"
- "An older man with a deep, gravelly voice speaking slowly and calmly"
- "A professional female voice with clear enunciation"

This enables:
1. Emotion-aware responses (cheerful for good news, calm for complaints)
2. Skill-specific voice personas (warm receptionist vs authoritative technician)
3. Dynamic voice adaptation based on conversation context

Architecture:
    [Groq/Claude Response] → [Voice Description Generator] → [Parler-TTS] → [Audio]
    
The LLM can output voice descriptions alongside text, enabling real-time voice modulation.
"""

import modal
import os
import io
import base64
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# VOICE DESCRIPTION SYSTEM
# =============================================================================

class Emotion(Enum):
    """Emotional states that affect voice delivery."""
    NEUTRAL = "neutral"
    WARM = "warm"
    EXCITED = "excited"
    CALM = "calm"
    CONCERNED = "concerned"
    APOLOGETIC = "apologetic"
    CONFIDENT = "confident"
    EMPATHETIC = "empathetic"
    URGENT = "urgent"
    CHEERFUL = "cheerful"


@dataclass
class VoicePersona:
    """A voice persona with base characteristics."""
    name: str
    base_description: str
    gender: str
    age_range: str
    speaking_rate: str  # slow, moderate, fast
    pitch: str  # low, medium, high
    
    def get_description(self, emotion: Emotion = Emotion.NEUTRAL) -> str:
        """Get full voice description with emotion modifier."""
        emotion_modifiers = {
            Emotion.NEUTRAL: "",
            Emotion.WARM: "speaking warmly and kindly",
            Emotion.EXCITED: "speaking with enthusiasm and energy",
            Emotion.CALM: "speaking in a soothing, relaxed manner",
            Emotion.CONCERNED: "speaking with genuine concern",
            Emotion.APOLOGETIC: "speaking apologetically and sincerely",
            Emotion.CONFIDENT: "speaking with confidence and authority",
            Emotion.EMPATHETIC: "speaking with deep empathy and understanding",
            Emotion.URGENT: "speaking with urgency and importance",
            Emotion.CHEERFUL: "speaking cheerfully and pleasantly",
        }
        
        modifier = emotion_modifiers.get(emotion, "")
        if modifier:
            return f"{self.base_description}, {modifier}"
        return self.base_description


# Pre-defined voice personas for different skills
VOICE_PERSONAS = {
    "receptionist": VoicePersona(
        name="Sarah",
        base_description="A young woman with a clear, pleasant voice speaking at a moderate pace with professional warmth",
        gender="female",
        age_range="25-35",
        speaking_rate="moderate",
        pitch="medium"
    ),
    "electrician": VoicePersona(
        name="Mike",
        base_description="A middle-aged man with a friendly, knowledgeable voice speaking clearly and confidently",
        gender="male",
        age_range="35-50",
        speaking_rate="moderate",
        pitch="medium-low"
    ),
    "plumber": VoicePersona(
        name="Tom",
        base_description="A man with a reassuring, calm voice speaking steadily with patience",
        gender="male",
        age_range="40-55",
        speaking_rate="moderate-slow",
        pitch="low"
    ),
    "lawyer": VoicePersona(
        name="Jennifer",
        base_description="A professional woman with a clear, articulate voice speaking with measured confidence",
        gender="female",
        age_range="35-50",
        speaking_rate="moderate",
        pitch="medium"
    ),
    "solar": VoicePersona(
        name="Alex",
        base_description="A young, enthusiastic person with an energetic voice speaking with genuine excitement",
        gender="neutral",
        age_range="25-35",
        speaking_rate="moderate-fast",
        pitch="medium"
    ),
    "general": VoicePersona(
        name="Jordan",
        base_description="A friendly person with a warm, natural voice speaking conversationally",
        gender="neutral",
        age_range="28-40",
        speaking_rate="moderate",
        pitch="medium"
    ),
}


# =============================================================================
# EMOTION DETECTION
# =============================================================================

def detect_emotion_from_context(
    user_message: str,
    agent_response: str,
    skill_id: str
) -> Emotion:
    """
    Detect appropriate emotion based on conversation context.
    
    This is a rule-based approach. For more sophistication,
    you could use an LLM to classify emotion.
    """
    user_lower = user_message.lower()
    response_lower = agent_response.lower()
    
    # Emergency/urgent situations
    urgent_keywords = ["emergency", "urgent", "asap", "right now", "immediately", "sparks", "flooding", "fire", "smoke"]
    if any(kw in user_lower for kw in urgent_keywords):
        return Emotion.URGENT
    
    # Complaints or frustration
    complaint_keywords = ["frustrated", "angry", "upset", "disappointed", "complaint", "problem", "issue", "not working"]
    if any(kw in user_lower for kw in complaint_keywords):
        return Emotion.EMPATHETIC
    
    # Apology situations
    apology_keywords = ["sorry", "apologize", "apologies", "unfortunately", "regret"]
    if any(kw in response_lower for kw in apology_keywords):
        return Emotion.APOLOGETIC
    
    # Good news / positive outcomes
    positive_keywords = ["great news", "good news", "congratulations", "excellent", "perfect", "wonderful"]
    if any(kw in response_lower for kw in positive_keywords):
        return Emotion.CHEERFUL
    
    # Technical explanations
    if skill_id in ["electrician", "plumber"] and len(agent_response) > 200:
        return Emotion.CONFIDENT
    
    # Legal intake (always professional, empathetic)
    if skill_id == "lawyer":
        return Emotion.EMPATHETIC
    
    # Sales/solar (enthusiastic)
    if skill_id == "solar":
        return Emotion.EXCITED
    
    # Default warm tone for most interactions
    return Emotion.WARM


# =============================================================================
# PARLER-TTS MODEL (Modal Deployment)
# =============================================================================

# Modal app for Parler-TTS
app = modal.App("hive215-parler-tts")

# Create image with Parler-TTS dependencies
parler_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "parler-tts",
    "soundfile",
    "numpy"
)


@app.cls(
    image=parler_image,
    gpu="T4",  # Parler-TTS runs well on T4
    scaledown_window=300,  # Keep warm for 5 minutes
)
class ParlerTTSModel:
    """
    Parler-TTS model for description-based speech synthesis.
    
    Usage:
        model = ParlerTTSModel()
        audio = model.synthesize(
            text="Hello, how can I help you today?",
            description="A warm female voice speaking clearly"
        )
    """
    
    @modal.enter()
    def load_model(self):
        """Load model on container start."""
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the mini model for faster inference
        # Options: "parler-tts/parler-tts-mini-v1" (faster) or "parler-tts/parler-tts-large-v1" (better quality)
        model_name = "parler-tts/parler-tts-mini-v1"
        
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

        print(f"Parler-TTS loaded on {device}")

    def _synthesize_internal(
        self,
        text: str,
        description: str,
        output_format: str = "wav"
    ) -> bytes:
        """Internal synthesis method - not exposed as Modal endpoint."""
        import torch
        import soundfile as sf
        import numpy as np

        # Tokenize inputs
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        # Generate audio
        with torch.no_grad():
            generation = self.model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
                do_sample=True,
                temperature=1.0,
            )

        # Convert to audio
        audio_arr = generation.cpu().numpy().squeeze()

        # Write to buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio_arr, samplerate=self.model.config.sampling_rate, format=output_format.upper())
        buffer.seek(0)

        return buffer.read()

    @modal.method()
    def synthesize(
        self,
        text: str,
        description: str,
        output_format: str = "wav"
    ) -> bytes:
        """
        Synthesize speech from text with voice description.

        Args:
            text: The text to speak
            description: Natural language description of the voice
            output_format: "wav" or "mp3"

        Returns:
            Audio bytes
        """
        return self._synthesize_internal(text, description, output_format)
    
    @modal.method()
    def synthesize_with_emotion(
        self,
        text: str,
        skill_id: str,
        emotion: str = "neutral",
        user_context: Optional[str] = None
    ) -> Tuple[bytes, str]:
        """
        Synthesize with automatic voice persona and emotion.
        
        Args:
            text: Text to speak
            skill_id: The skill ID (e.g., "electrician")
            emotion: Emotion name or "auto" for detection
            user_context: User's message for emotion detection
            
        Returns:
            (audio_bytes, description_used)
        """
        # Get voice persona
        persona = VOICE_PERSONAS.get(skill_id, VOICE_PERSONAS["general"])

        # Clean and determine emotion
        emotion_clean = emotion.strip().strip('"').strip("'").lower()

        if emotion_clean == "auto" and user_context:
            detected_emotion = detect_emotion_from_context(user_context, text, skill_id)
        elif emotion_clean == "neutral" or not emotion_clean:
            detected_emotion = Emotion.NEUTRAL
        else:
            # Try to match emotion enum
            try:
                detected_emotion = Emotion[emotion_clean.upper()]
            except KeyError:
                # Fallback to neutral if invalid emotion
                detected_emotion = Emotion.NEUTRAL
        
        # Build description
        description = persona.get_description(detected_emotion)

        # Synthesize using internal method
        audio = self._synthesize_internal(text, description)

        return audio, description


# =============================================================================
# FAST BRAIN INTEGRATION
# =============================================================================

class ParlerTTSClient:
    """
    Client for using Parler-TTS in your voice pipeline.
    
    Usage:
        client = ParlerTTSClient()
        
        # Simple synthesis
        audio = await client.speak(
            "Hello, how can I help?",
            skill_id="receptionist"
        )
        
        # With emotion
        audio = await client.speak(
            "I'm so sorry to hear that. Let me help.",
            skill_id="receptionist",
            emotion="empathetic"
        )
    """
    
    def __init__(self):
        self._model = None
    
    def _get_model(self):
        if self._model is None:
            self._model = modal.Cls.lookup("hive215-parler-tts", "ParlerTTSModel")()
        return self._model
    
    async def speak(
        self,
        text: str,
        skill_id: str = "general",
        emotion: str = "auto",
        user_context: Optional[str] = None
    ) -> bytes:
        """Generate speech with appropriate voice and emotion."""
        model = self._get_model()
        audio, description = await model.synthesize_with_emotion.remote.aio(
            text=text,
            skill_id=skill_id,
            emotion=emotion,
            user_context=user_context
        )
        return audio
    
    async def speak_with_description(
        self,
        text: str,
        description: str
    ) -> bytes:
        """Generate speech with custom description."""
        model = self._get_model()
        return await model.synthesize.remote.aio(text, description)


# =============================================================================
# LLM-DRIVEN VOICE DESCRIPTIONS
# =============================================================================

VOICE_DESCRIPTION_PROMPT = """You are a voice direction assistant. Given a response text and context, output a brief voice description for text-to-speech.

The description should include:
- Speaker characteristics (age, gender if appropriate)
- Emotional tone
- Speaking pace
- Any special delivery notes

Keep descriptions under 30 words. Be specific and natural.

Examples:
- "A warm female voice speaking with genuine concern, slightly slower pace"
- "An energetic young man speaking quickly with enthusiasm"  
- "A calm, professional voice with clear enunciation"

Context: {context}
Skill: {skill}
Response text: {text}

Voice description:"""


async def get_llm_voice_description(
    text: str,
    skill_id: str,
    context: str,
    groq_client
) -> str:
    """
    Use Groq to generate a dynamic voice description.
    
    This allows the LLM to pick voice characteristics based on
    the specific content and context of each response.
    """
    prompt = VOICE_DESCRIPTION_PROMPT.format(
        context=context,
        skill=skill_id,
        text=text[:200]  # First 200 chars for context
    )
    
    response = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()


# =============================================================================
# HYBRID TTS STRATEGY
# =============================================================================

class HybridTTSManager:
    """
    Manages TTS with fallback between Parler-TTS and Cartesia.
    
    Strategy:
    - Use Cartesia for speed-critical responses (System 1)
    - Use Parler-TTS for emotional/nuanced responses (System 2)
    - Fall back to Cartesia if Parler-TTS fails
    
    Usage:
        tts = HybridTTSManager(cartesia_key="...")
        
        audio = await tts.synthesize(
            text="I understand your frustration.",
            skill_id="receptionist",
            use_parler=True,  # Enable dynamic voice
            emotion="empathetic"
        )
    """
    
    def __init__(self, cartesia_api_key: Optional[str] = None):
        self.cartesia_key = cartesia_api_key or os.environ.get("CARTESIA_API_KEY")
        self.parler_client = ParlerTTSClient()
        
        # Cartesia voice IDs per skill (you'd configure these)
        self.cartesia_voices = {
            "receptionist": "a0e99841-438c-4a64-b679-ae501e7d6091",  # Example ID
            "electrician": "694f9389-aac1-45b6-b726-9d9369183238",
            "general": "a0e99841-438c-4a64-b679-ae501e7d6091",
        }
    
    async def synthesize(
        self,
        text: str,
        skill_id: str = "general",
        use_parler: bool = False,
        emotion: str = "auto",
        user_context: Optional[str] = None
    ) -> bytes:
        """
        Synthesize speech with automatic provider selection.
        """
        if use_parler:
            try:
                return await self.parler_client.speak(
                    text=text,
                    skill_id=skill_id,
                    emotion=emotion,
                    user_context=user_context
                )
            except Exception as e:
                print(f"Parler-TTS failed, falling back to Cartesia: {e}")
        
        # Fallback to Cartesia
        return await self._cartesia_synthesize(text, skill_id)
    
    async def _cartesia_synthesize(self, text: str, skill_id: str) -> bytes:
        """Synthesize with Cartesia (faster, less expressive)."""
        import httpx
        
        voice_id = self.cartesia_voices.get(skill_id, self.cartesia_voices["general"])
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.cartesia.ai/tts/bytes",
                headers={
                    "X-API-Key": self.cartesia_key,
                    "Cartesia-Version": "2024-06-10",
                    "Content-Type": "application/json"
                },
                json={
                    "model_id": "sonic-english",
                    "transcript": text,
                    "voice": {"mode": "id", "id": voice_id},
                    "output_format": {
                        "container": "wav",
                        "encoding": "pcm_s16le",
                        "sample_rate": 24000
                    }
                }
            )
            return response.content


# =============================================================================
# DEPLOYMENT EXAMPLE
# =============================================================================

DEPLOYMENT_EXAMPLE = '''
# In your deploy_groq.py, add Parler-TTS for emotional responses:

from parler_integration import HybridTTSManager, detect_emotion_from_context

tts = HybridTTSManager()

@app.function(secrets=[groq_secret, anthropic_secret, cartesia_secret])
@modal.web_endpoint(method="POST")
async def chat_with_voice(request: dict):
    """Generate response with dynamic TTS."""
    
    user_message = request["messages"][-1]["content"]
    skill_id = request.get("skill", "general")
    
    # Get text response from Fast Brain
    response = await fast_brain_inference(request)
    text = response["content"]
    system_used = response["system_used"]
    
    # Determine if we should use expressive TTS
    # Use Parler for System 2 responses (more time available)
    # Use Cartesia for System 1 (speed critical)
    use_expressive = (system_used == "deep")
    
    # Detect emotion for voice
    emotion = detect_emotion_from_context(user_message, text, skill_id)
    
    # Generate audio
    audio = await tts.synthesize(
        text=text,
        skill_id=skill_id,
        use_parler=use_expressive,
        emotion=emotion.value,
        user_context=user_message
    )
    
    return {
        "text": text,
        "audio_base64": base64.b64encode(audio).decode(),
        "system_used": system_used,
        "emotion": emotion.value,
        "voice_type": "parler" if use_expressive else "cartesia"
    }
'''


# =============================================================================
# TESTING / CLI
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_voice_descriptions():
        """Test voice persona descriptions."""
        print("=== Voice Persona Test ===\n")
        
        for skill_id, persona in VOICE_PERSONAS.items():
            print(f"Skill: {skill_id}")
            print(f"  Neutral: {persona.get_description(Emotion.NEUTRAL)}")
            print(f"  Empathetic: {persona.get_description(Emotion.EMPATHETIC)}")
            print(f"  Urgent: {persona.get_description(Emotion.URGENT)}")
            print()
    
    async def test_emotion_detection():
        """Test emotion detection."""
        print("=== Emotion Detection Test ===\n")
        
        test_cases = [
            ("There's sparks coming from my outlet!", "Stay away from that area. I'm dispatching someone now.", "electrician"),
            ("I'm really frustrated with the service", "I completely understand. Let me help make this right.", "receptionist"),
            ("What are your hours?", "We're open Monday through Friday, 8am to 6pm.", "receptionist"),
            ("Great, I'd like to schedule an appointment", "Wonderful! I have availability tomorrow at 2pm.", "solar"),
        ]
        
        for user_msg, response, skill in test_cases:
            emotion = detect_emotion_from_context(user_msg, response, skill)
            print(f"User: {user_msg[:50]}...")
            print(f"Response: {response[:50]}...")
            print(f"Detected emotion: {emotion.value}")
            print()
    
    # Run tests
    asyncio.run(test_voice_descriptions())
    asyncio.run(test_emotion_detection())
    
    print("\n=== Deployment Instructions ===")
    print("1. Deploy Parler-TTS model:")
    print("   modal deploy parler_integration.py")
    print("\n2. Add to your voice pipeline:")
    print(DEPLOYMENT_EXAMPLE[:500] + "...")
