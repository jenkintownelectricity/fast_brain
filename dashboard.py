"""
Skill Command Center Dashboard

A visual interface for:
- Testing different LLM providers side-by-side
- Configuring latency masking
- Creating and testing skills
- Monitoring performance
- Connecting to voice platforms
"""

import gradio as gr
import asyncio
import time
import json
from typing import Optional
from dataclasses import asdict

from skill_command_center import (
    SkillCommandCenter,
    LLMProvider,
    LLMConfig,
    Skill,
    LatencyMasker,
)

# ============================================================================
# LLM CLIENT IMPLEMENTATIONS
# ============================================================================

class BitNetClient:
    """Client for your local BitNet model on Modal"""

    def __init__(self):
        self.modal_available = False
        try:
            import modal
            self.modal_available = True
        except ImportError:
            pass

    async def generate(self, prompt: str, skill: Optional[Skill] = None):
        if not self.modal_available:
            yield "[BitNet: Modal not installed. Run: pip install modal]"
            return

        try:
            import modal
            lpu = modal.Cls.from_name("bitnet-lpu-v1", "VirtualLPU")()

            # Build prompt with skill context
            if skill:
                full_prompt = f"{skill.system_prompt}\n\nUser: {prompt}\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\nAssistant:"

            # Get response
            response = ""
            for chunk in lpu.chat.remote_gen(full_prompt, max_tokens=128):
                response += chunk
                yield chunk

        except Exception as e:
            yield f"[BitNet Error: {str(e)}]"


class GroqClient:
    """Client for Groq API (fast inference)"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
        except ImportError:
            pass

    async def generate(self, prompt: str, skill: Optional[Skill] = None):
        if not self.client:
            yield "[Groq: Install with 'pip install groq']"
            return

        try:
            messages = []
            if skill:
                messages.append({"role": "system", "content": skill.system_prompt})
            messages.append({"role": "user", "content": prompt})

            stream = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                stream=True,
                max_tokens=256,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"[Groq Error: {str(e)}]"


class OpenAIClient:
    """Client for OpenAI API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            pass

    async def generate(self, prompt: str, skill: Optional[Skill] = None):
        if not self.client:
            yield "[OpenAI: Install with 'pip install openai']"
            return

        try:
            messages = []
            if skill:
                messages.append({"role": "system", "content": skill.system_prompt})
            messages.append({"role": "user", "content": prompt})

            stream = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True,
                max_tokens=256,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"[OpenAI Error: {str(e)}]"


class AnthropicClient:
    """Client for Anthropic Claude API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            pass

    async def generate(self, prompt: str, skill: Optional[Skill] = None):
        if not self.client:
            yield "[Anthropic: Install with 'pip install anthropic']"
            return

        try:
            system = skill.system_prompt if skill else "You are a helpful assistant."

            with self.client.messages.stream(
                model="claude-3-haiku-20240307",
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            yield f"[Anthropic Error: {str(e)}]"


# ============================================================================
# MOCK CLIENT FOR TESTING
# ============================================================================

class MockClient:
    """Mock client for testing without API keys"""

    def __init__(self, name: str, delay: float = 0.5):
        self.name = name
        self.delay = delay

    async def generate(self, prompt: str, skill: Optional[Skill] = None):
        await asyncio.sleep(self.delay)

        if skill:
            response = f"[{self.name}] As a {skill.name}, I would say: "
        else:
            response = f"[{self.name}] "

        # Simulate streaming
        words = f"This is a simulated response from {self.name}. In production, this would be a real LLM response to: '{prompt[:50]}...'".split()

        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)


# ============================================================================
# DASHBOARD STATE
# ============================================================================

class DashboardState:
    def __init__(self):
        self.center = SkillCommandCenter()
        self.api_keys = {}
        self.active_providers = set()
        self.conversation_history = []
        self.test_results = []

        # Register mock clients by default
        self._register_mock_clients()

    def _register_mock_clients(self):
        """Register mock clients for testing"""
        for provider in LLMProvider:
            config = LLMConfig(
                provider=provider,
                model="mock",
                avg_first_token_ms=500,
                avg_tokens_per_sec=50,
            )
            client = MockClient(provider.value, delay=0.3 if provider == LLMProvider.GROQ else 1.0)
            self.center.register_llm(config, client)

    def set_api_key(self, provider: str, api_key: str):
        """Set API key for a provider and initialize real client"""
        self.api_keys[provider] = api_key

        if provider == "groq" and api_key:
            config = LLMConfig(
                provider=LLMProvider.GROQ,
                model="llama-3.1-8b-instant",
                api_key=api_key,
                avg_first_token_ms=100,
                avg_tokens_per_sec=300,
            )
            self.center.register_llm(config, GroqClient(api_key))
            self.active_providers.add("groq")

        elif provider == "openai" and api_key:
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4o-mini",
                api_key=api_key,
                avg_first_token_ms=300,
                avg_tokens_per_sec=80,
                cost_per_1k_tokens=0.00015,
            )
            self.center.register_llm(config, OpenAIClient(api_key))
            self.active_providers.add("openai")

        elif provider == "anthropic" and api_key:
            config = LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-haiku-20240307",
                api_key=api_key,
                avg_first_token_ms=400,
                avg_tokens_per_sec=100,
                cost_per_1k_tokens=0.00025,
            )
            self.center.register_llm(config, AnthropicClient(api_key))
            self.active_providers.add("anthropic")

    def register_skill(self, name: str, description: str, system_prompt: str,
                       cached_responses: str, filler_type: str):
        """Register a new skill"""
        # Parse cached responses (format: "pattern: response" per line)
        example_responses = {}
        for line in cached_responses.strip().split("\n"):
            if ": " in line:
                pattern, response = line.split(": ", 1)
                example_responses[pattern.strip()] = response.strip()

        skill = Skill(
            name=name,
            description=description,
            system_prompt=system_prompt,
            example_responses=example_responses,
            filler_type=filler_type,
        )
        self.center.register_skill(skill)
        return f"✅ Skill '{name}' registered with {len(example_responses)} cached responses"


# Global state
state = DashboardState()


# ============================================================================
# DASHBOARD UI FUNCTIONS
# ============================================================================

def save_api_keys(groq_key, openai_key, anthropic_key):
    """Save API keys and initialize clients"""
    results = []

    if groq_key:
        state.set_api_key("groq", groq_key)
        results.append("✅ Groq API key saved")

    if openai_key:
        state.set_api_key("openai", openai_key)
        results.append("✅ OpenAI API key saved")

    if anthropic_key:
        state.set_api_key("anthropic", anthropic_key)
        results.append("✅ Anthropic API key saved")

    if not results:
        results.append("No API keys provided. Using mock clients for testing.")

    return "\n".join(results)


async def test_single_provider(prompt: str, provider: str, use_masking: bool):
    """Test a single LLM provider"""
    provider_enum = LLMProvider(provider)

    start_time = time.time()
    response = ""
    first_token_time = None

    async for chunk in state.center.process_query(
        prompt,
        use_latency_masking=use_masking,
        force_provider=provider_enum,
    ):
        if first_token_time is None:
            first_token_time = time.time() - start_time
        response += chunk
        yield response

    total_time = time.time() - start_time

    # Add stats
    stats = f"\n\n---\n⏱️ First token: {first_token_time*1000:.0f}ms | Total: {total_time*1000:.0f}ms"
    yield response + stats


async def compare_providers(prompt: str, providers: list, use_masking: bool):
    """Compare multiple providers side by side"""
    results = {p: "" for p in providers}
    stats = {p: {"first_token": None, "start": time.time()} for p in providers}

    async def run_provider(provider):
        provider_enum = LLMProvider(provider)
        async for chunk in state.center.process_query(
            prompt,
            use_latency_masking=use_masking,
            force_provider=provider_enum,
        ):
            if stats[provider]["first_token"] is None:
                stats[provider]["first_token"] = time.time() - stats[provider]["start"]
            results[provider] += chunk

    # Run all providers concurrently
    await asyncio.gather(*[run_provider(p) for p in providers])

    # Format results
    output = ""
    for provider in providers:
        total_time = time.time() - stats[provider]["start"]
        first_token = stats[provider]["first_token"] or 0
        output += f"### {provider.upper()}\n"
        output += f"⏱️ First token: {first_token*1000:.0f}ms | Total: {total_time*1000:.0f}ms\n\n"
        output += results[provider] + "\n\n---\n\n"

    return output


def register_new_skill(name, description, system_prompt, cached_responses, filler_type):
    """Register a new skill from the UI"""
    return state.register_skill(name, description, system_prompt, cached_responses, filler_type)


def get_stats():
    """Get current stats as formatted string"""
    stats = state.center.stats
    return json.dumps(stats, indent=2)


def list_skills():
    """List all registered skills"""
    skills = state.center.skill_retriever.skills
    if not skills:
        return "No skills registered yet."

    output = ""
    for name, skill in skills.items():
        output += f"### {name}\n"
        output += f"**Description:** {skill.description}\n"
        output += f"**Filler Type:** {skill.filler_type}\n"
        output += f"**Cached Responses:** {len(skill.example_responses)}\n\n"

    return output


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_dashboard():
    """Create the Gradio dashboard"""

    with gr.Blocks(title="Skill Command Center", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎛️ Skill Command Center

        **Your hybrid LLM testing and voice AI control panel**

        Mix and match LLM providers, test latency masking, and create skills for narrow-role voice assistants.
        """)

        with gr.Tabs():
            # ============ TAB 1: API Keys ============
            with gr.TabItem("🔑 API Keys"):
                gr.Markdown("### Configure your LLM API keys")
                gr.Markdown("Leave blank to use mock clients for testing.")

                with gr.Row():
                    groq_key = gr.Textbox(label="Groq API Key", type="password")
                    openai_key = gr.Textbox(label="OpenAI API Key", type="password")
                    anthropic_key = gr.Textbox(label="Anthropic API Key", type="password")

                save_btn = gr.Button("Save API Keys", variant="primary")
                key_status = gr.Markdown()

                save_btn.click(save_api_keys, [groq_key, openai_key, anthropic_key], key_status)

            # ============ TAB 2: Single Provider Test ============
            with gr.TabItem("🧪 Test Single LLM"):
                gr.Markdown("### Test a single LLM provider")

                with gr.Row():
                    test_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="What are your hours?",
                        lines=2
                    )
                    test_provider = gr.Dropdown(
                        label="Provider",
                        choices=["bitnet", "groq", "openai", "anthropic"],
                        value="groq"
                    )

                use_masking = gr.Checkbox(label="Enable Latency Masking", value=True)
                test_btn = gr.Button("Run Test", variant="primary")
                test_output = gr.Markdown(label="Response")

                test_btn.click(
                    test_single_provider,
                    [test_prompt, test_provider, use_masking],
                    test_output
                )

            # ============ TAB 3: Compare Providers ============
            with gr.TabItem("⚖️ Compare Providers"):
                gr.Markdown("### Compare multiple LLMs side-by-side")

                compare_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Explain quantum computing in one sentence.",
                    lines=2
                )

                compare_providers_select = gr.CheckboxGroup(
                    label="Providers to compare",
                    choices=["bitnet", "groq", "openai", "anthropic"],
                    value=["groq", "openai"]
                )

                compare_masking = gr.Checkbox(label="Enable Latency Masking", value=False)
                compare_btn = gr.Button("Compare", variant="primary")
                compare_output = gr.Markdown(label="Results")

                compare_btn.click(
                    compare_providers,
                    [compare_prompt, compare_providers_select, compare_masking],
                    compare_output
                )

            # ============ TAB 4: Skills ============
            with gr.TabItem("🎯 Skills"):
                gr.Markdown("### Create and manage narrow-role voice assistant skills")

                with gr.Row():
                    with gr.Column():
                        skill_name = gr.Textbox(label="Skill Name", placeholder="receptionist")
                        skill_desc = gr.Textbox(label="Description", placeholder="Front desk receptionist")
                        skill_filler = gr.Dropdown(
                            label="Filler Type",
                            choices=["default", "technical", "customer_service", "scheduling", "sales"],
                            value="customer_service"
                        )

                    with gr.Column():
                        skill_system = gr.Textbox(
                            label="System Prompt",
                            placeholder="You are a friendly receptionist at a dental office...",
                            lines=4
                        )
                        skill_cached = gr.Textbox(
                            label="Cached Responses (pattern: response)",
                            placeholder="hours: We're open Monday through Friday, 9 AM to 5 PM.\nlocation: We're at 123 Main Street.",
                            lines=4
                        )

                register_btn = gr.Button("Register Skill", variant="primary")
                register_status = gr.Markdown()

                register_btn.click(
                    register_new_skill,
                    [skill_name, skill_desc, skill_system, skill_cached, skill_filler],
                    register_status
                )

                gr.Markdown("### Registered Skills")
                skills_list = gr.Markdown()
                refresh_skills_btn = gr.Button("Refresh Skills List")
                refresh_skills_btn.click(list_skills, [], skills_list)

            # ============ TAB 5: Latency Masking ============
            with gr.TabItem("🎭 Latency Masking"):
                gr.Markdown("""
                ### Latency Masking Configuration

                Configure the filler sounds and phrases used to mask LLM latency.
                These make slow models feel more natural and conversational.
                """)

                gr.Markdown("""
                **Current Filler Sounds:**
                - Hmm... Mmm... Umm... Ah... Well...

                **Current Thinking Phrases:**
                - Let me think about that...
                - That's a good question...
                - Let me check...
                - One moment...

                **Skill-Specific Fillers:**
                - Technical: "Let me look that up...", "Checking the docs..."
                - Customer Service: "I understand.", "Let me help with that..."
                - Scheduling: "Let me check the calendar...", "One moment..."
                - Sales: "Great question!", "Let me find the best option..."
                """)

            # ============ TAB 6: Stats ============
            with gr.TabItem("📊 Stats"):
                gr.Markdown("### Performance Statistics")

                stats_display = gr.Code(language="json", label="Current Stats")
                refresh_stats_btn = gr.Button("Refresh Stats")
                refresh_stats_btn.click(get_stats, [], stats_display)

            # ============ TAB 7: Voice Integration ============
            with gr.TabItem("🎙️ Voice Integration"):
                gr.Markdown("""
                ### Voice Platform Integration

                Connect the Skill Command Center to your voice platform.

                **Supported Platforms:**
                - LiveKit
                - Vapi
                - Twilio
                - Custom WebSocket

                **Integration Code:**
                ```python
                from skill_command_center import SkillCommandCenter

                center = SkillCommandCenter()

                # In your voice handler:
                async def on_user_speech(text: str):
                    async for chunk in center.process_query(text, use_latency_masking=True):
                        await tts.speak(chunk)
                ```

                **For LiveKit specifically:**
                ```python
                from livekit.agents import llm

                class HybridLLM(llm.LLM):
                    def __init__(self):
                        self.center = SkillCommandCenter()

                    async def chat(self, messages):
                        prompt = messages[-1].content
                        async for chunk in self.center.process_query(prompt):
                            yield llm.ChatChunk(content=chunk)
                ```
                """)

    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(share=True)  # share=True creates a public URL
