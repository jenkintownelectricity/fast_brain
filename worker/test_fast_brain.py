#!/usr/bin/env python3
"""
Fast Brain Integration Test

Tests the Fast Brain client and measures latency.

Usage:
    # Set environment variable first
    export FAST_BRAIN_URL=https://your-url.modal.run

    # Run test
    python worker/test_fast_brain.py
"""

import asyncio
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_brain.client import FastBrainClient


async def test_health_check():
    """Test health check endpoint"""
    print("\n=== Health Check ===")

    url = os.getenv("FAST_BRAIN_URL")
    if not url:
        print("ERROR: FAST_BRAIN_URL not set")
        return False

    client = FastBrainClient(url)

    try:
        health = await client.health_check()
        print(f"Status: {health.get('status')}")
        print(f"Model loaded: {health.get('model_loaded')}")
        print(f"Skills available: {health.get('skills_available', [])}")
        return health.get("status") == "healthy"
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    finally:
        await client.close()


async def test_non_streaming():
    """Test non-streaming chat"""
    print("\n=== Non-Streaming Test ===")

    url = os.getenv("FAST_BRAIN_URL")
    client = FastBrainClient(url)

    try:
        start = time.perf_counter()
        response = await client.chat(
            "What is the capital of France? Answer in one sentence.",
            max_tokens=50,
        )
        elapsed = (time.perf_counter() - start) * 1000

        print(f"Response: {response}")
        print(f"Total time: {elapsed:.1f}ms")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    finally:
        await client.close()


async def test_streaming():
    """Test streaming chat with latency measurement"""
    print("\n=== Streaming Test ===")

    url = os.getenv("FAST_BRAIN_URL")
    client = FastBrainClient(url)

    try:
        start = time.perf_counter()
        first_token_time = None
        full_response = ""
        token_count = 0

        print("Response: ", end="", flush=True)

        async for chunk in client.stream_chat(
            "Tell me a short joke about programming.",
            max_tokens=100,
        ):
            if chunk.text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
                token_count += 1

            if chunk.is_done:
                print()
                print(f"\n--- Metrics ---")
                ttfb = chunk.ttfb_ms or ((first_token_time - start) * 1000 if first_token_time else 0)
                print(f"TTFB: {ttfb:.1f}ms (target: <50ms)")
                if chunk.tokens_per_sec:
                    print(f"Throughput: {chunk.tokens_per_sec:.1f} tok/s (target: >500)")
                if chunk.total_tokens:
                    print(f"Total tokens: {chunk.total_tokens}")

                # Check targets
                if ttfb < 50:
                    print("TTFB target: PASS")
                else:
                    print(f"TTFB target: FAIL ({ttfb:.1f}ms > 50ms)")

        return True
    except Exception as e:
        print(f"\nERROR: {e}")
        return False
    finally:
        await client.close()


async def test_conversation():
    """Test multi-turn conversation"""
    print("\n=== Conversation Test ===")

    url = os.getenv("FAST_BRAIN_URL")
    client = FastBrainClient(url)

    history = []

    try:
        # Turn 1
        print("User: Hi, I'm looking to schedule an appointment.")
        history.append({"role": "user", "content": "Hi, I'm looking to schedule an appointment."})

        response1 = ""
        async for chunk in client.stream_chat(
            "Hi, I'm looking to schedule an appointment.",
            system_prompt="You are a helpful receptionist at a dental office.",
            max_tokens=100,
        ):
            if chunk.text:
                response1 += chunk.text
        print(f"Assistant: {response1}")
        history.append({"role": "assistant", "content": response1})

        # Turn 2
        print("\nUser: How about next Tuesday at 2pm?")
        response2 = ""
        async for chunk in client.stream_chat(
            "How about next Tuesday at 2pm?",
            system_prompt="You are a helpful receptionist at a dental office.",
            conversation_history=history,
            max_tokens=100,
        ):
            if chunk.text:
                response2 += chunk.text
        print(f"Assistant: {response2}")

        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    finally:
        await client.close()


async def main():
    """Run all tests"""
    print("=" * 50)
    print("Fast Brain Integration Tests")
    print("=" * 50)

    url = os.getenv("FAST_BRAIN_URL")
    if not url:
        print("\nERROR: FAST_BRAIN_URL environment variable not set")
        print("Deploy Fast Brain first:")
        print("  modal deploy fast_brain/deploy.py")
        print("Then set the URL:")
        print("  export FAST_BRAIN_URL=https://your-url.modal.run")
        return

    print(f"Testing: {url}")

    results = []

    # Run tests
    results.append(("Health Check", await test_health_check()))
    results.append(("Non-Streaming", await test_non_streaming()))
    results.append(("Streaming", await test_streaming()))
    results.append(("Conversation", await test_conversation()))

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} passed")


if __name__ == "__main__":
    asyncio.run(main())
