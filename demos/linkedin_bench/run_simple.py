#!/usr/bin/env python3
"""
Run a single LinkedIn task using a direct browser (no pool required).
Bypasses the browser pool requirement for testing on Hobbyist plan.

Usage:
    uv run python run_simple.py
    uv run python run_simple.py --task commenters_stripe_ai_post
    uv run python run_simple.py --task reactors_datadog_observability
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from linkedin_bench.kernel_runner import (
    ensure_browser_ready,
    run_claude_code,
    write_skill_file,
)
from linkedin_bench.skill_template import get_skill_content
from linkedin_bench.tasks import TASKS, get_task_by_id
from linkedin_bench.verifier import (
    calculate_reward,
    count_agent_steps,
    count_sleep_commands,
    verify_with_llm,
)


async def main():
    parser = argparse.ArgumentParser(description="Run a single LinkedIn task (no pool needed)")
    parser.add_argument("--task", type=str, default="commenters_stripe_ai_post", help="Task ID to run")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per task in seconds")
    parser.add_argument("--list", action="store_true", help="List available tasks")
    args = parser.parse_args()

    if args.list:
        print("Available tasks:")
        for task in TASKS:
            print(f"  {task.id}: {task.prompt}")
        return

    # Check env
    for var in ["KERNEL_API_KEY", "ANTHROPIC_API_KEY"]:
        if not os.environ.get(var):
            print(f"ERROR: {var} not set")
            sys.exit(1)

    task = get_task_by_id(args.task)
    skill = get_skill_content()

    print(f"\n{'=' * 60}")
    print(f"Task: {task.id}")
    print(f"Prompt: {task.prompt}")
    print(f"Expected: {task.expected}")
    print(f"Timeout: {args.timeout}s")
    print(f"{'=' * 60}\n")

    import httpx
    from kernel import AsyncKernel

    client = AsyncKernel(
        api_key=os.environ["KERNEL_API_KEY"],
        timeout=httpx.Timeout(300.0),
    )

    # Create a browser directly with the linkedin profile
    print("Creating browser with linkedin profile...")
    browser = await client.browsers.create(
        profile={"name": "linkedin"},
        stealth=True,
        timeout_seconds=max(args.timeout + 120, 600),
    )
    session_id = browser.session_id
    print(f"  Session: {session_id}")
    print(f"  Live view: {browser.browser_live_view_url}")

    try:
        # Ensure browser is ready
        print("Ensuring browser is ready...")
        await ensure_browser_ready(client, session_id)

        # Write skill file
        print("Writing skill file...")
        await write_skill_file(client, session_id, skill)

        # Run Claude Code
        print(f"Running Claude Code (timeout: {args.timeout}s)...")
        result = await run_claude_code(
            client,
            session_id,
            task.prompt,
            os.environ["ANTHROPIC_API_KEY"],
            timeout=args.timeout,
        )

        print(f"\n{'=' * 60}")
        print(f"Result: exit_code={result.exit_code}, elapsed={result.elapsed_seconds:.1f}s")
        print(f"{'=' * 60}")

        # Show last part of output
        print("\nAgent output (last 2000 chars):")
        print("-" * 40)
        print(result.output[-2000:])
        print("-" * 40)

        # Verify
        print("\nVerifying with LLM judge...")
        verification = await verify_with_llm(task, result.output, os.environ["ANTHROPIC_API_KEY"])

        num_steps = count_agent_steps(result.output)
        num_sleeps, total_sleep_ms = count_sleep_commands(result.output)
        reward = calculate_reward(
            correctness_score=verification.raw_score,
            elapsed_seconds=result.elapsed_seconds,
            num_agent_steps=num_steps,
            num_sleep_commands=num_sleeps,
            total_sleep_ms=total_sleep_ms,
            max_time=float(args.timeout),
        )

        print(f"\nVerification:")
        print(f"  Correct: {verification.correct}")
        print(f"  Extracted: {verification.extracted_answer}")
        print(f"  Reason: {verification.reason}")
        if num_sleeps > 0:
            print(f"  Sleep commands: {num_sleeps} ({total_sleep_ms}ms total)")
        print(f"  Steps: {num_steps}")
        print(f"  Reward: {reward:.3f}")

    finally:
        # Delete the browser
        print(f"\nCleaning up browser {session_id}...")
        try:
            import httpx as hx
            r = hx.delete(
                f"https://api.onkernel.com/browsers/{session_id}",
                headers={"Authorization": f"Bearer {os.environ['KERNEL_API_KEY']}"},
            )
            print(f"  Deleted ({r.status_code})")
        except Exception as e:
            print(f"  Warning: cleanup failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
