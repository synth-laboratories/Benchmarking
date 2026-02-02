#!/usr/bin/env python3
"""
Run evaluation on LinkedIn browser automation tasks.

This script runs Claude Code on LinkedIn tasks using Kernel cloud browsers
and evaluates the results. Use this to test the skill before running GEPA.

Usage:
    # Run a single task
    uv run python run_eval.py --task commenters_stripe_ai_post

    # Run all tasks
    uv run python run_eval.py --all

    # Run specific seeds
    uv run python run_eval.py --seeds 0 1 2
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Force unbuffered output for real-time streaming visibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from linkedin_bench.kernel_runner import (
    DEFAULT_POOL_NAME,
    get_or_create_pool,
    run_task_in_kernel,
)
from linkedin_bench.skill_template import get_skill_content
from linkedin_bench.tasks import TASKS, get_task_by_id, get_task_by_seed
from linkedin_bench.verifier import (
    calculate_reward,
    count_agent_steps,
    count_sleep_commands,
    verify_with_llm,
)


async def run_single_task(task_id: str, timeout: int | None = None, skill_content: str | None = None):
    """Run a single task and print results."""
    task = get_task_by_id(task_id)
    skill = skill_content or get_skill_content()
    
    # Use task timeout if not explicitly provided
    effective_timeout = timeout if timeout is not None else task.timeout

    print(f"\n{'=' * 60}")
    print(f"Task: {task.id}")
    print(f"Prompt: {task.prompt}")
    print(f"Expected: {task.expected}")
    print(f"Timeout: {effective_timeout}s")
    print(f"{'=' * 60}\n")

    result = await run_task_in_kernel(
        task_prompt=task.prompt,
        skill_content=skill,
        timeout=effective_timeout,
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
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    verification = await verify_with_llm(task, result.output, anthropic_key)

    num_steps = count_agent_steps(result.output)
    num_sleeps, total_sleep_ms = count_sleep_commands(result.output)
    reward = calculate_reward(
        correctness_score=verification.raw_score,
        elapsed_seconds=result.elapsed_seconds,
        num_agent_steps=num_steps,
        num_sleep_commands=num_sleeps,
        total_sleep_ms=total_sleep_ms,
        max_time=float(effective_timeout),
    )

    print(f"\nVerification:")
    print(f"  Correct: {verification.correct}")
    print(f"  Extracted: {verification.extracted_answer}")
    print(f"  Reason: {verification.reason}")
    if num_sleeps > 0:
        print(f"  Sleep commands: {num_sleeps} ({total_sleep_ms}ms total) - PENALIZED")
    print(f"  Steps: {num_steps}")
    print(f"  Reward: {reward:.3f}")

    return {
        "task_id": task.id,
        "correct": verification.correct,
        "reward": reward,
        "elapsed": result.elapsed_seconds,
        "steps": num_steps,
    }


async def run_all_tasks(timeout: int = 120):
    """Run all tasks and summarize results."""
    results = []

    for task in TASKS:
        try:
            result = await run_single_task(task.id, timeout)
            results.append(result)
        except Exception as e:
            print(f"Task {task.id} failed: {e}")
            results.append({
                "task_id": task.id,
                "correct": False,
                "reward": 0.0,
                "elapsed": 0,
                "steps": 0,
                "error": str(e),
            })

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    total_reward = 0
    correct_count = 0

    for r in results:
        status = "✓" if r.get("correct") else "✗"
        print(f"  {status} {r['task_id']}: reward={r['reward']:.3f}, elapsed={r['elapsed']:.1f}s")
        total_reward += r["reward"]
        if r.get("correct"):
            correct_count += 1

    print(f"\nTotal: {correct_count}/{len(results)} correct")
    print(f"Mean reward: {total_reward / len(results):.3f}")

    return results


async def run_seeds(seeds: list[int], timeout: int = 120):
    """Run specific seeds."""
    results = []

    for seed in seeds:
        task = get_task_by_seed(seed)
        try:
            result = await run_single_task(task.id, timeout)
            results.append(result)
        except Exception as e:
            print(f"Seed {seed} (task {task.id}) failed: {e}")
            results.append({
                "task_id": task.id,
                "correct": False,
                "reward": 0.0,
                "error": str(e),
            })

    return results


async def ensure_pool_exists():
    """Make sure the browser pool exists before running (optional - falls back to direct browsers)."""
    from kernel import AsyncKernel

    api_key = os.environ.get("KERNEL_API_KEY")
    if not api_key:
        print("ERROR: KERNEL_API_KEY not set")
        sys.exit(1)

    client = AsyncKernel(api_key=api_key)

    print(f"Checking browser pool '{DEFAULT_POOL_NAME}'...")
    try:
        await get_or_create_pool(client, DEFAULT_POOL_NAME, "linkedin", 10)
        print("Pool ready.")
    except Exception as e:
        print(f"Pool not available ({e}), will use direct browser creation instead.")


def main():
    parser = argparse.ArgumentParser(description="Run LinkedIn browser automation evaluation")
    parser.add_argument("--task", type=str, help="Run a specific task by ID")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--seeds", type=int, nargs="+", help="Run specific seeds")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per task (default: 120)")
    parser.add_argument("--list", action="store_true", help="List available tasks")

    args = parser.parse_args()

    if args.list:
        print("Available tasks:")
        for task in TASKS:
            print(f"  {task.id}: {task.prompt}")
        return

    # Check environment
    required_vars = ["KERNEL_API_KEY", "ANTHROPIC_API_KEY"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("Make sure .env file exists or export the variables.")
        sys.exit(1)

    async def run():
        await ensure_pool_exists()

        if args.task:
            await run_single_task(args.task, args.timeout)
        elif args.all:
            await run_all_tasks(args.timeout)
        elif args.seeds:
            await run_seeds(args.seeds, args.timeout)
        else:
            parser.print_help()

    asyncio.run(run())


if __name__ == "__main__":
    main()
