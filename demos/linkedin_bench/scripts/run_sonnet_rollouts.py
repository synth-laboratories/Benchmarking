#!/usr/bin/env python3
"""
Run multiple Sonnet rollouts per LinkedIn task and record outputs.

This script:
- Runs Claude Code (Sonnet) for each task N times
- Captures raw output to data/sonnet45_runs/<task_id>/run_<n>.txt
- Extracts a normalized answer via an LLM for consensus voting
- Appends a JSONL record for each run to data/sonnet45_runs/rollouts.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from linkedin_bench.kernel_runner import run_task_in_kernel
from linkedin_bench.skill_template import get_skill_content
from linkedin_bench.tasks import TASKS, get_task_by_id

import anthropic


DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_EXTRACT_MODEL = "claude-sonnet-4-5-20250929"


def normalize_answer(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return ""

    # Try JSON parsing first
    try:
        data = json.loads(text)
        return _normalize_json(data)
    except Exception:
        pass

    # Split into lines, remove bullets, sort
    lines = [re.sub(r"^[\-*\d.\)\s]+", "", line).strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        lines = sorted(lines)
        joined = " | ".join(lines)
        return _normalize_text(joined)

    return _normalize_text(text)


def _normalize_json(data) -> str:
    if isinstance(data, dict):
        items = [f"{_normalize_text(str(k))}:{_normalize_json(v)}" for k, v in sorted(data.items())]
        return "{" + ",".join(items) + "}"
    if isinstance(data, list):
        items = sorted(_normalize_json(v) for v in data)
        return "[" + ",".join(items) + "]"
    return _normalize_text(str(data))


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9#,:;\-\.\s]", "", text)
    return text.strip()


async def extract_answer(task_prompt: str, agent_output: str, model: str, api_key: str) -> str | None:
    client = anthropic.AsyncAnthropic(api_key=api_key)
    prompt = f"""Extract the final answer from the agent output.

TASK:\n{task_prompt}\n\nOUTPUT:\n{agent_output[-8000:]}\n\nReturn ONLY a JSON object with this schema (no markdown):\n{{\n  \"answer\": \"...\"\n}}\n"""
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0].text.strip()
        match = re.search(r"\{[\s\S]*\}", content)
        payload = json.loads(match.group() if match else content)
        return payload.get("answer")
    except Exception as e:
        return None


def extract_answer_from_output(agent_output: str) -> str | None:
    matches = re.findall(r"^\s*ANSWER:\s*(.+?)\s*$", agent_output, flags=re.MULTILINE)
    if matches:
        return matches[-1].strip()
    return None


async def run_one(task, run_index: int, args, lock: asyncio.Lock, output_root: Path):
    task_id = task.id
    skill = get_skill_content()
    start = time.time()

    task_prompt = (
        f"{task.prompt}\n\n"
        "IMPORTANT: After completing the task, respond with a single line in the exact format:\n"
        "ANSWER: <your final answer>\n"
        "Do not add any extra text after the ANSWER line."
    )

    result = await run_task_in_kernel(
        task_prompt=task_prompt,
        skill_content=skill,
        timeout=args.timeout,
        claude_model=args.model,
    )

    elapsed = time.time() - start

    # Save raw output
    task_dir = output_root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    output_path = task_dir / f"run_{run_index}.txt"
    output_path.write_text(result.output)

    # Extract answer
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    extracted = extract_answer_from_output(result.output)
    if extracted is None and anthropic_key:
        extracted = await extract_answer(task_prompt, result.output, args.extract_model, anthropic_key)

    normalized = normalize_answer(extracted)

    record = {
        "task_id": task_id,
        "run_index": run_index,
        "model": args.model,
        "extract_model": args.extract_model,
        "exit_code": result.exit_code,
        "elapsed_seconds": result.elapsed_seconds,
        "output_path": str(output_path),
        "extracted_answer": extracted,
        "normalized_answer": normalized,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    async with lock:
        with (output_root / "rollouts.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[{task_id} run {run_index}] done in {elapsed:.1f}s (exit={result.exit_code})")


async def main():
    parser = argparse.ArgumentParser(description="Run Sonnet rollouts for all LinkedIn tasks")
    parser.add_argument("--runs-per-task", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--extract-model", type=str, default=DEFAULT_EXTRACT_MODEL)
    parser.add_argument("--task", action="append", help="Run a specific task ID (can repeat)")
    parser.add_argument("--output-dir", type=str, default="data/sonnet45_runs")
    args = parser.parse_args()

    if not os.environ.get("KERNEL_API_KEY"):
        raise SystemExit("KERNEL_API_KEY not set")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set (needed for answer extraction)")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.task:
        tasks = [get_task_by_id(tid) for tid in args.task]
    else:
        tasks = TASKS

    # pre-clear rollouts file
    rollouts_path = output_root / "rollouts.jsonl"
    if rollouts_path.exists():
        rollouts_path.unlink()

    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def run_wrapped(task, run_index):
        async with semaphore:
            await run_one(task, run_index, args, lock, output_root)

    jobs = []
    for task in tasks:
        for run_index in range(1, args.runs_per_task + 1):
            jobs.append(run_wrapped(task, run_index))

    print(f"Running {len(jobs)} rollouts ({len(tasks)} tasks x {args.runs_per_task} runs) with concurrency {args.concurrency}")
    await asyncio.gather(*jobs)


if __name__ == "__main__":
    asyncio.run(main())
