#!/usr/bin/env python3
"""Run the Banking77 GEPA demo end-to-end via managed LocalAPI deploy.

Unlike run_demo.py, this script deploys the Banking77 LocalAPI to the managed
cloud backend (Modal), so no local server, tunnel, or cloudflared binary is needed.

Usage:
    uv run python demos/gepa_banking77/run_demo_async.py
"""
import asyncio
import json
import os
import shutil
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
from synth_ai.sdk.localapi.deploy import deploy_localapi, LocalAPIDeployResult
from synth_ai.sdk.harbor import HarborLimits
from synth_ai.sdk.optimization.internal.prompt_learning import PromptLearningJob
from synth_ai.sdk.optimization.internal.learning.prompt_learning_client import PromptLearningClient
from synth_ai.sdk.eval.job import EvalJob, EvalJobConfig, EvalResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYNTH_API_BASE = os.environ.get("SYNTH_BACKEND_URL", "https://api-dev.usesynth.ai")
API_KEY = os.environ.get("SYNTH_API_KEY", "")

BASELINE_SYSTEM_PROMPT = (
    "You are an expert banking assistant that classifies customer queries into "
    "banking intents. Given a customer message, respond with exactly one intent "
    "label from the provided list using the `banking77_classify` tool."
)
USER_PROMPT = (
    "Customer Query: {query}\n\n"
    "Available Intents:\n{available_intents}\n\n"
    "Classify this query into one of the above banking intents using the tool call."
)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs}s"


def _stage_localapi_context(target_dir: Path) -> None:
    demo_dir = Path(__file__).resolve().parent
    repo_root = demo_dir.parents[2]

    shutil.copy2(demo_dir / "localapi_banking77.py", target_dir / "localapi_banking77.py")
    shutil.copy2(demo_dir / "Dockerfile.banking77-localapi", target_dir / "Dockerfile.banking77-localapi")

    synth_ai_src = repo_root / "synth-ai" / "synth_ai"
    synth_ai_core_assets = repo_root / "synth-ai" / "synth_ai_core" / "assets"

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    synth_ai_dst = target_dir / "synth_ai"
    synth_ai_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(synth_ai_src / "__init__.py", synth_ai_dst / "__init__.py")
    if (synth_ai_src / "__main__.py").exists():
        shutil.copy2(synth_ai_src / "__main__.py", synth_ai_dst / "__main__.py")

    shutil.copytree(synth_ai_src / "data", synth_ai_dst / "data", ignore=ignore)
    (synth_ai_dst / "sdk").mkdir(parents=True, exist_ok=True)
    shutil.copy2(synth_ai_src / "sdk" / "__init__.py", synth_ai_dst / "sdk" / "__init__.py")
    shutil.copytree(
        synth_ai_src / "sdk" / "localapi",
        synth_ai_dst / "sdk" / "localapi",
        ignore=ignore,
    )

    (target_dir / "synth_ai_core").mkdir(parents=True, exist_ok=True)
    shutil.copytree(synth_ai_core_assets, target_dir / "synth_ai_core" / "assets")


# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

def deploy_banking77() -> LocalAPIDeployResult:
    """Deploy the Banking77 LocalAPI to the managed cloud and wait until ready."""
    print("Deploying Banking77 LocalAPI to managed cloud...", flush=True)
    deployment_name = f"banking77-gepa-demo-{int(time.time())}"
    with tempfile.TemporaryDirectory(prefix="banking77_localapi_") as tmp_dir:
        context_dir = Path(tmp_dir)
        _stage_localapi_context(context_dir)
        result = deploy_localapi(
            name=deployment_name,
            dockerfile_path="Dockerfile.banking77-localapi",
            context_dir=str(context_dir),
            entrypoint="uvicorn localapi_banking77:app --host 0.0.0.0 --port 8000",
            entrypoint_mode="command",
            description="Banking77 intent classification LocalAPI for GEPA demo",
            env_vars={"HF_HOME": "/tmp/hf", "BANKING77_LOG_LEVEL": "info"},
            limits=HarborLimits(timeout_s=600, cpu_cores=2, memory_mb=4096),
            backend_url=SYNTH_API_BASE,
            api_key=API_KEY,
            wait_for_ready=True,
            build_timeout_s=600.0,
            provider="cloud",
            port=8000,
        )
    print(f"LocalAPI deployment ready: {result.deployment_id} (status={result.status})")
    print(f"  task_app_url: {result.task_app_url}")
    return result


# ---------------------------------------------------------------------------
# GEPA config
# ---------------------------------------------------------------------------

def create_gepa_config(task_app_url: str) -> dict:
    """Build the GEPA prompt-learning config body."""
    return {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_id": "banking77",
            "task_app_url": task_app_url,
            "initial_prompt": {
                "id": "banking77_pattern",
                "name": "Banking77 Classification",
                "messages": [
                    {"role": "system", "order": 0, "pattern": BASELINE_SYSTEM_PROMPT},
                    {"role": "user", "order": 1, "pattern": USER_PROMPT},
                ],
                "wildcards": {"query": "REQUIRED", "available_intents": "OPTIONAL"},
            },
            "policy": {
                "model": "gpt-4.1-nano",
                "provider": "openai",
                "inference_mode": "synth_hosted",
                "temperature": 0.0,
                "max_completion_tokens": 256,
            },
            "env_config": {"split": "train"},
            "gepa": {
                "env_name": "banking77",
                "evaluation": {
                    "seeds": list(range(30)),
                    "validation_seeds": list(range(50, 80)),
                },
                "rollout": {"budget": 200, "max_concurrent": 20, "minibatch_size": 5},
                "mutation": {"rate": 0.3},
                "population": {
                    "initial_size": 3,
                    "num_generations": 4,
                    "children_per_generation": 2,
                },
                "archive": {"pareto_set_size": 20},
                "token": {"counting_model": "gpt-4"},
            },
        },
    }


# ---------------------------------------------------------------------------
# Event streaming callback
# ---------------------------------------------------------------------------

def make_on_status_update(job_id: str) -> Any:
    """Return an on_status_update callback that streams GEPA events."""
    last_event_seq = 0
    total_pareto_seeds = 20  # matches archive.pareto_set_size

    def on_status_update(status_data: dict[str, Any]) -> None:
        nonlocal last_event_seq

        current_status = status_data.get("status", "")
        if current_status.lower() in ("failed", "failure", "error"):
            error_msg = (
                status_data.get("error")
                or status_data.get("error_message")
                or status_data.get("failure_reason")
                or status_data.get("message")
                or "no error message in status response"
            )
            print(f"\n{'='*60}", flush=True)
            print(f"JOB FAILED: {error_msg}", flush=True)
            for k in ("error", "error_message", "error_details", "failure_reason", "traceback", "message"):
                v = status_data.get(k)
                if v:
                    print(f"  {k}: {v}", flush=True)
            print(f"{'='*60}", flush=True)
        elif current_status.lower() in ("cancelled", "canceled"):
            print("\nJOB CANCELLED", flush=True)

        try:
            response = httpx.get(
                f"{SYNTH_API_BASE}/api/prompt-learning/online/jobs/{job_id}/events",
                params={"since_seq": last_event_seq, "limit": 100},
                headers={"X-API-Key": API_KEY},
                timeout=30.0,
            )
            events = response.json().get("events", []) if response.status_code == 200 else []

            for event in events:
                event_type = event.get("type", "")
                event_seq = event.get("seq", 0)
                if event_seq > last_event_seq:
                    last_event_seq = event_seq

                data = event.get("data", {})
                message = event.get("message", "")

                def format_pareto_growth(growth: Any) -> str:
                    if not isinstance(growth, dict):
                        return ""
                    parts = []
                    for label, key in [("all", "all_time"), ("last1", "last_1"), ("last5", "last_5"), ("last20", "last_20")]:
                        val = growth.get(key)
                        if val is not None:
                            parts.append(f"{label}={val:.2f}")
                    return " ".join(parts)

                def format_seeds_outstanding(total_solved: Any) -> str:
                    if total_pareto_seeds <= 0 or total_solved is None:
                        return ""
                    try:
                        outstanding = max(0, total_pareto_seeds - int(total_solved))
                    except (TypeError, ValueError):
                        return ""
                    return f"outstanding={outstanding}/{total_pareto_seeds}"

                if event_type == "learning.policy.gepa.job.progress":
                    print(f"\n  {message}", flush=True)
                elif event_type == "learning.policy.gepa.rollout.started":
                    print(f"  {message}", flush=True)
                elif event_type == "learning.policy.gepa.candidate.evaluated":
                    if "evaluated" in message or "completed" in message:
                        version_id = data.get("version_id") or data.get("candidate_id", "")
                        accuracy = None
                        for key in ["accuracy", "acc", "reward", "train_accuracy", "best_score"]:
                            val = data.get(key)
                            if val is not None:
                                accuracy = val
                                break
                        if accuracy is None and isinstance(data.get("score"), dict):
                            for key in ["mean_reward", "reward", "accuracy"]:
                                val = data["score"].get(key)
                                if val is not None:
                                    accuracy = val
                                    break
                        accepted = data.get("accepted", True)
                        if accuracy is None and "acc=" in message:
                            try:
                                accuracy = float(message.split("acc=")[1].split()[0])
                            except (ValueError, IndexError):
                                pass
                        if "accepted=True" in message:
                            accepted = True
                        elif "accepted=False" in message:
                            accepted = False
                        if accuracy is not None:
                            status_char = "\u2713" if accepted else "\u2717"
                            print(f"  {status_char} Candidate {version_id}: mean reward = {accuracy:.2f}")
                        program_candidate = data.get("program_candidate", {}) if isinstance(data, dict) else {}
                        if isinstance(program_candidate, dict):
                            prompt_summary = program_candidate.get("prompt_summary")
                            if isinstance(prompt_summary, str) and prompt_summary.strip():
                                print(f"    prompt_summary: {prompt_summary.strip()[:200]}")
                            objectives = program_candidate.get("objectives")
                            if isinstance(objectives, dict):
                                print(f"    objectives: {objectives}")
                elif event_type == "learning.policy.gepa.phase.started":
                    print(f"\n  {message}", flush=True)
                elif event_type == "learning.policy.gepa.generation.started":
                    print(f"\n  {message}", flush=True)
                elif event_type in ("learning.policy.gepa.frontier.updated", "learning.policy.gepa.archive.updated"):
                    frontier_density = data.get("frontier_density")
                    frontier_size = data.get("frontier_size") or data.get("archive_size")
                    total_seeds_solved = data.get("total_seeds_solved")
                    pareto_growth = format_pareto_growth(data.get("pareto_growth"))
                    seeds_outstanding = format_seeds_outstanding(total_seeds_solved)
                    best_reward = data.get("best_reward")
                    details = []
                    if best_reward is not None:
                        details.append(f"best={best_reward:.3f}")
                    if frontier_density is not None:
                        details.append(f"density={frontier_density:.3f}")
                    if frontier_size is not None:
                        details.append(f"frontier={frontier_size}")
                    if total_seeds_solved is not None:
                        details.append(f"total_seeds={total_seeds_solved}")
                    if seeds_outstanding:
                        details.append(seeds_outstanding)
                    if pareto_growth:
                        details.append(f"growth[{pareto_growth}]")
                    label = "Frontier improved" if "archive" in event_type else "GEPA progress"
                    if details:
                        print(f"\n  {label}: {' | '.join(details)}")
                    else:
                        print(f"\n  {label} (raw): {data}")
                elif event_type in ("learning.policy.gepa.job.started", "learning.policy.gepa.job.queued"):
                    print(f"  {message}", flush=True)
                elif event_type.startswith("learning.policy.gepa."):
                    if "concurrency" not in event_type and message:
                        print(f"  [{event_type.split('.')[-1]}] {message}", flush=True)

        except Exception as exc:
            print(f"  [event-fetch warning] {type(exc).__name__}: {exc}", flush=True)

    return on_status_update


# ---------------------------------------------------------------------------
# Eval helper
# ---------------------------------------------------------------------------

async def run_eval_job(
    task_app_url: str,
    seeds: list[int],
    mode: str,
) -> EvalResult:
    """Run an eval job against a managed LocalAPI."""
    config = EvalJobConfig(
        task_app_url=task_app_url,
        backend_url=SYNTH_API_BASE,
        api_key=API_KEY,
        env_name="banking77",
        seeds=seeds,
        policy_config={
            "model": "gpt-4.1-nano",
            "provider": "openai",
            "inference_mode": "synth_hosted",
            "api_key": API_KEY,
        },
        env_config={"split": "test"},
        concurrency=10,
    )
    job = EvalJob(config)
    job_id = await asyncio.to_thread(job.submit)
    print(f"  {mode} eval job: {job_id}")

    start_time = time.time()
    last_event_seq = 0
    completed_seeds: set[int] = set()
    total_seeds = len(seeds)
    timeout = 600.0
    interval = 3.0

    while True:
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            print(f"[{int(elapsed // 60):02d}:{int(elapsed % 60):02d}] timeout", flush=True)
            break

        try:
            status_data = job.get_status()
            status = status_data.get("status", "pending")
        except Exception as e:
            print(f"  [error getting status: {e}]", flush=True)
            await asyncio.sleep(interval)
            continue

        try:
            response = httpx.get(
                f"{SYNTH_API_BASE}/api/eval/jobs/{job_id}/events",
                params={"since_seq": last_event_seq, "limit": 100},
                headers={"X-API-Key": API_KEY},
                timeout=10.0,
            )
            if response.status_code == 200:
                events = response.json()
                if isinstance(events, list):
                    for event in events:
                        event_seq = event.get("seq", 0)
                        if event_seq > last_event_seq:
                            last_event_seq = event_seq
                        if event.get("type") in ("eval.policy.seed.completed", "eval.policy.seed.failed"):
                            seed = event.get("data", {}).get("seed")
                            if seed is not None:
                                completed_seeds.add(seed)
        except Exception:
            pass

        mins, secs = divmod(int(elapsed), 60)
        completed_count = len(completed_seeds)

        if status in ("completed", "failed", "cancelled"):
            try:
                results_data = status_data.get("results", {})
                mean_reward = results_data.get("mean_reward")
                if mean_reward is not None:
                    print(f"[{mins:02d}:{secs:02d}] {status} | mean_reward: {mean_reward:.2f}", flush=True)
                else:
                    print(f"[{mins:02d}:{secs:02d}] {status}", flush=True)
            except Exception:
                print(f"[{mins:02d}:{secs:02d}] {status}", flush=True)
            break
        else:
            print(f"[{mins:02d}:{secs:02d}] {status} | {completed_count}/{total_seeds} completed", flush=True)

        await asyncio.sleep(interval)

    return job.poll_until_complete(timeout=10.0, interval=1.0, progress=False)


# ---------------------------------------------------------------------------
# Prompt extraction (same logic as run_demo.py)
# ---------------------------------------------------------------------------

def extract_system_prompt(prompt_results) -> str:
    """Extract system prompt from prompt results, handling multiple formats."""
    if prompt_results.top_prompts:
        top = prompt_results.top_prompts[0]
        if "full_text" in top and top["full_text"]:
            return top["full_text"]
        if "template" in top and top["template"]:
            template = top["template"]
            if "sections" in template:
                for section in template["sections"]:
                    if section.get("role") == "system":
                        return section.get("content", "")
            if "full_text" in template:
                return template["full_text"]
        if "system_prompt" in top:
            return top["system_prompt"]
        if "prompt" in top:
            return top["prompt"]

    if prompt_results.best_prompt:
        if isinstance(prompt_results.best_prompt, str):
            return prompt_results.best_prompt
        elif isinstance(prompt_results.best_prompt, dict):
            if "full_text" in prompt_results.best_prompt:
                return prompt_results.best_prompt["full_text"]
            if "content" in prompt_results.best_prompt:
                return prompt_results.best_prompt["content"]
            if "messages" in prompt_results.best_prompt:
                messages = prompt_results.best_prompt["messages"]
                if messages and isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict) and msg.get("role") == "system":
                            return msg.get("content") or msg.get("pattern", "")
                    if messages[0]:
                        return messages[0].get("content") or messages[0].get("pattern", "")

    if prompt_results.top_prompts:
        return f"[Could not extract prompt. Keys available: {list(prompt_results.top_prompts[0].keys())}]"
    return "[No prompts found in results]"


def extract_mean_reward(result: EvalResult) -> float | None:
    """Extract mean reward from an eval result, trying multiple fields."""
    mean_reward = getattr(result, "mean_reward", None)
    if mean_reward is None:
        mean_reward = getattr(result, "mean_score", None)
    if mean_reward is None:
        summary = result.raw.get("summary", {})
        mean_reward = summary.get("mean_reward")
    if mean_reward is None and result.seed_results:
        rewards = [
            r.get("outcome_reward") or r.get("reward_mean") or r.get("reward")
            for r in result.seed_results
            if isinstance(r, dict)
            and (r.get("outcome_reward") or r.get("reward_mean") or r.get("reward")) is not None
        ]
        if rewards:
            mean_reward = sum(rewards) / len(rewards)
    return mean_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    if not API_KEY:
        raise RuntimeError("SYNTH_API_KEY environment variable is required")

    os.environ["SYNTH_API_KEY"] = API_KEY

    timings: dict[str, float] = {}
    total_start = time.time()

    # ── 1. Deploy Banking77 LocalAPI to managed cloud ───────────────────
    deploy_start = time.time()
    deploy_result = await asyncio.to_thread(deploy_banking77)
    task_app_url = deploy_result.task_app_url
    timings["deploy"] = time.time() - deploy_start
    print(f"Deploy completed ({format_duration(timings['deploy'])})\n")

    # ── 2. Run GEPA optimization ────────────────────────────────────────
    config_body = create_gepa_config(task_app_url)
    print(f"Creating GEPA job (task_app_url={task_app_url})...")

    pl_job = PromptLearningJob.from_dict(
        config_dict=deepcopy(config_body),
        backend_url=SYNTH_API_BASE,
        skip_health_check=False,
    )

    job_id = await asyncio.to_thread(pl_job.submit)
    print(f"Job ID: {job_id}")

    on_status_update = make_on_status_update(job_id)

    optimization_start = time.time()
    try:
        gepa_result = pl_job.poll_until_complete(
            timeout=3600.0,
            interval=3.0,
            progress=False,
            on_status=on_status_update,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
        raise
    except Exception as e:
        print(f"\n\nERROR during poll_until_complete: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    timings["optimization"] = time.time() - optimization_start

    print(f"\nFINAL: {gepa_result.status.value} ({format_duration(timings['optimization'])})")

    if gepa_result.succeeded:
        best_reward = None
        if isinstance(gepa_result.raw, dict):
            best_reward = (
                gepa_result.raw.get("best_reward")
                or gepa_result.raw.get("best_avg_reward")
                or gepa_result.raw.get("best_train_reward")
            )
        if isinstance(best_reward, (int, float)):
            print(f"BEST REWARD: {float(best_reward):.1%}")
        else:
            print("BEST REWARD: N/A")
    elif gepa_result.failed:
        print(f"ERROR: {gepa_result.error}")
        if gepa_result.raw:
            print("\n--- Full error details from status ---")
            for key in ("error", "error_message", "error_details", "traceback", "failure_reason", "message"):
                if key in gepa_result.raw and gepa_result.raw[key]:
                    print(f"{key}: {gepa_result.raw[key]}")

        try:
            print("\n--- Fetching job events for error details ---")
            pl_client = PromptLearningClient(SYNTH_API_BASE, API_KEY)
            events = await pl_client.get_events(gepa_result.job_id, limit=100)
            error_events = [
                e for e in events
                if "error" in e.get("type", "").lower()
                or "fail" in e.get("type", "").lower()
                or e.get("data", {}).get("error")
            ]
            if error_events:
                for event in error_events[-3:]:
                    print(f"\n[{event.get('type')}] {event.get('message', '')}")
                    edata = event.get("data", {})
                    if edata.get("error"):
                        print(f"  error: {edata['error']}")
                    if edata.get("traceback"):
                        print(f"  traceback: {edata['traceback'][:500]}...")
            else:
                print("No error events found. Last events:")
                for event in events[-5:]:
                    print(f"  [{event.get('type')}] {event.get('message', '')[:100]}")
        except Exception as e:
            print(f"Could not fetch events: {e}")

    # ── 3. Evaluation ───────────────────────────────────────────────────
    eval_seeds = list(range(100, 120))

    if gepa_result.succeeded:
        print("GEPA Job Succeeded!\n")

        try:
            pl_client = PromptLearningClient(SYNTH_API_BASE, API_KEY)
            prompt_results = await pl_client.get_prompts(gepa_result.job_id)
            optimized_system = extract_system_prompt(prompt_results)

            if optimized_system.startswith("[Could not extract") or optimized_system.startswith("[No prompts"):
                print(f"Debug: top_prompts[0] = {prompt_results.top_prompts[0] if prompt_results.top_prompts else 'empty'}")
                print(f"Debug: best_prompt type = {type(prompt_results.best_prompt)}", flush=True)

                if prompt_results.optimized_candidates:
                    cand = prompt_results.optimized_candidates[0]
                    if isinstance(cand, dict):
                        for key in ["full_text", "prompt", "template", "content", "system_prompt"]:
                            if key in cand and cand[key]:
                                val = cand[key]
                                if isinstance(val, str) and len(val) > 20:
                                    optimized_system = val
                                    break
                                elif isinstance(val, dict):
                                    if "full_text" in val:
                                        optimized_system = val["full_text"]
                                        break
                                    elif "sections" in val:
                                        for sec in val["sections"]:
                                            if sec.get("role") == "system":
                                                optimized_system = sec.get("content", "")
                                                break

                if optimized_system.startswith("["):
                    print("\nWARNING: Could not extract optimized prompt. Using baseline for comparison.", flush=True)
                    optimized_system = BASELINE_SYSTEM_PROMPT

            best_train_reward = 0.0
            if isinstance(gepa_result.raw, dict):
                raw_best = gepa_result.raw.get("best_reward") or gepa_result.raw.get("best_avg_reward")
                if isinstance(raw_best, (int, float)):
                    best_train_reward = float(raw_best)

        except Exception as e:
            print(f"\nERROR extracting prompts: {e}", flush=True)
            import traceback
            traceback.print_exc()
            optimized_system = BASELINE_SYSTEM_PROMPT
            best_train_reward = 0.0
            if isinstance(gepa_result.raw, dict):
                raw_best = gepa_result.raw.get("best_reward") or gepa_result.raw.get("best_avg_reward")
                if isinstance(raw_best, (int, float)):
                    best_train_reward = float(raw_best)

        print("=" * 60)
        print("BASELINE SYSTEM PROMPT")
        print("=" * 60)
        print(BASELINE_SYSTEM_PROMPT)

        print("\n" + "=" * 60)
        print("OPTIMIZED SYSTEM PROMPT (from GEPA)")
        print("=" * 60)
        print(optimized_system[:800] + "..." if len(optimized_system) > 800 else optimized_system)

        print("\n" + "=" * 60)
        print("GEPA TRAINING RESULTS")
        print("=" * 60)
        print(f"Best Train Reward: {best_train_reward:.1%}" if best_train_reward else "Best Train Reward: N/A")

        print("\n" + "=" * 60)
        print(f"FORMAL EVAL JOBS (test split, seeds {eval_seeds[0]}-{eval_seeds[-1]})")
        print("=" * 60)

        # Both baseline and optimized eval use the same managed deployment
        print("\nRunning BASELINE eval job...")
        eval_start = time.time()
        baseline_result = await run_eval_job(
            task_app_url=task_app_url,
            seeds=eval_seeds,
            mode="baseline",
        )
        timings["baseline_eval"] = time.time() - eval_start

        if baseline_result.succeeded:
            br = extract_mean_reward(baseline_result)
            if br is not None:
                print(f"  Baseline eval reward: {br:.1%} ({format_duration(timings['baseline_eval'])})")
            else:
                print(f"  Baseline eval completed but no reward available ({format_duration(timings['baseline_eval'])})")
        else:
            print(f"  Baseline eval failed: {baseline_result.error}")

        print("\nRunning OPTIMIZED eval job...")
        eval_start = time.time()
        optimized_result = await run_eval_job(
            task_app_url=task_app_url,
            seeds=eval_seeds,
            mode="optimized",
        )
        timings["optimized_eval"] = time.time() - eval_start

        if optimized_result.succeeded:
            opr = extract_mean_reward(optimized_result)
            if opr is not None:
                print(f"  Optimized eval reward: {opr:.1%} ({format_duration(timings['optimized_eval'])})")
            else:
                print(f"  Optimized eval completed but no reward available ({format_duration(timings['optimized_eval'])})")
        else:
            print(f"  Optimized eval failed: {optimized_result.error}")

        if baseline_result.succeeded and optimized_result.succeeded:
            baseline_reward = extract_mean_reward(baseline_result)
            optimized_reward = extract_mean_reward(optimized_result)

            if baseline_reward is not None and optimized_reward is not None:
                print("\n" + "=" * 60)
                print("FINAL COMPARISON")
                print("=" * 60)
                print("Training:")
                print(f"  Best Train Reward: {best_train_reward:.1%}")

                print(f"\nEval (seeds {eval_seeds[0]}-{eval_seeds[-1]}, held-out):")
                print(f"  Baseline Reward:  {baseline_reward:.1%}")
                print(f"  Optimized Reward: {optimized_reward:.1%}")

                eval_lift = optimized_reward - baseline_reward
                print(f"  Lift:             {eval_lift:+.1%}")

                if eval_lift > 0:
                    print("\n>>> OPTIMIZATION GENERALIZES TO HELD-OUT DATA!")
                elif eval_lift == 0:
                    print("\n=== Same performance on held-out data")
                else:
                    print("\n<<< Baseline better on held-out (possible overfitting)")
            else:
                print("\n" + "=" * 60)
                print("FINAL COMPARISON")
                print("=" * 60)
                print("Eval jobs completed but rewards not available for comparison")
    else:
        print(f"Job did not succeed: {gepa_result.status.value}")

    # ── 4. Timing summary ──────────────────────────────────────────────
    timings["total"] = time.time() - total_start
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    if "deploy" in timings:
        print(f"  LocalAPI deploy:    {format_duration(timings['deploy'])}")
    if "optimization" in timings:
        print(f"  GEPA optimization:  {format_duration(timings['optimization'])}")
    if "baseline_eval" in timings:
        print(f"  Baseline eval:      {format_duration(timings['baseline_eval'])}")
    if "optimized_eval" in timings:
        print(f"  Optimized eval:     {format_duration(timings['optimized_eval'])}")
    print("  " + "\u2500" * 25)
    print(f"  Total:              {format_duration(timings['total'])}")

    print("\nDemo complete!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nDemo failed with error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
