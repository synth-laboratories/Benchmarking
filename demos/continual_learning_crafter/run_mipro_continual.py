#!/usr/bin/env python3
"""
Run MIPRO continual learning on Crafter with progressive episode splits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demos" / "gepa_crafter_vlm"))

from crafter_logic import CRAFTER_ALLOWED_ACTIONS
from data_splits import CRAFTER_SPLITS, split_config, split_stats
from demo_crafter_react import create_crafter_vlm_local_api
from synth_ai.core.env import mint_demo_api_key
from synth_ai.core.urls import BACKEND_URL_BASE
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port

try:
    from synth_ai.sdk.task.server import run_server_background
except ImportError:  # pragma: no cover
    from synth_ai.sdk.task import run_server_background


def resolve_backend_url() -> str:
    for env_var in ("SYNTH_URL", "SYNTH_BACKEND_URL", "RUST_BACKEND_URL"):
        env_url = (os.environ.get(env_var) or "").strip()
        if env_url:
            return env_url.rstrip("/")
    return BACKEND_URL_BASE.rstrip("/")


def wait_for_health_check_sync(
    host: str, port: int, api_key: str, timeout: float = 60.0
) -> None:
    health_url = f"http://{host}:{port}/health"
    headers = {"X-API-Key": api_key} if api_key else {}
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(health_url, headers=headers, timeout=5.0)
            if response.status_code in (200, 400):
                return
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Health check failed: {health_url} not ready after {timeout}s")


def build_initial_prompt() -> Dict[str, Any]:
    allowed_actions = ", ".join(CRAFTER_ALLOWED_ACTIONS)
    system_prompt = (
        "You are an agent playing Crafter, a survival crafting game. "
        "Your goal is to survive and unlock achievements by exploring, crafting, and building. "
        "You can see the game state through images. Analyze each image carefully to understand "
        "your surroundings, inventory, health, and available resources. "
        "Use the crafter_interact tool to execute actions. "
        "Key mechanics: use 'do' only when adjacent to a resource (tree, stone, cow, plant); "
        "it does nothing on grass or water. "
        "Craft progression: wood -> table -> wood_pickaxe -> stone -> stone_pickaxe -> iron tools. "
        "Sleep when energy is low to restore and unlock wake_up. "
        f"Available actions: {allowed_actions}. "
        "Only use these action names and return 2-5 actions per decision."
    )
    return {
        "id": "crafter_continual",
        "name": "Crafter Continual (Episode Splits)",
        "messages": [{"role": "system", "order": 0, "pattern": system_prompt}],
        "wildcards": {},
    }


def build_mipro_continual_config(
    *,
    task_app_url: str,
    train_seeds: List[int],
    val_seeds: List[int],
    min_rollouts_before_proposal: int,
    system_id: Optional[str] = None,
    system_name: Optional[str] = None,
) -> Dict[str, Any]:
    policy_model = os.environ.get("CRAFTER_POLICY_MODEL", "gpt-4.1-nano")
    policy_provider = os.environ.get("CRAFTER_POLICY_PROVIDER", "openai")
    proposer_model = os.environ.get("CRAFTER_PROPOSER_MODEL", "gpt-4.1-mini")
    proposer_provider = os.environ.get("CRAFTER_PROPOSER_PROVIDER", "openai")

    mipro_section: Dict[str, Any] = {
        "mode": "online",
        "bootstrap_train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "online_pool": train_seeds,
        "online_proposer_mode": "inline",
        "online_proposer_min_rollouts": min_rollouts_before_proposal,
        "online_proposer_max_candidates": 50,
        "online_rollouts_per_candidate": 4,
        "ontology": {
            "enabled": True,
            "reads": True,
            "writes": True,
            "batch_proposer": {
                "enabled": True,
                "min_rollouts": 6,
                "batch_size": 8,
                "model": proposer_model,
                "provider": proposer_provider,
                "temperature": 0.7,
                "max_tokens": 1024,
            },
        },
        "proposer": {
            "mode": "instruction_only",
            "model": proposer_model,
            "provider": proposer_provider,
            "temperature": 0.7,
            "max_tokens": 4096,  # Needs to be large enough for full JSON output
        },
    }

    if system_id:
        mipro_section["system_id"] = system_id
    if system_name:
        mipro_section["system_name"] = system_name

    return {
        "prompt_learning": {
            "algorithm": "mipro",
            "task_app_id": "crafter_continual",
            "task_app_url": task_app_url,
            "initial_prompt": build_initial_prompt(),
            "policy": {
                "model": policy_model,
                "provider": policy_provider,
                "inference_mode": "synth_hosted",
                "temperature": 0.0,
                "max_completion_tokens": 512,
            },
            "mipro": mipro_section,
        },
    }


def create_job(backend_url: str, api_key: str, config_body: Dict[str, Any]) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = httpx.post(
        f"{backend_url}/api/prompt-learning/online/jobs",
        json={"algorithm": "mipro", "config_body": config_body},
        headers=headers,
        timeout=60.0,
    )
    if response.status_code != 200:
        print(f"Error response: {response.status_code}")
        print(f"Response body: {response.text}")
    response.raise_for_status()
    payload = response.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError(f"Missing job_id in response: {payload}")
    return str(job_id)


def get_job_detail(
    backend_url: str, api_key: str, job_id: str, *, include_metadata: bool = True
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = httpx.get(
        f"{backend_url}/api/prompt-learning/online/jobs/{job_id}",
        params={"include_events": False, "include_snapshot": False, "include_metadata": include_metadata},
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def get_system_state(backend_url: str, api_key: str, system_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = httpx.get(
        f"{backend_url}/api/prompt-learning/online/mipro/systems/{system_id}/state",
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def extract_candidate_text(state: Dict[str, Any], candidate_id: str | None) -> str | None:
    if not candidate_id:
        return None
    candidates = state.get("candidates", {}) if isinstance(state, dict) else {}
    if not isinstance(candidates, dict):
        return None
    candidate = candidates.get(candidate_id)
    if not isinstance(candidate, dict):
        return None

    stage_payloads = candidate.get("stage_payloads", {})
    if isinstance(stage_payloads, dict) and stage_payloads:
        for payload in stage_payloads.values():
            if not isinstance(payload, dict):
                continue
            instruction_text = payload.get("instruction_text")
            if isinstance(instruction_text, str) and instruction_text.strip():
                return instruction_text.strip()
            instruction_lines = payload.get("instruction_lines")
            if isinstance(instruction_lines, list) and instruction_lines:
                joined = "\n".join(str(line) for line in instruction_lines)
                if joined.strip():
                    return joined.strip()

    deltas = candidate.get("deltas")
    if isinstance(deltas, dict):
        for key in ("instruction_text", "text", "content"):
            value = deltas.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    baseline_messages = candidate.get("baseline_messages")
    if isinstance(baseline_messages, list):
        for msg in baseline_messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "system":
                content = msg.get("content") or msg.get("pattern")
                if isinstance(content, str) and content.strip():
                    return content.strip()

    return None


def run_rollout(
    *,
    task_app_url: str,
    env_key: str,
    seed: int,
    inference_url: str,
    model: str,
    rollout_id: str,
    split_name: str,
    split_env_config: Dict[str, int | str],
) -> float:
    payload = {
        "trace_correlation_id": rollout_id,
        "env": {"seed": seed, "config": split_env_config},
        "policy": {"config": {"model": model, "provider": "openai", "inference_url": inference_url}},
    }
    headers = {"X-API-Key": env_key}
    response = httpx.post(
        f"{task_app_url}/rollout",
        json=payload,
        headers=headers,
        timeout=180.0,
    )
    response.raise_for_status()
    body = response.json()

    reward_info = body.get("reward_info", {}) if isinstance(body, dict) else {}
    reward = reward_info.get("outcome_reward")
    if reward is None and isinstance(body, dict):
        metrics = body.get("metrics", {}) or {}
        reward = metrics.get("outcome_reward")
        if reward is None:
            reward = (metrics.get("outcome_objectives") or {}).get("reward", 0.0)
    if reward is None:
        reward = (reward_info.get("outcome_objectives") or {}).get("reward", 0.0)

    return float(reward or 0.0)


def run_rollout_with_status(
    *,
    task_app_url: str,
    env_key: str,
    seed: int,
    inference_url: str,
    model: str,
    rollout_id: str,
    split_name: str,
    split_env_config: Dict[str, int | str],
    backend_url: str,
    api_key: str,
    system_id: str,
) -> tuple[float, str]:
    reward = run_rollout(
        task_app_url=task_app_url,
        env_key=env_key,
        seed=seed,
        inference_url=inference_url,
        model=model,
        rollout_id=rollout_id,
        split_name=split_name,
        split_env_config=split_env_config,
    )
    candidate_id = push_status(
        backend_url=backend_url,
        api_key=api_key,
        system_id=system_id,
        rollout_id=rollout_id,
        reward=reward,
    )
    return reward, candidate_id


def push_status(
    *,
    backend_url: str,
    api_key: str,
    system_id: str,
    rollout_id: str,
    reward: float,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"rollout_id": rollout_id, "status": "done", "reward": reward}
    response = httpx.post(
        f"{backend_url}/api/prompt-learning/online/mipro/systems/{system_id}/status",
        json=payload,
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()
    response_data = response.json() if response.content else {}
    candidate_id = response_data.get("candidate_id", "unknown")
    return str(candidate_id)


def new_rollout_id(split_name: str, seed: int) -> str:
    return f"trace_{split_name}_rollout_{seed}_{uuid.uuid4().hex[:6]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIPRO Continual Learning on Crafter")
    parser.add_argument("--backend-url", default=None, help="Backend URL")
    parser.add_argument("--local-host", default="localhost", help="Local API hostname")
    parser.add_argument("--local-port", type=int, default=8030, help="Local API port")
    parser.add_argument("--rollouts-per-split", type=int, default=20, help="Rollouts per split")
    parser.add_argument(
        "--max-concurrent-rollouts",
        type=int,
        default=4,
        help="Max concurrent rollouts per split",
    )
    parser.add_argument(
        "--min-proposal-rollouts",
        type=int,
        default=10,
        help="Min rollouts before proposer runs",
    )
    parser.add_argument("--model", default="gpt-4.1-nano", help="Policy model to use")
    parser.add_argument("--train-size", type=int, default=20, help="Training seeds count")
    parser.add_argument("--val-size", type=int, default=10, help="Validation seeds count")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--system-id", type=str, default=None, help="Reuse existing system_id")
    parser.add_argument("--system-name", type=str, default=None, help="Human-readable system name")
    args = parser.parse_args()

    backend_url = (args.backend_url or resolve_backend_url()).rstrip("/")
    print(f"Backend URL: {backend_url}")

    api_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if not api_key:
        print("Minting demo API key...")
        api_key = mint_demo_api_key(backend_url=backend_url)
    print(f"API Key: {api_key[:20]}...")

    try:
        r = httpx.get(f"{backend_url}/health", timeout=30)
        print(f"Backend health: {r.status_code}")
    except Exception as exc:
        print(f"WARNING: Backend health check failed: {exc}")

    env_key = ensure_localapi_auth(backend_base=backend_url, synth_api_key=api_key)
    os.environ["ENVIRONMENT_API_KEY"] = env_key
    print(f"Environment key: {env_key[:12]}...{env_key[-4:]}")

    port = acquire_port(args.local_port, on_conflict=PortConflictBehavior.FIND_NEW)
    if port != args.local_port:
        print(f"Port {args.local_port} in use, using port {port} instead")
    app = create_crafter_vlm_local_api(build_initial_prompt()["messages"][0]["pattern"])
    run_server_background(app, port)
    print(f"Waiting for local API on port {port}...")
    wait_for_health_check_sync("localhost", port, env_key, timeout=60.0)
    task_app_url = f"http://{args.local_host}:{port}"
    print(f"Local API URL: {task_app_url}")

    splits_summary = split_stats()
    print("\n" + "=" * 60)
    print("Crafter Continual Splits")
    for split in CRAFTER_SPLITS:
        stats = splits_summary[split.name]
        print(
            f"  {split.name}: seeds {stats['seed_start']}.."
            f"{stats['seed_start'] + stats['seed_count'] - 1} | "
            f"max_steps={stats['max_steps']} | max_turns={stats['max_turns']}"
        )
    print("=" * 60)

    train_seeds = list(range(args.train_size))
    val_seeds = list(range(args.train_size, args.train_size + args.val_size))

    config_body = build_mipro_continual_config(
        task_app_url=task_app_url,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        min_rollouts_before_proposal=args.min_proposal_rollouts,
        system_id=args.system_id,
        system_name=args.system_name,
    )

    print("\nCreating MIPRO online job...")
    start_time = time.time()
    job_id = create_job(backend_url, api_key, config_body)
    print(f"Job ID: {job_id}")

    detail = get_job_detail(backend_url, api_key, job_id, include_metadata=True)
    metadata = detail.get("metadata", {})
    system_id = metadata.get("mipro_system_id")
    proxy_url = metadata.get("mipro_proxy_url")

    if not system_id or not proxy_url:
        raise RuntimeError(f"Missing mipro_system_id or mipro_proxy_url in metadata: {metadata}")

    print(f"System ID: {system_id}")
    print(f"Proxy URL: {proxy_url}")

    all_results = {
        "method": "mipro_continual_crafter",
        "job_id": job_id,
        "system_id": system_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "backend_url": backend_url,
            "model": args.model,
            "rollouts_per_split": args.rollouts_per_split,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "min_proposal_rollouts": args.min_proposal_rollouts,
            "max_concurrent_rollouts": args.max_concurrent_rollouts,
            "splits": splits_summary,
        },
        "split_results": {},
        "checkpoints": [],
    }

    total_rollouts = 0
    total_reward = 0.0

    for split in CRAFTER_SPLITS:
        split_start = time.time()
        split_rollouts = 0
        split_reward = 0.0
        seeds = split.seeds()
        env_config = split_config(split)

        print(f"\n{'=' * 60}")
        print(f"SPLIT {split.name.upper()} ({split.description})")
        print(f"{'=' * 60}")

        last_candidate_id = "unknown"
        candidate_stats: Dict[str, Dict[str, Any]] = {}

        rollout_futures: Dict[Future, int] = {}
        with ThreadPoolExecutor(
            max_workers=args.max_concurrent_rollouts,
            thread_name_prefix="crafter_rollout",
        ) as rollout_executor:
            for i in range(args.rollouts_per_split):
                seed = seeds[i % len(seeds)]
                rollout_id = new_rollout_id(split.name, seed)
                inference_url = f"{proxy_url}/{rollout_id}/chat/completions"
                future = rollout_executor.submit(
                    run_rollout_with_status,
                    task_app_url=task_app_url,
                    env_key=env_key,
                    seed=seed,
                    inference_url=inference_url,
                    model=args.model,
                    rollout_id=rollout_id,
                    split_name=split.name,
                    split_env_config=env_config,
                    backend_url=backend_url,
                    api_key=api_key,
                    system_id=system_id,
                )
                rollout_futures[future] = i

            completed_count = 0
            for future in as_completed(rollout_futures):
                i = rollout_futures[future]
                try:
                    reward, candidate_id = future.result()
                    split_reward += reward
                    total_reward += reward
                    split_rollouts += 1
                    total_rollouts += 1

                    last_candidate_id = candidate_id
                    if candidate_id and candidate_id != "unknown":
                        candidate_stats.setdefault(candidate_id, {"count": 0, "total": 0.0})
                        candidate_stats[candidate_id]["count"] += 1
                        candidate_stats[candidate_id]["total"] += reward

                    completed_count += 1
                    if completed_count % 5 == 0:
                        running_score = split_reward / split_rollouts if split_rollouts else 0.0
                        print(
                            f"  Progress: {completed_count}/{args.rollouts_per_split} | "
                            f"Avg reward: {running_score:.3f} | Candidate: {last_candidate_id}"
                        )
                except Exception as exc:
                    print(f"  Error on rollout {i}: {exc}")

        split_elapsed = time.time() - split_start
        split_avg_reward = split_reward / split_rollouts if split_rollouts else 0.0

        state = get_system_state(backend_url, api_key, system_id)
        best_candidate_id = state.get("best_candidate_id")
        best_candidate_text = extract_candidate_text(state, best_candidate_id)

        # Diagnostic: print backend state for debugging proposals
        print(f"\n  [DEBUG] Backend state after {split.name}:")
        print(f"    rollout_count: {state.get('rollout_count', 'N/A')}")
        print(f"    reward_count: {state.get('reward_count', 'N/A')}")
        print(f"    proposal_seq: {state.get('proposal_seq', 'N/A')}")
        print(f"    proposal_pending: {state.get('proposal_pending', 'N/A')}")
        print(f"    last_proposal_rollout: {state.get('last_proposal_rollout', 'N/A')}")
        print(f"    num_candidates: {len(state.get('candidates', {}))}")
        candidate_ids = list(state.get("candidates", {}).keys())
        print(f"    candidate_ids: {candidate_ids[:5]}{'...' if len(candidate_ids) > 5 else ''}")
        events = state.get("events", [])
        proposer_events = [e for e in events if "propos" in str(e.get("event_type", "")).lower()]
        if proposer_events:
            print(f"    proposer_events: {len(proposer_events)}")
            for ev in proposer_events[-5:]:
                evt_type = ev.get('event_type', '')
                msg = ev.get('message', '')[:80]
                print(f"      - {evt_type}: {msg}")
                # Print error details if this is a failure event
                if "failed" in evt_type.lower():
                    data = ev.get("data", {})
                    error = data.get("error", data.get("reason", ""))
                    if error:
                        print(f"        ERROR: {str(error)[:200]}")

        checkpoint = {
            "split": split.name,
            "split_avg_reward": split_avg_reward,
            "total_rollouts_so_far": total_rollouts,
            "elapsed_seconds": split_elapsed,
            "best_candidate_id": best_candidate_id,
            "best_candidate_text": best_candidate_text[:500] + "..."
            if best_candidate_text and len(best_candidate_text) > 500
            else best_candidate_text,
            "candidate_stats": candidate_stats,
        }
        all_results["checkpoints"].append(checkpoint)
        all_results["split_results"][split.name] = {
            "avg_reward": split_avg_reward,
            "total": split_rollouts,
            "elapsed_seconds": split_elapsed,
            "config": env_config,
        }

        print(f"\n  Split {split.name} Summary:")
        print(f"    Avg reward: {split_avg_reward:.3f}")
        print(f"    Time: {split_elapsed:.1f}s")
        print(f"    Best candidate: {best_candidate_id}")
        if best_candidate_text:
            preview = (
                best_candidate_text[:200] + "..."
                if len(best_candidate_text) > 200
                else best_candidate_text
            )
            print(f"    Best prompt: {preview}")

        if split != CRAFTER_SPLITS[-1]:
            print("\nPausing 20s between splits...")
            time.sleep(20)

    total_elapsed = time.time() - start_time
    all_results["total_elapsed_seconds"] = total_elapsed
    all_results["final_avg_reward"] = total_reward / total_rollouts if total_rollouts else 0.0

    # Print all candidates with scores
    final_state = get_system_state(backend_url, api_key, system_id)
    candidates = final_state.get("candidates", {})
    
    # Also check candidate_stats from the demo's tracking
    print("\n" + "=" * 90)
    print("ALL CANDIDATES (from demo tracking)")
    print("=" * 90)
    for cid in sorted(candidate_stats.keys(), key=lambda x: (candidate_stats.get(x, {}).get("avg", 0) if candidate_stats.get(x, {}).get("count", 0) > 0 else -1), reverse=True):
        stats = candidate_stats.get(cid, {})
        count = stats.get("count", 0)
        total = stats.get("total", 0)
        avg = total / count if count > 0 else 0
        print(f"{cid[:55]:55} | avg: {avg:.4f} | n={count:2}")
    print("=" * 90)
    
    # Also print backend state candidates with their actual stats
    print("\nBackend candidate stats (from server):")
    sorted_cands = sorted(candidates.items(), key=lambda x: x[1].get("avg_reward") or 0, reverse=True)
    for cid, cdata in sorted_cands:
        rewards = cdata.get("rewards", [])
        avg_reward = cdata.get("avg_reward")
        n_rewards = len(rewards)
        avg_str = f"{avg_reward:.4f}" if avg_reward is not None else "N/A"
        stage_payloads = cdata.get("stage_payloads", {})
        prompt_preview = ""
        for sp in stage_payloads.values():
            if isinstance(sp, dict):
                prompt_preview = sp.get("instruction", "")[:50]
                break
        print(f"  {cid[:45]:45} | avg: {avg_str:>8} | n={n_rewards:2} | {prompt_preview}...")

    print("\n" + "=" * 70)
    print("MIPRO CONTINUAL LEARNING - FINAL RESULTS")
    print("=" * 70)
    for split in CRAFTER_SPLITS:
        sr = all_results["split_results"].get(split.name, {})
        avg_reward = sr.get("avg_reward", 0.0)
        elapsed = sr.get("elapsed_seconds", 0.0)
        print(f"{split.name:<6} | avg reward: {avg_reward:.3f} | time: {elapsed:.1f}s")
    print("-" * 70)
    print(f"Total rollouts: {total_rollouts}")
    print(f"Overall avg reward: {all_results['final_avg_reward']:.3f}")
    print(f"Total time: {total_elapsed:.1f}s")
    print("=" * 70)

    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"mipro_crafter_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
