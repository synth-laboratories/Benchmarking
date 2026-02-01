#!/usr/bin/env python3
"""
Run EngineBench GEPA with gpt-5-anno and print proposed prompts + scores.

This script:
- starts the EngineBench task app locally
- submits a GEPA prompt-learning job
- runs for 5 generations
- prints proposed prompts and their scores from job results
"""

import argparse
import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from localapi_engine_bench import INSTANCE_IDS, app
from synth_ai.core.env import mint_demo_api_key
from synth_ai.core.urls import BACKEND_URL_BASE
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.core.tunnels.tunneled_api import TunneledLocalAPI, TunnelBackend
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port

try:
    from synth_ai.sdk.task.server import run_server_background
except ImportError:  # pragma: no cover
    from synth_ai.sdk.task import run_server_background


def _wait_for_health(host: str, port: int, api_key: str, timeout: float = 30.0) -> None:
    url = f"http://{host}:{port}/health"
    headers = {"X-API-Key": api_key} if api_key else {}
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(url, headers=headers, timeout=5.0)
            if r.status_code in (200, 400):
                return
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Health check failed: {url}")


def _load_config(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as f:
        return tomllib.load(f)


def _score_from_candidate(candidate: Dict[str, Any]) -> Optional[float]:
    for key in (
        "score",
        "reward",
        "avg_reward",
        "best_score",
        "best_reward",
        "train_score",
        "validation_score",
        "fitness",
    ):
        value = candidate.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _prompt_from_candidate(candidate: Dict[str, Any]) -> Optional[str]:
    if isinstance(candidate.get("full_text"), str):
        return candidate.get("full_text")
    if isinstance(candidate.get("prompt"), str):
        return candidate.get("prompt")
    if isinstance(candidate.get("pattern"), str):
        return candidate.get("pattern")
    if isinstance(candidate.get("best_prompt"), str):
        return candidate.get("best_prompt")
    if isinstance(candidate.get("text"), str):
        return candidate.get("text")
    if isinstance(candidate.get("object"), dict):
        obj = candidate.get("object", {})
        if isinstance(obj.get("pattern"), str):
            return obj.get("pattern")
    return None


def _extract_candidates(results: Dict[str, Any]) -> List[Tuple[str, Optional[float], str]]:
    candidates: List[Tuple[str, Optional[float], str]] = []
    sources: Iterable[Tuple[str, Any]] = (
        ("top_prompts", results.get("top_prompts")),
        ("optimized_candidates", results.get("optimized_candidates")),
        ("frontier", results.get("frontier")),
        ("archive", results.get("archive")),
        ("candidates", results.get("candidates")),
    )
    for source_name, source in sources:
        if not isinstance(source, list):
            continue
        for idx, cand in enumerate(source):
            if not isinstance(cand, dict):
                continue
            score = _score_from_candidate(cand)
            prompt = _prompt_from_candidate(cand) or ""
            label = cand.get("id") or cand.get("candidate_id") or f"{source_name}#{idx+1}"
            candidates.append((str(label), score, prompt))
    return candidates


def _print_candidates(results: Dict[str, Any]) -> None:
    candidates = _extract_candidates(results)
    if not candidates:
        best_prompt = results.get("best_prompt")
        best_score = results.get("best_score") or results.get("best_reward")
        print("\nNo candidate list found in results payload.")
        if best_prompt:
            print("\nBest prompt:")
            print(best_prompt)
        if best_score is not None:
            print(f"Best score: {best_score}")
        return

    def sort_key(item: Tuple[str, Optional[float], str]) -> float:
        return item[1] if isinstance(item[1], (int, float)) else -1.0

    candidates = sorted(candidates, key=sort_key, reverse=True)

    print("\nProposed prompts + scores")
    print("=" * 90)
    for label, score, prompt in candidates:
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
        snippet = (prompt or "").strip().replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:180] + "..."
        print(f"{label:30} | score: {score_str:>8} | {snippet}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run EngineBench GEPA (gpt-5-anno)")
    parser.add_argument("--local", action="store_true", help="Use localhost backend")
    parser.add_argument("--local-host", type=str, default="localhost")
    parser.add_argument("--no-tunnel", action="store_true", help="Skip tunnel, use localhost for task app (requires local Temporal worker)")
    parser.add_argument("--port", type=int, default=8020, help="Task app port")
    parser.add_argument(
        "--config",
        type=str,
        default="enginebench_gepa.toml",
        help="Path to GEPA config file",
    )
    parser.add_argument("--model", type=str, default="gpt-5-nano", help="Policy model")
    parser.add_argument("--provider", type=str, default="openai", help="Policy provider")
    parser.add_argument("--proposer-model", type=str, default=None, help="Override proposer model")
    parser.add_argument("--budget", type=int, help="Override rollout budget")
    parser.add_argument("--generations", type=int, default=5, help="Override number of generations")
    args = parser.parse_args()

    backend_url = f"http://{args.local_host}:8000" if args.local else BACKEND_URL_BASE
    print(f"Backend: {backend_url}")
    print(f"Instances available: {len(INSTANCE_IDS)}")

    async with httpx.AsyncClient() as client:
        r = await client.get(f"{backend_url}/health", timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"Backend not healthy: {r.status_code}")

    api_key = os.environ.get("SYNTH_API_KEY", "")
    if not api_key:
        print("No SYNTH_API_KEY, minting demo key...")
        api_key = mint_demo_api_key(backend_url=backend_url)
        os.environ["SYNTH_API_KEY"] = api_key
    os.environ.setdefault("SYNTH_BACKEND_URL", backend_url)
    print(f"API Key: {api_key[:20]}...")

    env_key = ensure_localapi_auth(
        backend_base=backend_url,
        synth_api_key=api_key,
    )
    os.environ["ENVIRONMENT_API_KEY"] = env_key
    print(f"Environment key: {env_key[:12]}...")

    port = acquire_port(args.port, on_conflict=PortConflictBehavior.FIND_NEW)
    run_server_background(app, port)
    _wait_for_health(args.local_host, port, env_key)

    tunnel = None
    if args.local or args.no_tunnel:
        task_url = f"http://{args.local_host}:{port}"
    else:
        print("Setting up managed tunnel...")
        tunnel = await TunneledLocalAPI.create(
            local_port=port,
            backend=TunnelBackend.CloudflareManagedLease,
            api_key=api_key,
            env_api_key=env_key,
            progress=True,
        )
        task_url = tunnel.url
    print(f"Task app ready: {task_url}")

    keepalive_stop = threading.Event()

    def _keepalive_loop(url: str, stop: threading.Event) -> None:
        while not stop.wait(15.0):
            try:
                httpx.get(f"{url}/health", timeout=5.0)
            except Exception:
                pass

    if tunnel is not None:
        keepalive_thread = threading.Thread(
            target=_keepalive_loop, args=(task_url, keepalive_stop), daemon=True
        )
        keepalive_thread.start()

    config_path = Path(__file__).parent / args.config
    config_dict = _load_config(config_path)
    prompt_cfg = config_dict.get("prompt_learning")
    if not isinstance(prompt_cfg, dict):
        raise RuntimeError(f"Config {config_path} must contain a [prompt_learning] section")
    prompt_cfg["task_app_url"] = task_url
    model = args.model
    if model.endswith("-anno"):
        print(f"Model '{model}' is not supported for prompt learning; using 'gpt-5-nano' instead.")
        model = "gpt-5-nano"
    prompt_cfg.setdefault("policy", {})["model"] = model
    prompt_cfg.setdefault("policy", {})["provider"] = args.provider
    if isinstance(prompt_cfg.get("gepa"), dict):
        prompt_cfg["gepa"].pop("proposer", None)
    if args.proposer_model:
        print("Warning: proposer model override is not supported in GEPA config; ignoring.")

    if args.budget is not None:
        prompt_cfg.setdefault("gepa", {}).setdefault("rollout", {})["budget"] = args.budget
    if args.generations is not None:
        prompt_cfg.setdefault("gepa", {}).setdefault("population", {})["num_generations"] = (
            args.generations
        )
    os.environ["ENGINE_BENCH_MODEL"] = model
    prompt_cfg.setdefault("gepa", {}).setdefault("archive", {})
    pareto = prompt_cfg["gepa"]["archive"].get("pareto_set_size")
    if not isinstance(pareto, int) or pareto < 10:
        prompt_cfg["gepa"]["archive"]["pareto_set_size"] = 10

    print("\nSubmitting GEPA job...")
    job = PromptLearningJob.from_dict(
        config_dict=config_dict,
        backend_url=backend_url,
        api_key=api_key,
        task_app_api_key=env_key,
        skip_health_check=True,
    )
    job_id = job.submit()
    print(f"Job ID: {job_id}")

    print("Polling for results...")
    result = job.poll_until_complete(timeout=7200.0, interval=15.0, progress=True)
    keepalive_stop.set()
    print(f"Status: {result.status}")
    if result.failed:
        print(f"Job failed: {result.error}")
        if tunnel is not None:
            tunnel.close()
        return 1

    if result.best_reward is not None:
        print(f"Best score: {result.best_reward:.4f}")

    results = job.get_results()
    _print_candidates(results)

    if tunnel is not None:
        tunnel.close()
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
