#!/usr/bin/env python3
"""Run large-scale GEPA Crafter VLM demo against dev backend with QA outputs."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

from synth_ai.core.urls import BACKEND_URL_BASE, backend_health_url
from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig, EvalResult
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob, PromptLearningResult
from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.localapi.helpers import call_chat_completion_api, extract_api_key
from synth_ai.sdk.localapi.helpers import create_http_client_hooks
from synth_ai.sdk.task import run_server_background
from synth_ai.sdk.task import TaskInfo
from synth_ai.sdk.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse
from synth_ai.sdk.tunnels import wait_for_health_check
from synth_ai.sdk.tunnels.tunneled_api import TunnelBackend, TunneledLocalAPI

# Add script dir to path for local imports
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

from crafter_logic import (  # noqa: E402
    ACTION_STRING_TO_INT,
    CRAFTER_ALLOWED_ACTIONS,
    CrafterEnvironmentWrapper,
    CrafterScorer,
    CrafterVLMReActPolicy,
    normalize_action_name,
)


def _parse_int_list(value: str, default: List[int]) -> List[int]:
    raw = (value or "").strip()
    if not raw:
        return default
    items: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items or default


def _extract_system_prompt(prompt_payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(prompt_payload, dict):
        return None
    for msg in prompt_payload.get("messages", []) or []:
        if msg.get("role") == "system":
            return msg.get("pattern") or msg.get("content")
    for sec in prompt_payload.get("sections", []) or []:
        if sec.get("role") == "system":
            return sec.get("content")
    return None


def _prompt_text(candidate: Dict[str, Any]) -> str:
    if not isinstance(candidate, dict):
        return ""
    full_text = candidate.get("full_text")
    if full_text:
        return str(full_text)
    pattern = candidate.get("pattern")
    if isinstance(pattern, dict):
        prompt = _extract_system_prompt(pattern)
        if prompt:
            return prompt
    candidate_prompt = candidate.get("candidate") or candidate.get("best_prompt")
    if isinstance(candidate_prompt, dict):
        prompt = _extract_system_prompt(candidate_prompt)
        if prompt:
            return prompt
    return ""


def _qa_prompt(text: str, allowed_actions: List[str]) -> Dict[str, Any]:
    lowered = text.lower()
    required_checks = {
        "mentions_tool": "crafter_interact" in lowered,
        "mentions_action_limit": "2-5" in lowered or "2 to 5" in lowered,
        "mentions_do_adjacent": "adjacent" in lowered and "do" in lowered,
    }
    action_hits = []
    for action in allowed_actions:
        if action in lowered:
            action_hits.append(action)
    return {
        "length": len(text),
        "required_checks": required_checks,
        "missing_required": [k for k, v in required_checks.items() if not v],
        "actions_mentioned": sorted(set(action_hits)),
    }


def _summarize_candidates(candidates: List[Dict[str, Any]], allowed_actions: List[str]) -> Dict[str, Any]:
    qa_rows = []
    missing_required = 0
    for idx, cand in enumerate(candidates, start=1):
        prompt_text = _prompt_text(cand)
        qa = _qa_prompt(prompt_text, allowed_actions)
        qa_rows.append(
            {
                "rank": cand.get("rank"),
                "candidate_id": cand.get("candidate_id") or cand.get("version_id"),
                "train_accuracy": cand.get("train_accuracy"),
                "val_accuracy": cand.get("val_accuracy"),
                "qa": qa,
            }
        )
        if qa["missing_required"]:
            missing_required += 1
    return {
        "total_candidates": len(candidates),
        "missing_required_count": missing_required,
        "candidates": qa_rows,
    }


def create_crafter_vlm_local_api(system_prompt: str, default_model: str) -> Any:
    startup_http_client, shutdown_http_client = create_http_client_hooks(
        timeout=60.0,
        log_prefix="crafter_vlm_local_api",
    )

    async def rollout_executor(
        request: RolloutRequest, fastapi_request
    ) -> RolloutResponse:
        policy_config = request.policy.config or {}
        if not (policy_config.get("model") or "").strip():
            policy_config["model"] = default_model
        if not (policy_config.get("inference_url") or "").strip():
            policy_config["inference_url"] = os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            )

        seed = request.env.seed or 0
        env_config = request.env.config or {}
        max_steps = int(env_config.get("max_steps_per_episode", 200))
        max_turns = int(env_config.get("max_turns", 50))

        env = CrafterEnvironmentWrapper(seed=seed, max_steps=max_steps)
        observation = await env.reset()

        policy = CrafterVLMReActPolicy(
            system_prompt=system_prompt,
            use_vision=True,
            image_only_mode=True,
        )

        api_key = extract_api_key(fastapi_request, policy_config) or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        http_client = getattr(fastapi_request.app.state, "http_client", None)

        history: List[Dict[str, Any]] = []
        episode_rewards: List[float] = []

        for _turn in range(max_turns):
            messages = policy.build_messages(observation, history)
            response_text, response_json, tool_calls = await call_chat_completion_api(
                policy_config=policy_config,
                messages=messages,
                tools=policy.tools,
                tool_choice="required",
                api_key=api_key,
                http_client=http_client,
                enable_dns_preresolution=False,
                expected_tool_name="crafter_interact",
                log_prefix=None,
            )

            next_observation = observation
            tool_responses: List[Dict[str, Any]] = []
            if tool_calls:
                for tc in tool_calls:
                    tool_call_id = tc.get("id") or tc.get("tool_call_id")
                    tool_name = tc.get("function", {}).get("name")
                    actions_list: List[str] = []
                    if tool_name == "crafter_interact":
                        args_str = tc.get("function", {}).get("arguments", "{}")
                        try:
                            args = json.loads(args_str)
                            raw_actions = args.get("actions_list", [])
                            if isinstance(raw_actions, list):
                                actions_list = [str(a) for a in raw_actions if str(a).strip()]
                        except Exception:
                            actions_list = []
                    if not actions_list:
                        actions_list = ["noop"]

                    actions_list = actions_list[:5]
                    normalized_actions: List[str] = []
                    action_results: List[Dict[str, Any]] = []

                    for action_str in actions_list:
                        normalized = normalize_action_name(action_str) or "noop"
                        normalized_actions.append(normalized)
                        action = ACTION_STRING_TO_INT.get(normalized, 0)
                        next_observation = await env.step(action)
                        reward = next_observation.get("reward", 0.0)
                        episode_rewards.append(float(reward))
                        action_results.append(
                            {
                                "action": normalized,
                                "reward": reward,
                                "step_count": next_observation.get("step_count"),
                                "terminated": next_observation.get("terminated"),
                                "truncated": next_observation.get("truncated"),
                            }
                        )
                        if next_observation.get("terminated") or next_observation.get(
                            "truncated"
                        ):
                            break

                    if tool_call_id:
                        tool_responses.append(
                            {
                                "tool_call_id": tool_call_id,
                                "actions": normalized_actions,
                                "results": action_results,
                            }
                        )
                    if next_observation.get("terminated") or next_observation.get("truncated"):
                        break
            else:
                next_observation = await env.step(0)
                reward = next_observation.get("reward", 0.0)
                episode_rewards.append(float(reward))

            history.append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "tool_calls": tool_calls or [],
                }
            )
            if tool_responses:
                for response in tool_responses:
                    payload = {
                        "actions": response.get("actions", []),
                        "results": response.get("results", []),
                        "terminated": next_observation.get("terminated"),
                        "truncated": next_observation.get("truncated"),
                        "step_count": next_observation.get("step_count"),
                    }
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": response["tool_call_id"],
                            "content": json.dumps(payload),
                        }
                    )

            observation = next_observation
            if observation.get("terminated") or observation.get("truncated"):
                break

        outcome_reward, details = CrafterScorer.score_episode(
            observation,
            len(episode_rewards),
            max_steps,
        )

        metrics = RolloutMetrics(
            outcome_reward=outcome_reward,
            event_rewards=episode_rewards,
            details=details,
        )

        return RolloutResponse(
            metrics=metrics,
            trace_correlation_id=policy_config.get("trace_correlation_id"),
            trace=None,
            inference_url=policy_config.get("inference_url", ""),
        )

    def describe_taskset() -> Dict[str, Any]:
        return {
            "id": "crafter_vlm",
            "name": "Crafter VLM ReAct Agent",
            "splits": ["train", "test"],
            "description": "Vision-language model playing Crafter using image-only observations",
        }

    def provide_task_instances(seeds: List[int]):
        for seed in seeds:
            yield TaskInfo(
                task={"id": "crafter_vlm", "name": "Crafter VLM", "version": "1.0.0"},
                environment="crafter",
                dataset={"id": "crafter_vlm", "split": "train", "index": seed},
                inference={"supports_proxy": True, "tool": "crafter_interact"},
                limits={"max_turns": 50, "max_steps_per_episode": 200},
                task_metadata={"seed": seed, "format": "vlm_image_only"},
            )

    config = LocalAPIConfig(
        app_id="crafter_vlm",
        name="Crafter VLM ReAct Agent",
        description="Crafter local API for VLM ReAct agent with image-only observations.",
        base_task_info=TaskInfo(
            task={"id": "crafter_vlm", "name": "Crafter VLM", "version": "1.0.0"},
            environment="crafter",
            dataset={"id": "crafter_vlm", "splits": ["train", "test"]},
            inference={"supports_proxy": True, "tool": "crafter_interact"},
            limits={"max_turns": 50, "max_steps_per_episode": 200},
            task_metadata={"format": "vlm_image_only"},
        ),
        provide_taskset_description=describe_taskset,
        provide_task_instances=provide_task_instances,
        rollout=rollout_executor,
        app_state={},
        cors_origins=["*"],
        require_api_key=False,
        startup_hooks=[startup_http_client],
        shutdown_hooks=[shutdown_http_client],
    )

    return create_local_api(config)


async def _run_gepa_job(
    *,
    api_key: str,
    task_app_url: str,
    task_app_api_key: str,
    task_app_worker_token: Optional[str],
    baseline_prompt: str,
    policy_model: str,
    verifier_model: str,
    rollout_budget: int,
    num_generations: int,
    train_seeds: List[int],
    validation_seeds: List[int],
    max_concurrent: int,
    minibatch_size: int,
    population_size: int,
    children_per_generation: int,
    archive_size: int,
    pareto_set_size: int,
) -> PromptLearningResult:
    config_body = {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_url": task_app_url,
            "env_name": "crafter",
            "initial_prompt": {
                "messages": [{"role": "system", "order": 0, "pattern": baseline_prompt}],
                "wildcards": {},
            },
            "policy": {
                "inference_mode": "synth_hosted",
                "model": policy_model,
                "provider": "openai",
                "temperature": 0.0,
                "max_completion_tokens": 512,
            },
            "gepa": {
                "env_name": "crafter",
                "evaluation": {"seeds": train_seeds, "validation_seeds": validation_seeds},
                "rollout": {
                    "budget": rollout_budget,
                    "max_concurrent": max_concurrent,
                    "minibatch_size": minibatch_size,
                },
                "mutation": {"rate": 0.3},
                "population": {
                    "initial_size": population_size,
                    "num_generations": num_generations,
                    "children_per_generation": children_per_generation,
                },
                "archive": {"size": archive_size, "pareto_set_size": pareto_set_size},
                "token": {"counting_model": "gpt-4"},
            },
            "verifier": {
                "enabled": True,
                "reward_source": "verifier",
                "backend_base": SYNTH_API_BASE,
                "backend_provider": "openai",
                "backend_model": verifier_model,
                "verifier_graph_id": "zero_shot_verifier_crafter_vlm",
                "backend_event_enabled": False,
                "backend_outcome_enabled": True,
                "weight_env": 0.0,
                "weight_event": 0.0,
                "weight_outcome": 1.0,
            },
        }
    }

    job = PromptLearningJob.from_dict(
        config_dict=config_body,
        backend_url=SYNTH_API_BASE,
        api_key=api_key,
        task_app_api_key=task_app_api_key,
        task_app_worker_token=task_app_worker_token,
        skip_health_check=True,
    )
    job_id = job.submit()
    print(f"GEPA job created: {job_id}")
    result = job.poll_until_complete(timeout=7200.0, interval=5.0, progress=True)
    print(f"GEPA job finished: {result.status.value}")
    return result


async def _run_eval_job(
    *,
    api_key: str,
    task_app_url: str,
    task_app_api_key: str,
    task_app_worker_token: Optional[str],
    seeds: List[int],
    policy_model: str,
    concurrency: int,
) -> EvalResult:
    config = EvalJobConfig(
        task_app_url=task_app_url,
        backend_url=SYNTH_API_BASE,
        api_key=api_key,
        task_app_api_key=task_app_api_key,
        task_app_worker_token=task_app_worker_token,
        env_name="crafter",
        seeds=seeds,
        policy_config={"model": policy_model, "provider": "openai"},
        env_config={"max_steps_per_episode": 200, "max_turns": 50},
        concurrency=concurrency,
    )
    job = EvalJob(config)
    job_id = job.submit()
    print(f"Eval job submitted: {job_id}")
    return job.poll_until_complete(timeout=1800.0, interval=5.0, progress=True)


async def _create_tunnel(port: int, api_key: str, env_api_key: str) -> Tuple[str, Optional[str], TunneledLocalAPI]:
    tunnel = await TunneledLocalAPI.create(
        local_port=port,
        backend=TunnelBackend.SynthTunnel,
        api_key=api_key,
        env_api_key=env_api_key,
        backend_url=SYNTH_API_BASE,
        verify_dns=False,
        progress=True,
        reason="gepa_crafter_vlm_demo",
    )
    return tunnel.url, tunnel.worker_token, tunnel


async def main() -> None:
    load_dotenv(_THIS_DIR.parent.parent / ".env")

    api_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SYNTH_API_KEY not set")

    print(f"Backend: {SYNTH_API_BASE}")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(backend_health_url(SYNTH_API_BASE))
        resp.raise_for_status()
        print(f"Backend health: {resp.json()}")

    environment_api_key = ensure_localapi_auth(
        backend_base=SYNTH_API_BASE,
        synth_api_key=api_key,
    )

    allowed_actions = ", ".join(CRAFTER_ALLOWED_ACTIONS)
    baseline_prompt = (
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
        "Only use these action names and return 2-5 actions per decision. "
        "Strategy: move toward trees to collect wood; place a table once you have wood; "
        "craft a wood pickaxe, then collect stone and craft a stone pickaxe; "
        "progress toward iron tools and combat when safe."
    )

    baseline_port = int(os.environ.get("CRAFTER_TASK_APP_PORT", "8001"))
    optimized_port = int(os.environ.get("CRAFTER_OPT_TASK_APP_PORT", "8002"))

    policy_model = (os.environ.get("CRAFTER_POLICY_MODEL") or "").strip() or "gpt-4.1-nano"
    verifier_model = (os.environ.get("CRAFTER_VERIFIER_MODEL") or "").strip() or "gpt-5-nano"
    rollout_budget = int(os.environ.get("CRAFTER_ROLLOUT_BUDGET", "200"))
    num_generations = int(os.environ.get("CRAFTER_NUM_GENERATIONS", "5"))
    max_concurrent = int(os.environ.get("CRAFTER_MAX_CONCURRENT", "5"))
    minibatch_size = int(os.environ.get("CRAFTER_MINIBATCH_SIZE", "5"))
    population_size = int(os.environ.get("CRAFTER_POPULATION_SIZE", "5"))
    children_per_generation = int(os.environ.get("CRAFTER_CHILDREN_PER_GEN", "3"))
    archive_size = int(os.environ.get("CRAFTER_ARCHIVE_SIZE", "10"))
    pareto_set_size = int(os.environ.get("CRAFTER_PARETO_SET_SIZE", "15"))

    train_seeds = _parse_int_list(
        os.environ.get("CRAFTER_TRAIN_SEEDS", ""), list(range(40))
    )
    validation_seeds = _parse_int_list(
        os.environ.get("CRAFTER_VALIDATION_SEEDS", ""), list(range(80, 90))
    )
    eval_seeds = _parse_int_list(os.environ.get("CRAFTER_EVAL_SEEDS", ""), list(range(100, 130)))

    results_dir = _THIS_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    print("Starting baseline local API...")
    baseline_app = create_crafter_vlm_local_api(baseline_prompt, policy_model)
    run_server_background(baseline_app, port=baseline_port)
    await wait_for_health_check("127.0.0.1", baseline_port, environment_api_key, timeout=60.0)

    baseline_url, baseline_worker_token, baseline_tunnel = await _create_tunnel(
        baseline_port, api_key, environment_api_key
    )
    print(f"Baseline URL: {baseline_url}")

    try:
        print("Running GEPA job...")
        (results_dir / "gepa_config.json").write_text(
            json.dumps(
                {
                    "policy_model": policy_model,
                    "verifier_model": verifier_model,
                    "rollout_budget": rollout_budget,
                    "num_generations": num_generations,
                    "train_seeds": train_seeds,
                    "validation_seeds": validation_seeds,
                    "archive_size": archive_size,
                    "pareto_set_size": pareto_set_size,
                },
                indent=2,
            )
        )
        job_result = await _run_gepa_job(
            api_key=api_key,
            task_app_url=baseline_url,
            task_app_api_key=environment_api_key,
            task_app_worker_token=baseline_worker_token,
            baseline_prompt=baseline_prompt,
            policy_model=policy_model,
            verifier_model=verifier_model,
            rollout_budget=rollout_budget,
            num_generations=num_generations,
            train_seeds=train_seeds,
            validation_seeds=validation_seeds,
            max_concurrent=max_concurrent,
            minibatch_size=minibatch_size,
            population_size=population_size,
            children_per_generation=children_per_generation,
            archive_size=archive_size,
            pareto_set_size=pareto_set_size,
        )

        if not job_result.succeeded:
            raise RuntimeError(f"GEPA job failed: {job_result.error}")

        pl_client = PromptLearningClient(SYNTH_API_BASE, api_key)
        prompt_results = await pl_client.get_prompts(job_result.job_id)
        prompt_results_dict = asdict(prompt_results)
        (results_dir / "prompt_results.json").write_text(
            json.dumps(prompt_results_dict, indent=2)
        )

        optimized_prompt = _extract_system_prompt(prompt_results.best_prompt)
        if not optimized_prompt:
            raise RuntimeError("Failed to extract optimized prompt")
        (results_dir / "optimized_prompt.txt").write_text(optimized_prompt)

        print("Starting optimized local API...")
        optimized_app = create_crafter_vlm_local_api(optimized_prompt, policy_model)
        run_server_background(optimized_app, port=optimized_port)
        await wait_for_health_check("127.0.0.1", optimized_port, environment_api_key, timeout=60.0)

        optimized_url, optimized_worker_token, optimized_tunnel = await _create_tunnel(
            optimized_port, api_key, environment_api_key
        )
        print(f"Optimized URL: {optimized_url}")

        print("Running baseline eval...")
        baseline_eval = await _run_eval_job(
            api_key=api_key,
            task_app_url=baseline_url,
            task_app_api_key=environment_api_key,
            task_app_worker_token=baseline_worker_token,
            seeds=eval_seeds,
            policy_model=policy_model,
            concurrency=5,
        )

        print("Running optimized eval...")
        optimized_eval = await _run_eval_job(
            api_key=api_key,
            task_app_url=optimized_url,
            task_app_api_key=environment_api_key,
            task_app_worker_token=optimized_worker_token,
            seeds=eval_seeds,
            policy_model=policy_model,
            concurrency=5,
        )

        (results_dir / "eval_results.json").write_text(
            json.dumps(
                {"baseline": baseline_eval.raw, "optimized": optimized_eval.raw}, indent=2
            )
        )

        print("Running QA analysis...")
        top_prompts = prompt_results.top_prompts or []
        optimized_candidates = prompt_results.optimized_candidates or []

        qa_summary = {
            "job_id": job_result.job_id,
            "rollout_budget": rollout_budget,
            "num_generations": num_generations,
            "train_seeds": train_seeds,
            "validation_seeds": validation_seeds,
            "eval_seeds": eval_seeds,
            "best_reward": prompt_results.best_reward,
            "best_candidate": prompt_results.best_candidate,
            "baseline_eval": baseline_eval.raw,
            "optimized_eval": optimized_eval.raw,
            "top_prompt_qa": _summarize_candidates(top_prompts, CRAFTER_ALLOWED_ACTIONS),
            "optimized_candidates_qa": _summarize_candidates(
                optimized_candidates, CRAFTER_ALLOWED_ACTIONS
            ),
        }

        (results_dir / "quality_report.json").write_text(
            json.dumps(qa_summary, indent=2, default=str)
        )

        report_lines = [
            "# Crafter VLM GEPA QA Report",
            "",
            f"Job ID: {job_result.job_id}",
            f"Rollout budget: {rollout_budget}",
            f"Generations: {num_generations}",
            f"Train seeds: {len(train_seeds)}",
            f"Validation seeds: {len(validation_seeds)}",
            f"Eval seeds: {len(eval_seeds)}",
            "",
            "## Best Reward",
            f"{prompt_results.best_reward}",
            "",
            "## Candidate QA Summary",
            f"Top prompts: {qa_summary['top_prompt_qa']['total_candidates']}",
            f"Top prompts missing required checks: {qa_summary['top_prompt_qa']['missing_required_count']}",
            f"Optimized candidates: {qa_summary['optimized_candidates_qa']['total_candidates']}",
            f"Optimized candidates missing required checks: {qa_summary['optimized_candidates_qa']['missing_required_count']}",
        ]
        (results_dir / "quality_report.md").write_text("\n".join(report_lines))

        optimized_tunnel.close()
    finally:
        baseline_tunnel.close()

    print("Done.")


SYNTH_API_BASE = os.environ.get("SYNTH_BACKEND_URL", BACKEND_URL_BASE)

if __name__ == "__main__":
    asyncio.run(main())
