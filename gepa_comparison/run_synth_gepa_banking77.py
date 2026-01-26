#!/usr/bin/env python3
"""
Run Banking77 GEPA Comparison using Synth SDK

This script runs GEPA optimization on Banking77 using the synth-ai SDK directly.
Based on synth-ai/demos/gepa_banking77/run_demo.py

Usage:
    python run_synth_gepa_banking77.py
    python run_synth_gepa_banking77.py --local  # Use localhost backend
    python run_synth_gepa_banking77.py --rollouts 100 --model gpt-4.1-nano
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx

# Add synth-ai to path if running from monorepo
synth_ai_path = Path(__file__).resolve().parents[4] / "synth-ai"
if synth_ai_path.exists() and str(synth_ai_path) not in sys.path:
    sys.path.insert(0, str(synth_ai_path))

try:
    from datasets import load_dataset
    from fastapi import Request
    from openai import AsyncOpenAI
    from synth_ai.core.utils.env import mint_demo_api_key
    from synth_ai.data.enums import SuccessStatus
    from synth_ai.sdk.optimization.internal.prompt_learning import PromptLearningJob
    from synth_ai.sdk.optimization.internal.learning.prompt_learning_client import PromptLearningClient
    from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
    from synth_ai.sdk.localapi.auth import ensure_localapi_auth
    from synth_ai.sdk.localapi._impl.server import run_server_background
    from synth_ai.sdk.localapi._impl.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo
    from synth_ai.sdk.localapi._impl.trace_correlation_helpers import extract_trace_correlation_id
    from synth_ai.core.tunnels import (
        PortConflictBehavior,
        TunnelBackend,
        TunneledLocalAPI,
        acquire_port,
        cleanup_all,
    )
    from synth_ai.sdk.eval.job import EvalJob, EvalJobConfig
except ImportError as e:
    print(f"Error: Missing dependencies. {e}")
    print("Install with: pip install synth-ai datasets")
    sys.exit(1)


# Parse arguments
parser = argparse.ArgumentParser(description="Run Banking77 GEPA using Synth SDK")
parser.add_argument("--local", action="store_true", help="Use localhost backend (no tunnels)")
parser.add_argument("--local-host", type=str, default="localhost", help="Local API hostname")
parser.add_argument("--model", type=str, default="gpt-4.1-nano", help="Model to use")
parser.add_argument("--rollouts", type=int, default=100, help="Rollout budget")
parser.add_argument("--generations", type=int, default=4, help="Number of generations")
parser.add_argument("--train-size", type=int, default=30, help="Training seeds count")
parser.add_argument("--val-size", type=int, default=20, help="Validation seeds count")
parser.add_argument("--skip-validation", action="store_true", help="Skip GEPA validation phase")
parser.add_argument("--output", type=str, default="banking77_synth_gepa_results.json", help="Output file")
parser.add_argument(
    "--tunnel-backend",
    type=str,
    choices=["lease", "quick"],
    default="lease",
    help="Tunnel backend for dev runs: lease (managed) or quick (trycloudflare)",
)
args = parser.parse_args()

LOCAL_MODE = args.local
LOCAL_HOST = args.local_host

# Backend configuration
if LOCAL_MODE:
    SYNTH_API_BASE = "http://localhost:8000"
    TUNNEL_BACKEND = TunnelBackend.Localhost
    LOCAL_API_PORT = 8013
    print("="*60)
    print("RUNNING IN LOCAL MODE")
    print("="*60)
else:
    SYNTH_API_BASE = os.environ.get("RUST_BACKEND_URL", "https://api-dev.usesynth.ai")
    if args.tunnel_backend == "quick":
        TUNNEL_BACKEND = TunnelBackend.CloudflareQuickTunnel
    else:
        TUNNEL_BACKEND = TunnelBackend.CloudflareManagedLease
    LOCAL_API_PORT = 8001

print(f"Backend: {SYNTH_API_BASE}")
print(f"Tunnel backend: {TUNNEL_BACKEND.value}")
print(f"Model: {args.model}")
print(f"Rollouts: {args.rollouts}")
print(f"Generations: {args.generations}")

# Check backend health
r = httpx.get(f"{SYNTH_API_BASE}/health", timeout=30)
if r.status_code != 200:
    print(f"ERROR: Backend not healthy (status {r.status_code})")
    sys.exit(1)
print(f"Backend health: {r.json()}")

# Get API Key
API_KEY = os.environ.get("SYNTH_API_KEY", "")
if not API_KEY:
    print("No SYNTH_API_KEY found, minting demo key...")
    API_KEY = mint_demo_api_key(backend_url=SYNTH_API_BASE)
    print(f"Demo API Key: {API_KEY[:25]}...")
else:
    print(f"Using SYNTH_API_KEY: {API_KEY[:20]}...")

os.environ["SYNTH_API_KEY"] = API_KEY

# Ensure Environment Key
ENVIRONMENT_API_KEY = ensure_localapi_auth(
    backend_base=SYNTH_API_BASE,
    synth_api_key=API_KEY,
)
print(f"Env key ready: {ENVIRONMENT_API_KEY[:12]}...{ENVIRONMENT_API_KEY[-4:]}")


# Banking77 configuration
APP_ID = "banking77"
APP_NAME = "Banking77 Intent Classification"

BANKING77_LABELS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed",
    "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
    "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal",
    "pending_top_up", "pending_transfer", "pin_blocked", "receiving_money",
    "Refund_not_showing_up", "request_refund", "reverted_card_payment?",
    "supported_cards_and_currencies", "terminate_account", "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits",
    "top_up_reverted", "topping_up_by_card", "transaction_charged_twice",
    "transfer_fee_charged", "transfer_into_account", "transfer_not_received_by_recipient",
    "transfer_timing", "unable_to_verify_identity", "verify_my_identity",
    "verify_source_of_funds", "verify_top_up", "virtual_card_not_working",
    "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]

TOOL_NAME = "banking77_classify"
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Return the predicted banking77 intent label.",
        "parameters": {
            "type": "object",
            "properties": {"intent": {"type": "string"}},
            "required": ["intent"],
        },
    },
}


def format_available_intents(label_names: list) -> str:
    return "\n".join(f"{i + 1}. {label}" for i, label in enumerate(label_names))


def extract_system_prompt(prompt: Any, fallback: str) -> str:
    if not prompt:
        return fallback
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        messages = prompt.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if message.get("role") == "system":
                    return message.get("content") or message.get("pattern") or fallback
        return prompt.get("content") or prompt.get("pattern") or fallback
    if hasattr(prompt, "messages"):
        messages = getattr(prompt, "messages")
        if isinstance(messages, list):
            for message in messages:
                role = getattr(message, "role", None) or message.get("role")
                if role == "system":
                    content = getattr(message, "content", None) or message.get("content")
                    pattern = getattr(message, "pattern", None) or message.get("pattern")
                    return content or pattern or fallback
    return fallback


async def check_health_with_retry(
    api_url: str,
    env_api_key: str,
    retries: int = 3,
    backoff: float = 2.0,
) -> bool:
    headers = {"X-API-Key": env_api_key} if env_api_key else {}
    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{api_url}/health", headers=headers)
            if response.status_code == 200:
                return True
        except Exception as exc:
            print(f"  Health check attempt {attempt} failed: {exc}")
        await asyncio.sleep(backoff * attempt)
    return False


async def classify_banking77_query(
    query: str,
    system_prompt: str,
    available_intents: str,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    inference_url: str | None = None,
) -> str:
    user_msg = (
        f"Customer Query: {query}\n\n"
        f"Available Intents:\n{available_intents}\n\n"
        f"Classify this query into one of the above banking intents using the tool call."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    if inference_url:
        # Ensure we have an API key when using Synth backend
        if not api_key:
            raise ValueError(
                f"API key required when using inference_url={inference_url}. "
                "Set policy.config.api_key or ensure API_KEY is set."
            )
        default_headers = {"X-API-Key": api_key}
        client = AsyncOpenAI(
            base_url=inference_url,
            api_key="synth-interceptor",  # Placeholder - real auth via X-API-Key header
            default_headers=default_headers,
        )
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        )
        tool_call = response.choices[0].message.tool_calls[0]
        args_raw = tool_call.function.arguments
    else:
        # Only use api_key if it looks like an OpenAI key (starts with sk-)
        # Otherwise fall back to OPENAI_API_KEY env var
        openai_key = api_key if (api_key and api_key.startswith("sk-")) else None
        client = AsyncOpenAI(api_key=openai_key) if openai_key else AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        )
        tool_call = response.choices[0].message.tool_calls[0]
        args_raw = tool_call.function.arguments

    if not args_raw:
        raise RuntimeError("No tool call arguments returned from model")

    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    return args["intent"]


class Banking77Dataset:
    # Load directly from GitHub CSV (HuggingFace dataset scripts no longer supported)
    _DATA_URLS = {
        "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
        "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
    }

    def __init__(self):
        self._cache = {}
        self._label_names = None

    def _load_split(self, split: str):
        if split not in self._cache:
            url = self._DATA_URLS.get(split)
            if not url:
                raise ValueError(f"Unknown split: {split}. Available: {list(self._DATA_URLS.keys())}")
            ds = load_dataset("csv", data_files=url, split="train")
            self._cache[split] = ds
            # Build label names from unique categories
            if self._label_names is None:
                self._label_names = sorted(set(ds["category"]))
        return self._cache[split]

    def ensure_ready(self, splits):
        for split in splits:
            self._load_split(split)

    def size(self, split: str) -> int:
        return len(self._load_split(split))

    def sample(self, *, split: str, index: int) -> dict:
        ds = self._load_split(split)
        idx = index % len(ds)
        row = ds[idx]
        # CSV has 'category' field with string labels
        label_text = row.get("category", "unknown")
        return {"index": idx, "split": split, "text": str(row.get("text", "")), "label": label_text}

    @property
    def label_names(self) -> list:
        if self._label_names is None:
            self._load_split("train")
        return self._label_names or []


def create_banking77_local_api(system_prompt: str):
    dataset = Banking77Dataset()
    dataset.ensure_ready(["train", "test"])

    async def run_rollout(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
        split = request.env.config.get("split", "train")
        seed = request.env.seed
        sample = dataset.sample(split=split, index=seed)

        policy_config = request.policy.config or {}
        inference_url = policy_config.get("inference_url")
        api_key = policy_config.get("api_key")
        
        # Debug: log what we received
        print(f"[DEBUG rollout] seed={seed}, inference_url={inference_url}, api_key={api_key[:20] + '...' if api_key else None}")
        
        # If inference_url is missing but we're in synth_hosted mode, construct it
        # and use the global API_KEY for auth
        if not inference_url:
            inference_url = f"{SYNTH_API_BASE}/v1"
            print(f"[DEBUG rollout] No inference_url provided, using fallback: {inference_url}")
        if not api_key:
            api_key = API_KEY
            print(f"[DEBUG rollout] No api_key provided, using global API_KEY")
        
        prompt_override = (
            policy_config.get("system_prompt")
            or policy_config.get("instruction")
            or policy_config.get("prompt")
        )
        active_system_prompt = prompt_override or system_prompt

        start = time.perf_counter()
        predicted_intent = await classify_banking77_query(
            query=sample["text"],
            system_prompt=active_system_prompt,
            available_intents=format_available_intents(dataset.label_names),
            model=policy_config.get("model", "gpt-4o-mini"),
            api_key=api_key,
            inference_url=inference_url,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        expected_intent = sample["label"]
        is_correct = (
            predicted_intent.lower().replace("_", " ").strip()
            == expected_intent.lower().replace("_", " ").strip()
        )
        reward = 1.0 if is_correct else 0.0

        # Use trace_correlation_id directly from request (now a required field)
        trace_correlation_id = request.trace_correlation_id

        return RolloutResponse(
            reward_info=RolloutMetrics(
                outcome_reward=reward,
                outcome_objectives={"reward": reward, "latency_ms": latency_ms},
                instance_objectives=[{"reward": reward, "latency_ms": latency_ms}],
                details={"latency_ms": latency_ms},
            ),
            trace=None,
            trace_correlation_id=trace_correlation_id,
            inference_url=str(inference_url or ""),
            success_status=SuccessStatus.SUCCESS,
        )

    def provide_taskset_description():
        return {
            "splits": ["train", "test"],
            "sizes": {"train": dataset.size("train"), "test": dataset.size("test")},
        }

    def provide_task_instances(seeds):
        for seed in seeds:
            sample = dataset.sample(split="train", index=seed)
            yield TaskInfo(
                task={"id": APP_ID, "name": APP_NAME},
                dataset={"id": APP_ID, "split": sample["split"], "index": sample["index"]},
                inference={"tool": TOOL_NAME},
                limits={"max_turns": 1},
                task_metadata={"query": sample["text"], "expected_intent": sample["label"]},
            )

    return create_local_api(
        LocalAPIConfig(
            app_id=APP_ID,
            name=APP_NAME,
            description=f"{APP_NAME} local API for classifying customer queries into banking intents.",
            provide_taskset_description=provide_taskset_description,
            provide_task_instances=provide_task_instances,
            rollout=run_rollout,
            cors_origins=["*"],
        )
    )


def wait_for_health_check_sync(host: str, port: int, api_key: str, timeout: float = 30.0) -> None:
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


async def evaluate_prompt(
    api_url: str,
    prompt: str,
    seeds: list[int],
    label: str,
    env_api_key: str,
    *,
    split: str,
    inference_url: str | None,
    policy_api_key: str | None,
    model: str,
    fallback_api_key: str | None = None,  # Global API_KEY as fallback
) -> float:
    """Evaluate a prompt on a set of seeds and return accuracy."""
    correct = 0
    total = len(seeds)
    
    print(f"  Evaluating {label} on {total} seeds...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, seed in enumerate(seeds):
            if (i + 1) % 5 == 0:
                print(f"    Progress: {i+1}/{total}")
            
            try:
                request_body = {
                    "trace_correlation_id": f"eval-{label}-{uuid.uuid4()}",
                    "env": {"config": {"split": split}, "seed": seed},
                    "policy": {
                        "config": {
                            "model": model,
                            "inference_url": inference_url,
                            "api_key": policy_api_key or fallback_api_key,  # Fallback to global API_KEY if None
                            "system_prompt": prompt,
                        }
                    },
                }
                # Debug: Log if API key is missing
                if not (policy_api_key or fallback_api_key):
                    print(f"    WARNING: No API key available for evaluation call {i+1}/{total}")
                response = await client.post(
                    f"{api_url}/rollout",
                    json=request_body,
                    headers={"X-API-Key": env_api_key},
                )
                
                # Check if the response is successful
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success") or result.get("reward", 0) > 0:
                        correct += 1
            except Exception as e:
                error_msg = str(e)
                # Provide more helpful error messages
                if "401" in error_msg or "Invalid API key" in error_msg or "AuthenticationError" in error_msg:
                    print(f"    Authentication error on seed {seed}: API key may be invalid or expired")
                    print(f"      Inference URL: {inference_url}")
                    print(f"      API key present: {bool(policy_api_key or fallback_api_key)}")
                    print(f"      Policy API key: {policy_api_key[:20] + '...' if policy_api_key else 'None'}")
                    print(f"      Fallback API key: {fallback_api_key[:20] + '...' if fallback_api_key else 'None'}")
                else:
                    print(f"    Error on seed {seed}: {e}")
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"  {label} Accuracy: {accuracy:.1%} ({correct}/{total})")
    return accuracy


async def main():
    baseline_system_prompt = "You are an expert banking assistant that classifies customer queries into banking intents. Given a customer message, respond with exactly one intent label from the provided list using the `banking77_classify` tool."
    user_prompt = "Customer Query: {query}\n\nAvailable Intents:\n{available_intents}\n\nClassify this query into one of the above banking intents using the tool call."
    train_seeds = list(range(args.train_size))
    val_seeds_eval = list(range(50, 50 + args.val_size))
    val_seeds_gepa = [] if args.skip_validation else list(val_seeds_eval)

    print("\n" + "="*60)
    print("Setting up Banking77 Local API")
    print("="*60)

    baseline_app = create_banking77_local_api(baseline_system_prompt)
    baseline_port = acquire_port(LOCAL_API_PORT, on_conflict=PortConflictBehavior.FIND_NEW)
    if baseline_port != LOCAL_API_PORT:
        print(f"Port {LOCAL_API_PORT} in use, using port {baseline_port} instead")

    run_server_background(baseline_app, baseline_port)
    print(f"Waiting for local API on port {baseline_port}...")
    wait_for_health_check_sync("localhost", baseline_port, ENVIRONMENT_API_KEY, timeout=30.0)
    print("Local API ready!")

    if LOCAL_MODE:
        print(f"\nUsing {LOCAL_HOST} (no tunnel)...")
        baseline_local_api_url = f"http://{LOCAL_HOST}:{baseline_port}"
        baseline_tunnel = None
    else:
        print("\nProvisioning Cloudflare tunnel...")
        baseline_tunnel = await TunneledLocalAPI.create(
            local_port=baseline_port,
            backend=TUNNEL_BACKEND,
            api_key=API_KEY,
            env_api_key=ENVIRONMENT_API_KEY,
            backend_url=SYNTH_API_BASE,
            progress=True,
        )
        baseline_local_api_url = baseline_tunnel.url

    print(f"Local API URL: {baseline_local_api_url}")

    print("\n" + "="*60)
    print("Running GEPA Optimization")
    print("="*60)

    config_body = {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_id": "banking77",
            "task_app_url": baseline_local_api_url,
            "initial_prompt": {
                "id": "banking77_pattern",
                "name": "Banking77 Classification",
                "messages": [
                    {"role": "system", "order": 0, "pattern": baseline_system_prompt},
                    {"role": "user", "order": 1, "pattern": user_prompt},
                ],
                "wildcards": {"query": "REQUIRED", "available_intents": "OPTIONAL"},
            },
            "policy": {
                "model": args.model,
                "provider": "openai",
                "inference_mode": "synth_hosted",
                "temperature": 0.0,
                "max_completion_tokens": 256,
            },
            "env_config": {"split": "train"},
            "gepa": {
                "env_name": "banking77",
                "evaluation": {
                    "seeds": train_seeds,
                    "validation_seeds": val_seeds_gepa,
                },
                "rollout": {
                    "budget": args.rollouts,
                    "max_concurrent": 100,
                    "minibatch_size": 5
                },
                "mutation": {"rate": 0.3},
                "population": {
                    "initial_size": 3,
                    "num_generations": args.generations,
                    "children_per_generation": 2,
                },
                "archive": {"pareto_set_size": min(20, args.train_size - 3)},  # Leave at least 3 for feedback
                "token": {"counting_model": "gpt-4"},
            },
        },
    }

    print(f"Creating GEPA job...")
    print(f"  Model: {args.model}")
    print(f"  Rollouts: {args.rollouts}")
    print(f"  Generations: {args.generations}")
    print(f"  Train seeds: {args.train_size}")
    val_status = "0 (skipped)" if args.skip_validation else str(args.val_size)
    print(f"  Val seeds: {val_status}")

    pl_job = PromptLearningJob.from_dict(
        config_dict=deepcopy(config_body),
        backend_url=SYNTH_API_BASE,
    )

    job_id = pl_job.submit()
    print(f"\nJob ID: {job_id}")
    print("Streaming events in real-time...\n")

    start_time = time.time()

    # Use SSE streaming to get real-time events
    gepa_result = await pl_job.stream_until_complete_async(
        timeout=3600.0,
        interval=10.0,  # Interval for status checks (events stream in real-time via SSE)
    )
    
    elapsed = time.time() - start_time
    print(f"\n" + "="*60)
    print(f"Job completed in {elapsed:.1f}s")
    print(f"Status: {gepa_result.status.value}")
    print("="*60)

    results = {
        "job_id": job_id,
        "status": gepa_result.status.value,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model": args.model,
            "rollouts": args.rollouts,
            "generations": args.generations,
            "train_size": args.train_size,
            "val_size": args.val_size,
        },
    }

    if gepa_result.succeeded:
        # Get prompts and evaluate
        try:
            print("\n" + "="*60)
            print("Fetching optimized prompts and evaluating...")
            print("="*60)
            
            print(f"Verifying local API is accessible at {baseline_local_api_url}...")
            tunnel_reprovisioned = False
            health_ok = await check_health_with_retry(baseline_local_api_url, ENVIRONMENT_API_KEY)
            if not health_ok and not LOCAL_MODE:
                print("  Health check failed, re-provisioning tunnel...")
                baseline_tunnel = await TunneledLocalAPI.create(
                    local_port=baseline_port,
                    backend=TUNNEL_BACKEND,
                    api_key=API_KEY,
                    env_api_key=ENVIRONMENT_API_KEY,
                    backend_url=SYNTH_API_BASE,
                    progress=True,
                )
                baseline_local_api_url = baseline_tunnel.url
                tunnel_reprovisioned = True
                health_ok = await check_health_with_retry(baseline_local_api_url, ENVIRONMENT_API_KEY)
            print(f"  Health check: {'ok' if health_ok else 'failed'}")
            results["tunnel_reprovisioned"] = tunnel_reprovisioned
            
            pl_client = PromptLearningClient(SYNTH_API_BASE, API_KEY)
            prompt_results = await pl_client.get_prompts(job_id)
            
            print(f"\nFetched prompt results:")
            print(f"  top_prompts: {len(prompt_results.top_prompts) if prompt_results.top_prompts else 0}")
            print(f"  optimized_candidates: {len(prompt_results.optimized_candidates) if prompt_results.optimized_candidates else 0}")
            print(f"  validation_results: {len(prompt_results.validation_results) if prompt_results.validation_results else 0}")
            print(f"  attempted_candidates: {len(prompt_results.attempted_candidates) if prompt_results.attempted_candidates else 0}")
            print(f"  total_rollouts: {prompt_results.total_rollouts}")
            print(f"  total_proposal_calls: {prompt_results.total_proposal_calls}")

            # Add the new metrics to results
            results["total_rollouts"] = prompt_results.total_rollouts
            results["total_proposal_calls"] = prompt_results.total_proposal_calls
            
            # Debug: print what we have
            if hasattr(prompt_results, '__dict__'):
                print(f"\nPrompt results fields: {list(prompt_results.__dict__.keys())}")
            
            # Debug: Check version_tree for actual prompts
            if hasattr(prompt_results, 'version_tree') and prompt_results.version_tree:
                vt = prompt_results.version_tree
                print(f"\nDEBUG version_tree type: {type(vt)}")
                if isinstance(vt, dict):
                    print(f"DEBUG version_tree keys: {list(vt.keys())[:10]}")
                    # Try to find prompts in version_tree
                    for k, v in list(vt.items())[:3]:
                        print(f"  {k}: {type(v)} = {repr(str(v))[:200]}")
            
            # Use GEPA's validation results instead of re-evaluating
            # Extract baseline accuracy from validation results
            baseline_prompt = baseline_system_prompt + "\n\n" + user_prompt
            baseline_acc = None
            
            # Extract validation accuracies from validation_results
            val_accuracies = {}
            if prompt_results.validation_results:
                print(f"\nProcessing {len(prompt_results.validation_results)} validation results...")
                for i, val_result in enumerate(prompt_results.validation_results):
                    print(f"  Val result {i}: {type(val_result)}")
                    if isinstance(val_result, dict):
                        print(f"    Keys: {list(val_result.keys())}")
                        print(f"    Content: {val_result}")
                        # Store all accuracies we find
                        version_id = val_result.get("version_id", f"result_{i}")
                        accuracy = val_result.get("accuracy") or val_result.get("val_accuracy") or val_result.get("val_reward")
                        if accuracy is not None:
                            val_accuracies[version_id] = accuracy
                            print(f"    → Stored {version_id}: {accuracy:.1%}")
                    elif hasattr(val_result, '__dict__'):
                        attrs = val_result.__dict__
                        version_id = attrs.get("version_id", f"result_{i}")
                        accuracy = attrs.get("accuracy") or attrs.get("val_accuracy") or attrs.get("val_reward")
                        if accuracy is not None:
                            val_accuracies[version_id] = accuracy
                            print(f"  Stored {version_id} (object): {accuracy:.1%}")
            
            # Look for baseline in the stored accuracies
            if baseline_acc is None:
                if "baseline" in val_accuracies:
                    baseline_acc = val_accuracies["baseline"]
                    print(f"\nUsing baseline from validation_results: {baseline_acc:.1%}")
                elif "result_0" in val_accuracies:
                    # First result might be baseline
                    baseline_acc = val_accuracies["result_0"]
                    print(f"\nUsing result_0 as baseline: {baseline_acc:.1%}")
            
            # Get best and second best candidates with their validation accuracies
            best_prompt = None
            best_acc = None
            second_prompt = None
            second_acc = None
            
            # First try optimized_candidates
            if prompt_results.optimized_candidates and len(prompt_results.optimized_candidates) > 0:
                print(f"\nProcessing {len(prompt_results.optimized_candidates)} optimized candidates...")
                # Best candidate
                best_candidate = prompt_results.optimized_candidates[0]
                if isinstance(best_candidate, dict):
                    best_prompt = best_candidate.get("prompt")
                    best_acc = best_candidate.get("val_reward") or best_candidate.get("val_accuracy") or best_candidate.get("accuracy")
                elif hasattr(best_candidate, '__dict__'):
                    attrs = best_candidate.__dict__
                    best_prompt = attrs.get("prompt")
                    best_acc = attrs.get("val_reward") or attrs.get("val_accuracy") or attrs.get("accuracy")
                else:
                    best_prompt = str(best_candidate)
                
                print(f"\n[1/2] Best candidate:")
                print(f"  Validation accuracy: {best_acc:.1%}" if best_acc else "  Validation accuracy: not available")
                if best_prompt:
                    print(f"  Prompt length: {len(best_prompt)}")
                
                # Second best candidate
                if len(prompt_results.optimized_candidates) > 1:
                    second_candidate = prompt_results.optimized_candidates[1]
                    if isinstance(second_candidate, dict):
                        second_prompt = second_candidate.get("prompt")
                        second_acc = second_candidate.get("val_reward") or second_candidate.get("val_accuracy") or second_candidate.get("accuracy")
                    elif hasattr(second_candidate, '__dict__'):
                        attrs = second_candidate.__dict__
                        second_prompt = attrs.get("prompt")
                        second_acc = attrs.get("val_reward") or attrs.get("val_accuracy") or attrs.get("accuracy")
                    else:
                        second_prompt = str(second_candidate)
                    
                    print(f"\n[2/2] Second best candidate:")
                    print(f"  Validation accuracy: {second_acc:.1%}" if second_acc else "  Validation accuracy: not available")
                    if second_prompt:
                        print(f"  Prompt length: {len(second_prompt)}")
                else:
                    print("\n[2/2] Second best candidate not available")
            
            # Fallback: Try to get from validation_results using stored accuracies
            if best_acc is None:
                print("\nFalling back to stored validation accuracies...")
                # Try various keys for optimized candidates
                for key in ["optimized_0", "result_1", "top_0", "candidate_0"]:
                    if key in val_accuracies:
                        best_acc = val_accuracies[key]
                        print(f"  Found best candidate as '{key}': {best_acc:.1%}")
                        break
            
            if second_acc is None and len(val_accuracies) > 2:
                for key in ["optimized_1", "result_2", "top_1", "candidate_1"]:
                    if key in val_accuracies:
                        second_acc = val_accuracies[key]
                        print(f"  Found second best as '{key}': {second_acc:.1%}")
                        break
            
            # Try using best_score from top-level if available
            if best_acc is None and hasattr(prompt_results, 'best_score') and prompt_results.best_score:
                best_acc = prompt_results.best_score
                print(f"\nUsing best_score from API: {best_acc:.1%}")
            
            # Try using best_prompt content if available
            if not best_prompt and hasattr(prompt_results, 'best_prompt') and prompt_results.best_prompt:
                raw_best_prompt = prompt_results.best_prompt
                print(f"\nDEBUG: raw best_prompt type={type(raw_best_prompt)}")
                print(f"DEBUG: raw best_prompt repr={repr(raw_best_prompt)[:500]}")
                
                # Handle dict format (messages array)
                if isinstance(raw_best_prompt, dict):
                    if "messages" in raw_best_prompt:
                        for msg in raw_best_prompt["messages"]:
                            if msg.get("role") == "system":
                                best_prompt = msg.get("content", "")
                                break
                    elif "content" in raw_best_prompt:
                        best_prompt = raw_best_prompt["content"]
                    else:
                        best_prompt = str(raw_best_prompt)
                elif isinstance(raw_best_prompt, str) and len(raw_best_prompt) > 10:
                    best_prompt = raw_best_prompt
                else:
                    print(f"WARNING: best_prompt from API is only {len(str(raw_best_prompt))} chars, ignoring")
                    
                if best_prompt:
                    print(f"Using best_prompt from API (length: {len(best_prompt)})")
            
            # Full evaluation via EvalJob (kicks off eval jobs through backend)
            # This uses the same auth flow as GEPA - backend creates interceptor URLs
            print("\n" + "="*60)
            print("RUNNING FULL EVALUATION VIA EVAL JOBS")
            print("="*60)
            
            def run_eval_job_sync(prompt: str, label: str, seeds: list[int]) -> tuple[float, float]:
                """Run an eval job through the backend and return (mean accuracy, duration_seconds).
                
                EvalJob uses sync httpx, so this is a sync function.
                """
                print(f"\nSubmitting eval job for {label}...")
                eval_start = time.time()
                
                # Create eval job config - uses the tunneled local API
                eval_config = EvalJobConfig(
                    task_app_url=baseline_tunnel.url if baseline_tunnel else baseline_local_api_url,
                    backend_url=SYNTH_API_BASE,
                    api_key=API_KEY,
                    task_app_api_key=ENVIRONMENT_API_KEY,
                    env_name="banking77",
                    seeds=seeds,
                    policy_config={
                        "model": args.model,
                        "system_prompt": prompt,
                    },
                    env_config={"split": "train"},
                    timeout=120.0,
                )
                
                # Create and submit job
                eval_job = EvalJob(eval_config)
                job_id = eval_job.submit()
                print(f"  Eval job submitted: {job_id}")
                
                # Poll until complete with progress
                result = eval_job.poll_until_complete(
                    timeout=600.0,
                    interval=5.0,
                    progress=True,
                )
                
                eval_duration = time.time() - eval_start
                
                if result.succeeded:
                    mean_reward = result.mean_reward or 0.0
                    print(f"  {label} completed: accuracy={mean_reward:.1%} ({eval_duration:.1f}s)")
                    return mean_reward, eval_duration
                else:
                    print(f"  {label} failed: {result.error}")
                    return 0.0, eval_duration
            
            # Run baseline evaluation (sync - EvalJob uses sync httpx)
            full_baseline_acc, baseline_eval_duration = run_eval_job_sync(
                prompt=baseline_system_prompt,
                label="Baseline (full eval)",
                seeds=val_seeds_eval,
            )
            
            # Run best candidate evaluation
            best_system_prompt = extract_system_prompt(best_prompt, baseline_system_prompt)
            full_best_acc, best_eval_duration = run_eval_job_sync(
                prompt=best_system_prompt,
                label="Best Candidate (full eval)",
                seeds=val_seeds_eval,
            )

            # Store results
            gepa_duration = elapsed  # Use the tracked elapsed time from script
            results["evaluations"] = {
                "baseline": {
                    "prompt": baseline_prompt,
                    "accuracy": baseline_acc,
                    "prompt_length": len(baseline_prompt)
                },
                "full_eval_baseline": {
                    "accuracy": full_baseline_acc,
                    "seed_count": len(val_seeds_eval),
                    "duration_seconds": baseline_eval_duration,
                    "source": "eval_job",
                },
                "full_eval_best_candidate": {
                    "accuracy": full_best_acc,
                    "seed_count": len(val_seeds_eval),
                    "duration_seconds": best_eval_duration,
                    "source": "eval_job",
                },
            }
            results["timing"] = {
                "gepa_seconds": gepa_duration,
                "baseline_eval_seconds": baseline_eval_duration,
                "best_eval_seconds": best_eval_duration,
                "total_seconds": gepa_duration + baseline_eval_duration + best_eval_duration,
            }
            
            if best_acc is not None:
                results["evaluations"]["best_candidate"] = {
                    "prompt": best_prompt if best_prompt else "(prompt not available)",
                    "accuracy": best_acc,
                    "prompt_length": len(best_prompt) if best_prompt else None,
                    "improvement_over_baseline": best_acc - baseline_acc if (best_acc and baseline_acc) else None
                }
            
            if second_acc is not None:
                results["evaluations"]["second_best_candidate"] = {
                    "prompt": second_prompt if second_prompt else "(prompt not available)",
                    "accuracy": second_acc,
                    "prompt_length": len(second_prompt) if second_prompt else None,
                    "improvement_over_baseline": second_acc - baseline_acc if (second_acc and baseline_acc) else None
                }
            
            # Print summary table - show eval job results (not GEPA validation)
            print("\n" + "="*60)
            print("FINAL EVALUATION RESULTS (Full Eval Jobs)")
            print("="*60)
            print(f"{'Prompt':<20} {'Accuracy':<12} {'Length':<10} {'Improvement':<12}")
            print("-"*60)
            print(f"{'Baseline':<20} {full_baseline_acc:>11.1%} {len(baseline_prompt):>9}  {'-':<12}")
            if full_best_acc is not None:
                improvement = full_best_acc - full_baseline_acc
                prompt_len = len(best_prompt) if best_prompt else 0
                print(f"{'Best Candidate':<20} {full_best_acc:>11.1%} {prompt_len:>9}  {improvement:>+11.1%}")
            print("="*60)
            
            # Print timing summary
            print("\n" + "="*60)
            print("TIMING SUMMARY")
            print("="*60)
            print(f"GEPA optimization:     {gepa_duration:>7.1f}s")
            print(f"Baseline eval job:     {baseline_eval_duration:>7.1f}s")
            print(f"Best candidate eval:   {best_eval_duration:>7.1f}s")
            print(f"{'─'*30}")
            total_time = gepa_duration + baseline_eval_duration + best_eval_duration
            print(f"Total:                 {total_time:>7.1f}s")
            print("="*60)
            
        except Exception as e:
            print(f"\nError during evaluation: {e}")
            import traceback
            traceback.print_exc()
            results["evaluation_error"] = str(e)
    else:
        print(f"Error: {gepa_result.error}")
        results["error"] = gepa_result.error

    # Save results to results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"banking77_synth_gepa_{timestamp}.json"
    output_path = results_dir / output_filename
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")

    # Cleanup
    if not LOCAL_MODE:
        print("\nCleaning up tunnels...")
        cleanup_all()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
