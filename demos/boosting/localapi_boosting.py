#!/usr/bin/env python3
"""Local API for GEPA prompt optimization over LLM-based tabular featurization.

Recipe-based approach: The LLM generates feature engineering formulas (a recipe)
once per prompt, and Python executes those formulas on every row. This ensures
the features are actual derived computations rather than copied raw values.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

import httpx
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from xgboost import XGBClassifier
except Exception as exc:
    raise RuntimeError(
        "xgboost is required for this demo. Install with: uv pip install xgboost"
    ) from exc

from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.localapi._impl.validators import normalize_inference_url
from synth_ai.sdk.localapi.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo

APP_ID = "boosting"
APP_NAME = "Boosting Featurization (GEPA + XGBoost)"
TOOL_NAME = "feature_recipe"
NUM_FEATURES = int(os.getenv("BOOSTING_NUM_FEATURES", "5"))

DEFAULT_SYSTEM_PROMPT = (
    "You are a feature engineer for a diabetes prediction task.\n"
    "\n"
    "The dataset has 8 numeric columns from the Pima Indians Diabetes study:\n"
    "  preg: number of pregnancies\n"
    "  plas: plasma glucose concentration (2h oral glucose tolerance test)\n"
    "  pres: diastolic blood pressure (mm Hg)\n"
    "  skin: triceps skin fold thickness (mm)\n"
    "  insu: 2-hour serum insulin (mu U/ml)\n"
    "  mass: BMI (weight in kg / height in m^2)\n"
    "  pedi: diabetes pedigree function (genetic risk score)\n"
    "  age: age in years\n"
    "\n"
    "Your job: output a feature_recipe tool call containing a JSON array of exactly "
    f"{NUM_FEATURES} formula strings. Each formula is a Python/math expression using "
    "column names as variables. The formulas will be evaluated per-row with numpy.\n"
    "\n"
    "Good formulas combine multiple columns to capture predictive interactions:\n"
    "  - Ratios: 'plas / (mass + 1)'\n"
    "  - Products: 'age * pedi'\n"
    "  - Log transforms: 'log(insu + 1)'\n"
    "  - Polynomials: 'mass ** 2'\n"
    "  - Composite: '(plas * mass) / (age + 1)'\n"
    "\n"
    "Allowed functions: log, sqrt, abs, exp (applied element-wise).\n"
    "Do NOT use raw column names alone — every formula must combine 2+ columns or "
    "apply a non-trivial transform."
)

DEFAULT_USER_PROMPT = (
    "Available columns:\n{column_list}\n\n"
    f"Return a tool call with exactly {NUM_FEATURES} formula strings in the 'formulas' array.\n"
    "Each formula is a Python expression using column names. It will be evaluated per-row."
)

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Return a feature engineering recipe: a list of formula strings that "
            "will be evaluated per-row to create derived features."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "formulas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": NUM_FEATURES,
                    "maxItems": NUM_FEATURES,
                    "description": (
                        "Python/math expressions using column names as variables. "
                        "Example: ['worst_radius / mean_radius', 'log(mean_area + 1)']"
                    ),
                }
            },
            "required": ["formulas"],
        },
    },
}

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 1337,
}

TRAIN_SIZE = int(os.getenv("BOOSTING_TRAIN_SIZE", "200"))
VAL_SIZE = int(os.getenv("BOOSTING_VAL_SIZE", "80"))


# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------

COLUMN_NAMES = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    column_names: list[str]
    train_idx: np.ndarray
    val_idx: np.ndarray


def _load_dataset() -> DatasetBundle:
    dataset = fetch_openml(data_id=37, as_frame=False, parser="auto")
    X = dataset.data.astype(float)
    y = (dataset.target == "tested_positive").astype(int)

    rng = np.random.default_rng(1337)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    train_size = min(TRAIN_SIZE, len(indices))
    val_size = min(VAL_SIZE, len(indices) - train_size)

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]

    return DatasetBundle(
        X=X, y=y, column_names=COLUMN_NAMES,
        train_idx=train_idx, val_idx=val_idx,
    )


DATASET = _load_dataset()


# ---------------------------------------------------------------------------
# Formula evaluation (safe, restricted)
# ---------------------------------------------------------------------------

# Allowed identifiers: column names + math functions
SAFE_FUNCTIONS = {
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "exp": np.exp,
}


def _validate_formula(formula: str) -> bool:
    """Check that a formula only uses allowed tokens."""
    # Remove known safe tokens and see if anything remains
    cleaned = formula
    for name in COLUMN_NAMES:
        cleaned = cleaned.replace(name, " ")
    for func in SAFE_FUNCTIONS:
        cleaned = cleaned.replace(func, " ")
    # Only numbers, operators, parens, whitespace should remain
    cleaned = re.sub(r"[\s\d\.\+\-\*/\(\)\,\^]+", "", cleaned)
    # Replace ** (power) marker
    cleaned = cleaned.replace("**", "")
    return len(cleaned.strip()) == 0


def evaluate_formulas(
    formulas: list[str],
    X: np.ndarray,
    column_names: list[str],
) -> np.ndarray:
    """Evaluate formula strings on data matrix. Returns (n_rows, n_formulas)."""
    n_rows = X.shape[0]
    n_feats = len(formulas)
    result = np.zeros((n_rows, n_feats), dtype=float)

    # Build namespace: column_name -> column_vector
    namespace = dict(SAFE_FUNCTIONS)
    for col_idx, col_name in enumerate(column_names):
        namespace[col_name] = X[:, col_idx]

    for i, formula in enumerate(formulas):
        if not _validate_formula(formula):
            # Invalid formula -> zeros
            continue
        try:
            values = eval(formula, {"__builtins__": {}}, namespace)
            if isinstance(values, (int, float)):
                result[:, i] = values
            else:
                arr = np.asarray(values, dtype=float)
                # Handle NaN/Inf
                arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
                result[:, i] = arr
        except Exception:
            # Formula error -> zeros
            pass

    return result


# ---------------------------------------------------------------------------
# LLM call to get recipe
# ---------------------------------------------------------------------------

def _extract_system_prompt(policy_config: dict[str, Any] | None, fallback: str) -> str:
    if not policy_config:
        return fallback

    for key in ("system_prompt", "prompt", "system"):
        value = policy_config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    messages = policy_config.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content") or msg.get("pattern")
                if isinstance(content, str) and content.strip():
                    return content.strip()

    prompt = policy_config.get("prompt")
    if isinstance(prompt, dict):
        msg_list = prompt.get("messages") or []
        for msg in msg_list:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content") or msg.get("pattern")
                if isinstance(content, str) and content.strip():
                    return content.strip()

    return fallback


def _render_messages(
    *,
    system_prompt: str,
    column_list: str,
    policy_config: dict[str, Any] | None,
) -> list[dict[str, str]]:
    messages = policy_config.get("messages") if policy_config else None
    if isinstance(messages, list) and messages:
        rendered = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or "user"
            content = msg.get("content") or msg.get("pattern") or ""
            if "{column_list}" in content:
                content = content.replace("{column_list}", column_list)
            rendered.append({"role": role, "content": content})
        if rendered:
            return rendered

    user_prompt = DEFAULT_USER_PROMPT.replace("{column_list}", column_list)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


_RECIPE_CACHE: dict[str, list[str]] = {}


async def _get_recipe(
    *,
    system_prompt: str,
    inference_url: str,
    model: str,
    api_key: str | None,
    policy_config: dict[str, Any] | None,
) -> list[str]:
    """Call the LLM once to get a feature recipe (list of formula strings)."""
    # Cache by trace_correlation_id (unique per GEPA trial) if available,
    # otherwise by system_prompt hash. The interceptor modifies the system
    # prompt at the LLM call level, so the local API sees the same prompt
    # text for every candidate — only the inference URL / trace ID differs.
    trace_id = (policy_config or {}).get("trace_correlation_id")
    cache_key = trace_id or hashlib.sha1(system_prompt.encode()).hexdigest()
    if cache_key in _RECIPE_CACHE:
        return _RECIPE_CACHE[cache_key]

    column_list = "\n".join(f"  {name}" for name in COLUMN_NAMES)
    messages = _render_messages(
        system_prompt=system_prompt,
        column_list=column_list,
        policy_config=policy_config,
    )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "tools": [TOOL_SCHEMA],
        "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
        "temperature": 0,
    }

    url = normalize_inference_url(inference_url)
    timeout_seconds = float(os.getenv("BOOSTING_LLM_TIMEOUT", "120"))

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            try:
                error_json = response.json()
                error_msg = str(error_json.get("error", {}).get("message", error_json))
            except Exception:
                error_msg = response.text[:500]
            raise RuntimeError(f"Proxy error ({response.status_code}): {error_msg}")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("No choices returned from model")
        tool_calls = choices[0].get("message", {}).get("tool_calls", [])
        if not tool_calls:
            raise RuntimeError("No tool calls returned from model")
        args_raw = tool_calls[0].get("function", {}).get("arguments")

    if not args_raw:
        raise RuntimeError("No tool call arguments returned from model")

    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    formulas = args.get("formulas") if isinstance(args, dict) else None
    if not isinstance(formulas, list):
        raise RuntimeError("Tool call did not return a 'formulas' list")

    # Ensure we have exactly NUM_FEATURES formulas
    formulas = [str(f) for f in formulas[:NUM_FEATURES]]
    while len(formulas) < NUM_FEATURES:
        formulas.append("0.0")

    _RECIPE_CACHE[cache_key] = formulas
    return formulas


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_SCORE_CACHE: dict[str, dict[str, Any]] = {}


async def score_prompt(
    *,
    system_prompt: str,
    inference_url: str,
    model: str,
    api_key: str | None,
    policy_config: dict[str, Any] | None,
) -> dict[str, Any]:
    trace_id = (policy_config or {}).get("trace_correlation_id")
    prompt_key = trace_id or hashlib.sha1(system_prompt.encode()).hexdigest()
    cached = _SCORE_CACHE.get(prompt_key)
    if cached:
        return cached

    start = time.perf_counter()

    # Get recipe from LLM (one call per prompt, cached)
    try:
        formulas = await _get_recipe(
            system_prompt=system_prompt,
            inference_url=inference_url,
            model=model,
            api_key=api_key,
            policy_config=policy_config,
        )
    except Exception as exc:
        return {
            "auc": 0.5,
            "accuracy": 0.0,
            "invalid_rate": 1.0,
            "reward": 0.0,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "formulas": [],
            "error": str(exc),
        }

    # Evaluate formulas on train and val data
    X_train = evaluate_formulas(formulas, DATASET.X[DATASET.train_idx], DATASET.column_names)
    X_val = evaluate_formulas(formulas, DATASET.X[DATASET.val_idx], DATASET.column_names)

    # Count invalid formulas (all-zero columns)
    invalid_count = sum(1 for i in range(X_train.shape[1]) if np.all(X_train[:, i] == 0))
    invalid_rate = invalid_count / NUM_FEATURES

    # Train and score
    model_obj = XGBClassifier(**XGB_PARAMS)
    model_obj.fit(X_train, DATASET.y[DATASET.train_idx])

    proba = model_obj.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(DATASET.y[DATASET.val_idx], proba))
    acc = float(accuracy_score(DATASET.y[DATASET.val_idx], preds))

    reward = auc - 0.1 * invalid_rate
    latency_ms = (time.perf_counter() - start) * 1000

    result = {
        "auc": auc,
        "accuracy": acc,
        "invalid_rate": invalid_rate,
        "reward": reward,
        "latency_ms": latency_ms,
        "formulas": formulas,
    }
    _SCORE_CACHE[prompt_key] = result
    return result


# ---------------------------------------------------------------------------
# Local API
# ---------------------------------------------------------------------------

def create_boosting_local_api(system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    async def run_rollout(request: RolloutRequest, fastapi_request) -> RolloutResponse:
        policy_config = request.policy.config or {}
        inference_url = policy_config.get("inference_url")
        if not inference_url:
            raise ValueError("No inference_url provided in policy config")

        resolved_prompt = _extract_system_prompt(policy_config, system_prompt)
        trace_id = policy_config.get("trace_correlation_id", "?")
        try:
            result = await score_prompt(
                system_prompt=resolved_prompt,
                inference_url=inference_url,
                model=policy_config.get("model", "gpt-4.1-nano"),
                api_key=policy_config.get("api_key"),
                policy_config=policy_config,
            )
            print(f"[rollout] trace={trace_id[:20]}  auc={result['auc']:.4f}  formulas={result.get('formulas', [])[:2]}...", flush=True)
        except Exception as exc:
            print(f"[rollout] trace={trace_id[:20]}  ERROR: {exc}", flush=True)
            raise

        return RolloutResponse(
            reward_info=RolloutMetrics(
                outcome_reward=float(result["reward"]),
                outcome_objectives={
                    "auc": result["auc"],
                    "accuracy": result["accuracy"],
                    "invalid_rate": result["invalid_rate"],
                },
                instance_objectives=[
                    {
                        "auc": result["auc"],
                        "accuracy": result["accuracy"],
                        "invalid_rate": result["invalid_rate"],
                    }
                ],
                details={
                    "latency_ms": result["latency_ms"],
                    "formulas": result.get("formulas", []),
                },
            ),
            trace=None,
            trace_correlation_id=request.trace_correlation_id,
        )

    def provide_taskset_description():
        return {
            "splits": ["train", "val"],
            "sizes": {"train": len(DATASET.train_idx), "val": len(DATASET.val_idx)},
            "num_features": NUM_FEATURES,
            "column_names": COLUMN_NAMES,
        }

    def provide_task_instances(seeds: Iterable[int]):
        for seed in seeds:
            yield TaskInfo(
                task={"id": APP_ID, "name": APP_NAME},
                dataset={"id": APP_ID, "split": "train", "index": int(seed)},
                inference={"tool": TOOL_NAME},
                limits={"max_turns": 1},
                task_metadata={"seed": int(seed)},
            )

    return create_local_api(
        LocalAPIConfig(
            app_id=APP_ID,
            name=APP_NAME,
            description="GEPA optimizes LLM feature recipe prompts with downstream XGBoost scoring.",
            provide_taskset_description=provide_taskset_description,
            provide_task_instances=provide_task_instances,
            rollout=run_rollout,
            cors_origins=["*"],
        )
    )


__all__ = ["create_boosting_local_api", "score_prompt", "DEFAULT_SYSTEM_PROMPT", "COLUMN_NAMES"]
