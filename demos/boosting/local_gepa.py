#!/usr/bin/env python3
"""Local GEPA-style prompt evolution for recipe-based boosting featurizer.

The LLM generates feature formulas (a recipe), Python evaluates them on the data,
and XGBoost trains on the derived features. GEPA evolves the system prompt that
guides recipe generation.

Usage:
  uv run python demos/boosting/local_gepa.py
  uv run python demos/boosting/local_gepa.py --generations 4 --children 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import httpx
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from synth_ai.core.utils.env import mint_demo_api_key
from localapi_boosting import (
    COLUMN_NAMES,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    NUM_FEATURES,
    SAFE_FUNCTIONS,
    TOOL_SCHEMA,
    XGB_PARAMS,
    evaluate_formulas,
    _validate_formula,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYNTH_API_BASE = os.environ.get("SYNTH_BACKEND_URL", "https://api-dev.usesynth.ai")
TRAIN_SIZE = 120
VAL_SIZE = 60
FEATURIZER_MODEL = "gpt-4.1-nano"
PROPOSER_MODEL = "gpt-4.1-mini"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    dataset = fetch_openml(data_id=37, as_frame=False, parser="auto")
    X = dataset.data.astype(float)
    y = (dataset.target == "tested_positive").astype(int)
    rng = np.random.default_rng(1337)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    train_idx = indices[:TRAIN_SIZE]
    val_idx = indices[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
    return X, y, train_idx, val_idx


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

async def get_recipe(
    system_prompt: str, api_key: str, inference_url: str,
) -> list[str]:
    """Call the LLM once to get a feature recipe."""
    column_list = "\n".join(f"  {name}" for name in COLUMN_NAMES)
    user_prompt = DEFAULT_USER_PROMPT.replace("{column_list}", column_list)

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": FEATURIZER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "tools": [TOOL_SCHEMA],
        "tool_choice": {"type": "function", "function": {"name": "feature_recipe"}},
        "temperature": 0,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(inference_url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()

    tool_calls = data["choices"][0]["message"].get("tool_calls", [])
    if not tool_calls:
        raise RuntimeError("No tool calls")
    args_raw = tool_calls[0]["function"]["arguments"]
    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    formulas = args.get("formulas", [])

    formulas = [str(f) for f in formulas[:NUM_FEATURES]]
    while len(formulas) < NUM_FEATURES:
        formulas.append("0.0")
    return formulas


async def propose_prompts(
    population: list[dict], api_key: str, inference_url: str,
    num_children: int = 2,
) -> list[str]:
    """Ask a proposer LLM to generate improved system prompts."""
    pop_summary = ""
    for i, member in enumerate(sorted(population, key=lambda m: m["auc"], reverse=True)):
        pop_summary += f"\n--- Candidate {i+1} (AUC={member['auc']:.4f}, Acc={member['accuracy']:.4f}, invalid={member['invalid_formulas']}/{NUM_FEATURES}) ---\n"
        pop_summary += f"System prompt:\n{member['prompt']}\n"
        pop_summary += f"Generated formulas: {member['formulas']}\n"

    propose_prompt = f"""You are optimizing system prompts for an LLM-based feature recipe generator.

TASK: The featurizer LLM ({FEATURIZER_MODEL}) receives a list of column names from the
Pima Indians Diabetes dataset (8 columns: preg, plas, pres, skin, insu, mass, pedi, age)
and outputs a "recipe" — a list of {NUM_FEATURES} Python formula strings like
'plas / (mass + 1)' or 'log(insu + 1)'. These formulas are evaluated per-row with numpy,
then XGBoost classifies diabetes positive vs negative.

The user prompt provides the column list. The system prompt guides what formulas to generate.

Allowed formula elements: column names ({', '.join(COLUMN_NAMES)}), operators (+,-,*,/,**),
functions (log, sqrt, abs, exp), and numeric constants.

CURRENT POPULATION (ranked by validation AUC):
{pop_summary}

ANALYSIS NOTES:
- Look at which formulas actually worked (high AUC) vs which produced errors (invalid count)
- Formulas that reference correct column names and use valid Python syntax score better
- Glucose (plas) is the single strongest predictor — interactions with it are often useful
- BMI (mass) * age captures metabolic aging effects
- Insulin (insu) has many zeros — log(insu + 1) handles this
- pedi (genetic risk) interactions with metabolic features are medically meaningful
- The model is small ({FEATURIZER_MODEL}) — be VERY explicit about exact formulas to generate

GOAL: Generate {num_children} NEW system prompts that will produce better formula recipes.
Each prompt should clearly specify which formulas the LLM should output, potentially
listing the exact formulas or describing the computation patterns precisely.

Return a JSON object with a "prompts" array of {num_children} new system prompt strings."""

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": PROPOSER_MODEL,
        "messages": [{"role": "user", "content": propose_prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.9,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(inference_url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Proposer error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    result = json.loads(content)
    prompts = result.get("prompts", [])
    return prompts[:num_children]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

async def score_prompt(
    system_prompt: str, api_key: str, inference_url: str,
    X: np.ndarray, y: np.ndarray,
    train_idx: np.ndarray, val_idx: np.ndarray,
) -> dict:
    formulas = await get_recipe(system_prompt, api_key, inference_url)

    X_train = evaluate_formulas(formulas, X[train_idx], COLUMN_NAMES)
    X_val = evaluate_formulas(formulas, X[val_idx], COLUMN_NAMES)

    # Count invalid (all-zero) columns
    invalid_count = sum(1 for i in range(X_train.shape[1]) if np.all(X_train[:, i] == 0))

    clf = XGBClassifier(**XGB_PARAMS)
    clf.fit(X_train, y[train_idx])
    proba = clf.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y[val_idx], proba))
    acc = float(accuracy_score(y[val_idx], preds))

    return {
        "auc": auc,
        "accuracy": acc,
        "invalid_formulas": invalid_count,
        "formulas": formulas,
    }


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--children", type=int, default=3)
    parser.add_argument("--initial-size", type=int, default=2)
    args = parser.parse_args()

    print(f"=== Recipe-Based GEPA for XGBoost Feature Engineering ===")
    print(f"Backend: {SYNTH_API_BASE}")
    print(f"Featurizer: {FEATURIZER_MODEL}, Proposer: {PROPOSER_MODEL}")
    print(f"Train={TRAIN_SIZE}, Val={VAL_SIZE}, Features={NUM_FEATURES}")
    print(f"Generations={args.generations}, Children/gen={args.children}")

    api_key = os.environ.get("SYNTH_API_KEY", "")
    if not api_key:
        print("Minting demo API key...")
        api_key = mint_demo_api_key(backend_url=SYNTH_API_BASE)

    inference_url = f"{SYNTH_API_BASE}/v1/chat/completions"
    X, y, train_idx, val_idx = load_data()

    # Raw features reference
    clf_raw = XGBClassifier(**XGB_PARAMS)
    clf_raw.fit(X[train_idx], y[train_idx])
    raw_auc = float(roc_auc_score(y[val_idx], clf_raw.predict_proba(X[val_idx])[:, 1]))
    print(f"\nReference — Raw features ({X.shape[1]}) AUC: {raw_auc:.4f}")

    # Initialize population
    population: list[dict] = []

    initial_prompts = [DEFAULT_SYSTEM_PROMPT]
    if args.initial_size > 1:
        print("\nGenerating initial population variants...")
        seed_pop = [{"prompt": DEFAULT_SYSTEM_PROMPT, "auc": 0.0, "accuracy": 0.0, "invalid_formulas": 0, "formulas": []}]
        try:
            extra = await propose_prompts(seed_pop, api_key, inference_url, num_children=args.initial_size - 1)
            initial_prompts.extend(extra)
        except Exception as exc:
            print(f"  Proposer failed ({exc}), using single initial prompt")

    print(f"\nEvaluating {len(initial_prompts)} initial prompts...")
    for i, prompt in enumerate(initial_prompts):
        print(f"\n[Init {i+1}/{len(initial_prompts)}]")
        print(f"  System prompt: {prompt[:120]}...")
        t0 = time.time()
        result = await score_prompt(prompt, api_key, inference_url, X, y, train_idx, val_idx)
        elapsed = time.time() - t0
        print(f"  Formulas: {result['formulas']}")
        print(f"  AUC={result['auc']:.4f}  Acc={result['accuracy']:.4f}  Invalid={result['invalid_formulas']}/{NUM_FEATURES}  Time={elapsed:.1f}s")
        population.append({"prompt": prompt, **result, "generation": 0})

    # Evolution loop
    for gen in range(1, args.generations + 1):
        print(f"\n{'='*60}")
        print(f"GENERATION {gen}/{args.generations}")
        print(f"{'='*60}")

        print("Proposing new candidates...")
        try:
            new_prompts = await propose_prompts(population, api_key, inference_url, num_children=args.children)
        except Exception as exc:
            print(f"  Proposer failed: {exc}")
            continue

        for i, prompt in enumerate(new_prompts):
            print(f"\n[Gen {gen}, Child {i+1}/{len(new_prompts)}]")
            print(f"  System prompt: {prompt[:150]}...")
            t0 = time.time()
            result = await score_prompt(prompt, api_key, inference_url, X, y, train_idx, val_idx)
            elapsed = time.time() - t0
            print(f"  Formulas: {result['formulas']}")
            print(f"  AUC={result['auc']:.4f}  Acc={result['accuracy']:.4f}  Invalid={result['invalid_formulas']}/{NUM_FEATURES}  Time={elapsed:.1f}s")
            population.append({"prompt": prompt, **result, "generation": gen})

        # Keep top performers
        population.sort(key=lambda m: m["auc"], reverse=True)
        if len(population) > 10:
            population = population[:10]

        best = population[0]
        print(f"\n  Best so far: AUC={best['auc']:.4f} (gen {best['generation']})")

    # Final report
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Reference — Raw features ({X.shape[1]}) AUC: {raw_auc:.4f}\n")
    print(f"{'Rank':<6} {'Gen':<5} {'AUC':>8} {'Acc':>8} {'Inv':>5}")
    print("-" * 40)
    for rank, member in enumerate(population, 1):
        print(f"{rank:<6} {member['generation']:<5} {member['auc']:>8.4f} {member['accuracy']:>8.4f} {member['invalid_formulas']:>5}")

    # Print full details of top 3
    for rank, member in enumerate(population[:3], 1):
        print(f"\n{'='*70}")
        print(f"RANK {rank} — AUC={member['auc']:.4f}, Gen={member['generation']}")
        print(f"{'='*70}")
        print(f"SYSTEM PROMPT:\n{member['prompt']}\n")
        print(f"FORMULAS:")
        for j, f in enumerate(member['formulas']):
            valid = _validate_formula(f)
            print(f"  [{j}] {f}  {'OK' if valid else 'INVALID'}")

    # Save results
    results_file = os.path.join(os.path.dirname(__file__), "local_gepa_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "raw_auc": raw_auc,
            "population": population,
            "config": {
                "featurizer_model": FEATURIZER_MODEL,
                "proposer_model": PROPOSER_MODEL,
                "num_features": NUM_FEATURES,
                "train_size": TRAIN_SIZE,
                "val_size": VAL_SIZE,
                "generations": args.generations,
                "children_per_generation": args.children,
            },
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
