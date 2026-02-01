# Boosting demo (GEPA + XGBoost)

This demo uses GEPA to optimize a featurization prompt for tabular data. An LLM
generates a feature engineering **recipe** (a list of formula strings like
`worst_radius / mean_radius`), Python evaluates those formulas per-row with numpy,
and a frozen XGBoost model scores on validation AUC. GEPA evolves the system prompt
to discover better formula recipes.

## What it does
- Uses the Breast Cancer dataset from scikit-learn (fast, local, numeric).
- LLM outputs a recipe of `BOOSTING_NUM_FEATURES` formula strings (one LLM call per prompt).
- Formulas are evaluated per-row with numpy to produce derived features.
- XGBoost trains with fixed hyperparameters and scores on a fixed validation split.
- GEPA evolves the system prompt to maximize validation AUC.

## Quickstart

### Local GEPA (no backend needed)
```bash
uv run python demos/boosting/local_gepa.py
uv run python demos/boosting/local_gepa.py --generations 4 --children 3
```

### Full GEPA (via Synth backend)
```bash
uv run python demos/boosting/run_demo.py
uv run python demos/boosting/run_demo.py --local
```

## Requirements
```bash
uv pip install xgboost scikit-learn
```

## Environment variables
- `SYNTH_API_KEY`: Synth API key (auto-minted if missing)
- `SYNTH_BACKEND_URL`: override backend URL
- `BOOSTING_INFERENCE_URL`: inference URL for quick baseline vs optimized scoring
- `BOOSTING_NUM_FEATURES`: feature vector length (default: 10)
- `BOOSTING_TRAIN_SIZE`: number of training rows (default: 200)
- `BOOSTING_VAL_SIZE`: number of validation rows (default: 80)

## Architecture

```
System prompt (optimized by GEPA)
  │
  ▼
LLM (gpt-4.1-nano) ──► Recipe: ["worst_radius / mean_radius", "log(mean_area + 1)", ...]
  │                      (one LLM call per prompt candidate)
  ▼
numpy eval per row ──► Feature matrix (n_rows × NUM_FEATURES)
  │
  ▼
XGBoost (frozen params) ──► AUC score ──► GEPA reward
```

Key design choice: the LLM outputs formula *strings*, not computed values. This
ensures features are actual derived computations (ratios, products, logs) rather
than copied raw values, and makes each evaluation fast (single LLM call).

## Notes
- The XGBoost training procedure is frozen (`XGB_PARAMS` in `localapi_boosting.py`).
- GEPA only optimizes the prompt; data splits and model hyperparameters stay fixed.
- Column names follow the pattern: `mean_radius`, `se_radius`, `worst_radius`, etc.
- Allowed formula functions: `log`, `sqrt`, `abs`, `exp`.
