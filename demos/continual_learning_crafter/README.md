# Continual Learning: Crafter (MIPRO Online)

This demo adapts the continual learning workflow to **Crafter**, using a VLM ReAct agent
and progressive episode horizons. We run MIPRO online across multiple splits and track
how the prompt and ontology evolve as the environment gets harder.

## Goal

Measure continual learning performance across increasing episode difficulty:

- **Split 1 (short):** short horizon episodes
- **Split 2 (medium):** mid-length episodes
- **Split 3 (long):** full horizon episodes

Each split uses a different seed range and max step budget to simulate a distribution shift.

## Usage

```bash
cd demos/continual_learning_crafter

# Quick run (20 rollouts per split)
uv run python run_mipro_continual.py

# Customize rollouts and model
uv run python run_mipro_continual.py \
  --rollouts-per-split 30 \
  --model gpt-4.1-nano \
  --train-size 20 \
  --val-size 10
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--rollouts-per-split` | `20` | Rollouts per split |
| `--model` | `gpt-4.1-nano` | Policy model for the agent |
| `--train-size` | `20` | Bootstrap training seeds |
| `--val-size` | `10` | Validation seeds |
| `--output` | auto | Output JSON path |
| `--backend-url` | auto | Backend URL override |
| `--system-id` | (new) | Reuse an existing MIPRO system |
| `--system-name` | (none) | Human-readable system label |

## Notes

- This demo reuses the Crafter VLM local API from `demos/gepa_crafter_vlm`.
- The split configs live in `data_splits.py` and control both seed ranges and episode limits.
