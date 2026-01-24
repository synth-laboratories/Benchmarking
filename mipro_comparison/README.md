# MIPRO Comparison Benchmark

Compare MIPRO prompt optimization implementations on Banking77 intent classification:

1. **Synth MIPRO** - Synth's implementation via synth-ai SDK
2. **DSPy MIPROv2** - Stanford's implementation from DSPy library

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Synth API key (for Synth MIPRO)

### 1. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY=sk-your-openai-key
export SYNTH_API_KEY=sk_live_your-synth-key
```

Or create a `.env` file:
```
OPENAI_API_KEY=sk-your-openai-key
SYNTH_API_KEY=sk_live_your-synth-key
```

### 3. Run Synth MIPRO

```bash
# Quick test (small scale)
python run_synth_mipro_banking77.py --rollouts 3 --train-size 5 --val-size 3

# Full benchmark
python run_synth_mipro_banking77.py --rollouts 100 --train-size 100 --val-size 50
```

**Options:**
- `--rollouts N` - Number of rollouts per candidate (default: 100)
- `--train-size N` - Training set size (default: 100)
- `--val-size N` - Validation set size (default: 50)
- `--model MODEL` - Model to use (default: gpt-4.1-nano)
- `--mode MODE` - "offline" or "online" (default: offline)
- `--local` - Use local backend instead of api-dev.usesynth.ai

### 4. Run DSPy MIPROv2

```bash
# Quick test
python run_dspy_mipro_banking77.py --trials 5 --train-size 5 --val-size 3

# Full benchmark
python run_dspy_mipro_banking77.py --trials 20 --train-size 100 --val-size 50
```

**Options:**
- `--trials N` - Number of optimization trials (default: 20)
- `--train-size N` - Training set size (default: 100)
- `--val-size N` - Validation set size (default: 50)
- `--model MODEL` - Model to use (default: gpt-4o-mini)

### 5. Run Full Comparison

```bash
python run_comparison.py --all
```

## How It Works

### Synth MIPRO

1. Starts a local Banking77 task app on port 8015
2. Creates a Cloudflare tunnel to expose the task app
3. Submits optimization job to Synth backend
4. Backend orchestrates rollouts through the tunnel
5. Returns optimized prompt with best score

The task app handles:
- Receiving rollout requests with candidate prompts
- Calling OpenAI via Synth's interceptor (for usage tracking)
- Evaluating intent classification accuracy
- Returning rewards to the optimizer

### DSPy MIPROv2

1. Loads Banking77 dataset directly
2. Creates DSPy modules for classification
3. Runs Bayesian optimization over prompt space
4. Returns optimized prompt configuration

## Results

### Benchmark Comparison (Jan 2026)

| Metric | Synth MIPRO | DSPy MIPROv2 |
|--------|-------------|--------------|
| Model | gpt-4.1-nano | gpt-4.1-nano |
| Best Score | 60% | 92% |
| Elapsed | 38.1s | 75.9s |
| Train Size | 50 | 50 |
| Val Size | 25 | 25 |
| Mode | instruction-only | few-shot + instruction |

**Notes:**
- DSPy uses few-shot examples in addition to instruction tuning
- Synth MIPRO currently only supports instruction-only optimization
- Results saved to `results/` directory as JSON

### Output Files

```
results/
  banking77_synth_mipro_YYYYMMDD_HHMMSS.json
  banking77_dspy_mipro_YYYYMMDD_HHMMSS.json
```

## Troubleshooting

### "Network unreachable" errors

The Rust backend needs to reach your Cloudflare tunnel. If you see this error, check Railway network settings for the `infra-api-dev` service.

### "trial not registered" errors

This was fixed in the Jan 2026 update. Ensure you're using the latest `rust_backend` deployment which writes trials to both Postgres AND Redis.

### Tunnel connection issues

Upgrade synth-ai to 0.7.0+:
```bash
pip install synth-ai>=0.7.0
```

### Missing SYNTH_API_KEY

Get your API key from https://usesynth.ai and set:
```bash
export SYNTH_API_KEY=sk_live_your-key
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Test Script    │────▶│  Python Backend  │────▶│  Rust Backend   │
│  (local)        │     │  (api-dev)       │     │  (infra-api-dev)│
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
        ┌─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Task App       │◀────│  Cloudflare      │◀────│  Rollout calls  │
│  (local:8015)   │     │  Tunnel          │     │                 │
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  Interceptor    │────▶│  OpenAI API      │
│  (infra-api-dev)│     │                  │
└─────────────────┘     └──────────────────┘
```

## References

- MIPRO Paper: https://arxiv.org/abs/2406.11695
- DSPy Documentation: https://dspy.ai
- Banking77 Dataset: https://huggingface.co/datasets/PolyAI/banking77
- Synth AI: https://usesynth.ai
