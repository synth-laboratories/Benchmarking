# Banking77 MIPRO Comparison Benchmark

This benchmark compares MIPRO implementations on the Banking77 intent classification task:

1. **Synth MIPRO** - Our implementation using the synth-ai SDK
2. **DSPy MIPROv2** - Stanford's implementation from the DSPy library

## Dataset

**Banking77** is a banking intent classification dataset with:
- 77 different banking intent classes
- ~13,000 customer queries
- Task: Classify customer query into one of 77 banking intents

Source: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)

## Benchmark Configuration

All methods use comparable configuration:

| Parameter | Synth MIPRO | DSPy MIPROv2 |
|-----------|-------------|--------------|
| Task Model | gpt-4.1-nano | gpt-4o-mini |
| Training Set | 100 examples | 100 examples |
| Validation Set | 50 examples | 50 examples |
| Optimization | 50 rollouts | 20 trials |
| Mode | Offline | Batch |
| Metric | Intent accuracy | Intent accuracy |

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Or use the setup script
./setup.sh
```

### 2. Run Comparison

```bash
# Run all three methods
python run_comparison.py --all \
    --model gpt-4.1-nano \
    --dspy-model gpt-4o-mini \
    --rollouts 50 \
    --trials 20 \
    --train-size 100 \
    --val-size 50
```

### 3. Execute Each Method Individually

#### Synth MIPRO

```bash
python run_synth_mipro_banking77.py \
    --model gpt-4.1-nano \
    --rollouts 50 \
    --train-size 100 \
    --val-size 50 \
    --mode offline
```

#### DSPy MIPROv2

```bash
python run_dspy_mipro_banking77.py \
    --model gpt-4o-mini \
    --trials 20 \
    --train-size 100 \
    --val-size 50 \
    --auto light
```

## Results Structure

The benchmark outputs results in JSON format:

```json
{
  "timestamp": "2026-01-23T...",
  "config": {
    "train_size": 100,
    "val_size": 50
  },
  "synth_mipro": {
    "method": "synth_mipro",
    "status": "succeeded",
    "elapsed_seconds": 120.5,
    "results": {
      "best_score": 0.82
    }
  },
  "dspy_miprov2": {
    "method": "dspy_miprov2",
    "status": "succeeded",
    "elapsed_seconds": 180.3,
    "results": {
      "baseline_accuracy": 0.65,
      "optimized_accuracy": 0.80,
      "improvement": 0.15
    }
  }
}
```

## Expected Performance

Based on typical MIPRO results:

| Method | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Synth MIPRO | ~65% | ~80-85% | +15-20% |
| DSPy MIPROv2 | ~65% | ~78-82% | +13-17% |

*Note: Results vary based on model, seed, and configuration.*

## Implementation Details

### Synth MIPRO

Synth MIPRO supports two modes:

**Offline Mode** (used in benchmark):
- Backend orchestrates all rollouts
- Bootstrap phase generates few-shot examples
- Proposer generates candidate instructions
- Candidates evaluated on validation set

**Online Mode**:
- User drives rollouts locally
- Proxy-based candidate selection
- Real-time adaptation

Key features:
- Integration with Synth's prompt learning infrastructure
- Streaming optimization events
- Cloudflare tunnel for local task apps

### DSPy MIPROv2

DSPy MIPROv2 uses a three-stage approach:

1. **Bootstrap Phase**: Generate few-shot demonstrations from training data
2. **Proposal Phase**: LLM proposes grounded instructions based on task
3. **Optimization Phase**: Bayesian surrogate model searches for best combination

Auto presets:
- `light`: Fewer trials, faster optimization
- `medium`: Balanced exploration
- `heavy`: More thorough search

## Customization

### Different Model

```bash
# Use GPT-4o for Synth
python run_synth_mipro_banking77.py --model gpt-4o

# Use Claude for DSPy  
python run_dspy_mipro_banking77.py --model claude-3-sonnet-20240229
```

### More Optimization

```bash
# More rollouts for Synth
python run_synth_mipro_banking77.py --rollouts 200

# More trials for DSPy
python run_dspy_mipro_banking77.py --trials 50 --auto heavy
```

### Larger Dataset

```bash
python run_comparison.py --train-size 500 --val-size 200
```

## Environment Variables

### Required for Synth MIPRO
- `SYNTH_API_KEY`: Synth API key
- `SYNTH_BACKEND_URL` (optional): Backend URL (defaults to api-dev.usesynth.ai)

### Required for DSPy MIPROv2
- `OPENAI_API_KEY`: OpenAI API key (or appropriate provider key)

## Troubleshooting

### "SYNTH_API_KEY required"
Set your Synth API key: `export SYNTH_API_KEY=sk_live_your_key`

### "Backend not healthy"
Check the backend URL and ensure you have network access.

### "dspy module not found"
Install DSPy: `pip install dspy-ai>=2.4.0`

### "Rate limit exceeded"
Reduce concurrency or add delays between requests.

## References

- MIPRO Paper: https://arxiv.org/abs/2406.11695
- DSPy Documentation: https://dspy.ai
- DSPy MIPROv2 API: https://dspy.ai/api/optimizers/MIPROv2/
- Banking77 Dataset: https://huggingface.co/datasets/PolyAI/banking77
- Synth AI: https://usesynth.ai
