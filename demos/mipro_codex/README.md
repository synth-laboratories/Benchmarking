# MiPRO Online Demo — EngineBench (Codex)

Runs MiPRO prompt optimization in online mode for the EngineBench coding task.
Each rollout dispatches a coding agent (OpenCode, Codex CLI, or Claude Code)
to implement Pokemon TCG card mechanics from stub files.

## Quick Start

```bash
export SYNTH_API_KEY=sk_live_...
export OPENAI_API_KEY=sk-...

# Default: 15 rollouts with opencode agent
uv run python demos/mipro_codex/run_online_demo.py --rollouts 15

# Use Codex CLI agent
uv run python demos/mipro_codex/run_online_demo.py --rollouts 15 --agent codex

# Use Claude Code agent
uv run python demos/mipro_codex/run_online_demo.py --rollouts 15 --agent claude_code --model claude-3-5-haiku-20241022

# Custom model and train/val split
uv run python demos/mipro_codex/run_online_demo.py --rollouts 30 --model gpt-4.1-mini --train-size 10 --val-size 5
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--rollouts` | 15 | Number of rollouts to run |
| `--agent` | opencode | Agent type: `opencode`, `codex`, `claude_code` |
| `--agent-timeout` | 300 | Per-rollout agent timeout (seconds) |
| `--model` | gpt-4.1-mini | Model for inference |
| `--train-size` | 10 | Training seed count |
| `--val-size` | 5 | Validation seed count |
| `--min-proposal-rollouts` | 6 | Min rollouts before new proposals |
| `--backend-url` | auto | Backend URL override |
| `--output` | auto | Output JSON path |
| `--save-traces` | none | Directory for V4 traces |

## Environment Variables

| Variable | Description |
|---|---|
| `SYNTH_API_KEY` | Required. Synth API key |
| `CODEX_POLICY_MODEL` | Override policy model (default: gpt-4.1-mini) |
| `CODEX_POLICY_PROVIDER` | Override policy provider (default: openai) |
| `CODEX_PROPOSER_MODEL` | Override proposer model (default: gpt-4.1-mini) |
| `CODEX_PROPOSER_PROVIDER` | Override proposer provider (default: openai) |
| `SYNTH_BACKEND_URL` | Override backend URL |

## Key Differences from Banking77 Demo

- **Longer timeouts**: 600s HTTP timeout (vs 120s) — coding agents need more time
- **Fewer rollouts per candidate**: 3 (vs 10) — each rollout is expensive
- **Higher token limit**: 4096 max completion tokens (vs 256)
- **Per-rollout progress**: Prints status after every rollout (vs every 10)
- **Agent selection**: Supports opencode, codex, and claude_code agents
