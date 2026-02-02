# GEPA LinkedIn Corporate Monitoring

Optimize site-specific browser automation skills using [Synth GEPA](https://usesynth.ai), [Claude Code](https://claude.ai/code), and [Kernel cloud browsers](https://www.kernel.sh).

This demo shows how to use GEPA (Generative Evolutionary Prompt Adaptation) to iteratively improve browser automation skills that Claude Code discovers and uses. The result is a more reliable and faster browser agent for LinkedIn corporate monitoring tasks.

## What is GEPA?

GEPA is a prompt optimization algorithm that evolves prompts over multiple generations using LLM-guided mutations and selection. It's like genetic algorithms, but for prompts.

In this demo, GEPA optimizes a "skill file" that tells Claude Code how to automate LinkedIn:
- Navigation patterns and URLs
- Element selectors and interaction strategies
- Wait timing and error recovery
- Output formatting for answer extraction

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GEPA Optimization Loop                         │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Synth      │───►│   Task App   │───►│   Kernel     │                   │
│  │   Backend    │    │   (FastAPI)  │    │   Browser    │                   │
│  │              │    │              │    │   Pool       │                   │
│  │  - Proposes  │    │  - Writes    │    │              │                   │
│  │    skill     │    │    skill to  │    │  - Chrome    │                   │
│  │    mutations │    │    VM        │    │    with CDP  │                   │
│  │              │    │              │    │              │                   │
│  │  - Evaluates │    │  - Runs      │    │  - agent-    │                   │
│  │    rollouts  │    │    Claude    │    │    browser   │                   │
│  │              │    │    Code      │    │              │                   │
│  │  - Selects   │◄───│              │◄───│  - Claude    │                   │
│  │    best      │    │  - LLM       │    │    Code CLI  │                   │
│  │    variants  │    │    verifies  │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                             │
│  Repeat for N generations until skill converges to optimal performance      │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Synth Backend** proposes skill variations using LLM-guided mutations
2. **Task App** receives rollout requests and orchestrates execution
3. **Kernel Browser Pool** provides pre-authenticated VMs with Chrome + claude + agent-browser installed
4. **Claude Code** reads the skill file and automates the browser
5. **LLM Verifier** scores each attempt for correctness and efficiency
6. **GEPA** selects the best-performing skill variants for the next generation

## Features

- **Fully sandboxed** - Uses Kernel's cloud browsers which are full Linux VMs
- **Browser profiles** - Pre-authenticated sessions (e.g., logged into LinkedIn)
- **Parallel rollouts** - Browser pool supports concurrent GEPA evaluations
- **Efficiency rewards** - Faster execution, fewer steps, and no arbitrary sleeps = higher reward
- **Browser reuse** - VMs are reused across rollouts, preserving network cache
- **Real-time streaming** - See Claude's tool calls and reasoning as they happen

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) for package management
- API keys for:
  - [Synth](https://www.usesynth.ai) - GEPA optimization
  - [Kernel](https://www.kernel.sh) - Cloud browsers
  - [Anthropic](https://platform.claude.com/) - for running Claude Code and the LLM judge.

### Installation

```bash
# From the monorepo root
cd Benchmarking/demos/linkedin_bench
uv sync
```

### Configuration

Create a `.env` file with your API keys:

```bash
SYNTH_API_KEY=sk_synth_user_...
KERNEL_API_KEY=sk_...
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Create a Kernel profile that is authenticated into LinkedIn

```bash
# Using Kernel CLI: npm install -g @onkernel/cli
kernel profiles create --name linkedin
kernel browsers create --stealth --profile-name linkedin --save-changes

```

Then navigate to the live view URL, log in to LinkedIn. When you're finished:

``` bash
kernel browsers delete <session id>

```

You now have a logged-in profile you can use across any number of browser sessions. 


### Create Browser Pool

Before running, create a browser pool with your authenticated profile:

```bash
kernel browser-pools create \
  --name agent-gepa \
  --profile-name linkedin \
  --stealth \
  --size 10
```


### Test the Skill

Run a single task to verify everything works:

```bash
# List available tasks
uv run python run_eval.py --list

# Run one task
uv run python run_eval.py --task commenters_stripe_ai_post

# Run all tasks
uv run python run_eval.py --all
```

### Run GEPA Optimization

The Synth backend needs to reach your task app. The script automatically creates a SynthTunnel to expose your local task app:

```bash
# Default: Uses SynthTunnel automatically
uv run python run_gepa.py

# Override settings
uv run python run_gepa.py --generations 5 --budget 50

# Use local Synth backend (for development)
uv run python run_gepa.py --local
```

**Workaround if SynthTunnel fails** (see `problems/synthtunnel.md`):

```bash
# Use ngrok instead
ngrok http 8030 --url your-subdomain.ngrok-free.app
export TASK_APP_URL=https://your-subdomain.ngrok-free.app
uv run python run_gepa.py
```

The optimized skill will be saved to `output/optimized_skill_<timestamp>.md`.

## Model Strategy

We will try Claude Code with **Haiku** first for faster, cheaper rollouts. Once we have a working baseline, we will re-run with **Sonnet** for higher accuracy and coverage.

## Task Source

This task set is derived from the corporate monitoring spec in:

```
https://gist.github.com/rgarcia/b8816e26fe6be17b8bb515ff0559f7e7
```

## Tasks

This demo includes 10 LinkedIn corporate monitoring tasks:

| Task ID | Description | Expected Result |
|---------|-------------|-----------------|
| `commenters_stripe_ai_post` | Stripe AI post commenters | List of commenter names + titles |
| `reactors_datadog_observability` | Datadog observability reactors by function | Engineering/Sales/Marketing counts |
| `exec_engagement_recent_posts` | Exec commenters on recent Datadog posts | Post URLs with exec names/titles |
| `notion_eng_leadership_census` | Notion engineering leaders | Names + titles |
| `figma_to_canva_departures_2024` | Figma → Canva 2024 movers | Five names + current titles |
| `anthropic_product_new_hires_jan_2026` | Anthropic January 2026 product hires | Names + titles |
| `anthropic_top_posts_dec_2024` | Anthropic top posts (Dec 2024) | Ranked posts with reactions |
| `notion_hashtag_frequency` | Notion hashtag counts | Counts for #AI/#hiring/#product |
| `snowflake_shared_connections` | Snowflake 2nd-degree connections | Count + shared connections |
| `hubspot_top_followed_employees` | HubSpot top follower counts | Three names + follower counts |

> Note: These tasks are intentionally open-ended. For deterministic scoring, capture
> post URLs and ground-truth snapshots, then update the `expected` fields in
> `src/linkedin_bench/tasks.py`.

## Reward Calculation

The reward function balances correctness with efficiency:

```python
reward = correctness * (1.0 - time_penalty - step_penalty - sleep_penalty)
```

Where:
- `correctness` = 1.0 if LLM judge says correct, 0.0 otherwise
- `time_penalty` = min(0.2, 0.1 × elapsed / max_time)
- `step_penalty` = min(0.2, 0.1 × steps / max_steps)
- `sleep_penalty` = min(0.15, 0.02 × num_sleeps + 0.01 × sleep_seconds)

The sleep penalty discourages arbitrary `sleep` or `agent-browser wait` commands, pushing the agent toward smarter waiting strategies (like waiting for specific elements).

**Example rewards:**
| Scenario | Reward |
|----------|--------|
| Correct, fast, no sleeps | ~0.90 |
| Correct, slow, no sleeps | ~0.75 |
| Correct, fast, 2s sleep | ~0.85 |
| Incorrect | 0.00 |

This encourages skills that are **correct**, **efficient**, and **avoid arbitrary delays**.

## Project Structure

```
.
├── pyproject.toml              # Python dependencies
├── .env                        # API keys (gitignored)
├── linkedin_gepa.toml          # GEPA configuration
├── run_eval.py                 # Test tasks manually
├── run_gepa.py                 # Run GEPA optimization
├── skills/
│   └── linkedin.com/
│       └── SKILL.md            # Initial skill (edit this!)
├── src/
│   └── linkedin_bench/
│       ├── __init__.py
│       ├── tasks.py            # Task definitions
│       ├── skill_template.py   # Skill loader (reads from skills/)
│       ├── verifier.py         # LLM-based result verification
│       ├── kernel_runner.py    # Kernel SDK wrapper
│       └── task_app.py         # FastAPI task app for GEPA
└── output/                     # Optimized skills saved here
```

The initial skill lives in `skills/linkedin.com/SKILL.md` so you can:
- Edit it directly as markdown
- Diff it against GEPA-optimized versions in `output/`

## Adapting for Your Own Site

This repo serves as a template. To optimize browser automation skills for a different site:

### 1. Create an authenticated browser profile

```bash
kernel profiles create --name mysite
kernel browsers create --profile-name mysite --save-changes
# Navigate to the live view URL, log in, then delete the browser
kernel browsers delete <session-id>
```

### 2. Create a browser pool

```bash
kernel browser-pools create --name mysite-pool --profile-name mysite --size 5
```

### 3. Add your skill file

Create `skills/mysite.com/SKILL.md` with instructions for automating your site. Include:
- How to connect to Chrome (`agent-browser --cdp ws://127.0.0.1:9222`)
- Common navigation patterns and URLs
- How to find and extract data from the page
- Expected output format

### 4. Define verifiable tasks

Create a tasks module with `Task` definitions. Each task needs:
- A clear prompt describing what to do
- A natural language expectation the LLM judge can evaluate
- A reasonable timeout

```python
from dataclasses import dataclass

@dataclass
class Task:
    id: str
    prompt: str
    expected: str
    timeout: int = 120

TASKS = [
    Task(
        id="get_account_balance",
        prompt="Navigate to my account page and get my current balance.",
        expected="A dollar amount (e.g., $1,234.56)",
        timeout=120,
    ),
]
```

### 5. Update configuration

- Rename/copy `src/linkedin_bench/` to match your site (e.g., `src/mysite_bench/`)
- Update `DEFAULT_POOL_NAME` in `kernel_runner.py` to your pool name
- Update `skill_template.py` to point to your skill file path
- Update `pyproject.toml` to reference your new package
- Adjust GEPA settings in your TOML config as needed

## How Claude Code Finds the Skill

Claude Code automatically discovers skill files at:

```
~/.claude/skills/<domain>/SKILL.md
```

The task app writes the skill content to this path on each rollout. Claude sees the skill in its context and follows its instructions.

## Key Implementation Details

### agent-browser with CDP

Since Claude Code runs inside the Kernel VM (same machine as Chrome), we use CDP (Chrome DevTools Protocol) for direct browser control:

```bash
agent-browser --cdp http://127.0.0.1:9223 <command>
```

**Important:** Use `http://127.0.0.1:9223` (not `localhost`, `ws://`, or just a port number). The HTTP URL connects to the CDP endpoint on port 9223.

This is faster and more reliable than remote connections.

### Browser Pool Lifecycle

1. `pool.acquire()` - Get a browser session from the pool
2. `ensure_ready()` - Reset browser state and install tools if needed:
   - Clean up old skill files and Claude temp data
   - Check if Claude Code and agent-browser are installed (skip if already present)
   - Close extra tabs and navigate to new tab page
3. `fs.write_file()` - Write skill to correct path
4. `process.spawn()` - Run Claude Code with task prompt (streaming output)
5. `pool.release(reuse=true)` - Return browser to pool, preserving VM state for next use

### LLM Verification

The verifier uses Claude to interpret natural language expectations:

```python
TASK: Get my LinkedIn follower count
EXPECTED: Around 1218 followers (±5%)
AGENT OUTPUT: "ANSWER: 1,205"

# LLM evaluates: 1205 is within 5% of 1218 → correct=True
```

This is more flexible than exact matching and tolerates minor variations.

## Troubleshooting

### SynthTunnel WebSocket 404 error

If you see `TunnelError: ws connect failed: HTTP error: 404 Not Found`, the SynthTunnel relay endpoint isn't responding. See `problems/synthtunnel.md` for details.

**Workaround:** Use ngrok instead:

```bash
ngrok http 8030 --url your-subdomain.ngrok-free.app
export TASK_APP_URL=https://your-subdomain.ngrok-free.app
uv run python run_gepa.py
```

### GEPA job fails immediately

The Synth backend needs to reach your task app. If SynthTunnel isn't working and you haven't set `TASK_APP_URL`:

```bash
# Expose your task app via ngrok
ngrok http 8030
export TASK_APP_URL=https://xxxxx.ngrok.io
uv run python run_gepa.py
```

### "Pool not found"

Create the pool first:
```bash
kernel browser-pools create --name agent-gepa --profile-name linkedin --size 10
```

### "Claude Code not installed"

The task app auto-installs Claude Code on first run. If it fails:
```bash
kernel ssh <session-id>
curl -fsSL https://claude.ai/install.sh | bash
```

### Slow performance

- **Disable stealth mode** for faster page loads: `kernel browser-pools update agent-gepa --stealth=false`
- Increase pool size for more parallelism
- Discard idle browsers after config changes: `kernel browser-pools update agent-gepa --discard-all-idle`

## License

MIT

## Contributing

Contributions welcome! Please open an issue first to discuss major changes.
