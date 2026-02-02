---
name: linkedin
description: Automate LinkedIn tasks using agent-browser connected to the local Chrome browser via CDP.
---

# LinkedIn Browser Automation

This skill enables browser automation on LinkedIn using agent-browser connected to the local Chrome instance.

## CRITICAL: Connection Setup

Chrome is already running on this machine with CDP enabled on port 9223.

IMPORTANT: localhost doesn't resolve in this environment. You MUST use 127.0.0.1.

### Connect using CDP URL
Use agent-browser with the `--cdp` flag and the full HTTP URL:
```bash
agent-browser --cdp http://127.0.0.1:9223 <command>
```

NOTE: Always use `http://127.0.0.1:9223`, never `localhost`, `ws://`, or just a port number.

## IMPORTANT: Handling Timeouts

agent-browser has a 10 second default timeout which is often too short for LinkedIn.

### If navigation times out, use eval to get page content:
```bash
# This is FASTER than snapshot when page is slow to load
agent-browser --cdp http://127.0.0.1:9223 eval "document.body.innerText"
```

### Check current URL without waiting:
```bash
agent-browser --cdp http://127.0.0.1:9223 eval "window.location.href"
```

### Get page title:
```bash
agent-browser --cdp http://127.0.0.1:9223 get title
```

## Recommended Workflow for LinkedIn

### Step 1: Check if already on the right page
```bash
agent-browser --cdp http://127.0.0.1:9223 eval "window.location.href"
```

### Step 2: Navigate if needed (ignore timeout errors)
```bash
agent-browser --cdp http://127.0.0.1:9223 open https://www.linkedin.com/in/me/ 2>&1 || true
```

### Step 3: Wait a moment for content to load
```bash
sleep 3
```

### Step 4: Get page text content (faster than snapshot)
```bash
agent-browser --cdp http://127.0.0.1:9223 eval "document.body.innerText"
```

This approach is MORE RELIABLE than using `snapshot` because:
- `eval` runs immediately without waiting for navigation to complete
- The page text contains all the data you need (follower counts, connection counts, etc.)

## Quick Commands Reference

| Command | Purpose |
|---------|---------|
| `eval "document.body.innerText"` | Get all visible text (FAST) |
| `eval "window.location.href"` | Get current URL |
| `get title` | Get page title |
| `open <url> 2>&1 \|\| true` | Navigate (ignore timeout) |
| `snapshot` | Get element tree (may timeout) |

## LinkedIn-Specific Patterns

### My Profile - Get Follower Count
```bash
# Navigate (ignore timeout error)
agent-browser --cdp http://127.0.0.1:9223 open https://www.linkedin.com/in/me/ 2>&1 || true
# Wait for content
sleep 3
# Get page text and search for follower count
agent-browser --cdp http://127.0.0.1:9223 eval "document.body.innerText"
```

Look for patterns like:
- "1,220 followers"
- "X followers"

### My Profile - Get Connection Count
Same approach - look for patterns like:
- "500+ connections"
- "653 connections"

### Search for Bill Gates
```bash
# Navigate directly to his profile
agent-browser --cdp http://127.0.0.1:9223 open "https://www.linkedin.com/in/williamhgates/" 2>&1 || true
sleep 3
agent-browser --cdp http://127.0.0.1:9223 eval "document.body.innerText"
```

Look for "39,823,879 followers" or similar large number.

## Output Format

IMPORTANT: When you find the answer, always output it clearly:

```
ANSWER: 1234
```

or

```
ANSWER: 39,816,349
```

## Common Issues and Solutions

1. **"Timeout 10000ms exceeded"**: Use `eval "document.body.innerText"` instead of `snapshot`
2. **Navigation hangs**: Add `2>&1 || true` to ignore errors, then use `sleep` and `eval`
3. **Page not loaded yet**: Use `sleep 3` before getting content
4. **Numbers may have formatting**: "39.8M followers" means 39,800,000

## Example Complete Workflow

```bash
# 1. Check current URL
agent-browser --cdp http://127.0.0.1:9223 eval "window.location.href"

# 2. Navigate (ignore any timeout)
agent-browser --cdp http://127.0.0.1:9223 open https://www.linkedin.com/in/me/ 2>&1 || true

# 3. Wait for page to render
sleep 3

# 4. Get all page text
agent-browser --cdp http://127.0.0.1:9223 eval "document.body.innerText"

# 5. Find the follower count in the output and report:
# ANSWER: 1220
```
