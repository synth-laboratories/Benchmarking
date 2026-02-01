# Test Results: Local vs Daytona Sandboxing

## Test Configuration
- Seeds: 5 (requested) / 2 (quick validation)
- Model: gpt-4o-mini
- Timeout: 120s per seed
- Agent: opencode (default)

## Local Mode Test

**Status**: ✅ Running
**Command**: 
```bash
uv run python demos/engine_bench/run_eval.py --local --seeds 2 --model gpt-4o-mini --timeout 120
```

**Expected Behavior**:
1. Starts task app locally on localhost:8017
2. Submits eval job with 2 seeds
3. Task app creates temp directories for each rollout
4. Agents run locally, execute Rust code, run cargo tests
5. Results collected and displayed

## Daytona Mode Test

**Status**: ⚠️ Requires DAYTONA_API_KEY
**Command**:
```bash
export DAYTONA_API_KEY=your_key
uv run python demos/engine_bench/run_eval.py --daytona --seeds 2 --model gpt-4o-mini --timeout 120
```

**Expected Behavior**:
1. Provisions Daytona sandbox
2. Uploads task app to /app/task_app.py
3. Installs dependencies (fastapi, uvicorn, synth-ai)
4. Starts task app on 0.0.0.0:8000
5. Gets preview URL (https://8000-sandbox123.proxy.daytona.work)
6. Submits eval job using preview URL
7. Task app runs in sandbox, agents execute in sandbox
8. Results collected and displayed
9. Sandbox cleaned up

## Validation Checklist

- [x] Local mode script runs without errors
- [ ] Local mode completes eval job successfully
- [ ] Daytona mode provisions sandbox successfully
- [ ] Daytona mode uploads task app successfully
- [ ] Daytona mode gets preview URL successfully
- [ ] Daytona mode completes eval job successfully
- [ ] Both modes produce comparable results

## Next Steps

1. Wait for local mode test to complete
2. Set DAYTONA_API_KEY and run Daytona test
3. Compare results from both modes
4. Verify cleanup happens correctly
