# Eval Job Failure Analysis

## Summary

For eval job `eval_a23158af4d614e52`, seeds 0-3 failed with `outcome_reward=None` in the API, but the actual issue is **constraint violations** that triggered the fail-fast gate.

## Root Cause

The deckbuilder task app implements a **fail-fast validity gate** (see `localapi_deckbuilder.py:850-867`):

```python
unsatisfied = [r for r in constraint_results if not r.get("satisfied")]
if unsatisfied:
    # Treat ANY constraint failure as an invalid deck
    return RolloutResponse(
        metrics=RolloutMetrics(
            outcome_reward=0.0,  # Fail-fast: reward is 0 if ANY constraint fails
            details={
                "error": "Invalid deck (failed task requirements)",
                "constraint_score": constraint_score,
                "constraint_results": constraint_results,
                "failed_constraints": unsatisfied,
                ...
            },
        ),
    )
```

**Any constraint failure results in `outcome_reward=0.0`**, regardless of how many constraints were satisfied.

## Actual Results (from console logs)

| Seed | Challenge | Constraint Score | Satisfied/Total | Outcome | Reason |
|------|-----------|------------------|-----------------|---------|--------|
| 0 | Basic Deck Building | 0.71 | 3/5 | **FAILED** | 2 constraints failed |
| 1 | Gardevoir Deck | 0.61 | 4/6 | **FAILED** | 2 constraints failed |
| 2 | Flygon Deck | 0.78 | 5/6 | **FAILED** | 1 constraint failed |
| 3 | No Pokemon-ex Challenge | 0.78 | 5/6 | **FAILED** | 1 constraint failed |
| 4 | Dual Evolution Lines | 1.00 | 5/5 | **PASSED** | All constraints satisfied |

## Why API Shows `outcome_reward=None`

The backend extracts `outcome_reward` from the rollout response using `RolloutMetricsStrict.from_response()` (see `backend/app/routes/eval/job_service.py:838`). 

**Issue**: The backend is not correctly extracting `outcome_reward` from the rollout response for seeds 0-3, resulting in `None` instead of `0.0`.

**Expected behavior**: The task app returns `outcome_reward=0.0` for failed constraints, but the backend isn't parsing it correctly.

## Constraint Details

To see which specific constraints failed, you would need to:

1. **Check the rollout response details** (stored in `trajectory` field, but not exposed in API)
2. **Check console logs** (shows constraint breakdown during execution)
3. **Inspect trace data** (contains LLM call, but rollout response is stored separately)

## Common Constraint Failures

Based on the constraint scores, likely failures include:

- **Deck size**: Not exactly 60 cards
- **Copy limits**: More than 4 copies of a card
- **Evolution lines**: Missing required evolution line cards
- **Ratios**: Energy/Pokemon/Trainer ratios outside required ranges
- **Pokemon-ex limits**: Too many Pokemon-ex cards (for no-ex challenge)

## Recommendations

1. **Fix backend extraction**: Ensure `RolloutMetricsStrict.from_response()` correctly extracts `outcome_reward=0.0` from failed constraint responses
2. **Expose details in API**: Include `trajectory` or `details` in the eval results API so users can see which constraints failed
3. **Improve error messages**: Surface constraint failure details in the `error` field or a separate `constraint_failures` field
4. **Debugging**: Add logging to show which constraints failed for each seed

## Next Steps

To debug further:
1. Check the actual rollout response stored in the database (trajectory field)
2. Verify the response structure matches what `RolloutMetricsStrict` expects
3. Add better error handling/logging in the backend extraction logic
