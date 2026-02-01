# MIPRO Online Latency Breakdown

This document provides a comprehensive breakdown of all latency sources in the MIPRO online demo, starting with database query analysis.

## Overview

The MIPRO online demo experiences significant latency (3.5-7.6 seconds per rollout), which we're investigating systematically. This document breaks down each component.

---

## Online MIPRO Architecture

### System Components

```
┌─────────────────┐
│  Demo Script    │  (run_demo.py)
│  - Creates job  │
│  - Registers    │
│    candidates   │
│  - Runs         │
│    rollouts     │
└────────┬────────┘
         │
         │ HTTP API calls
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Rust Backend (Port 8090)                               │
│  ┌───────────────────────────────────────────────────┐ │
│  │  API Routes                                       │ │
│  │  - POST /api/prompt-learning/online/jobs          │ │
│  │  - POST /api/.../mipro/systems/{id}/candidates │ │
│  │  - POST /api/.../mipro/systems/{id}/status      │ │
│  └───────────────────────────────────────────────────┘ │
│                                                          │
│  ┌───────────────────────────────────────────────────┐ │
│  │  MIPRO Proxy                                      │ │
│  │  GET /api/mipro/v1/{system_id}/{rollout_id}/     │ │
│  │      chat/completions                             │ │
│  │  - Selects candidate (TPE or active)             │ │
│  │  - Adds x-mipro-candidate-id header              │ │
│  └───────────────────────────────────────────────────┘ │
│                                                          │
│  ┌───────────────────────────────────────────────────┐ │
│  │  MIPRO Online Service                             │ │
│  │  - Manages system state                           │ │
│  │  - Stores candidates in blob storage              │ │
│  │  - Updates systems table (Postgres)               │ │
│  │  - TPE optimizer                                  │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
         │
         │ Database (Postgres/PlanetScale)
         │ - systems table (system_id, org_id, state_ref)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Blob Storage (Wasabi/S3)                              │
│  - System state snapshots                              │
│  - Candidate definitions                               │
│  - Rollout history                                     │
└─────────────────────────────────────────────────────────┘

         │
         │ Inference requests
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Task App (Banking77)                                  │
│  Port 8016                                              │
│  ┌───────────────────────────────────────────────────┐ │
│  │  POST /rollout                                    │ │
│  │  - Gets test example (by seed)                    │ │
│  │  - Calls LLM via proxy URL                        │ │
│  │  - Compares prediction to ground truth            │ │
│  │  - Returns reward (1.0 = correct, 0.0 = wrong)    │ │
│  │  - Includes candidate_id in response metadata    │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Flow Sequence

```
1. Job Creation
   Demo Script → POST /api/prompt-learning/online/jobs
   → Rust Backend creates system_id, initializes baseline
   → Saves state to S3, inserts into systems table
   → Returns system_id and proxy_url

2. Candidate Registration
   Demo Script → POST /api/.../mipro/systems/{id}/candidates
   → MIPRO Service loads state from S3
   → Applies deltas to baseline prompt
   → Saves updated state to S3, updates systems table
   → Returns candidate_id

3. Rollout Execution (per rollout)
   Demo Script → POST {task_app}/rollout
   → Task App calls proxy URL for LLM inference
   → Proxy: start_rollout() → selects candidate → forwards to LLM
   → Task App calculates reward → returns to demo script

4. Status Updates (per rollout)
   Demo Script → POST /api/.../mipro/systems/{id}/status (reward)
   → MIPRO Service loads state, updates candidate rewards, updates TPE
   → Saves state to S3, updates systems table
   → Demo Script → POST /api/.../mipro/systems/{id}/status (done)
   → MIPRO Service marks rollout complete
```

---

## Core Algorithm Pseudocode

### `start_rollout(system_id, correlation_id)` - Candidate Selection

```rust
async fn start_rollout(system_id: String, correlation_id: Option<String>) -> Result<(Candidate, String)> {
    // LATENCY: Load state from S3 + DB query (~400-900ms)
    state = load_state(system_id).await?;
    
    rollout_batch = get_batch_size_from_config(state.config);
    
    // Candidate selection strategy
    if rollout_batch.is_some() {
        // BATCH MODE: Use same candidate for N rollouts
        if state.active_candidate_remaining == 0 || state.active_candidate_id.is_none() {
            candidate = select_candidate_with_tpe(&mut state).await
                .unwrap_or_else(|| state.choose_candidate().clone());
            state.active_candidate_id = Some(candidate.candidate_id.clone());
            state.active_candidate_remaining = rollout_batch;
        } else {
            candidate = state.active_candidate().clone();
        }
        state.active_candidate_remaining -= 1;
    } else {
        // PROBABILISTIC MODE: TPE selects candidate per rollout
        candidate = select_candidate_with_tpe(&mut state).await
            .unwrap_or_else(|| state.choose_candidate().clone());
    }
    
    // Create rollout record
    rollout_id = correlation_id.unwrap_or_else(|| generate_uuid());
    rollout = MiproRollout {
        rollout_id: rollout_id.clone(),
        candidate_id: Some(candidate.candidate_id.clone()),
        status: "started",
        ...
    };
    state.rollouts.insert(rollout_id.clone(), rollout);
    state.rollout_seq += 1;
    
    // LATENCY: Save state to S3 + DB update (~300-400ms)
    save_state(&state).await?;
    
    return (candidate, rollout_id);
}
```

**Key Operations:**
- `load_state()`: DB query (200-740ms) + S3 fetch (90-180ms) = **290-920ms**
- `select_candidate_with_tpe()`: TPE optimization (~1-10ms)
- `save_state()`: S3 upload (90-125ms) + DB update (200-420ms) = **290-545ms**
- **Total**: ~580-1465ms per `start_rollout` call

### `apply_status(system_id, payload)` - Reward Processing

```rust
async fn apply_status(payload: MiproStatusPayload) -> Result<MiproSystemState> {
    system_id = payload.system_id;
    
    // LATENCY: Load state from S3 + DB query (~400-900ms)
    state = load_state(system_id).await?;
    
    rollout_id = payload.rollout_id.or(payload.attempt_id).unwrap_or(generate_uuid());
    
    // Validate candidate_id exists
    if let Some(candidate_id) = payload.candidate_id {
        if !state.candidates.contains_key(&candidate_id) {
            return Err("unknown candidate_id");
        }
    }
    
    // Get or create rollout record
    if !state.rollouts.contains_key(&rollout_id) {
        state.rollouts.insert(rollout_id.clone(), MiproRollout {
            rollout_id: rollout_id.clone(),
            candidate_id: payload.candidate_id.clone(),
            status: "pending",
            ...
        });
    }
    
    rollout_entry = state.rollouts.get_mut(&rollout_id);
    
    // Update rollout status
    rollout_entry.status = payload.status.clone();
    rollout_entry.reward = payload.reward.or(rollout_entry.reward);
    rollout_entry.candidate_id = payload.candidate_id.or(rollout_entry.candidate_id);
    
    status = rollout_entry.status.clone();
    candidate_id = rollout_entry.candidate_id.clone();
    
    if status == "done" {
        // LATENCY: Save state (~500-1200ms)
        state.bump_version();
        save_state(&state).await?;
        return Ok(state);
    }
    
    if payload.status == "reward" {
        if payload.candidate_id.is_none() {
            return Err("candidate_id required for reward status");
        }
        
        reward = payload.reward.unwrap();
        candidate_id = payload.candidate_id.unwrap();
        
        // Update candidate reward history
        candidate = state.candidates.get_mut(&candidate_id);
        candidate.rewards.push(reward);
        candidate.avg_reward = calculate_average(candidate.rewards);
        
        // Update best candidate if improved
        if candidate.avg_reward > state.best_candidate_avg_reward {
            state.best_candidate_id = Some(candidate_id.clone());
            state.best_candidate_avg_reward = candidate.avg_reward;
        }
        
        // Update TPE optimizer
        update_tpe_optimizer(&mut state.tpe, candidate, reward);
        
        // Record events
        record_event(&mut state, "mipro.reward.received", ...).await?;
        record_event(&mut state, "mipro.tpe.update", ...).await?;
    }
    
    // LATENCY: Save state (~300-2247ms, can be very slow!)
    state.bump_version();
    save_state(&state).await?;
    
    return Ok(state);
}
```

**Key Operations:**
- `load_state()`: DB query (200-740ms) + S3 fetch (90-180ms) = **290-920ms**
- `update_candidate_reward()`: In-memory update (~1ms)
- `update_tpe_optimizer()`: TPE computation (~1-10ms)
- `save_state()`: S3 upload (100-1954ms ⚠️) + DB update (200-550ms) = **300-2504ms**
- **Total**: ~590-3434ms per `apply_status` call (reward) + ~940-1880ms (done)

### `select_candidate_with_tpe(state)` - TPE-Based Selection

```rust
async fn select_candidate_with_tpe(state: &mut MiproSystemState) -> Option<Candidate> {
    // Need at least 2 candidates with rewards to use TPE
    candidates_with_rewards = state.candidates.values()
        .filter(|c| c.rewards.len() >= 2)
        .collect();
    
    if candidates_with_rewards.len() < 2 {
        return None; // Fall back to random selection
    }
    
    // Split candidates into "good" and "bad" based on quantile
    gamma = state.config.tpe_gamma.unwrap_or(0.25);
    quantile_threshold = calculate_quantile(candidates_with_rewards, gamma);
    
    good_candidates = candidates_with_rewards.iter()
        .filter(|c| c.avg_reward >= quantile_threshold)
        .collect();
    bad_candidates = candidates_with_rewards.iter()
        .filter(|c| c.avg_reward < quantile_threshold)
        .collect();
    
    // Build probability distributions for each dimension
    dimensions = extract_dimensions(state.candidates);
    for dimension in dimensions {
        good_values = extract_values(good_candidates, dimension);
        bad_values = extract_values(bad_candidates, dimension);
        
        // Fit Parzen estimators
        good_dist = fit_parzen_estimator(good_values);
        bad_dist = fit_parzen_estimator(bad_values);
        
        dimension_distributions[dimension] = (good_dist, bad_dist);
    }
    
    // Sample N candidates from search space
    n_candidates = state.config.tpe_n_candidates.unwrap_or(24);
    proposed_candidates = [];
    for _ in 0..n_candidates {
        // Sample from search space
        candidate_params = sample_from_search_space(state.config.search_space);
        
        // Calculate expected improvement (EI)
        good_prob = calculate_probability(candidate_params, dimension_distributions, "good");
        bad_prob = calculate_probability(candidate_params, dimension_distributions, "bad");
        
        ei = good_prob / (bad_prob + epsilon);
        proposed_candidates.push((candidate_params, ei));
    }
    
    // Select candidate with highest EI
    best_params = proposed_candidates.max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    // Find or create candidate matching these params
    candidate = find_or_create_candidate(state, best_params);
    
    return Some(candidate);
}
```

**Key Operations:**
- TPE computation: ~1-10ms (in-memory, no I/O)
- This is **NOT** a latency bottleneck

### `load_state(system_id)` - State Loading

```rust
async fn load_state(system_id: String) -> Result<MiproSystemState> {
    // LATENCY: DB query (~200-740ms)
    record = db.get_mipro_system(system_id).await?;
    
    state_ref = record.state_ref.ok_or("state_ref missing")?;
    
    // LATENCY: S3 fetch (~90-180ms)
    state_json = s3_client.get_object(state_ref).await?;
    
    state = deserialize_json(state_json)?;
    
    return Ok(state);
}
```

**Total Latency**: ~290-920ms per load

### `save_state(state)` - State Persistence

```rust
async fn save_state(state: &MiproSystemState) -> Result<MiproStateRef> {
    // Serialize state to JSON
    state_json = serialize_to_json(state)?;
    
    // LATENCY: S3 upload (~90-1954ms ⚠️)
    // Gets slower as state grows!
    s3_key = format!("mipro/state/{}/{}.json", state.system_id, state.version);
    s3_url = s3_client.put_object(s3_key, state_json).await?;
    
    // LATENCY: DB update (~200-550ms)
    db.upsert_mipro_system(
        state.system_id,
        state.org_id,
        state.status,
        state.mode,
        state.version,
        Some(s3_url)
    ).await?;
    
    return MiproStateRef { s3_url };
}
```

**Total Latency**: ~290-2504ms per save (can be very slow!)

---

## Per-Rollout Latency Breakdown (Pseudocode)

```python
# Demo script pseudocode
for seed in range(num_rollouts):
    rollout_id = f"trace_rollout_{seed}_{uuid}"
    
    # 1. START_ROLLOUT (when proxy receives request)
    #    LATENCY: ~620-1087ms
    #    - load_state: 290-920ms
    #    - select_candidate: 1-10ms
    #    - save_state: 290-545ms
    
    response = task_app.post("/rollout", {
        "trace_correlation_id": rollout_id,
        "policy": {
            "config": {
                "inference_url": f"{proxy_url}/{rollout_id}/chat/completions"
            }
        }
    })
    
    # 2. LLM API CALL
    #    LATENCY: ~1300-2800ms
    #    - Network RTT: 50-200ms
    #    - LLM processing: 500-3000ms
    #    - Network RTT back: 50-200ms
    
    reward = response.reward_info.outcome_reward
    candidate_id = response.metadata.mipro_candidate_id
    
    # 3. STATUS UPDATE (reward)
    #    LATENCY: ~630-2623ms
    #    - load_state: 290-920ms
    #    - update_rewards: 1ms
    #    - update_tpe: 1-10ms
    #    - save_state: 300-2504ms ⚠️
    
    backend.post(f"/api/.../mipro/systems/{system_id}/status", {
        "rollout_id": rollout_id,
        "status": "reward",
        "reward": reward,
        "candidate_id": candidate_id
    })
    
    # 4. STATUS UPDATE (done)
    #    LATENCY: ~940-1880ms
    #    - load_state: 290-920ms
    #    - save_state: 500-1200ms
    
    backend.post(f"/api/.../mipro/systems/{system_id}/status", {
        "rollout_id": rollout_id,
        "status": "done",
        "candidate_id": candidate_id
    })
    
    # TOTAL PER ROLLOUT: ~3490-7590ms (3.5-7.6 seconds)
```

---

## Database Query Latency

### PlanetScale Dashboard vs. Actual Latency

**Key Finding**: PlanetScale's dashboard shows **database-side query execution time** (p50: 0.1ms, p95: 0.2ms), but this doesn't include:
- Network round-trip time (RTT) to PlanetScale
- Connection pool acquisition time
- TLS handshake overhead
- Result serialization/deserialization
- Connection pooling overhead (PgBouncer)

### DB Queries in MIPRO Flow

#### 1. `get_mipro_system` (SELECT query)
**Query**: `SELECT to_jsonb(t) AS payload FROM (SELECT * FROM systems WHERE system_id = $1::uuid LIMIT 1) t`

**When called**: 
- During `load_state()` → `load_system_record()` → `get_mipro_system()`
- Called **twice per rollout** (once in `start_rollout`, once in `apply_status`)

**Measured latency breakdown** (from timing logs):
- **Total**: 200-740ms
- **Breakdown** (instrumentation added, investigating why logs not appearing):
  - Connection acquisition: ?ms (likely 50-200ms based on PlanetScale dashboard discrepancy)
  - Query execution: ~0.1ms (what PlanetScale shows)
  - Result parsing: ?ms (likely <10ms)
  - **Network RTT**: Likely 50-150ms per query (cross-region?)

**PlanetScale dashboard**: Shows this query as very fast (sub-millisecond), but actual end-to-end latency is 200-740ms.

#### 2. `upsert_mipro_system` (INSERT/UPDATE query)
**Query**: 
```sql
WITH inserted AS (
    INSERT INTO systems (system_id, org_id, status, mode, version, state_ref)
    VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6)
    ON CONFLICT (system_id) DO UPDATE SET
        org_id = EXCLUDED.org_id,
        status = EXCLUDED.status,
        mode = EXCLUDED.mode,
        version = EXCLUDED.version,
        state_ref = EXCLUDED.state_ref,
        updated_at = now()
    RETURNING *
) SELECT to_jsonb(inserted) AS payload FROM inserted
```

**When called**: 
- During `save_state()` → `upsert_mipro_system()`
- Called **twice per rollout** (once in `start_rollout`, once in `apply_status`)

**Measured latency breakdown** (from timing logs):
- **Total**: 200-550ms
- **Breakdown** (instrumentation added, investigating why logs not appearing):
  - Connection acquisition: ?ms (likely 50-200ms)
  - Query execution: ~0.1ms (what PlanetScale shows)
  - Result parsing: ?ms (likely <10ms)
  - **Network RTT**: Likely 50-150ms per query (cross-region?)

**PlanetScale dashboard**: Shows this query as very fast, but actual end-to-end latency is 200-550ms.

### DB Latency Components

Based on the discrepancy between PlanetScale dashboard (0.1ms) and actual measurements (200-740ms), the latency breakdown likely includes:

1. **Connection Pool Acquisition** (~50-200ms?)
   - Waiting for available connection from pool
   - PgBouncer connection checkout
   - Pool exhaustion delays

2. **Network RTT** (~50-150ms?)
   - Round-trip time to PlanetScale servers
   - Cross-region latency (if app and DB in different regions)
   - TLS handshake overhead

3. **Query Execution** (~0.1-1ms)
   - Actual database query execution (what PlanetScale shows)
   - Very fast for simple SELECT/INSERT queries

4. **Result Serialization** (~1-10ms?)
   - Converting PostgreSQL rows to JSONB
   - Parsing JSONB in Rust
   - Type conversions

5. **PgBouncer Overhead** (~10-50ms?)
   - Transaction pooling overhead
   - Protocol conversion
   - Connection multiplexing

### Hypothesis: Cross-Region Latency

If the Rust backend is running locally (e.g., San Francisco) and PlanetScale is in a different region (e.g., AWS us-east-1), network RTT could be 50-150ms per query, multiplied by 4 queries per rollout = 200-600ms just for network latency.

### Next Steps

1. ✅ Add detailed timing instrumentation to DB queries (connection acquisition, query execution, parsing)
2. ⏳ Run demo and capture DB timing breakdown
3. ⏳ Check PlanetScale region vs. app region
4. ⏳ Measure direct network latency to PlanetScale host
5. ⏳ Analyze connection pool metrics

---

## S3/Blob Storage Latency

### Operations

#### 1. `fetch_state` (S3 GET)
**When called**: During `load_state()` → `store.fetch_state()`
**Frequency**: Twice per rollout

**Measured latency**: 90-180ms per fetch
- Network latency to S3/Wasabi
- Object retrieval
- JSON deserialization

#### 2. `put_state` (S3 PUT)
**When called**: During `save_state()` → `store.put_state()`
**Frequency**: Twice per rollout

**Measured latency**: 90-1954ms per upload ⚠️
- **Normal**: 90-370ms
- **Worst case**: Up to 1954ms (almost 2 seconds!)
- Gets slower as state JSON grows larger

**State growth impact**:
- Early rollouts: ~90-125ms
- Later rollouts: ~100-1954ms
- State includes: all candidates, all rollouts, TPE state, events
- Each rollout adds more data → larger JSON → slower uploads

---

## LLM API Latency

### Operations

**When called**: During task app → proxy → LLM API call
**Frequency**: Once per rollout

**Measured latency**: 1300-2800ms per call
- Network round-trip: ~50-200ms
- LLM processing time: ~500-3000ms (model dependent)
- This is expected and unavoidable

---

## Complete Per-Rollout Breakdown

Based on actual measurements from 3 rollouts:

| Operation | Min | Max | Avg (est) | Notes |
|-----------|-----|-----|-----------|-------|
| **START_ROLLOUT** | | | | |
| └─ load_state (db) | 200ms | 740ms | ~350ms | PlanetScale query |
| └─ load_state (s3) | 90ms | 180ms | ~120ms | S3 fetch |
| └─ save_state (s3) | 90ms | 125ms | ~100ms | S3 upload |
| └─ save_state (db) | 200ms | 420ms | ~300ms | PlanetScale update |
| **LLM API CALL** | 1300ms | 2800ms | ~2000ms | Inference |
| **STATUS UPDATE (reward)** | | | | |
| └─ load_state (db) | 200ms | 740ms | ~350ms | PlanetScale query |
| └─ load_state (s3) | 90ms | 180ms | ~120ms | S3 fetch |
| └─ save_state (s3) | 100ms | 1954ms | ~300ms | ⚠️ Can be very slow! |
| └─ save_state (db) | 200ms | 550ms | ~300ms | PlanetScale update |
| **STATUS UPDATE (done)** | | | | |
| └─ load_state (db) | 200ms | 370ms | ~280ms | PlanetScale query |
| └─ load_state (s3) | 90ms | 180ms | ~120ms | S3 fetch |
| └─ save_state (s3) | 100ms | 370ms | ~200ms | S3 upload |
| └─ save_state (db) | 200ms | 430ms | ~300ms | PlanetScale update |
| **TOTAL PER ROLLOUT** | **3490ms** | **7590ms** | **~5500ms** | |

---

## Bottleneck Analysis

### Ranked by Impact

1. **S3 Uploads** (especially as state grows) - 90-1954ms ⚠️⚠️⚠️
   - Biggest bottleneck
   - Gets worse over time as state grows
   - Can take almost 2 seconds for a single upload

2. **Database Operations** - 200-740ms ⚠️⚠️
   - Much slower than PlanetScale dashboard suggests
   - Likely network/pool overhead, not query execution
   - Cross-region latency possible

3. **Sequential Processing** - Multiplies total time ⚠️
   - No parallelization
   - Each rollout waits for previous to complete

4. **LLM API Calls** - 1300-2800ms
   - Expected and unavoidable
   - Not the main bottleneck

5. **State Loading** - 320-915ms
   - Can be cached in memory
   - Currently loads full state twice per rollout

---

## Optimization Opportunities

### 1. In-Memory State Cache
- **Impact**: Eliminate load_state before each operation
- **Savings**: ~400-900ms per operation
- **Total per rollout**: Save ~800-1800ms
- **New total**: ~2700-5800ms per rollout

### 2. Batch State Updates
- **Impact**: Save state once per N rollouts instead of 2x per rollout
- **Savings**: ~300-2000ms per rollout (1 save instead of 2)
- **New total**: ~2400-4800ms per rollout

### 3. Parallel Rollouts
- **Impact**: Run 10 rollouts concurrently
- **Savings**: 100 rollouts in ~55 seconds instead of ~550 seconds
- **Speedup**: ~10x

### 4. DB Connection Pool Optimization
- **Impact**: Reduce connection acquisition time
- **Potential**: If pool wait is high, optimize pool size/config
- **Investigation needed**: Measure actual pool acquisition time

### 5. Cross-Region Optimization
- **Impact**: If app and DB are in different regions, move to same region
- **Potential savings**: 50-150ms per query × 4 queries = 200-600ms per rollout

### Combined Optimizations
- **Estimated**: ~30-60 seconds for 100 rollouts
- **vs current**: ~350-760 seconds
- **Speedup**: ~6-25x

---

## Next Steps

1. ✅ Add detailed DB query timing (connection acquisition, query execution, parsing)
2. ⏳ **IN PROGRESS**: Investigate why DB timing logs aren't appearing (code added but logs not showing)
   - Check if backend was rebuilt with new code
   - Verify eprintln! is working for other timing logs
   - Check if DB queries are taking a different code path
3. ⏳ Analyze DB timing breakdown once logs appear
4. ⏳ Check PlanetScale region vs. app region
5. ⏳ Measure direct network latency to PlanetScale host
6. ⏳ Analyze connection pool metrics
7. ⏳ Implement optimizations based on findings

## Current Status

- ✅ Comprehensive latency document created
- ✅ DB query timing instrumentation added to code
- ⚠️ DB timing logs not appearing yet (needs investigation)
- ✅ S3 timing working (90-1954ms measured)
- ✅ LLM timing working (1300-2800ms measured)
- ✅ Overall rollout timing working (3.5-7.6 seconds measured)


GOAL
```text
BEFORE (current: synchronous state load/save on every hop)

┌───────────────┐
│ Demo Script   │
└──────┬────────┘
       │ 1) POST /jobs
       ▼
┌───────────────────────────────┐
│ Rust Backend (API)            │
└──────┬────────────────────────┘
       │
       │ 2) POST /candidates
       ▼
┌───────────────────────────────┐
│ MIPRO Online Service          │
│ (state is “blob-of-truth”)    │
└──────┬───────────────┬────────┘
       │               │
       │ load_state()  │ save_state()
       ▼               ▼
┌───────────────┐   ┌───────────────┐
│ PlanetScale    │   │ Wasabi/S3      │
│ systems table  │   │ state.json     │
│ SELECT state_ref│  │ GET/PUT object │
└──────┬────────┘   └──────┬────────┘
       │                   │
       └─────(2x/rollout)──┘

Per rollout path:

┌───────────────┐
│ Task App      │
│ POST /rollout │
└──────┬────────┘
       │ calls inference_url (proxy)
       ▼
┌───────────────────────────────┐
│ MIPRO Proxy                   │
│ start_rollout()               │
│ - load_state  (DB+S3)         │
│ - select cand (TPE)           │
│ - save_state  (S3+DB)         │  <-- synchronous
└──────┬────────────────────────┘
       │ forward to LLM
       ▼
┌───────────────┐
│ LLM Provider   │
└──────┬────────┘
       │ response
       ▼
┌───────────────┐
│ Task App      │
│ computes reward
└──────┬────────┘
       │ POST /status (reward)
       ▼
┌───────────────────────────────┐
│ MIPRO Online Service          │
│ apply_status(reward)          │
│ - load_state (DB+S3)          │
│ - update agg/TPE              │
│ - save_state (S3+DB)          │  <-- synchronous, blob grows
└──────┬────────────────────────┘
       │ POST /status (done)
       ▼
┌───────────────────────────────┐
│ MIPRO Online Service          │
│ apply_status(done)            │
│ - load_state (DB+S3)          │
│ - save_state (S3+DB)          │  <-- synchronous
└───────────────────────────────┘
```

```text
AFTER (proposed: hot state cache + single-writer + batched snapshot uploads)

Key ideas:
- “Hot state” lives in-memory (per system_id), not in S3 on every call
- Writes are append-only events + write-behind snapshots
- S3 uploads are batched (every N events or T seconds)
- DB is metadata/pointer, not per-rollout

                   consistent-hash(system_id)
                         ┌───────────────────────────────┐
                         │ Router / LB                   │
                         │ (routes same system_id to     │
                         │ same worker shard)            │
                         └──────────────┬────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────┐
│ MIPRO System Actor Shard (single-writer per system_id)          │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ In-Memory Hot State Cache (small, stable size)            │  │
│  │ - candidates: ids + params + running aggregates           │  │
│  │ - tpe summary/stats                                      │  │
│  │ - active_candidate_id, counters, best_candidate_id        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Append-Only Event Log (fast)                              │  │
│  │ - rollout_started(system_id, rollout_id, candidate_id)    │  │
│  │ - reward_received(system_id, rollout_id, candidate_id, r) │  │
│  │ - (optional) done                                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Snapshotter (write-behind)                                │  │
│  │ - every N events or every T seconds                       │  │
│  │ - serialize hot state (binary+zstd)                       │  │
│  │ - PUT snapshot to S3                                      │  │
│  │ - update systems table pointer/version in DB              │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────┬───────────────────────────────┬────────────────┘
                │                               │
                │ snapshot (batched)            │ metadata pointer (batched)
                ▼                               ▼
        ┌───────────────┐               ┌───────────────────┐
        │ Wasabi/S3      │               │ PlanetScale        │
        │ snapshots/     │               │ systems table      │
        │ system/version │               │ (state_ref, ver)   │
        └───────────────┘               └───────────────────┘


Per rollout path (critical path has ~0 DB/S3):

┌───────────────┐
│ Task App      │
│ POST /rollout │
└──────┬────────┘
       │ calls inference_url (proxy)
       ▼
┌───────────────────────────────┐
│ MIPRO Proxy / Actor           │
│ start_rollout()               │
│ - read hot state from memory  │
│ - pick candidate (TPE)        │
│ - append event: rollout_started
│ - RETURN candidate_id fast    │
│ (NO S3 PUT, NO DB query)      │
└──────┬────────────────────────┘
       │ forward to LLM
       ▼
┌───────────────┐
│ LLM Provider   │
└──────┬────────┘
       │ response
       ▼
┌───────────────┐
│ Task App      │
│ computes reward
└──────┬────────┘
       │ POST /status (reward_done)  (preferred)
       ▼
┌───────────────────────────────┐
│ MIPRO Actor                   │
│ apply_status(reward_done)     │
│ - update aggregates/TPE in mem│
│ - append event: reward_received
│ - RETURN fast                 │
│ (NO S3 PUT, NO DB query)      │
└───────────────────────────────┘

Background (amortized):
- every N events or T seconds:
  - write snapshot to S3
  - update DB pointer once
```

```text
OPTIONAL: “batched S3 upload + cache” without full actor routing (smaller change)

┌───────────────────────────────┐
│ Rust Backend (per instance)   │
│ - LRU cache: system_id->state │
│ - dirty flag + debounce timer │
│ - load_state on cache miss    │
│ - save_state only on flush    │
└──────┬───────────────┬────────┘
       │               │
cache miss:        periodic flush:
DB+S3 GET          S3 PUT + DB UPSERT
       ▼               ▼
 PlanetScale          Wasabi/S3

Pros: easiest drop-in.
Cons: correctness gets tricky across multiple instances unless you add:
- sticky routing per system_id, or
- distributed lock/lease, or
- “single-writer” via actor shards.
```
