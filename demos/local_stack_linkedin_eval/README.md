# Local Stack LinkedIn Eval Demo (SynthTunnel + Interceptor)

This demo shows a **local-stack** eval run of the LinkedIn benchmark while:
- exposing the **task app** via **SynthTunnel (websocket-based)**, and
- routing **LLM traffic through the local interceptor** using a public tunnel.

It validates:
1) job submission through local Rust backend  
2) cloud kernel → local interceptor routing  
3) trace capture + hydration

---

## Prereqs

- `cloudflared` installed (`brew install cloudflared`)
- Local infra running (Postgres, Redis, MinIO, Temporal)
- `KERNEL_API_KEY` + `ANTHROPIC_API_KEY` exported
- Local dev API key seeded (`sk_dev_...`)

---

## 1) Start infra + tunnel

```bash
cd backend
./local_dev.sh up
```

In a **separate terminal**, start a quick tunnel to the Rust backend:

```bash
cd backend
./local_dev.sh tunnel 8080
```

Copy the `https://*.trycloudflare.com` URL it prints.

---

## 2) Start backend services with public URL

```bash
cd backend
export EXTERNAL_BACKEND_URL="https://<YOUR-TUNNEL>.trycloudflare.com"
./local_dev.sh services up
```

> This ensures the backend generates an interceptor URL reachable by cloud workers.

---

## 3) Run the LinkedIn eval (1 seed)

```bash
cd Benchmarking/demos/linkedin_bench

export SYNTH_API_KEY="sk_dev_00000000000000000000000000000001"
export SYNTH_BACKEND_URL="http://127.0.0.1:8080"

# Force the task app to use the public interceptor URL from the tunnel
export INTERCEPTOR_BASE_OVERRIDE="https://<YOUR-TUNNEL>.trycloudflare.com"

uv run python test_interceptor_eval.py
```

This script:
- starts the local task app
- opens a SynthTunnel (websocket)
- submits a 1-seed eval
- verifies traces

---

## 4) Verify traces

```bash
curl -s \
  -H "Authorization: Bearer $SYNTH_API_KEY" \
  "http://127.0.0.1:8080/api/eval/jobs/<JOB_ID>/traces/list" | python3 -m json.tool
```

Expected output includes `trace_id`, `trace_s3_key`, and `trace_s3_url`.

---

## Notes

- **SynthTunnel uses websocket transport** by default (`TunneledLocalAPI.create(...)`).
- The `INTERCEPTOR_BASE_OVERRIDE` path is necessary for cloud kernels to reach your local interceptor.
- If the tunnel URL changes, re-export it and restart the backend services.
