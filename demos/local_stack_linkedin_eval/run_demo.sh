#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -z "${INTERCEPTOR_BASE_OVERRIDE:-}" ]]; then
  echo "ERROR: INTERCEPTOR_BASE_OVERRIDE is not set."
  echo "Start a Cloudflare tunnel to localhost:8080 and export it, e.g.:"
  echo "  export INTERCEPTOR_BASE_OVERRIDE=\"https://<tunnel>.trycloudflare.com\""
  exit 1
fi

export SYNTH_BACKEND_URL="${SYNTH_BACKEND_URL:-http://127.0.0.1:8080}"
export SYNTH_API_KEY="${SYNTH_API_KEY:-sk_dev_00000000000000000000000000000001}"

cd "${ROOT_DIR}/Benchmarking/demos/linkedin_bench"
uv run python test_interceptor_eval.py
