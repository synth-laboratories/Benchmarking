#!/bin/bash
# Retry GEPA run, waiting for Anthropic API credits to become available.
# Usage: ./retry_gepa.sh

set -euo pipefail
cd "$(dirname "$0")"

set -a
source .env 2>/dev/null || true
set +a

MAX_RETRIES=144  # 12 hours of retries (144 * 5 min)
RETRY_INTERVAL=300  # 5 minutes between checks

check_credits() {
    uv run python -c "
import anthropic, os
client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
try:
    resp = client.messages.create(model='claude-haiku-3-5-20241022', max_tokens=5, messages=[{'role':'user','content':'hi'}])
    print('OK')
except Exception as e:
    if 'credit balance' in str(e).lower():
        print('NO_CREDITS')
    else:
        print('ERROR: ' + str(e))
" 2>/dev/null
}

echo "============================================="
echo "GEPA Retry Script"
echo "Waiting for Anthropic API credits..."
echo "============================================="

for i in $(seq 1 $MAX_RETRIES); do
    STATUS=$(check_credits)
    if [ "$STATUS" = "OK" ]; then
        echo "[$(date)] Credits available! Starting GEPA run..."
        exec uv run python run_gepa.py \
            --backend-url https://api-dev.usesynth.ai \
            --generations 3 \
            --budget 20 \
            --timeout 120
    elif [ "$STATUS" = "NO_CREDITS" ]; then
        echo "[$(date)] No credits yet (attempt $i/$MAX_RETRIES). Retrying in ${RETRY_INTERVAL}s..."
        sleep $RETRY_INTERVAL
    else
        echo "[$(date)] Unexpected status: $STATUS"
        sleep $RETRY_INTERVAL
    fi
done

echo "[$(date)] Gave up after $MAX_RETRIES attempts."
exit 1
