# SynthTunnel WebSocket 404 Issue

## Problem

When attempting to use SynthTunnel (the relay-based tunnel that doesn't require cloudflared), the WebSocket agent connection fails with a 404 error.

## Environment

- synth-ai SDK: 0.7.7 (built from source at `vendor/synth-ai`)
- Date: 2026-01-31

## Steps to Reproduce

1. Create a SynthTunnel lease via the API:

```python
from synth_ai.core.tunnels.synth_tunnel import SynthTunnelClient, get_client_instance_id

client = SynthTunnelClient(api_key)
lease = await client.create_lease(
    client_instance_id=get_client_instance_id(),
    local_host="127.0.0.1",
    local_port=8030,
)
```

2. The lease is created successfully and returns:

```json
{
  "lease_id": "...",
  "public_url": "https://infra-api.usesynth.ai/s/rt_...",
  "agent_connect": {
    "transport": "ws",
    "url": "wss://infra-api.usesynth.ai/agent",
    "agent_token": "eyJ..."
  }
}
```

3. When the SDK tries to connect to the WebSocket agent URL:

```python
async with session.ws_connect(
    "wss://infra-api.usesynth.ai/agent",
    headers={"Authorization": f"Bearer {agent_token}"},
) as ws:
    ...
```

4. The connection fails with:

```
WSServerHandshakeError: 404, message='Invalid response status', url='wss://infra-api.usesynth.ai/agent'
```

## Investigation

- The relay health endpoint works: `https://infra-api.usesynth.ai/health` returns 200
- The `/agent` endpoint returns 404 for both HTTP GET and WebSocket upgrade requests
- Tested both `infra-api.usesynth.ai` and `st.usesynth.ai` - same result
- The lease creation succeeds, so the backend API is working
- The issue is specifically with the WebSocket relay endpoint

## Workaround

Use ngrok to expose the local task app instead:

```bash
ngrok http 8030 --url your-subdomain.ngrok-free.app
```

Then set `TASK_APP_URL` environment variable to the ngrok URL.

## Next Steps

- Contact Synth support to report the issue
- Check if there's a different relay endpoint that should be used
- Check if there are additional configuration steps required
