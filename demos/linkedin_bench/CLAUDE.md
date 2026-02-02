# Browser Agent GEPA

## Infrastructure (Railway)

Railway project: `refreshing-healing`

### Service URLs

| Service | Dev | Prod |
|---------|-----|------|
| Python Backend (API gateway) | `https://api-dev.usesynth.ai` | `https://api.usesynth.ai` |
| Rust Backend (infra-api) | `https://infra-api-dev.usesynth.ai` | `https://infra-api.usesynth.ai` |

### Railway Service Names

| Service | Dev | Prod |
|---------|-----|------|
| Python Backend | `api-dev` | `api` |
| Rust Backend | `infra-api-dev` | `infra-api` |

### Architecture

- The **Python backend** (`api.usesynth.ai` / `api-dev.usesynth.ai`) is the API gateway. All client requests (SDK, `run_gepa.py`) should target this URL as `backend_url`.
- The **Rust backend** (`infra-api.usesynth.ai` / `infra-api-dev.usesynth.ai`) handles GEPA/MIPRO execution, interceptor, SynthTunnel relay, and graph service.
- Credentials (ENVIRONMENT_API_KEY) are stored in the **Python backend's** database. The Rust backend fetches them via `EnvKeyClient` → `GET {python_backend}/api/v1/env-keys`.
- `ensure_localapi_auth()` in the SDK uploads credentials to whatever `backend_url` is provided. This MUST be the Python backend URL so credentials end up in the correct store.

### Running GEPA

```bash
# Dev
uv run python run_gepa.py --backend-url https://api-dev.usesynth.ai --generations 2 --budget 20 --timeout 120

# Prod
uv run python run_gepa.py --generations 2 --budget 20 --timeout 120
```

**Important**: `--backend-url` must point to the **Python backend** (not the Rust backend directly) so that credentials are stored and fetched from the same place.
