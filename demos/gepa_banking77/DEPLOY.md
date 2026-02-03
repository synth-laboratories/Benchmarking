# Banking77 LocalAPI Deploy

This deploys the `localapi_banking77.py` task app to the cloud and emits a stable `task_app_url`.

From the synth-ai repo root:

```bash
export SYNTH_API_KEY=sk_live_...

synth localapi deploy \
  --name banking77-localapi \
  --app localapi_banking77:app \
  --dockerfile Dockerfile.banking77-localapi \
  --context demos/gepa_banking77 \
  --wait
```

Use the returned `task_app_url` in GEPA/MIPRO configs. For managed LocalAPI deployments,
health checks can authenticate with `SYNTH_API_KEY`, while the backend supplies
`ENVIRONMENT_API_KEY` to the task app for rollout auth.

For managed LocalAPI (Modal), the backend injects `ENVIRONMENT_API_KEY` into the task app.
If the key is missing in backend credentials, the deploy helper will auto-create it
using your `SYNTH_API_KEY`.

Note: the Dockerfile expects `synth_ai/` and `synth_ai_core/assets` in the build context
(from the repo's `synth-ai` package). The `run_demo_async.py` script stages a temporary
context that includes those files automatically.
