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

Use the returned `task_app_url` in GEPA/MIPRO configs. Harbor auth uses `SYNTH_API_KEY`
as the task app API key.
