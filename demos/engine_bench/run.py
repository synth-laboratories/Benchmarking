#!/usr/bin/env python3
"""
Entry point for the EngineBench Task App server.

This module allows running the task app via: python -m engine_bench.run

When deployed as a Docker container in an Environment Pool, the pool will
call the /health, /info, and /rollout endpoints.
"""
import os
import uvicorn

# Import the app from localapi_engine_bench
from engine_bench.localapi_engine_bench import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"[engine_bench] Starting task app server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
