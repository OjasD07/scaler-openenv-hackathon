from __future__ import annotations

import argparse
from typing import Any

import requests
from fastapi.testclient import TestClient

from email_triage_env.baseline import run_baseline
from email_triage_env.server.app import app


def _validate_local() -> dict[str, Any]:
    client = TestClient(app)
    reset = client.post("/reset", json={"task_id": 1, "seed": 7})
    reset.raise_for_status()
    manifest = client.get("/manifest")
    manifest.raise_for_status()
    baseline = run_baseline().model_dump()
    return {
        "mode": "local",
        "reset_ok": True,
        "manifest": manifest.json(),
        "baseline": baseline,
    }


def _validate_remote(base_url: str) -> dict[str, Any]:
    session = requests.Session()
    reset = session.post(f"{base_url}/reset", json={"task_id": 1, "seed": 7}, timeout=30)
    reset.raise_for_status()
    tasks = session.get(f"{base_url}/tasks", timeout=30)
    tasks.raise_for_status()
    manifest = session.get(f"{base_url}/manifest", timeout=30)
    manifest.raise_for_status()
    return {
        "mode": "remote",
        "reset_ok": True,
        "tasks": tasks.json(),
        "manifest": manifest.json(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the OpenEnv email triage submission.")
    parser.add_argument("--base-url", help="Optional live base URL to validate instead of the local app.")
    args = parser.parse_args()

    result = _validate_remote(args.base_url.rstrip("/")) if args.base_url else _validate_local()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
