from __future__ import annotations

from typing import Any

import requests


class OpenEnvClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self, *, task_id: int = 3, email_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"task_id": task_id}
        if email_id is not None:
            payload["email_id"] = email_id
        response = requests.post(f"{self.base_url}/reset", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}/step", json={"action": action}, timeout=30)
        response.raise_for_status()
        return response.json()

    def state(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/state", timeout=30)
        response.raise_for_status()
        return response.json()

    def tasks(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/tasks", timeout=30)
        response.raise_for_status()
        return response.json()

    def grader(self, action: dict[str, Any], *, email_id: str | None = None, task_id: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": action}
        if email_id is not None:
            payload["email_id"] = email_id
        if task_id is not None:
            payload["task_id"] = task_id
        response = requests.post(f"{self.base_url}/grader", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

