from __future__ import annotations

import random
from typing import Any

from fastapi.testclient import TestClient

from email_triage_env.server.app import app


def random_action(rng: random.Random) -> dict[str, Any]:
    return {
        "category": rng.choice(["spam", "support", "billing", "sales", "internal"]),
        "priority": rng.choice(["low", "medium", "high"]),
        "department": rng.choice(["support_team", "sales_team", "finance", "ignore"]),
        "action": rng.choice(["reply", "forward", "archive", "escalate"]),
    }


def main() -> int:
    rng = random.Random(7)
    client = TestClient(app)

    reset_response = client.post("/reset", json={})
    assert reset_response.status_code == 200, reset_response.text
    observation = reset_response.json()["observation"]
    print(f"reset -> {observation['current_email']['email_id']}")

    rewards: list[float] = []
    done = False
    steps = 0
    while not done and steps < 20:
        response = client.post("/step", json={"action": random_action(rng)})
        assert response.status_code == 200, response.text
        payload = response.json()
        reward = float(payload["reward"])
        rewards.append(reward)
        done = bool(payload["done"])
        steps += 1
        print(f"step {steps}: reward={reward}")

    manifest = client.get("/manifest")
    assert manifest.status_code == 200, manifest.text
    manifest_payload = manifest.json()
    print(f"manifest -> task_count={manifest_payload['task_count']}, tools={manifest_payload['supported_tools']}")
    print(f"done={done}, steps={steps}, rewards={rewards}")
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
