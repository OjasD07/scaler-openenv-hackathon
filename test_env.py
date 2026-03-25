from __future__ import annotations

import random
import sys
from typing import Any

import requests


BASE_URL = "http://127.0.0.1:8000"


def random_action(rng: random.Random) -> dict[str, Any]:
    return {
        "category": rng.choice(["spam", "support", "billing", "sales", "internal"]),
        "priority": rng.choice(["low", "medium", "high"]),
        "department": rng.choice(["support_team", "sales_team", "finance", "ignore"]),
        "action": rng.choice(["reply", "forward", "archive", "escalate"]),
    }


def main() -> int:
    rng = random.Random(7)
    session = requests.Session()

    reset_response = session.post(f"{BASE_URL}/reset", json={}, timeout=30)
    reset_response.raise_for_status()
    observation = reset_response.json()["observation"]
    print(f"reset -> {observation['current_email']['email_id']}")

    rewards: list[float] = []
    done = False
    steps = 0
    while not done and steps < 20:
        response = session.post(f"{BASE_URL}/step", json={"action": random_action(rng)}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        reward = float(payload["reward"])
        rewards.append(reward)
        done = bool(payload["done"])
        steps += 1
        print(f"step {steps}: reward={reward}")

    print(f"done={done}, steps={steps}, rewards={rewards}")
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
