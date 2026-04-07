from __future__ import annotations

import json
import os
from typing import Any

import requests
from openai import OpenAI

from email_triage_env.models import EmailAction


TASKS = (1, 2, 3)
TASK_NAMES = {1: "easy", 2: "medium", 3: "hard"}
DEFAULT_MODEL_NAME = "gpt-4.1-mini"
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
SEED = 7
REQUEST_TIMEOUT = 30


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _make_session(hf_token: str) -> requests.Session:
    session = requests.Session()
    if hf_token:
        session.headers.update({"Authorization": f"Bearer {hf_token}"})
    return session


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _fallback_action(email: dict[str, Any]) -> EmailAction:
    text = " ".join(
        str(email.get(field, ""))
        for field in ("sender", "subject", "email_text", "noisy_text")
    ).lower()

    if any(marker in text for marker in ["unsubscribe", "gift card", "limited time offer", "click this link", "spam"]):
        return EmailAction(category="spam", priority="low", department="ignore", action="archive")
    if any(marker in text for marker in ["charged twice", "refund", "invoice", "payment", "billing", "bank"]):
        return EmailAction(category="billing", priority="high" if "urgent" in text or "asap" in text else "medium", department="finance", action="reply")
    if any(marker in text for marker in ["pricing", "proposal", "demo", "enterprise", "trial", "contract", "seats"]):
        return EmailAction(category="sales", priority="medium", department="sales_team", action="forward")
    if any(marker in text for marker in ["internal", "policy", "approval", "meeting", "review", "launch"]):
        return EmailAction(category="internal", priority="medium", department="ignore", action="archive")
    return EmailAction(category="support", priority="high" if any(marker in text for marker in ["urgent", "blocked", "outage", "down", "cannot", "asap"]) else "medium", department="support_team", action="reply")


def _predict_action(client: OpenAI | None, model_name: str, task_id: int, observation: dict[str, Any]) -> EmailAction:
    email = observation["current_email"]
    prompt = {
        "task_id": task_id,
        "current_email": {
            "email_id": email.get("email_id"),
            "sender": email.get("sender"),
            "subject": email.get("subject"),
            "email_text": email.get("email_text"),
            "difficulty": email.get("difficulty"),
            "thread_id": email.get("thread_id"),
            "tags": email.get("tags", []),
        },
        "inbox_summary": observation.get("inbox_summary", []),
        "allowed_values": {
            "category": ["spam", "support", "billing", "sales", "internal"],
            "priority": ["low", "medium", "high"],
            "department": ["support_team", "sales_team", "finance", "ignore"],
            "action": ["reply", "forward", "archive", "escalate"],
        },
    }

    if client is None:
        return _fallback_action(email)

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an OpenEnv email triage policy. "
                        "Return only minified JSON with keys category, priority, department, action, use_tool, tool_input. "
                        "Use null for optional fields when not needed."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Choose the best structured action for the email below.\n"
                        "Be deterministic and conservative.\n"
                        f"{json.dumps(prompt, separators=(',', ':'))}"
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or ""
        parsed = _extract_json(content)
        return EmailAction.model_validate(parsed)
    except Exception:
        return _fallback_action(email)


def _reset_episode(session: requests.Session, base_url: str, task_id: int) -> dict[str, Any]:
    response = session.post(
        f"{base_url}/reset",
        json={"task_id": task_id, "seed": SEED},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if "observation" not in payload:
        raise RuntimeError("Reset response missing observation")
    return payload["observation"]


def _step_episode(
    session: requests.Session,
    base_url: str,
    action: EmailAction,
) -> dict[str, Any]:
    response = session.post(
        f"{base_url}/step",
        json={"action": action.model_dump(exclude_none=True)},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _empty_result(task_id: int, error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task_id": task_id,
        "task_name": TASK_NAMES.get(task_id, str(task_id)),
        "steps": 0,
        "total_reward": 0.0,
        "average_reward": 0.0,
    }
    if error:
        result["error"] = error
    return result


def run_task(
    session: requests.Session,
    client: OpenAI | None,
    base_url: str,
    model_name: str,
    task_id: int,
) -> dict[str, Any]:
    try:
        observation = _reset_episode(session, base_url, task_id)
    except Exception as exc:
        return _empty_result(task_id, f"reset failed: {exc}")

    total_reward = 0.0
    step_count = 0
    done = False

    while not done:
        try:
            action = _predict_action(client, model_name, task_id, observation)
            result = _step_episode(session, base_url, action)
        except Exception as exc:
            return _empty_result(task_id, f"step failed: {exc}")

        try:
            total_reward += float(result.get("reward", 0.0))
            step_count += 1
            done = bool(result.get("done", False))
            observation = result["observation"]
        except Exception as exc:
            return _empty_result(task_id, f"response parsing failed: {exc}")

    return {
        "task_id": task_id,
        "task_name": TASK_NAMES.get(task_id, str(task_id)),
        "steps": step_count,
        "total_reward": round(total_reward, 3),
        "average_reward": round(total_reward / step_count, 3) if step_count else 0.0,
    }


def main() -> int:
    base_url = API_BASE_URL.rstrip("/")
    model_name = MODEL_NAME
    hf_token = HF_TOKEN

    session = _make_session(hf_token)
    client = OpenAI(api_key=hf_token) if hf_token else None
    _ = LOCAL_IMAGE_NAME

    print("START")

    results = []
    for task_id in TASKS:
        result = run_task(session, client, base_url, model_name, task_id)
        results.append(result)
        status = "error" if "error" in result else "ok"
        print(
            "STEP "
            f"task_id={result['task_id']} "
            f"task_name={result['task_name']} "
            f"status={status} "
            f"steps={result['steps']} "
            f"total_reward={result['total_reward']} "
            f"average_reward={result['average_reward']}"
        )

    overall_total = sum(item["total_reward"] for item in results)
    overall_steps = sum(item["steps"] for item in results)

    print(
        "END "
        f"tasks={len(results)} "
        f"overall_total_reward={round(overall_total, 3)} "
        f"overall_average_reward={round(overall_total / len(results), 3) if results else 0.0} "
        f"overall_average_step_reward={round(overall_total / overall_steps, 3) if overall_steps else 0.0}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
