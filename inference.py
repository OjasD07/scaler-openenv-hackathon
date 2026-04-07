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
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = os.getenv("BENCHMARK", "email-triage-env")
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


def _format_action(action: EmailAction) -> str:
    return json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"))


def _log_start(task_name: str, model_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={model_name}", flush=True)


def _log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def run_task(
    session: requests.Session,
    client: OpenAI | None,
    base_url: str,
    model_name: str,
    task_id: int,
) -> dict[str, Any]:
    task_name = TASK_NAMES.get(task_id, str(task_id))
    rewards: list[float] = []
    step_count = 0
    done = False
    success = False
    score = 0.0

    _log_start(task_name, model_name)
    try:
        observation = _reset_episode(session, base_url, task_id)
        while not done:
            action = _predict_action(client, model_name, task_id, observation)
            action_str = _format_action(action)

            try:
                result = _step_episode(session, base_url, action)
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
                observation = result["observation"]
                rewards.append(reward)
                step_count += 1
                _log_step(step_count, action_str, reward, done, None)
            except Exception as exc:
                _log_step(step_count + 1, action_str, 0.0, False, str(exc))
                break

        score = round(sum(rewards) / len(rewards), 2) if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success = done and not rewards or (done and score >= 0.1)
        return {
            "task_id": task_id,
            "task_name": task_name,
            "steps": step_count,
            "total_reward": round(sum(rewards), 3),
            "average_reward": round(sum(rewards) / step_count, 3) if step_count else 0.0,
            "score": score,
            "success": success,
        }
    except Exception as exc:
        score = 0.0
        return {
            "task_id": task_id,
            "task_name": task_name,
            "steps": step_count,
            "total_reward": round(sum(rewards), 3),
            "average_reward": round(sum(rewards) / step_count, 3) if step_count else 0.0,
            "score": score,
            "success": False,
            "error": str(exc),
        }
    finally:
        _log_end(success=success, steps=step_count, score=score, rewards=rewards)


def main() -> int:
    base_url = _require_env("API_BASE_URL").rstrip("/")
    api_key = _require_env("API_KEY")
    model_name = MODEL_NAME

    session = requests.Session()
    client = OpenAI(base_url=base_url, api_key=api_key)
    _ = LOCAL_IMAGE_NAME

    for task_id in TASKS:
        run_task(session, client, base_url, model_name, task_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
