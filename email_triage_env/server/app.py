from __future__ import annotations

import logging
import os
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException

from ..baseline import run_baseline
from ..models import EmailAction, GradeRequest, ResetRequest
from ..tasks import get_email_by_id
from .environment import EmailTriageEnvironment

logging.basicConfig(
    level=getattr(logging, os.getenv("EMAIL_TRIAGE_LOG_LEVEL", "WARNING").upper(), logging.WARNING),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="OpenEnv Email Triage Environment", version="2.2.0")
env = EmailTriageEnvironment()
SCORE_FLOOR = 0.001
SCORE_CEILING = 0.999
DEFAULT_STEP_ACTION: dict[str, Any] = {
    "category": "support",
    "priority": "medium",
    "department": "support_team",
    "action": "reply",
    "use_tool": None,
    "tool_input": None,
}


def _normalize_step_payload(payload: dict[str, Any] | None) -> EmailAction:
    raw = dict(payload or {})
    nested = raw.get("action")
    if isinstance(nested, dict):
        raw = {**nested, **{key: value for key, value in raw.items() if key != "action"}}

    normalized = dict(DEFAULT_STEP_ACTION)
    for key in DEFAULT_STEP_ACTION:
        if key in raw and raw[key] is not None:
            normalized[key] = raw[key]

    return EmailAction.model_validate(normalized)


def _strict_score(value: float) -> float:
    score = round(float(value), 3)
    if score <= SCORE_FLOOR:
        return SCORE_FLOOR
    if score >= SCORE_CEILING:
        return SCORE_CEILING
    return score


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "email_triage_env", "status": "ready"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "ready": True,
        "dataset_size": len(env.dataset),
        "task_count": len(env.tasks()),
        "supported_tools": env.manifest()["supported_tools"],
        "version": app.version,
    }


@app.get("/tasks")
def list_tasks() -> dict[str, Any]:
    return env.tasks_payload()


@app.get("/manifest")
def manifest() -> dict[str, Any]:
    return env.manifest()


@app.post("/reset")
def reset(request: ResetRequest | None = Body(default=None)) -> dict[str, Any]:
    try:
        observation = env.reset(
            seed=request.seed if request is not None else None,
            episode_id=request.episode_id if request is not None else None,
            task_id=request.task_id if request is not None else None,
            email_id=request.email_id if request is not None else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid request: {exc}") from exc
    return {"observation": observation.model_dump(), "state": env.state().model_dump()}


@app.post("/step")
def step(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    try:
        env.ensure_initialized()
        result = env.step(_normalize_step_payload(payload))
        return result.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid action: {exc}") from exc


@app.get("/state")
def state() -> dict[str, Any]:
    env.ensure_initialized()
    return env.state().model_dump()


@app.post("/grader")
def grader(request: GradeRequest) -> dict[str, Any]:
    try:
        env.ensure_initialized()
        if request.email_data is not None:
            email_data = request.email_data
        elif request.email_id is not None:
            email_data = get_email_by_id(request.email_id)
        else:
            email_data = env.state().email_data

        task_id = request.task_id or env.state().task_id
        score, breakdown = env.grade(request.action, email_data=email_data, task_id=task_id)
        score = _strict_score(float(score))
        return {
            "score": score,
            "details": {
                "breakdown": breakdown,
                "task_id": task_id,
                "email_id": email_data.email_id,
            },
        }
    except ValueError as exc:
        return {
            "score": 0.0,
            "details": {
                "error": f"Invalid grader request: {exc}",
            },
        }
    except Exception as exc:
        return {
            "score": 0.0,
            "details": {
                "error": str(exc),
            },
        }


@app.get("/baseline")
def baseline() -> dict[str, Any]:
    return run_baseline().model_dump()


@app.get("/episode_log")
def episode_log() -> dict[str, Any]:
    env.ensure_initialized()
    return env.episode_log()


@app.get("/sample_action")
def sample_action() -> dict[str, Any]:
    env.ensure_initialized()
    return env.sample_action()


def main() -> None:
    uvicorn.run(
        "email_triage_env.server.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "7860")),
    )
