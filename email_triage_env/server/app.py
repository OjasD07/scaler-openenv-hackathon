from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException

from ..baseline import run_baseline
from ..models import EmailAction, GradeRequest, ResetRequest, StepRequest
from ..tasks import get_email_by_id
from .environment import EmailTriageEnvironment

logging.basicConfig(
    level=getattr(logging, os.getenv("EMAIL_TRIAGE_LOG_LEVEL", "WARNING").upper(), logging.WARNING),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="OpenEnv Email Triage Environment", version="1.0.0")
env = EmailTriageEnvironment()


def _resolve_step_action(request: StepRequest) -> EmailAction:
    if isinstance(request.action, EmailAction):
        return request.action

    missing_fields = [
        field_name
        for field_name in ("category", "priority", "department", "action")
        if getattr(request, field_name) is None
    ]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise HTTPException(status_code=400, detail=f"Missing required step fields: {missing}")

    return EmailAction(
        category=request.category,
        priority=request.priority,
        department=request.department,
        action=request.action,
        use_tool=request.use_tool,
        tool_input=request.tool_input,
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "email_triage_env", "status": "ready"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "ready": True,
        "dataset_size": len(env.dataset),
    }


@app.get("/tasks")
def list_tasks() -> dict[str, Any]:
    return env.tasks_payload()


@app.post("/reset")
def reset(request: ResetRequest) -> dict[str, Any]:
    try:
        observation = env.reset(task_id=request.task_id, email_id=request.email_id, seed=request.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid request: {exc}") from exc
    return {"observation": observation.model_dump(), "state": env.state().model_dump()}


@app.post("/step")
def step(request: StepRequest) -> dict[str, Any]:
    try:
        env.ensure_initialized()
        result = env.step(_resolve_step_action(request))
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
        return {
            "score": score,
            "breakdown": breakdown,
            "task_id": task_id,
            "email_id": email_data.email_id,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid grader request: {exc}") from exc


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
