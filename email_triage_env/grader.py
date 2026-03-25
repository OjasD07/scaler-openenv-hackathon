from __future__ import annotations

from .models import EmailAction, EmailExample
from .server.environment import EmailTriageEnvironment


def grade_action(
    action: EmailAction,
    *,
    email_data: EmailExample | None = None,
    task_id: int | None = None,
) -> float:
    env = EmailTriageEnvironment()
    if email_data is not None:
        env.reset(task_id=task_id or 3, email_id=email_data.email_id)
    else:
        env.reset(task_id=task_id or 3)
    score, _ = env.grade(action=action, email_data=email_data, task_id=task_id)
    return score
