from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from ..models import (
    BaselineScores,
    EmailAction,
    EmailExample,
    EmailObservation,
    EnvironmentState,
    GraderBreakdown,
    StepResponse,
)
from ..tasks import DATASET, TASKS, get_email_by_id, get_task_definition

IMPORTANT_CATEGORIES = {"support", "billing", "sales", "internal"}
AVAILABLE_TOOLS = ["lookup_order", "check_payment", "get_user_history"]
CATEGORY_SIMILARITY: dict[tuple[str, str], float] = {
    ("billing", "support"): 0.5,
    ("support", "billing"): 0.5,
    ("sales", "internal"): 0.35,
    ("internal", "sales"): 0.35,
    ("support", "internal"): 0.25,
    ("internal", "support"): 0.25,
    ("billing", "internal"): 0.2,
    ("internal", "billing"): 0.2,
    ("sales", "support"): 0.15,
    ("support", "sales"): 0.15,
}

logger = logging.getLogger(__name__)
SCORE_FLOOR = 0.001
SCORE_CEILING = 0.999


@dataclass(slots=True)
class _RuntimeState:
    inbox: list[EmailExample] = field(default_factory=list)
    current_email_index: int = 0
    processed: list[bool] = field(default_factory=list)
    task_id: int = 3
    step_count: int = 0
    episode_history: list[dict[str, Any]] = field(default_factory=list)
    pending_tool_result: dict[str, Any] | None = None
    done: bool = False
    history: list[str] = field(default_factory=list)


class EmailTriageEnvironment:
    """Deterministic OpenEnv-style email triage environment."""

    def __init__(self) -> None:
        self.dataset = list(DATASET)
        self._cursor = 0
        self._episode_counter = 0
        self._runtime = _RuntimeState()

    def _strict_score(self, value: float) -> float:
        score = round(float(value), 3)
        if score <= SCORE_FLOOR:
            return SCORE_FLOOR
        if score >= SCORE_CEILING:
            return SCORE_CEILING
        return score

    def _episode_size(self, task_id: int) -> int:
        return 3 if task_id == 1 else 4 if task_id == 2 else 5

    def _selection_rng(self, seed: int | None) -> random.Random:
        derived_seed = seed if seed is not None else self._episode_counter
        return random.Random(derived_seed)

    def _resolve_task_id(self, task_id: int | None, rng: random.Random) -> int:
        if task_id is not None:
            get_task_definition(task_id)
            return task_id
        return rng.choice([1, 2, 3])

    def _episode_start(self, email_id: str | None, rng: random.Random) -> int:
        if email_id is not None:
            for index, email in enumerate(self.dataset):
                if email.email_id == email_id:
                    return index
            raise ValueError(f"Unknown email_id: {email_id}")
        return rng.randrange(len(self.dataset))

    def _build_inbox(self, task_id: int, email_id: str | None, rng: random.Random) -> list[EmailExample]:
        if not self.dataset:
            raise RuntimeError("Dataset is empty")

        start = self._episode_start(email_id, rng)
        size = self._episode_size(task_id)
        inbox: list[EmailExample] = []
        for offset in range(size):
            inbox.append(self.dataset[(start + offset) % len(self.dataset)])

        if email_id is None:
            self._cursor = (start + size) % len(self.dataset)
        return inbox

    def _current_email(self) -> EmailExample:
        if not self._runtime.inbox:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        index = min(self._runtime.current_email_index, len(self._runtime.inbox) - 1)
        return self._runtime.inbox[index]

    def _summary_for_email(self, email: EmailExample) -> str:
        thread_suffix = f" [{email.thread_id}]" if email.thread_id else ""
        return f"{email.email_id}: {email.subject[:48]}{thread_suffix}"

    def _dataset_summary(self) -> dict[str, Any]:
        difficulty_counts: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
        thread_groups: dict[str, int] = {}
        for email in self.dataset:
            difficulty_counts[email.difficulty] = difficulty_counts.get(email.difficulty, 0) + 1
            if email.thread_id:
                thread_groups[email.thread_id] = thread_groups.get(email.thread_id, 0) + 1
        return {
            "dataset_size": len(self.dataset),
            "email_ids": [email.email_id for email in self.dataset],
            "sample_email": self.dataset[0].model_dump(),
            "sample_emails": [email.model_dump() for email in self.dataset[:3]],
            "difficulty_counts": difficulty_counts,
            "difficulty_distribution": difficulty_counts,
            "thread_group_counts": thread_groups,
        }

    def _observation(self) -> EmailObservation:
        email = self._current_email()
        self._runtime.history = self._runtime.history or []
        return EmailObservation(
            current_email=email,
            inbox_summary=[self._summary_for_email(item) for item in self._runtime.inbox],
            remaining_emails=sum(1 for processed in self._runtime.processed if not processed),
            history=list(self._runtime.history),
            step_count=self._runtime.step_count,
            tool_result=self._runtime.pending_tool_result,
        )

    def _state(self) -> EnvironmentState:
        email = self._current_email()
        return EnvironmentState(
            inbox=list(self._runtime.inbox),
            current_email_index=self._runtime.current_email_index,
            processed=list(self._runtime.processed),
            target_category=email.category,
            target_priority=email.priority,
            target_department=email.department,
            target_action=email.action,
            email_data=email,
            step_count=self._runtime.step_count,
            task_id=self._runtime.task_id,
            episode_history=list(self._runtime.episode_history),
            available_tools=list(AVAILABLE_TOOLS),
            pending_tool_result=self._runtime.pending_tool_result,
        )

    def _tool_lookup_order(self, email: EmailExample, tool_input: dict[str, Any] | None) -> dict[str, Any]:
        order_hint = (tool_input or {}).get("order_id") or email.email_id
        state = "delivered" if "delivered" in email.email_text.lower() else "shipping" if "order" in email.subject.lower() else "unknown"
        return {
            "tool": "lookup_order",
            "order_id": order_hint,
            "order_status": state,
            "requires_follow_up": email.category in {"support", "billing"},
        }

    def _tool_check_payment(self, email: EmailExample, tool_input: dict[str, Any] | None) -> dict[str, Any]:
        lower = f"{email.subject} {email.email_text}".lower()
        duplicate = any(keyword in lower for keyword in ["charged twice", "duplicate charge", "refund", "payment issue"])
        return {
            "tool": "check_payment",
            "account_id": (tool_input or {}).get("account_id", email.email_id),
            "payment_state": "duplicate_charge" if duplicate else "settled",
            "requires_finance_review": duplicate,
        }

    def _tool_user_history(self, email: EmailExample, tool_input: dict[str, Any] | None) -> dict[str, Any]:
        stable_score = sum(ord(char) for char in email.sender)
        prior_tickets = stable_score % 5
        last_category = ["support", "billing", "sales", "internal", "spam"][stable_score % 5]
        return {
            "tool": "get_user_history",
            "user_id": (tool_input or {}).get("user_id", email.sender),
            "prior_tickets": prior_tickets,
            "last_category": last_category,
            "vip_customer": stable_score % 7 == 0,
        }

    def _use_tool(self, email: EmailExample, tool_name: str | None, tool_input: dict[str, Any] | None) -> dict[str, Any] | None:
        if tool_name is None:
            return None
        if tool_name == "lookup_order":
            return self._tool_lookup_order(email, tool_input)
        if tool_name == "check_payment":
            return self._tool_check_payment(email, tool_input)
        if tool_name == "get_user_history":
            return self._tool_user_history(email, tool_input)
        raise ValueError(f"Unknown tool: {tool_name}")

    def _is_urgent(self, email: EmailExample) -> bool:
        if email.priority == "high":
            return True
        lower = f"{email.subject} {email.email_text}".lower()
        return any(keyword in lower for keyword in ["urgent", "asap", "immediately", "today", "blocked", "outage"])

    def _category_similarity(self, true_category: str, predicted_category: str) -> float:
        if true_category == predicted_category:
            return 1.0
        return CATEGORY_SIMILARITY.get((true_category, predicted_category), 0.0)

    def _task_weights(self, task_id: int) -> dict[str, float]:
        if task_id == 1:
            return {"category": 1.0, "priority": 0.0, "department": 0.0, "action": 0.0}
        if task_id == 2:
            return {"category": 0.6, "priority": 0.4, "department": 0.0, "action": 0.0}
        return {"category": 0.3, "priority": 0.2, "department": 0.3, "action": 0.2}

    def _severity_penalty(self, email: EmailExample, action: EmailAction, breakdown: GraderBreakdown, task_id: int) -> float:
        penalty = 0.0
        if task_id < 2:
            return penalty
        urgent = self._is_urgent(email)
        important = email.category in IMPORTANT_CATEGORIES

        if urgent and action.priority != email.priority:
            penalty -= 0.4
            breakdown.severity = "urgent_priority_miss"

        if email.category == "spam" and action.category in IMPORTANT_CATEGORIES:
            penalty -= 0.3
            breakdown.severity = "spam_marked_important"

        if important and action.category == "spam":
            penalty -= 0.5
            breakdown.severity = "important_marked_spam"

        return penalty

    def _score(self, action: EmailAction) -> tuple[float, dict[str, Any]]:
        email = self._current_email()
        task_id = self._runtime.task_id
        weights = self._task_weights(task_id)
        breakdown = GraderBreakdown(category=0, priority=0, department=0, action=0)

        score = 0.0
        category_similarity = self._category_similarity(email.category, action.category)
        if category_similarity == 1.0:
            score += weights["category"]
            breakdown.category = 1
            breakdown.category_partial = 1.0
        elif category_similarity > 0.0:
            score += weights["category"] * category_similarity
            breakdown.category_partial = round(category_similarity, 3)
        else:
            score -= 0.2

        if task_id >= 2 and action.priority == email.priority:
            score += weights["priority"]
            breakdown.priority = 1

        if task_id >= 3:
            if action.department == email.department:
                score += weights["department"]
                breakdown.department = 1
            if action.action == email.action:
                score += weights["action"]
                breakdown.action = 1

        score += self._severity_penalty(email, action, breakdown, task_id)
        score -= 0.05 * self._runtime.step_count

        if self._runtime.pending_tool_result is not None and action.use_tool is not None:
            score += 0.05
            breakdown.tool_used = 1

        return self._strict_score(score), breakdown.model_dump()

    def reset(
        self,
        seed: int | None = None,
        episode_id: int | None = None,
        task_id: int | None = None,
        email_id: str | None = None,
        **kwargs: Any,
    ) -> EmailObservation:
        if task_id is None:
            task_id = episode_id
        if task_id is None and "episode_id" in kwargs:
            task_id = kwargs["episode_id"]
        rng = self._selection_rng(seed)
        resolved_task_id = self._resolve_task_id(task_id, rng)
        inbox = self._build_inbox(resolved_task_id, email_id, rng)
        self._episode_counter += 1
        self._runtime.task_id = resolved_task_id
        self._runtime.inbox = inbox
        self._runtime.current_email_index = 0
        self._runtime.processed = [False] * len(inbox)
        self._runtime.step_count = 0
        self._runtime.episode_history = []
        self._runtime.pending_tool_result = None
        self._runtime.done = False
        self._runtime.history = [
            f"reset(task_id={resolved_task_id})",
            f"inbox_size={len(inbox)}",
            f"seed={seed if seed is not None else self._episode_counter - 1}",
        ]
        logger.info(
            "reset selected email_id=%s task_id=%s seed=%s",
            self._current_email().email_id,
            resolved_task_id,
            seed,
        )
        return self._observation()

    def ensure_initialized(self, task_id: int = 1) -> EmailObservation:
        if not self._runtime.inbox:
            return self.reset(task_id=task_id)
        return self._observation()

    def step(self, action: EmailAction) -> StepResponse:
        if not self._runtime.inbox:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        if self._runtime.done:
            self._runtime.history.append("invalid_step_after_done")
            return StepResponse(
                observation=self._observation(),
                reward=0.0,
                done=True,
                info={
                    "invalid_action": True,
                    "reason": "step called after done",
                    "penalty": -0.5,
                    "episode_history": list(self._runtime.episode_history),
                },
                state=self._state(),
            )

        current_email = self._current_email()
        logger.info(
            "step email_id=%s action=%s",
            current_email.email_id,
            action.model_dump(),
        )
        self._runtime.step_count += 1
        tool_result = self._use_tool(current_email, action.use_tool, action.tool_input)
        self._runtime.pending_tool_result = tool_result
        reward, breakdown = self._score(action)
        logger.debug(
            "reward_breakdown email_id=%s breakdown=%s reward=%s",
            current_email.email_id,
            breakdown,
            reward,
        )

        log_entry = {
            "email": current_email.model_dump(),
            "agent_action": action.model_dump(),
            "correct_action": {
                "category": current_email.category,
                "priority": current_email.priority,
                "department": current_email.department,
                "action": current_email.action,
            },
            "reward": reward,
            "tool_result": tool_result,
        }
        self._runtime.episode_history.append(log_entry)
        self._runtime.processed[self._runtime.current_email_index] = True
        self._runtime.history.append(
            f"processed={current_email.email_id}, category={action.category}, priority={action.priority}, department={action.department}, action={action.action}"
        )
        if tool_result is not None:
            self._runtime.history.append(f"tool={action.use_tool}")

        next_index = self._runtime.current_email_index + 1
        if next_index >= len(self._runtime.inbox):
            self._runtime.done = True
        else:
            self._runtime.current_email_index = next_index

        observation = self._observation()
        info = {
            "task_id": self._runtime.task_id,
            "email_id": current_email.email_id,
            "breakdown": breakdown,
            "episode_history": list(self._runtime.episode_history),
            "tool_result": tool_result,
        }
        return StepResponse(observation=observation, reward=reward, done=self._runtime.done, info=info, state=self._state())

    def state(self) -> EnvironmentState:
        return self._state()

    def tasks(self) -> list[dict[str, Any]]:
        return [task.model_dump() for task in TASKS]

    def tasks_payload(self) -> dict[str, Any]:
        summary = self._dataset_summary()
        return {
            "tasks": self.tasks(),
            "email_ids": summary["email_ids"],
            "sample_email": summary["sample_email"],
            "sample_emails": summary["sample_emails"],
            "difficulty_distribution": summary["difficulty_distribution"],
            "dataset_summary": summary,
        }

    def episode_log(self) -> dict[str, Any]:
        return {
            "task_id": self._runtime.task_id,
            "done": self._runtime.done,
            "step_count": self._runtime.step_count,
            "current_email_index": self._runtime.current_email_index,
            "processed": list(self._runtime.processed),
            "episode_history": list(self._runtime.episode_history),
            "history": list(self._runtime.history),
            "pending_tool_result": self._runtime.pending_tool_result,
        }

    def grade(self, action: EmailAction, email_data: EmailExample | None = None, task_id: int | None = None) -> tuple[float, dict[str, Any]]:
        original = self._runtime
        temp = _RuntimeState()
        if task_id is not None:
            get_task_definition(task_id)
        temp.task_id = task_id or self._runtime.task_id or 3
        temp.inbox = [email_data or self._current_email()]
        temp.current_email_index = 0
        temp.processed = [False]
        temp.step_count = 1
        temp.pending_tool_result = None
        temp.done = False
        self._runtime = temp
        try:
            score, breakdown = self._score(action)
            return score, breakdown
        finally:
            self._runtime = original

    def sample_action(self) -> dict[str, Any]:
        email = self._current_email() if self._runtime.inbox else self.dataset[0]
        action = {
            "category": email.category,
            "priority": email.priority,
            "department": email.department,
            "action": email.action,
        }
        return action

    def baseline_scores(self, score_fn) -> BaselineScores:
        task_scores: dict[int, list[float]] = {1: [], 2: [], 3: []}
        for task in TASKS:
            for email in self.dataset:
                task_scores[task.task_id].append(score_fn(email, task.task_id))

        averages = {
            task_id: sum(values) / len(values) if values else 0.0 for task_id, values in task_scores.items()
        }
        overall = sum(averages.values()) / 3.0
        return BaselineScores(
            task_1=self._strict_score(averages[1]),
            task_2=self._strict_score(averages[2]),
            task_3=self._strict_score(averages[3]),
            average=self._strict_score(overall),
            mode="heuristic",
        )
