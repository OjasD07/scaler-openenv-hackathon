from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Category = Literal["spam", "support", "billing", "sales", "internal"]
Priority = Literal["low", "medium", "high"]
Department = Literal["support_team", "sales_team", "finance", "ignore"]
ActionType = Literal["reply", "forward", "archive", "escalate"]


class EmailExample(BaseModel):
    email_id: str
    email_text: str
    sender: str
    subject: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    noisy_text: Optional[str] = None
    thread_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    category: Category
    priority: Priority
    department: Department
    action: ActionType


class EmailObservation(BaseModel):
    current_email: EmailExample
    inbox_summary: list[str] = Field(default_factory=list)
    remaining_emails: int = 0
    history: list[str] = Field(default_factory=list)
    step_count: int = 0
    tool_result: Optional[dict[str, Any]] = None


class EmailAction(BaseModel):
    category: Category
    priority: Priority
    department: Department
    action: ActionType
    use_tool: Optional[Literal["lookup_order", "check_payment", "get_user_history"]] = None
    tool_input: Optional[dict[str, Any]] = None


class EnvironmentState(BaseModel):
    inbox: list[EmailExample]
    current_email_index: int
    processed: list[bool]
    target_category: Category
    target_priority: Priority
    target_department: Department
    target_action: ActionType
    email_data: EmailExample
    step_count: int = 0
    task_id: int = 3
    episode_history: list[dict[str, Any]] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=lambda: ["lookup_order", "check_payment", "get_user_history"])
    pending_tool_result: Optional[dict[str, Any]] = None


class TaskDefinition(BaseModel):
    task_id: int
    name: str
    description: str
    required_fields: list[str]
    level: Literal["easy", "medium", "hard"]


class ResetRequest(BaseModel):
    task_id: Optional[int] = Field(
        default=None,
        description="Optional task selector. Defaults to a seeded random task when omitted.",
        examples=[1, 2, 3],
    )
    email_id: Optional[str] = Field(
        default=None,
        description="Optional email identifier. Defaults to a seeded random email when omitted.",
        examples=["em-001", "em-025"],
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional deterministic seed for episode sampling.",
        examples=[7, 42],
    )


class StepRequest(BaseModel):
    action: EmailAction = Field(
        default_factory=lambda: EmailAction(
            category="support",
            priority="medium",
            department="support_team",
            action="reply",
        )
    )


class GradeRequest(BaseModel):
    action: EmailAction
    email_id: Optional[str] = None
    task_id: Optional[int] = Field(default=None)
    email_data: Optional[EmailExample] = None


class StepResponse(BaseModel):
    observation: EmailObservation
    reward: float
    done: bool
    info: dict[str, Any]
    state: EnvironmentState


class BaselineScores(BaseModel):
    task_1: float
    task_2: float
    task_3: float
    average: float
    mode: str


class GraderBreakdown(BaseModel):
    category: int
    priority: int
    department: int
    action: int
    category_partial: float = 0.0
    tool_used: int = 0
    severity: str = "normal"
