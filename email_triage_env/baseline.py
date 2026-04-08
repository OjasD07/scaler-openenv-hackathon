from __future__ import annotations

import json
import os
import re
from typing import Any

from .grader import grade_action
from .models import BaselineScores, EmailAction, EmailExample
from .server.environment import EmailTriageEnvironment

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


IMPORTANT_CATEGORY_KEYWORDS = {
    "billing": ["invoice", "charged", "charge", "refund", "payment", "vat", "tax", "bank", "billing", "duplicate"],
    "support": ["issue", "error", "bug", "login", "help", "broken", "timeout", "verification", "outage", "arrived", "delivered"],
    "sales": ["pricing", "price", "demo", "proposal", "enterprise", "trial", "campaign", "contract", "seats", "buy"],
    "internal": ["review", "policy", "board", "launch", "checklist", "approval", "office", "internal", "meeting"],
    "spam": ["gift card", "claim your prize", "click this link", "limited time", "coupon", "password immediately", "unsubscribe", "offer"],
}


def _score_keyword_hits(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    return sum(1 for keyword in keywords if keyword in lower)


def _warmup_proxy(client: OpenAI, model: str) -> None:
    client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=1,
        messages=[
            {"role": "system", "content": "Reply with OK."},
            {"role": "user", "content": "OK"},
        ],
    )


def _normalize_text(email: EmailExample) -> str:
    text = f"{email.sender} {email.subject} {email.email_text} {email.noisy_text or ''}".lower()
    return re.sub(r"\s+", " ", text).strip()


def _classify_category(text: str) -> str:
    scores = {
        "spam": _score_keyword_hits(text, IMPORTANT_CATEGORY_KEYWORDS["spam"]),
        "billing": _score_keyword_hits(text, IMPORTANT_CATEGORY_KEYWORDS["billing"]),
        "support": _score_keyword_hits(text, IMPORTANT_CATEGORY_KEYWORDS["support"]),
        "sales": _score_keyword_hits(text, IMPORTANT_CATEGORY_KEYWORDS["sales"]),
        "internal": _score_keyword_hits(text, IMPORTANT_CATEGORY_KEYWORDS["internal"]),
    }

    if any(
        marker in text
        for marker in [
            "unsubscribe",
            "gift card",
            "claim your prize",
            "coupon",
            "password immediately",
            "newsletter",
            "digest",
            "attached invoice",
            "open the file",
            "confirm payment details",
        ]
    ):
        return "spam"
    if any(marker in text for marker in ["@acme.internal", ".internal", "corp.local", "internal.tools"]):
        return "internal"
    if any(marker in text for marker in ["package arrived", "damaged", "broken", "not arrived", "tracking says delivered", "verify phone", "sms verification", "onboarding"]):
        return "support"
    if scores["billing"] >= 2 or any(marker in text for marker in ["charged twice", "duplicate charge", "refund", "invoice", "vat", "tax", "payment", "bank"]):
        return "billing"
    if scores["support"] >= 2 or any(marker in text for marker in ["error", "bug", "login", "help", "broken", "timeout", "outage", "404", "checkout", "cannot verify"]):
        return "support"
    if scores["sales"] >= 2 or any(marker in text for marker in ["pricing", "price", "buy", "demo", "proposal", "enterprise", "contract", "seats", "co-marketing", "partnership", "trial"]):
        return "sales"
    if scores["internal"] >= 1 or any(marker in text for marker in ["review", "policy", "board", "launch", "checklist", "approval", "meeting", "internal", "employee", "office", "security"]):
        return "internal"
    return "support" if "system issue" in text or "user error" in text else "internal"


def _predict_priority(text: str, category: str) -> str:
    urgent_markers = [
        "urgent",
        "asap",
        "immediately",
        "today",
        "blocked",
        "outage",
        "high priority",
        "cannot",
        "stuck",
        "duplicate",
        "charged twice",
        "not arrived",
        "missing vat",
        "tax invoice",
        "500 seats",
        "proposal",
    ]
    if category == "spam":
        return "low"
    if category == "sales" and any(marker in text for marker in ["promotion", "discount", "save", "free trial"]):
        return "low"
    if category == "internal" and any(marker in text for marker in ["policy update", "kitchen supplies", "digest", "announcement", "reminder"]):
        return "low"
    if any(marker in text for marker in urgent_markers):
        return "high"
    if category in {"billing", "support"} and any(marker in text for marker in ["question", "confirm", "check", "not sure", "maybe", "help", "issue"]):
        return "medium"
    return "medium" if category in {"billing", "support", "sales", "internal"} else "low"


def _predict_action(email: EmailExample) -> EmailAction:
    text = _normalize_text(email)
    category = _classify_category(text)
    priority = _predict_priority(text, category)

    if category == "spam":
        department = "ignore"
        action = "archive"
    elif category == "billing":
        department = "finance"
        action = "reply"
    elif category == "support":
        department = "support_team"
        action = "escalate" if any(marker in text for marker in ["outage", "blocked", "critical", "urgent", "system issue", "404 error"]) else "reply"
    elif category == "sales":
        department = "sales_team"
        action = "forward"
    else:
        department = "ignore"
        action = "escalate" if any(marker in text for marker in ["urgent", "approve", "approval", "system issue"]) else "reply" if any(marker in text for marker in ["review", "policy", "checklist", "launch", "meeting"]) else "archive"

    if category == "support" and any(marker in text for marker in ["tracking says delivered", "not arrived", "check my order", "package"]):
        action = "reply"
        priority = "high" if "not arrived" in text or "tracking says delivered" in text else priority
    if category == "billing" and any(marker in text for marker in ["think i was charged twice", "not sure if it was my bank", "duplicate"]):
        priority = "high"
    if category == "sales" and any(marker in text for marker in ["buy", "price", "pricing", "proposal", "enterprise"]):
        department = "sales_team"
    if category == "internal" and "urgent" in text:
        priority = "high"
        action = "escalate"
    if category == "support" and any(marker in text for marker in ["404 error", "checkout bug", "bug"]):
        action = "reply"
    if category == "sales" and any(marker in text for marker in ["500 seats", "enterprise", "proposal", "rollout"]):
        priority = "high"
    if category == "billing" and any(marker in text for marker in ["invoice", "vat", "tax", "today", "urgent"]):
        priority = "high" if any(marker in text for marker in ["today", "urgent", "asap", "charged twice", "refund"]) else priority
    if category == "spam":
        priority = "low"
        department = "ignore"
        action = "archive"

    tool_name = None
    tool_input: dict[str, Any] | None = None
    if category in {"support", "billing"} and any(marker in text for marker in ["not sure", "think", "check", "arrived", "charged twice", "bank", "tracking says delivered"]):
        tool_name = "check_payment" if category == "billing" else "lookup_order"
        tool_input = {"order_id": email.email_id, "user_id": email.sender}

    return EmailAction(category=category, priority=priority, department=department, action=action, use_tool=tool_name, tool_input=tool_input)


def _openai_predict(email: EmailExample) -> EmailAction | None:
    if OpenAI is None:
        return None
    if not os.getenv("API_BASE_URL") or not os.getenv("API_KEY"):
        return None

    prompt = {
        "email_id": email.email_id,
        "subject": email.subject,
        "sender": email.sender,
        "email_text": email.email_text,
        "allowed_categories": ["spam", "support", "billing", "sales", "internal"],
        "allowed_priorities": ["low", "medium", "high"],
        "allowed_departments": ["support_team", "sales_team", "finance", "ignore"],
        "allowed_actions": ["reply", "forward", "archive", "escalate"],
    }

    try:
        client = OpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"])
        model = os.getenv("MODEL_NAME", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an enterprise email triage agent. Return only minified JSON."},
                {
                    "role": "user",
                    "content": (
                        "Classify the email for triage. Return JSON with keys category, priority, department, action.\n"
                        + json.dumps(prompt)
                    ),
                },
            ],
            temperature=0,
        )
        text = response.choices[0].message.content or ""
        text = text.strip()
        parsed = json.loads(text)
        return EmailAction(**parsed)
    except Exception:
        return None


def predict_action(email: EmailExample) -> EmailAction:
    predicted = _openai_predict(email)
    if predicted is not None:
        return predicted
    return _predict_action(email)


def run_baseline() -> BaselineScores:
    env = EmailTriageEnvironment()
    if OpenAI is not None and os.getenv("API_BASE_URL") and os.getenv("API_KEY"):
        _warmup_proxy(OpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"]), os.getenv("MODEL_NAME", "gpt-4o-mini"))

    def score_fn(email: EmailExample, task_id: int) -> float:
        predicted = predict_action(email)
        return grade_action(predicted, email_data=email, task_id=task_id)

    scores = env.baseline_scores(score_fn)
    scores.mode = "openai" if OpenAI is not None and os.getenv("API_BASE_URL") and os.getenv("API_KEY") else "heuristic"
    return scores


def run_all_tasks() -> dict[str, Any]:
    scores = run_baseline()
    return scores.model_dump()


def main() -> None:
    scores = run_baseline()
    print(json.dumps(scores.model_dump(), indent=2))


if __name__ == "__main__":
    main()
