from __future__ import annotations

import json
import re
from typing import Any

from .models import BaselineScores, EmailAction, EmailExample
from .server.environment import EmailTriageEnvironment


IMPORTANT_CATEGORY_KEYWORDS = {
    "billing": ["invoice", "charged", "charge", "refund", "payment", "vat", "tax", "bank", "billing", "duplicate"],
    "support": ["issue", "error", "bug", "login", "help", "broken", "timeout", "verification", "outage", "arrived", "delivered"],
    "sales": ["pricing", "price", "demo", "proposal", "enterprise", "trial", "campaign", "contract", "seats", "buy"],
    "internal": ["review", "policy", "board", "launch", "checklist", "approval", "office", "internal", "meeting"],
    "spam": ["gift card", "claim your prize", "click this link", "limited time", "coupon", "password immediately", "unsubscribe", "offer"],
}

SPAM_MARKERS = [
    "unsubscribe",
    "gift card",
    "claim your prize",
    "coupon",
    "password immediately",
    "newsletter",
    "digest",
    "click this link",
    "limited time offer",
    "attached invoice",
    "open the file",
    "confirm payment details",
]

INTERNAL_MARKERS = [
    "@acme.internal",
    "acme.local",
    ".internal",
    "corp.local",
    "internal.tools",
    "policy",
    "board",
    "launch",
    "checklist",
    "approval",
    "meeting",
    "review",
    "employee",
    "office",
    "security",
]

SALES_MARKERS = [
    "pricing",
    "price",
    "buy",
    "purchase",
    "demo",
    "proposal",
    "enterprise",
    "contract",
    "seats",
    "partnership",
    "quote",
    "trial",
]

SUPPORT_MARKERS = [
    "login",
    "help",
    "bug",
    "error",
    "timeout",
    "outage",
    "down",
    "broken",
    "access",
    "checkout",
    "not arrived",
    "tracking says delivered",
    "user error",
    "system issue",
]

BILLING_MARKERS = [
    "charged twice",
    "double charge",
    "duplicate charge",
    "duplicate billing",
    "incorrect billing",
    "refund",
    "invoice",
    "payment",
    "billing",
    "money back",
    "tax invoice",
]


def _score_keyword_hits(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    return sum(1 for keyword in keywords if keyword in lower)


def _normalize_text(email: EmailExample) -> str:
    text = f"{email.sender} {email.subject} {email.email_text} {email.noisy_text or ''}".lower()
    return re.sub(r"\s+", " ", text).strip()


def _classify_category(text: str) -> str:
    if any(marker in text for marker in SPAM_MARKERS):
        return "spam"

    if any(marker in text for marker in INTERNAL_MARKERS[:4]):
        return "internal"

    scores = {
        "sales": _score_keyword_hits(text, SALES_MARKERS) * 2
        + _score_keyword_hits(text, ["rollout blocker", "partnership ask", "want to buy", "need proposal", "compare pricing"]),
        "billing": _score_keyword_hits(text, BILLING_MARKERS) * 2
        + _score_keyword_hits(text, ["money back", "duplicate billing", "incorrect billing", "payment reset", "bank says pending", "month-end close", "statement update"]),
        "support": _score_keyword_hits(text, SUPPORT_MARKERS) * 2
        + _score_keyword_hits(text, ["package", "tracking", "onboarding", "checkout bug", "system down", "blocked", "affecting everyone", "login issue", "trial issue", "cannot log in", "failing"]),
        "internal": _score_keyword_hits(text, INTERNAL_MARKERS[4:]) * 2,
    }

    if "pricing" in text or "proposal" in text or "buy" in text or "partnership" in text or "seats" in text or "quote" in text:
        scores["sales"] += 3
    if "charged twice" in text or "duplicate billing" in text or "incorrect billing" in text or "refund" in text:
        scores["billing"] += 3
    if "outage" in text or "system down" in text or "blocked" in text or "checkout bug" in text or "login issue" in text:
        scores["support"] += 3
    if "policy update" in text or "board deck" in text or "launch checklist" in text or "compliance audit" in text:
        scores["internal"] += 4
    if "trial issue" in text or "login issue" in text or "cannot log in" in text:
        scores["support"] += 2
    if "invoice still pending" in text or "month-end close" in text or "statement update" in text:
        scores["billing"] += 4

    if scores["sales"] >= max(scores["support"], scores["billing"], scores["internal"]):
        return "sales"
    if scores["billing"] >= max(scores["support"], scores["internal"]):
        return "billing"
    if scores["support"] >= scores["internal"]:
        return "support"
    return "internal"


def _predict_priority(text: str, category: str) -> str:
    if category == "spam":
        return "low"

    urgent_markers = ["urgent", "asap", "immediately", "today", "blocked", "outage", "high priority", "cannot", "stuck"]
    severe_markers = [
        "charged twice",
        "duplicate billing",
        "incorrect billing",
        "double charge",
        "outage",
        "system down",
        "blocked",
        "not arrived",
        "tracking says delivered",
        "needs attention today",
        "affecting everyone",
    ]

    if category == "sales" and any(marker in text for marker in ["promotion", "discount", "save", "free trial", "newsletter", "digest"]):
        return "low"

    if category == "internal" and "policy update" in text and any(marker in text for marker in ["access issue", "cannot access", "report"]):
        return "medium"

    if category == "internal" and any(marker in text for marker in ["kitchen supplies", "digest", "announcement", "reminder"]):
        return "low"

    if any(marker in text for marker in urgent_markers):
        return "high"

    if any(marker in text for marker in severe_markers):
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
        action = "escalate" if any(
            marker in text
            for marker in [
                "outage",
                "blocked",
                "critical",
                "urgent",
                "system issue",
                "404 error",
                "down",
                "affecting everyone",
                "multiple teams",
                "checkout bug",
                "payment errors",
                "needs attention today",
            ]
        ) else "reply"
    elif category == "sales":
        department = "sales_team"
        action = "forward"
    else:
        department = "ignore"
        action = "escalate" if any(marker in text for marker in ["urgent", "approve", "approval", "system issue"]) else "reply" if any(marker in text for marker in ["review", "policy", "checklist", "launch", "meeting"]) else "archive"

    if category == "support" and any(marker in text for marker in ["tracking says delivered", "not arrived", "check my order", "package"]):
        action = "reply"
        priority = "high" if "not arrived" in text or "tracking says delivered" in text else priority
    if category == "support" and any(marker in text for marker in ["system down", "outage", "blocked", "affecting everyone", "multiple teams"]):
        action = "escalate"
        priority = "high"
    if category == "billing" and any(marker in text for marker in ["think i was charged twice", "not sure if it was my bank", "duplicate"]):
        priority = "high"
    if category == "billing" and any(marker in text for marker in ["incorrect billing", "duplicate billing", "charged twice", "double charge"]):
        priority = "high"
    if category == "sales" and any(marker in text for marker in ["buy", "price", "pricing", "proposal", "enterprise"]):
        department = "sales_team"
        action = "forward"
        if any(marker in text for marker in ["buy price", "pricing sheet", "trial account is broken"]):
            priority = "medium"
    if category == "internal" and "urgent" in text:
        priority = "high"
        action = "escalate"
    if category == "internal" and any(marker in text for marker in ["system is down", "system down", "blocked", "please escalate"]):
        priority = "high"
        action = "escalate"
    if category == "support" and any(marker in text for marker in ["404 error", "checkout bug", "bug", "trial issue", "login issue"]):
        if any(marker in text for marker in ["checkout bug", "down", "blocked", "affecting everyone", "multiple teams"]):
            action = "escalate"
    if category == "sales" and any(marker in text for marker in ["500 seats", "enterprise", "proposal", "rollout"]):
        priority = "high"
    if category == "billing" and any(marker in text for marker in ["invoice", "vat", "tax", "today", "urgent"]):
        priority = "high" if any(marker in text for marker in ["today", "urgent", "asap", "charged twice", "duplicate", "incorrect billing"]) else priority
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


def predict_action(email: EmailExample) -> EmailAction:
    return _predict_action(email)


def run_baseline() -> BaselineScores:
    env = EmailTriageEnvironment()
    def score_fn(email: EmailExample, task_id: int) -> float:
        predicted = predict_action(email)
        score, _ = env.grade(predicted, email_data=email, task_id=task_id)
        return score

    scores = env.baseline_scores(score_fn)
    scores.mode = "heuristic"
    return scores


def run_all_tasks() -> dict[str, Any]:
    scores = run_baseline()
    return scores.model_dump()


def main() -> None:
    scores = run_baseline()
    print(json.dumps(scores.model_dump(), indent=2))


if __name__ == "__main__":
    main()
