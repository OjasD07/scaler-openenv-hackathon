# OpenEnv Email Triage Environment

`email_triage_env` is a deterministic OpenEnv-compatible RL environment that simulates enterprise email triage across multi-email inbox episodes.

## What Is Email Triage

Email triage is the workflow of taking incoming messages, understanding intent, estimating urgency, routing to the right team, and deciding the right action.

This environment models the same operational flow used in support, billing, sales, security, and internal operations queues.

This environment is intentionally designed to challenge modern LLM agents by introducing:
- Multi-email decision-making instead of single-step classification
- Ambiguous and conflicting intents within the same email
- Adversarial phrasing and noisy inputs
- Temporal trade-offs via step-based penalties

Unlike standard classification benchmarks, agents must reason about intent priority, urgency, and downstream consequences across an episode.

## What’s Stronger In This Revision

- A self-describing `/manifest` endpoint for tooling and reviewers
- Richer `/tasks` metadata, including supported tools and dataset summaries
- A self-contained smoke test that exercises the FastAPI app directly
- A root-level Dockerfile in the GitHub repo for easier deployment

## Validation

- Run `python validate.py` for a one-command local check.
- Run `python -m unittest discover -s tests` to execute the automated smoke tests.

## System Architecture

```text
FastAPI Server
   |
   +--> /reset  -> load deterministic inbox episode
   +--> /step   -> score one email, advance to next
   +--> /state  -> inspect internal episode state
   +--> /tasks  -> task metadata + dataset summary
   +--> /grader -> deterministic grading API
   +--> /episode_log -> full trajectory inspection
   +--> /baseline -> heuristic/OpenAI baseline scores
   |
   +--> EmailTriageEnvironment
           |
           +--> synthetic dataset
           +--> reward shaping
           +--> episode logging
           +--> tool simulation
```

## Environment Overview

The environment follows the OpenEnv-style interface:

- `reset()`
- `step(action)`
- `state()`

Each episode now contains multiple emails. The agent processes one email at a time until the inbox is exhausted.

## Observation Schema

```json
{
  "current_email": {
    "email_id": "em-001",
    "subject": "Charged twice for order 88412",
    "sender": "billing@shopnova.com",
    "email_text": "I was charged twice for my order 88412...",
    "difficulty": "easy"
  },
  "inbox_summary": [
    "em-001: Charged twice for order 88412",
    "em-002: Login issue on my account",
    "em-003: Limited time offer on premium plans"
  ],
  "remaining_emails": 3,
  "history": ["reset(task_id=3)", "inbox_size=5"],
  "step_count": 1,
  "tool_result": {
    "tool": "lookup_order",
    "order_status": "shipping"
  }
}
```

## Action Schema

`POST /step` accepts either of these shapes:

```json
{
  "action": {
    "category": "billing",
    "priority": "high",
    "department": "finance",
    "action": "reply",
    "use_tool": "check_payment",
    "tool_input": {
      "account_id": "acct_123"
    }
  }
}
```

```json
{
  "category": "billing",
  "priority": "high",
  "department": "finance",
  "action": "reply",
  "use_tool": "check_payment",
  "tool_input": {
    "account_id": "acct_123"
  }
}
```

Allowed values:

- `category`: `spam`, `support`, `billing`, `sales`, `internal`
- `priority`: `low`, `medium`, `high`
- `department`: `support_team`, `sales_team`, `finance`, `ignore`
- `action`: `reply`, `forward`, `archive`, `escalate`
- `use_tool`: `lookup_order`, `check_payment`, `get_user_history`

## Internal State Schema

```json
{
  "inbox": ["..."],
  "current_email_index": 0,
  "processed": [false, false, false],
  "target_category": "billing",
  "target_priority": "high",
  "target_department": "finance",
  "target_action": "reply",
  "email_data": { "...": "..." },
  "step_count": 1,
  "task_id": 3,
  "episode_history": [
    {
      "email": { "...": "..." },
      "agent_action": { "...": "..." },
      "correct_action": { "...": "..." },
      "reward": 0.95
    }
  ],
  "available_tools": ["lookup_order", "check_payment", "get_user_history"]
}
```

## Tasks

| Task | Name | Required Fields |
|---|---|---|
| Task 1 | `easy` | `category` |
| Task 2 | `medium` | `category`, `priority` |
| Task 3 | `hard` | `category`, `priority`, `department`, `action` |

## Reward Explanation

Reward is dense, deterministic, and shaped for realistic triage behavior.

| Component | Effect |
|---|---|
| Correct category | `+0.3` on hard task, task-aware scaling on easier tasks |
| Similar category | partial credit via category similarity matrix |
| Correct priority | `+0.2` on medium/hard |
| Correct department | `+0.3` on hard |
| Correct action | `+0.2` on hard |
| Wrong category | `-0.2` |
| Urgent email with wrong priority | `-0.4` |
| Spam marked as important | `-0.3` |
| Important email marked as spam | `-0.5` |
| Time cost | `-0.05 * step_count` |
| Optional tool bonus | small deterministic bonus when a tool is used appropriately |

Final reward is clamped to `[0.0, 1.0]`.

## Example Episode Walkthrough

1. `POST /reset` loads a deterministic inbox with 3 to 5 emails.
2. The agent sees the first `current_email`, plus a short `inbox_summary`.
3. The agent submits an action, optionally using a tool.
4. The environment scores the triage decision, logs the episode entry, and advances to the next email.
5. The observation returned by `step()` includes the next email and any tool result from the previous decision.
6. The episode ends once all emails in the inbox are processed.

## Synthetic Dataset

The dataset contains 38 deterministic synthetic emails with:

- clear support, billing, sales, spam, and internal cases
- 8+ ambiguous examples
- noisy text variants
- severity variation
- realistic operational phrasing

## Ground Truth Rules

When an email contains overlapping cues, the label is determined by intent priority rather than raw keyword count:

1. Spam indicators override all other intents.
2. Billing intent wins when the core ask is refund, incorrect billing, invoice correction, money back, or payment reconciliation.
3. Support intent wins when the core ask is login, access, outage, bug, broken flow, or troubleshooting.
4. Sales intent wins when the core ask is pricing, proposal, seats, purchase, demo, or contract negotiation.
5. Internal intent wins when the message is primarily about company operations, approvals, policy, or internal coordination.

Priority is also intent-aware:

- "Not urgent" does not override a clearly blocked, down, or outage-driven workflow.
- If the email says the system is down or the user is blocked, priority is high even with hedging language.
- Mixed-intent emails inherit priority from the primary operational risk, not the most frequent keyword.

Examples include:

- "I think I was charged twice but not sure if it's my bank"
- "Can you check my order? It hasn't arrived but tracking says delivered"
- "This might be urgent, not sure if system issue or user error"

## Why This Is Real-World Useful

- It mirrors how enterprise inboxes are actually handled.
- It rewards both classification quality and operational judgment.
- It supports ambiguity instead of assuming every email is obvious.
- It introduces tool use, which is common in real triage workflows.
- It creates a multi-step decision process rather than a single-label toy problem.

## Comparison With Basic Classification Systems

Basic classifiers only predict a label.

This environment requires:

- category prediction
- urgency estimation
- routing decisions
- action selection
- optional tool-assisted reasoning
- episode-level progression through multiple emails

That makes it much closer to a production triage assistant than a standard text classifier.

## Baseline Agent

`baseline.py` runs a deterministic heuristic baseline that is fast, reproducible, and independent of external model credentials.

## Submission Runner

`inference.py` is the root-level submission script. It uses the same deterministic policy and optionally warms the injected OpenAI-compatible proxy when credentials are available. If `ENV_BASE_URL` is not reachable, it falls back to the local FastAPI app so the baseline can still reproduce.

Optional environment variables:

- `API_BASE_URL`
- `API_KEY`
- `MODEL_NAME`
- `ENV_BASE_URL` only if your local environment server is not on `http://127.0.0.1:8000`
- `LOCAL_IMAGE_NAME` only if you use `from_docker_image()`

The script emits structured stdout in the required format:

- `[START]`
- `[STEP]`
- `[END]`

`API_KEY` and `API_BASE_URL` are used when the grader injects proxy credentials. Without them, the script still runs through the deterministic triage policy.

Current deterministic heuristic scores:

- Task 1: `0.950`
- Task 2: `0.931`
- Task 3: `0.936`
- Average: `0.939`

Run it locally:

```bash
python -m email_triage_env.baseline
```

## Setup

```bash
pip install -r requirements.txt
uvicorn email_triage_env.server.app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `GET /episode_log`
- `GET /sample_action`
- `GET /health`
- `GET /baseline`

### Reset Behavior

- `POST /reset` accepts an empty JSON body.
- If `email_id` is omitted, the environment selects a seeded random email from the dataset.
- If `task_id` is omitted, the environment selects a seeded random task.
- If `seed` is provided, the episode selection is reproducible.

## Quick Test Script

Run the built-in smoke test without Swagger:

```bash
python test_env.py
```

## Docker

Build and run from the repository root:

```bash
docker build -f server/Dockerfile -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

## Hugging Face Spaces

This repository is ready for deployment as a Docker Space using `server/Dockerfile`.
It is currently deployed at: https://email-triage-env.ojasdeshpande.in/ or https://ojasd07-email-triage-env.hf.space/docs

## Author
Ojas Deshpande
[contact.ojasdeshpande@gmail.com]
