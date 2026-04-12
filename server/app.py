# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Email Triage RL Environment.

Exposes both the standard OpenEnv WebSocket/HTTP endpoints and three
demo-friendly REST endpoints documented in the project spec:

    GET  /reset  — sample a new email, return its text
    POST /step   — submit an action, receive reward
    GET  /auto   — full LLM pipeline: classify → decide → reward → reply

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import random
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with '\n    uv sync\n'"
    ) from e

# These imports work in both local package mode and Docker (PYTHONPATH=/app/env)
try:
    from ..models import EmailTriageAction, EmailTriageObservation  # local package
    from ..llm import classify_email, decide_action, generate_reply
    from .email_triage_env_environment import EmailTriageEnvironment, _load_emails
except ImportError:
    from models import EmailTriageAction, EmailTriageObservation  # type: ignore[import]
    from llm import classify_email, decide_action, generate_reply  # type: ignore[import]
    from server.email_triage_env_environment import EmailTriageEnvironment, _load_emails  # type: ignore[import]

# ---------------------------------------------------------------------------
# OpenEnv core app (provides /ws, /step, /reset, /state, /schema)
# ---------------------------------------------------------------------------
app = create_app(
    EmailTriageEnvironment,
    EmailTriageAction,
    EmailTriageObservation,
    env_name="email_triage_env",
    max_concurrent_envs=4,
)

# ---------------------------------------------------------------------------
# Lightweight shared state for the REST demo endpoints
# (separate from the WebSocket session managed by OpenEnv)
# ---------------------------------------------------------------------------
_emails = _load_emails()
_current_email: Optional[dict] = None


class ActionRequest(BaseModel):
    action: str
    value: Optional[str] = None  # required for reply / route_to_department


# ---------------------------------------------------------------------------
# Demo REST endpoints
# ---------------------------------------------------------------------------
router = APIRouter(tags=["Email Triage Demo"])

VALID_ACTIONS = {
    "reply", "mark_spam", "mark_important", "ignore",
    "route_to_department", 'high-priority "crisis"',
}


@router.get("/reset", summary="Sample a new email to triage")
def reset_email():
    """
    Sample a random email from the dataset and set it as the current email.

    Returns the email text so the caller (or UI) knows what to act on.
    """
    global _current_email
    _current_email = random.choice(_emails)
    return {
        "email": _current_email["email_text"],
        "category": _current_email["category"],
        "correct_steps": _current_email["steps"],
    }


@router.post("/step", summary="Apply an action to the current email")
def step_email(body: ActionRequest):
    """
    Apply the supplied action to the current email and receive a reward.

    Call /reset first to load an email.
    """
    if _current_email is None:
        raise HTTPException(status_code=400, detail="Call /reset first to load an email.")
    if body.action not in VALID_ACTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action '{body.action}'. Choose from: {sorted(VALID_ACTIONS)}",
        )

    correct_steps = _current_email["steps"]
    first_correct = correct_steps[0]["action"] if correct_steps else None
    reward = 10.0 if body.action == first_correct else -5.0

    reply = None
    if body.action == "reply":
        reply = generate_reply(_current_email["email_text"])

    return {
        "action_taken": body.action,
        "value": body.value,
        "correct_steps": correct_steps,
        "reward": reward,
        "reply": reply,
    }


@router.get("/auto", summary="Full LLM-driven auto-triage pipeline")
def auto_run():
    """
    End-to-end demo:
      1. Sample a random email
      2. Classify it (facebook/bart-large-mnli)
      3. Apply deterministic policy to choose an action
      4. Compute shaped reward against ground truth
      5. Generate a reply with GPT-2 if action == "reply"

    This is the full Email -> Model -> Category -> Policy -> Action ->
    Reward + Reply pipeline from the spec.
    """
    global _current_email
    _current_email = random.choice(_emails)
    email_text = _current_email["email_text"]
    correct_steps = _current_email["steps"]
    first_correct = correct_steps[0]["action"] if correct_steps else None

    category = classify_email(email_text)
    action = decide_action(category)
    reward = 10.0 if action == first_correct else -5.0

    reply = None
    if action == "reply":
        reply = generate_reply(email_text)

    return {
        "email": email_text,
        "category": category,
        "action": action,
        "correct_steps": correct_steps,
        "reward": reward,
        "reply": reply,
    }


app.include_router(router)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for: uv run --project . server"""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
