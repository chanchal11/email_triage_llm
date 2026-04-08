# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Email Triage RL Environment — multi-step episode version.

Episode flow
------------
1. reset()  → agent receives the email text and category
2. step(action) × N  → agent submits actions one at a time (up to MAX_STEPS)
3. Episode ends when:
     • all correct steps have been matched, OR
     • the agent submits a wrong/extra action, OR
     • MAX_STEPS reached

Reward shaping (per step)
-------------------------
  Correct action in correct position : +10
  Correct action in wrong position   : +2   (partial credit)
  Wrong action                       : -5
  Missing required action after done : -3 each

New actions (beyond original 4)
--------------------------------
  route_to_department  value = department name
  high-priority "crisis"  (no value, must be first step for crisis emails)
"""

import json
import os
import random
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        ActionStep,
        EmailTriageAction,
        EmailTriageObservation,
        MAX_STEPS,
    )
    from ..llm import generate_reply
except ImportError:
    from models import ActionStep, EmailTriageAction, EmailTriageObservation, MAX_STEPS  # type: ignore[import]
    from llm import generate_reply  # type: ignore[import]

# ---------------------------------------------------------------------------
# Dataset loading  (checks JSONL first, then legacy CSV)
# ---------------------------------------------------------------------------
_TRAIN_JSONL_PATHS = [
    Path(os.environ["EMAIL_TRIAGE_TRAIN"]) if "EMAIL_TRIAGE_TRAIN" in os.environ else None,
    Path(__file__).parent.parent.parent / "docs" / "train_data.jsonl",
    Path("/app/docs/train_data.jsonl"),
]


def _load_emails() -> list[dict]:
    """Load email records from train_data.jsonl.

    Each record must have:
        email_text : str
        category   : str
        steps      : list[dict]  — ordered correct action steps
    """
    for p in _TRAIN_JSONL_PATHS:
        if p is not None and p.exists():
            records = []
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            if records:
                print(f"[ENV] Loaded {len(records)} emails from {p}")
                return records

    raise FileNotFoundError(
        "No training dataset found. "
        "Set EMAIL_TRIAGE_TRAIN=/path/to/train_data.jsonl "
        "or place docs/train_data.jsonl in the repo root."
    )


# ---------------------------------------------------------------------------
# Per-step reward helper
# ---------------------------------------------------------------------------

def _step_reward(submitted: ActionStep, correct_steps: list[ActionStep], step_idx: int) -> float:
    """Score one submitted action step against the ground-truth sequence."""
    if step_idx < len(correct_steps):
        expected = correct_steps[step_idx]
        if submitted.action == expected.action:
            # Check value for actions that have one (ignore value for reply — per spec)
            if submitted.action == "route_to_department":
                if submitted.value and expected.value and submitted.value.lower() == expected.value.lower():
                    return 10.0           # perfect match
                return 2.0               # right action, wrong dept
            return 10.0                  # correct (reply / crisis / mark_*)
        # Check if it appears anywhere later in the sequence (partial credit)
        if any(s.action == submitted.action for s in correct_steps[step_idx + 1:]):
            return 2.0
        return -5.0                      # completely wrong action
    # Agent submitted an extra step beyond the correct sequence
    return -3.0


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class EmailTriageEnvironment(Environment):
    """
    Multi-step RL environment for email triage.

    State   : one email (email_text, category, correct steps)
    Actions : one EmailTriageAction per step (up to MAX_STEPS)
    Reward  : per-step shaped reward (see _step_reward)

    Call reset() to start a new episode, then step() repeatedly.

    Example:
        >>> env = EmailTriageEnvironment()
        >>> obs = env.reset()
        >>> # obs.email_text contains the email; obs.correct_steps shows what to do
        >>> obs = env.step(EmailTriageAction(action='high-priority "crisis"'))
        >>> obs = env.step(EmailTriageAction(action="mark_important"))
        >>> obs = env.step(EmailTriageAction(action="route_to_department", value="Tech Support"))
        >>> # obs.episode_done is True after all correct steps submitted
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._emails = _load_emails()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_email: dict | None = None
        self._correct_steps: list[ActionStep] = []
        self._step_idx: int = 0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> EmailTriageObservation:
        """Start a new episode — sample a random email from the training set."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_email = random.choice(self._emails)
        self._correct_steps = [
            ActionStep.from_dict(s) for s in self._current_email["steps"]
        ]
        self._step_idx = 0

        return EmailTriageObservation(
            email_text=self._current_email["email_text"],
            subject=self._current_email.get("subject", ""),
            sender=self._current_email.get("sender", ""),
            category=self._current_email.get("category", ""),
            action_taken="",
            action_value=None,
            correct_steps=[s.to_dict() for s in self._correct_steps],
            step=0,
            steps_remaining=len(self._correct_steps),
            episode_done=False,
            reply=None,
            done=False,
            reward=0.0,
        )

    def step(self, action: EmailTriageAction) -> EmailTriageObservation:
        """Submit one triage action for the current email."""
        if self._current_email is None:
            raise RuntimeError("call reset() before step()")

        submitted = action.to_step()
        reward = _step_reward(submitted, self._correct_steps, self._step_idx)

        self._state.step_count += 1
        self._step_idx += 1

        # Generate a reply when the action is 'reply'
        reply_text: str | None = None
        if submitted.action == "reply":
            effective_reply = submitted.value or generate_reply(self._current_email["email_text"])
            reply_text = effective_reply

        # Episode ends: all expected steps consumed, wrong action, or MAX_STEPS reached
        all_submitted = self._step_idx >= len(self._correct_steps)
        over_limit = self._step_idx >= MAX_STEPS
        episode_done = all_submitted or over_limit or reward < 0

        steps_remaining = max(0, len(self._correct_steps) - self._step_idx)

        return EmailTriageObservation(
            email_text=self._current_email["email_text"],
            subject=self._current_email.get("subject", ""),
            sender=self._current_email.get("sender", ""),
            category=self._current_email.get("category", ""),
            action_taken=submitted.action,
            action_value=submitted.value,
            correct_steps=[s.to_dict() for s in self._correct_steps],
            step=self._step_idx,
            steps_remaining=steps_remaining,
            episode_done=episode_done,
            reply=reply_text,
            done=episode_done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state

