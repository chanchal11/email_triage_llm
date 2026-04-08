# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Email Triage RL Environment Implementation.

Each episode presents the agent with one email sampled from the dataset.
The agent selects an action (reply / mark_spam / mark_important / ignore)
and receives a shaped reward computed against the ground-truth label.

When action == "reply" the environment also generates an LLM reply.
"""

import csv
import random
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Support two execution contexts:
#   1. Local dev: run as package  -> use relative imports
#   2. Docker:    PYTHONPATH=/app/env -> bare module names work
try:
    from ..models import EmailTriageAction, EmailTriageObservation  # local package
    from ..llm import compute_reward, generate_reply
except ImportError:
    from models import EmailTriageAction, EmailTriageObservation  # type: ignore[import]
    from llm import compute_reward, generate_reply  # type: ignore[import]

# Path to the evaluation dataset — works both in Docker and local dev
_CSV_PATHS = [
    Path(__file__).parent.parent.parent / "docs" / "email_test_data.csv",
    Path("/app/docs/email_test_data.csv"),
]


def _load_emails() -> list:
    """Load emails from the CSV dataset, trying known paths in order."""
    for csv_path in _CSV_PATHS:
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
    # Fallback: small in-memory dataset so the server can still start
    return [
        {"email_text": "Urgent meeting at 5 PM today", "true_label": "mark_important"},
        {"email_text": "You have won a lottery! Claim now", "true_label": "mark_spam"},
        {"email_text": "Can we schedule a meeting tomorrow?", "true_label": "reply"},
        {"email_text": "Big sale on electronics this weekend", "true_label": "ignore"},
        {"email_text": "Reminder: project deadline tomorrow", "true_label": "mark_important"},
        {"email_text": "Exclusive offer just for you!!!", "true_label": "mark_spam"},
        {"email_text": "Team meeting rescheduled to Friday", "true_label": "reply"},
    ]


class EmailTriageEnvironment(Environment):
    """
    RL environment for email triage.

    State  : one email (email_text, true_label)
    Actions: reply | mark_spam | mark_important | ignore
    Reward : shaped reward from compute_reward() in llm.py

    Each episode is a single step (one email -> one action -> reward -> done).
    Call reset() to load the next email.

    Example:
        >>> env = EmailTriageEnvironment()
        >>> obs = env.reset()
        >>> print(obs.email_text)
        >>>
        >>> obs = env.step(EmailTriageAction(action="reply"))
        >>> print(obs.reward, obs.reply)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._emails = _load_emails()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_email = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> EmailTriageObservation:
        """
        Start a new episode by sampling a random email from the dataset.

        Returns:
            Observation containing the email text; reward is 0.0 at this stage.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_email = random.choice(self._emails)

        return EmailTriageObservation(
            email_text=self._current_email["email_text"],
            subject=self._current_email.get("subject", ""),
            sender=self._current_email.get("sender", ""),
            category="",
            action_taken="",
            correct_action=self._current_email["true_label"],
            reply=None,
            done=False,
            reward=0.0,
            step=0,
        )

    def step(self, action: EmailTriageAction) -> EmailTriageObservation:
        """
        Execute the agent's action on the current email.

        Args:
            action: EmailTriageAction with the chosen action string.

        Returns:
            Observation with reward signal and (if action=="reply") a GPT-2 reply.
        """
        if self._current_email is None:
            raise RuntimeError("call reset() before step()")

        self._state.step_count += 1
        correct = self._current_email["true_label"]
        reward = compute_reward(action.action, correct)

        reply = None
        if action.action == "reply":
            reply = generate_reply(self._current_email["email_text"])

        return EmailTriageObservation(
            email_text=self._current_email["email_text"],
            subject=self._current_email.get("subject", ""),
            sender=self._current_email.get("sender", ""),
            category="",
            action_taken=action.action,
            correct_action=correct,
            reply=reply,
            done=True,
            reward=reward,
            step=self._state.step_count,
        )

    @property
    def state(self) -> State:
        return self._state
