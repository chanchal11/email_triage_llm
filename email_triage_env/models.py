# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Email Triage RL Environment.

Defines the action space (what the agent can do) and observation space
(what the agent sees after each step), following the OpenEnv interface.
"""

from typing import Literal, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation

# Valid actions the RL agent can take on an email
ACTION_SPACE = ["reply", "mark_spam", "mark_important", "ignore"]

# Email categories the LLM can assign
CATEGORY_SPACE = ["work", "spam", "personal", "promotion", "urgent"]


class EmailTriageAction(Action):
    """Action taken by the RL agent on a given email."""

    action: Literal["reply", "mark_spam", "mark_important", "ignore"] = Field(
        ..., description="Action to take: reply, mark_spam, mark_important, or ignore"
    )


class EmailTriageObservation(Observation):
    """Observation returned after each environment step."""

    email_text: str = Field(default="", description="Body text of the email")
    subject: str = Field(default="", description="Email subject (if available)")
    sender: str = Field(default="", description="Email sender (if available)")
    category: str = Field(
        default="", description="LLM-assigned category: work/spam/personal/promotion/urgent"
    )
    action_taken: str = Field(default="", description="Action the agent chose")
    correct_action: str = Field(
        default="", description="Ground-truth correct action for this email"
    )
    reply: Optional[str] = Field(
        default=None, description="Auto-generated reply text (only when action=reply)"
    )
    step: int = Field(default=0, description="Current episode step count")
