# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Email Triage RL Environment.

Defines the action space (what the agent can do) and observation space
(what the agent sees after each step), following the OpenEnv interface.

Action types
------------
- reply                    : respond to the email (value = suggested reply text)
- mark_spam                : flag as spam / scam (no value)
- mark_important           : flag as important (no value)
- ignore                   : low-priority, take no action (no value)
- route_to_department      : forward to a specific team (value = department name)
- high-priority "crisis"   : trigger crisis escalation (no value; always first step)

Departments: HR | Management | Tech Support | Billing & Finance | Legal |
             Business Team | Customer Support | Sales | Operations | Security
"""

from typing import Any, Literal, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_SPACE = [
    "reply",
    "mark_spam",
    "mark_important",
    "ignore",
    "route_to_department",
    'high-priority "crisis"',
]

CATEGORY_SPACE = ["work", "spam", "personal", "promotion", "urgent"]

DEPARTMENT_SPACE = [
    "HR",
    "Management",
    "Tech Support",
    "Billing & Finance",
    "Legal",
    "Business Team",
    "Customer Support",
    "Sales",
    "Operations",
    "Security",
]

# Actions that carry a free-text or enumerated value
VALUE_ACTIONS = {"reply", "route_to_department"}

# Maximum number of steps per episode
MAX_STEPS = 5


# ---------------------------------------------------------------------------
# Individual action step
# ---------------------------------------------------------------------------

class ActionStep:
    """
    One step in a multi-step triage sequence.

    Attributes:
        action: The action name (see ACTION_SPACE).
        value:  Optional payload:
                  - reply              → reply text
                  - route_to_department → department name (see DEPARTMENT_SPACE)
                  - all others         → None
    """

    def __init__(self, action: str, value: Optional[str] = None):
        if action not in ACTION_SPACE:
            raise ValueError(f"Unknown action '{action}'. Must be one of {ACTION_SPACE}")
        if action in VALUE_ACTIONS and value is None:
            raise ValueError(f"Action '{action}' requires a 'value'.")
        if action not in VALUE_ACTIONS and value is not None:
            raise ValueError(f"Action '{action}' does not accept a 'value'.")
        self.action = action
        self.value = value

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"action": self.action}
        if self.value is not None:
            d["value"] = self.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ActionStep":
        return cls(action=d["action"], value=d.get("value"))

    def __repr__(self) -> str:
        if self.value:
            return f"ActionStep(action={self.action!r}, value={self.value!r})"
        return f"ActionStep(action={self.action!r})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ActionStep):
            return NotImplemented
        return self.action == other.action and self.value == other.value


# ---------------------------------------------------------------------------
# OpenEnv action type  (wraps a single step submitted at a time)
# ---------------------------------------------------------------------------

class EmailTriageAction(Action):
    """Action submitted by the RL agent for the *current* triage step."""

    action: str = Field(
        ...,
        description=(
            "Action to take. One of: reply | mark_spam | mark_important | "
            "ignore | route_to_department | high-priority \"crisis\""
        ),
    )
    value: Optional[str] = Field(
        default=None,
        description=(
            "Payload for actions that need one: "
            "reply text (action=reply) or department name (action=route_to_department)."
        ),
    )

    def to_step(self) -> ActionStep:
        return ActionStep(action=self.action, value=self.value)


# ---------------------------------------------------------------------------
# OpenEnv observation type
# ---------------------------------------------------------------------------

class EmailTriageObservation(Observation):
    """Observation returned after each environment step."""

    email_text: str = Field(default="", description="Body text of the email")
    subject: str = Field(default="", description="Email subject (if available)")
    sender: str = Field(default="", description="Email sender (if available)")
    category: str = Field(
        default="",
        description="Email category: work | spam | personal | promotion | urgent",
    )
    # Current step's action
    action_taken: str = Field(default="", description="Action the agent chose this step")
    action_value: Optional[str] = Field(
        default=None, description="Value attached to the action (reply text or department)"
    )
    # Ground-truth steps for the entire episode
    correct_steps: list = Field(
        default_factory=list,
        description="Ordered list of correct ActionStep dicts for this email",
    )
    # Progress
    step: int = Field(default=0, description="Current step index within the episode (0-based)")
    steps_remaining: int = Field(default=0, description="How many more steps are expected")
    episode_done: bool = Field(default=False, description="True when the episode is complete")
    # Reply generated by LLM (only when action=reply)
    reply: Optional[str] = Field(
        default=None, description="Auto-generated reply text (only when action=reply)"
    )
