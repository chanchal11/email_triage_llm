# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Email Triage RL Environment Client.

Wraps the remote EmailTriageEnvironment via the OpenEnv WebSocket protocol,
giving callers a clean Python interface identical to a local environment.

Example (server already running on port 8000):
    >>> from email_triage_env import EmailTriageEnv, EmailTriageAction
    >>>
    >>> with EmailTriageEnv(base_url="http://localhost:8000") as client:
    ...     obs = client.reset()
    ...     print("Email:", obs.email_text)
    ...
    ...     result = client.step(EmailTriageAction(action="reply"))
    ...     print("Reward:", result.reward)
    ...     print("Reply:", result.observation.reply)
"""

from typing import Dict, Optional

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core.env_client import EnvClient

from .models import EmailTriageAction, EmailTriageObservation


class EmailTriageEnv(EnvClient[EmailTriageAction, EmailTriageObservation, State]):
    """
    Client for the Email Triage RL Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with EmailTriageEnv(base_url="http://localhost:8000") as client:
        ...     obs = client.reset()
        ...     result = client.step(EmailTriageAction(action="mark_spam"))
        ...     print(result.reward)

    Example with Docker:
        >>> client = EmailTriageEnv.from_docker_image("email_triage_env-env:latest")
        >>> try:
        ...     obs = client.reset()
        ...     result = client.step(EmailTriageAction(action="reply"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: EmailTriageAction) -> Dict:
        """Convert EmailTriageAction to JSON payload for the WebSocket step message."""
        return {"action": action.action}

    def _parse_result(self, payload: Dict) -> StepResult[EmailTriageObservation]:
        """Parse the server's JSON response into a typed StepResult."""
        obs_data = payload.get("observation", {})
        observation = EmailTriageObservation(
            email_text=obs_data.get("email_text", ""),
            subject=obs_data.get("subject", ""),
            sender=obs_data.get("sender", ""),
            category=obs_data.get("category", ""),
            action_taken=obs_data.get("action_taken", ""),
            correct_action=obs_data.get("correct_action", ""),
            reply=obs_data.get("reply"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            step=obs_data.get("step", 0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server JSON into a State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
