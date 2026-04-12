# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Email Triage RL Environment — LLM + RL + OpenEnv."""

from .client import EmailTriageEnv
from .models import (
    ACTION_SPACE,
    CATEGORY_SPACE,
    DEPARTMENT_SPACE,
    MAX_STEPS,
    ActionStep,
    EmailTriageAction,
    EmailTriageObservation,
)

__all__ = [
    "ActionStep",
    "EmailTriageAction",
    "EmailTriageObservation",
    "EmailTriageEnv",
    "ACTION_SPACE",
    "CATEGORY_SPACE",
    "DEPARTMENT_SPACE",
    "MAX_STEPS",
]
