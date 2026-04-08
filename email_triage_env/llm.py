# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LLM module for Email Triage.

Provides:
  - classify_email()  : zero-shot classification via facebook/bart-large-mnli
  - decide_action()   : deterministic policy mapping category → action
  - generate_reply()  : text generation via gpt2
  - compute_reward()  : nuanced RL reward signal
"""

from __future__ import annotations

import os
# Force transformers to use PyTorch backend only.
# Prevents broken TensorFlow installations from intercepting the pipeline import.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")

from typing import Optional

# ---------------------------------------------------------------------------
# Lazy-loaded singletons — models are large; load only on first use
# ---------------------------------------------------------------------------
_classifier = None
_generator = None

EMAIL_CATEGORIES = ["work", "spam", "personal", "promotion", "urgent"]

# Maps category → correct action (deterministic policy baseline)
CATEGORY_TO_ACTION: dict[str, str] = {
    "work": "reply",
    "spam": "mark_spam",
    "personal": "ignore",
    "promotion": "ignore",
    "urgent": "mark_important",
}

# Reward table: (action_taken, correct_action) → reward
# Designed so that:
#   - exact match                     →  +10
#   - missing urgent (mark it spam)   →  -10  (dangerous mistake)
#   - ignoring urgent                 →   -8  (bad)
#   - replying to urgent              →   -2  (close, but inefficient)
#   - most other mismatches           →   -5
_REWARD_TABLE: dict[tuple[str, str], float] = {
    ("mark_spam", "mark_important"): -10.0,
    ("ignore", "mark_important"): -8.0,
    ("reply", "mark_important"): -2.0,
    ("mark_important", "mark_spam"): -7.0,
    ("reply", "mark_spam"): -7.0,
}


def _get_classifier():
    global _classifier
    if _classifier is None:
        from transformers import pipeline  # type: ignore

        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _classifier


def _get_generator():
    global _generator
    if _generator is None:
        from transformers import pipeline  # type: ignore

        _generator = pipeline(
            "text-generation",
            model="gpt2",
            pad_token_id=50256,  # suppress padding warning
        )
    return _generator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_email(email_text: str) -> str:
    """
    Zero-shot classification of an email into one of the predefined categories.

    Args:
        email_text: Raw email body.

    Returns:
        One of: "work", "spam", "personal", "promotion", "urgent".
    """
    classifier = _get_classifier()
    result = classifier(email_text, candidate_labels=EMAIL_CATEGORIES)
    return result["labels"][0]  # highest-confidence label


def decide_action(category: str) -> str:
    """
    Deterministic policy: maps a category to the canonical action.

    Args:
        category: Email category string.

    Returns:
        One of: "reply", "mark_spam", "mark_important", "ignore".
    """
    return CATEGORY_TO_ACTION.get(category, "ignore")


def generate_reply(email_text: str, max_new_tokens: int = 80) -> str:
    """
    Generate a short professional reply to an email using GPT-2.

    Args:
        email_text: The email body to reply to.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        Generated reply string (fallback used when output is too short).
    """
    generator = _get_generator()
    prompt = f"Professional email reply to: {email_text}\nReply:"
    output = generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)

    raw: str = output[0]["generated_text"]
    # Strip the prompt prefix so only the reply is returned
    reply = raw[len(prompt):].strip()

    if len(reply) < 15:
        return "Thank you for your email. I will review it and get back to you shortly."
    return reply


def compute_reward(action: str, correct_action: str) -> float:
    """
    Compute a nuanced RL reward for the agent's chosen action.

    The reward is designed to:
      - Strongly reward exact matches (+10)
      - Penalise dangerous mistakes more heavily (e.g. marking urgent as spam)
      - Apply moderate penalties for other wrong actions (-5 default)

    Args:
        action: The action the agent chose.
        correct_action: The ground-truth correct action.

    Returns:
        Float reward value.
    """
    if action == correct_action:
        return 10.0
    return _REWARD_TABLE.get((action, correct_action), -5.0)


def classify_and_act(email_text: str) -> tuple[str, str, Optional[str]]:
    """
    Convenience helper: classify an email, decide action, and generate a reply
    if the action is "reply".

    Returns:
        (category, action, reply_or_None)
    """
    category = classify_email(email_text)
    action = decide_action(category)
    reply: Optional[str] = None
    if action == "reply":
        reply = generate_reply(email_text)
    return category, action, reply
