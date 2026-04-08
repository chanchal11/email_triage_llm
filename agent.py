#!/usr/bin/env python3
"""
Email Triage RL Agent — Q-Learning Demo
========================================

This script demonstrates reinforcement learning applied to email triage.

The *state* is the LLM-assigned email category (5 possible values).
The *action space* is {reply, mark_spam, mark_important, ignore} (4 actions).
A Q-table (5 x 4) maps every (state, action) pair to an expected reward.

Training loop (contextual bandit, no state transitions):
  For each episode:
    1. Sample a random email from the dataset
    2. Classify it with the LLM (or use ground-truth label for speed)
    3. Choose an action with epsilon-greedy exploration
    4. Compute a shaped reward via compute_reward()
    5. Update the Q-table with a Bellman-style TD update

Usage:
    python agent.py                          # run with LLM classification (slow)
    python agent.py --no-llm                 # use ground-truth labels (fast, for demo)
    python agent.py --episodes 100 --no-llm  # more episodes, no LLM
    python agent.py --server http://localhost:8000  # run against live server via REST
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTIONS = ["reply", "mark_spam", "mark_important", "ignore"]
CATEGORIES = ["work", "spam", "personal", "promotion", "urgent"]

# Ground-truth label → category mapping (used when --no-llm is set)
LABEL_TO_CATEGORY: dict[str, str] = {
    "reply": "work",
    "mark_spam": "spam",
    "mark_important": "urgent",
    "ignore": "promotion",
}

CSV_PATH = Path(
    os.environ.get("EMAIL_TRIAGE_DATA", str(Path(__file__).parent / "docs" / "email_test_data.csv"))
)


# ---------------------------------------------------------------------------
# Q-Learning Agent
# ---------------------------------------------------------------------------
class QAgent:
    """
    Tabular Q-learning agent for the email triage contextual bandit.

    Each state is an email category string; the Q-table maps
    (category, action) -> expected cumulative reward.
    """

    def __init__(
        self,
        alpha: float = 0.3,      # learning rate
        gamma: float = 0.9,      # discount factor (single-step -> ~irrelevant but kept for generality)
        epsilon: float = 1.0,    # initial exploration rate
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.97,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q[state][action] -> float
        self.Q: dict[str, dict[str, float]] = defaultdict(
            lambda: {a: 0.0 for a in ACTIONS}
        )

        # Tracking
        self.episode_rewards: list[float] = []
        self.episode_actions: list[str] = []

    def choose_action(self, state: str) -> str:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)          # explore
        return max(self.Q[state], key=self.Q[state].get)  # exploit

    def update(self, state: str, action: str, reward: float, next_state: Optional[str] = None):
        """
        TD(0) Q-table update.

        For a single-step bandit next_state is None and the update simplifies to:
            Q(s,a) <- Q(s,a) + alpha * (reward - Q(s,a))
        """
        if next_state is not None:
            best_next = max(self.Q[next_state].values())
        else:
            best_next = 0.0

        old = self.Q[state][action]
        td_target = reward + self.gamma * best_next
        self.Q[state][action] = old + self.alpha * (td_target - old)

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def best_policy(self) -> dict[str, str]:
        """Return the greedy policy derived from the current Q-table."""
        return {
            state: max(actions, key=actions.get)
            for state, actions in self.Q.items()
        }

    def save(self, path: str | Path) -> None:
        """
        Persist the Q-table and hyperparameters to a JSON file.

        Args:
            path: File path (e.g. "models/qtable.json").
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_table": {state: dict(actions) for state, actions in self.Q.items()},
            "hyperparameters": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
            "training_stats": {
                "episodes": len(self.episode_rewards),
                "avg_reward": (
                    sum(self.episode_rewards) / len(self.episode_rewards)
                    if self.episode_rewards else 0.0
                ),
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[SAVE] Q-table saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "QAgent":
        """
        Restore a Q-table from a JSON file saved by save().

        Args:
            path: File path to load from.

        Returns:
            QAgent instance with the restored Q-table.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)

        hp = payload.get("hyperparameters", {})
        agent = cls(
            alpha=hp.get("alpha", 0.3),
            gamma=hp.get("gamma", 0.9),
            epsilon=hp.get("epsilon", 0.05),    # start greedy after load
            epsilon_min=hp.get("epsilon_min", 0.05),
            epsilon_decay=hp.get("epsilon_decay", 0.97),
        )
        for state, actions in payload.get("q_table", {}).items():
            agent.Q[state] = {a: actions.get(a, 0.0) for a in ACTIONS}
        print(f"[LOAD] Q-table loaded ← {path}")
        return agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_emails() -> list[dict]:
    """Load emails from the CSV dataset (path set by EMAIL_TRIAGE_DATA or default)."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}\n"
            "Set EMAIL_TRIAGE_DATA=/path/to/emails.csv or place "
            "docs/email_test_data.csv in the repo root."
        )
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"[DATA] Loaded {len(rows)} emails from {CSV_PATH}")
    return rows


def compute_reward_local(action: str, correct_action: str) -> float:
    """Reward function (mirrors llm.py so this script is importable standalone)."""
    _REWARD_TABLE = {
        ("mark_spam", "mark_important"): -10.0,
        ("ignore", "mark_important"): -8.0,
        ("reply", "mark_important"): -2.0,
        ("mark_important", "mark_spam"): -7.0,
        ("reply", "mark_spam"): -7.0,
    }
    if action == correct_action:
        return 10.0
    return _REWARD_TABLE.get((action, correct_action), -5.0)


def get_category_from_label(true_label: str) -> str:
    """Map a true_label string to its canonical category (fast path, no LLM)."""
    return LABEL_TO_CATEGORY.get(true_label, "personal")


def get_category_from_llm(email_text: str) -> str:
    """Classify via facebook/bart-large-mnli (slow, requires transformers)."""
    try:
        from email_triage_env.llm import classify_email  # type: ignore
    except ImportError:
        # Fallback when running outside the package
        _root = Path(__file__).parent / "email_triage_env"
        sys.path.insert(0, str(_root))
        from llm import classify_email  # type: ignore
    return classify_email(email_text)


def running_average(values: list[float], window: int = 10) -> list[float]:
    """Compute a simple moving average."""
    avgs = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        avgs.append(sum(values[start : i + 1]) / (i - start + 1))
    return avgs


# ---------------------------------------------------------------------------
# Server-backed mode: use the /auto endpoint instead of local env
# ---------------------------------------------------------------------------
def run_server_episodes(server_url: str, episodes: int):
    """Run episodes against a live FastAPI server using the /auto endpoint."""
    try:
        import requests
    except ImportError:
        print("ERROR: 'requests' is required for --server mode. pip install requests")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Server mode: {server_url}/auto  ({episodes} episodes)")
    print(f"{'='*60}")

    rewards = []
    for ep in range(1, episodes + 1):
        try:
            resp = requests.get(f"{server_url}/auto", timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"Episode {ep}: ERROR — {exc}")
            continue

        reward = data.get("reward", 0)
        rewards.append(reward)
        correct = data.get("reward", 0) > 0
        print(
            f"[{ep:03d}] email={data.get('email','')[:40]!r:<42} "
            f"category={data.get('category',''):12s} "
            f"action={data.get('action',''):15s} "
            f"reward={reward:+5.1f}  {'✓' if correct else '✗'}"
        )

    avg = sum(rewards) / len(rewards) if rewards else 0
    print(f"\nAverage reward over {len(rewards)} episodes: {avg:+.2f}")


# ---------------------------------------------------------------------------
# Local Q-learning training loop
# ---------------------------------------------------------------------------
def run_local_training(episodes: int, use_llm: bool, seed: int, preload_path: Optional[str] = None):
    """Train a Q-learning agent directly against the local environment."""
    random.seed(seed)
    emails = load_emails()

    if preload_path:
        agent = QAgent.load(preload_path)
    else:
        agent = QAgent()

    print(f"\n{'='*60}")
    print(f"Q-Learning Email Triage Agent")
    print(f"Episodes: {episodes}  |  LLM: {use_llm}  |  Seed: {seed}")
    print(f"{'='*60}\n")

    for ep in range(1, episodes + 1):
        email = random.choice(emails)
        email_text = email["email_text"]
        true_label = email["true_label"]

        # Determine state (category)
        if use_llm:
            state = get_category_from_llm(email_text)
        else:
            state = get_category_from_label(true_label)

        # Agent picks action
        action = agent.choose_action(state)

        # Environment returns reward
        reward = compute_reward_local(action, true_label)

        # Q-table update (single-step bandit: no next_state)
        agent.update(state, action, reward)
        agent.decay_epsilon()

        agent.episode_rewards.append(reward)
        agent.episode_actions.append(action)

        # Progress logging
        correct_mark = "✓" if reward > 0 else "✗"
        if ep <= 20 or ep % (episodes // 10 + 1) == 0 or ep == episodes:
            print(
                f"[{ep:04d}] {correct_mark} "
                f"category={state:12s} "
                f"action={action:15s} "
                f"true={true_label:15s} "
                f"reward={reward:+5.1f}  "
                f"ε={agent.epsilon:.3f}"
            )

    # ---------------------------------------------------------------------------
    # Results summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Training Complete — Results")
    print(f"{'='*60}")

    total = len(agent.episode_rewards)
    wins = sum(1 for r in agent.episode_rewards if r > 0)
    avg_reward = sum(agent.episode_rewards) / total

    print(f"  Episodes       : {total}")
    print(f"  Correct actions: {wins}/{total} ({100*wins/total:.1f}%)")
    print(f"  Average reward : {avg_reward:+.2f}")
    print(f"  Final epsilon  : {agent.epsilon:.4f}")

    print("\nLearned Q-Policy (greedy):")
    for state, best_action in sorted(agent.best_policy().items()):
        q_vals = "  ".join(f"{a}={agent.Q[state][a]:+.1f}" for a in ACTIONS)
        print(f"  {state:12s} → {best_action:15s}  [{q_vals}]")

    print("\nFirst-10-episode vs Last-10-episode average reward:")
    first_10 = sum(agent.episode_rewards[:10]) / 10
    last_10 = sum(agent.episode_rewards[-10:]) / 10
    improvement = last_10 - first_10
    print(f"  First 10: {first_10:+.2f}")
    print(f"  Last 10 : {last_10:+.2f}")
    print(f"  Change  : {improvement:+.2f}  ({'improved' if improvement > 0 else 'no improvement'})")

    return agent


# ---------------------------------------------------------------------------
# Test / evaluation loop (uses saved weights, epsilon=0 → fully greedy)
# ---------------------------------------------------------------------------
def run_test(model_path: str, use_llm: bool, seed: int):
    """Load a saved Q-table and evaluate it on the full dataset."""
    agent = QAgent.load(model_path)
    agent.epsilon = 0.0          # pure exploitation — no random actions

    random.seed(seed)
    emails = load_emails()

    print(f"\n{'='*60}")
    print(f"Evaluation (greedy policy loaded from {model_path})")
    print(f"LLM: {use_llm}  |  Dataset: {len(emails)} emails")
    print(f"{'='*60}\n")

    total_reward = 0.0
    correct = 0
    rows = []

    for email in emails:
        email_text = email["email_text"]
        true_label = email["true_label"]

        if use_llm:
            state = get_category_from_llm(email_text)
        else:
            state = get_category_from_label(true_label)

        # Greedy action
        if state in agent.Q:
            action = max(agent.Q[state], key=agent.Q[state].get)
        else:
            action = "ignore"    # unknown state fallback

        reward = compute_reward_local(action, true_label)
        total_reward += reward
        if reward > 0:
            correct += 1

        mark = "✓" if reward > 0 else "✗"
        print(
            f"{mark} category={state:12s}  action={action:15s}  "
            f"true={true_label:15s}  reward={reward:+5.1f}  "
            f"email={email_text[:40]!r}"
        )
        rows.append({"email": email_text, "category": state, "action": action,
                     "correct_action": true_label, "reward": reward})

    n = len(emails)
    print(f"\n{'='*60}")
    print(f"  Emails tested  : {n}")
    print(f"  Correct actions: {correct}/{n} ({100*correct/n:.1f}%)")
    print(f"  Total reward   : {total_reward:+.1f}")
    print(f"  Avg reward     : {total_reward/n:+.2f}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Email Triage RL Agent (Q-Learning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of training episodes (default: 50)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM classification; use ground-truth labels as state (much faster)",
    )
    parser.add_argument(
        "--server", type=str, default="",
        help="If set, run against this server URL (e.g. http://localhost:8000) via /auto endpoint",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save", type=str, default="models/qtable.json",
        help="Path to save Q-table weights after training (default: models/qtable.json)",
    )
    parser.add_argument(
        "--load", type=str, default="",
        help="Path to load Q-table weights from before training (resume training)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Evaluate a saved model (use with --load); no training is done",
    )
    args = parser.parse_args()

    if args.server:
        run_server_episodes(args.server.rstrip("/"), args.episodes)
    elif args.test:
        if not args.load:
            print("ERROR: --test requires --load <path>")
            sys.exit(1)
        run_test(args.load, use_llm=not args.no_llm, seed=args.seed)
    else:
        agent = run_local_training(
            episodes=args.episodes,
            use_llm=not args.no_llm,
            seed=args.seed,
            preload_path=args.load or None,
        )
        if args.save:
            agent.save(args.save)


if __name__ == "__main__":
    main()
