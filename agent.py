#!/usr/bin/env python3
"""
Email Triage RL Agent — Q-Learning Demo (multi-step edition)
=============================================================

Trains a Q-learning agent to predict the *ordered sequence of triage actions*
for each email, supporting the full expanded action set:

  reply | mark_spam | mark_important | ignore | route_to_department | high-priority "crisis"

The *state* is the email category (5 possible values).
For routing actions the state is enriched with the category, so the Q-table
learns (category, step_idx) → best_action.

Training loop (multi-step contextual bandit):
  For each episode:
    1. Sample a random email from train_data.jsonl
    2. For each correct step in the ground-truth sequence (up to MAX_STEPS):
         a. Choose an action with epsilon-greedy exploration
         b. Compute shaped per-step reward
         c. Update Q-table with TD(0) update
    3. Decay epsilon

Usage:
    python agent.py --no-llm --episodes 200       # fast demo
    python agent.py --save models/qtable.json      # train & save
    python agent.py --test --load models/qtable.json   # evaluate
    python agent.py --server http://localhost:8000  # vs live server
"""

from __future__ import annotations

import argparse
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
CRISIS_ACTION = 'high-priority "crisis"'

ACTIONS = [
    CRISIS_ACTION,
    "mark_important",
    "mark_spam",
    "ignore",
    "reply",
    "route_to_department",
]

DEPARTMENTS = [
    "HR", "Management", "Tech Support", "Billing & Finance",
    "Legal", "Business Team", "Customer Support", "Sales",
    "Operations", "Security",
]

CATEGORIES = ["work", "spam", "personal", "promotion", "urgent"]
MAX_STEPS = 5

DOCS = Path(__file__).parent / "docs"
TRAIN_JSONL = Path(os.environ.get("EMAIL_TRIAGE_TRAIN", str(DOCS / "train_data.jsonl")))
TEST_JSONL  = Path(os.environ.get("EMAIL_TRIAGE_TEST",  str(DOCS / "test_data.jsonl")))


# ---------------------------------------------------------------------------
# Q-Learning Agent
# ---------------------------------------------------------------------------
class QAgent:
    """
    Tabular Q-learning agent for multi-step email triage.

    State key: "<category>:<step_idx>"  (e.g. "urgent:0", "spam:0")
    Action   : one of ACTIONS (for route_to_department, the value is
                chosen greedily from a sub-table keyed by category)
    """

    def __init__(
        self,
        alpha: float = 0.3,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.97,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q[state_key][action] → expected reward
        self.Q: dict[str, dict[str, float]] = defaultdict(
            lambda: {a: 0.0 for a in ACTIONS}
        )
        # Sub-table for route_to_department value (department) selection
        self.Q_dept: dict[str, dict[str, float]] = defaultdict(
            lambda: {d: 0.0 for d in DEPARTMENTS}
        )

        self.episode_rewards: list[float] = []

    def _state_key(self, category: str, step_idx: int) -> str:
        return f"{category}:{step_idx}"

    def choose_action(self, category: str, step_idx: int) -> tuple[str, Optional[str]]:
        """Epsilon-greedy: returns (action, value_or_None)."""
        key = self._state_key(category, step_idx)
        if random.random() < self.epsilon:
            action = random.choice(ACTIONS)
        else:
            action = max(self.Q[key], key=self.Q[key].get)

        value: Optional[str] = None
        if action == "route_to_department":
            if random.random() < self.epsilon:
                value = random.choice(DEPARTMENTS)
            else:
                value = max(self.Q_dept[category], key=self.Q_dept[category].get)
        elif action == "reply":
            value = "Thank you for your email. I will get back to you shortly."

        return action, value

    def update(
        self,
        category: str,
        step_idx: int,
        action: str,
        reward: float,
        dept_value: Optional[str] = None,
    ):
        key = self._state_key(category, step_idx)
        old = self.Q[key][action]
        self.Q[key][action] = old + self.alpha * (reward - old)

        if action == "route_to_department" and dept_value is not None:
            old_d = self.Q_dept[category][dept_value]
            self.Q_dept[category][dept_value] = old_d + self.alpha * (reward - old_d)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_table":     {k: dict(v) for k, v in self.Q.items()},
            "q_dept":      {k: dict(v) for k, v in self.Q_dept.items()},
            "hyperparameters": {
                "alpha": self.alpha, "gamma": self.gamma,
                "epsilon": self.epsilon, "epsilon_min": self.epsilon_min,
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
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        hp = payload.get("hyperparameters", {})
        agent = cls(
            alpha=hp.get("alpha", 0.3),
            gamma=hp.get("gamma", 0.9),
            epsilon=hp.get("epsilon", 0.05),
            epsilon_min=hp.get("epsilon_min", 0.05),
            epsilon_decay=hp.get("epsilon_decay", 0.97),
        )
        for k, v in payload.get("q_table", {}).items():
            agent.Q[k] = {a: v.get(a, 0.0) for a in ACTIONS}
        for k, v in payload.get("q_dept", {}).items():
            agent.Q_dept[k] = {d: v.get(d, 0.0) for d in DEPARTMENTS}
        print(f"[LOAD] Q-table loaded ← {path}")
        return agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_emails(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {jsonl_path}\n"
            "Run: python scripts/generate_data.py"
        )
    records = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[DATA] Loaded {len(records)} emails from {jsonl_path}")
    return records


def step_reward(
    action: str,
    value: Optional[str],
    correct_steps: list[dict],
    step_idx: int,
) -> float:
    """Compute reward for one submitted step vs ground-truth sequence."""
    if step_idx >= len(correct_steps):
        return -3.0   # extra step beyond expected sequence

    exp = correct_steps[step_idx]
    exp_action = exp.get("action", "")
    exp_value  = exp.get("value")

    if action == exp_action:
        if action == "route_to_department":
            if value and exp_value and value.lower() == exp_value.lower():
                return 10.0
            return 2.0    # right action, wrong department
        return 10.0       # correct (reply value not checked per spec)

    # Partial credit if this action appears later in the sequence
    if any(s.get("action") == action for s in correct_steps[step_idx + 1:]):
        return 2.0

    return -5.0


# ---------------------------------------------------------------------------
# Server-backed mode
# ---------------------------------------------------------------------------

def run_server_episodes(server_url: str, episodes: int):
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
        print(
            f"[{ep:03d}] email={data.get('email','')[:40]!r:<42} "
            f"reward={reward:+5.1f}"
        )

    avg = sum(rewards) / len(rewards) if rewards else 0
    print(f"\nAverage reward over {len(rewards)} episodes: {avg:+.2f}")


# ---------------------------------------------------------------------------
# Local multi-step Q-learning training loop
# ---------------------------------------------------------------------------

def run_local_training(
    episodes: int,
    seed: int,
    preload_path: Optional[str] = None,
):
    random.seed(seed)
    emails = load_emails(TRAIN_JSONL)

    agent = QAgent.load(preload_path) if preload_path else QAgent()

    print(f"\n{'='*60}")
    print(f"Multi-step Q-Learning Email Triage Agent")
    print(f"Episodes: {episodes}  |  Seed: {seed}")
    print(f"{'='*60}\n")

    for ep in range(1, episodes + 1):
        email = random.choice(emails)
        category = email.get("category", "work")
        correct_steps = email.get("steps", [])

        ep_reward = 0.0
        all_correct = True

        for step_idx in range(min(len(correct_steps), MAX_STEPS)):
            action, value = agent.choose_action(category, step_idx)
            reward = step_reward(action, value, correct_steps, step_idx)
            agent.update(category, step_idx, action, reward, value if action == "route_to_department" else None)
            ep_reward += reward
            if reward < 5.0:
                all_correct = False

        agent.decay_epsilon()
        agent.episode_rewards.append(ep_reward)

        if ep <= 20 or ep % max(1, episodes // 10) == 0 or ep == episodes:
            mark = "✓" if all_correct else "✗"
            first_correct = correct_steps[0] if correct_steps else {}
            print(
                f"[{ep:04d}] {mark} "
                f"cat={category:12s} "
                f"first_expected={first_correct.get('action','?'):28s} "
                f"ep_reward={ep_reward:+6.1f}  ε={agent.epsilon:.3f}"
            )

    # Summary
    total = len(agent.episode_rewards)
    avg   = sum(agent.episode_rewards) / total
    print(f"\n{'='*60}")
    print(f"Training Complete — {total} episodes, avg reward/episode: {avg:+.2f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    return agent


# ---------------------------------------------------------------------------
# Evaluation on test set
# ---------------------------------------------------------------------------

def run_test(model_path: str, seed: int):
    agent = QAgent.load(model_path)
    agent.epsilon = 0.0  # greedy

    random.seed(seed)
    emails = load_emails(TEST_JSONL)

    print(f"\n{'='*60}")
    print(f"Evaluation (greedy policy)  | {len(emails)} test emails")
    print(f"{'='*60}\n")

    total_reward = 0.0
    perfect_episodes = 0

    for email in emails:
        category = email.get("category", "work")
        email_text = email["email_text"]
        correct_steps = email.get("steps", [])

        ep_reward = 0.0
        all_ok = True
        pred_summary = []

        for step_idx in range(min(len(correct_steps), MAX_STEPS)):
            key = f"{category}:{step_idx}"
            if key in agent.Q:
                action = max(agent.Q[key], key=agent.Q[key].get)
            else:
                action = "ignore"  # unseen state fallback

            value: Optional[str] = None
            if action == "route_to_department":
                if category in agent.Q_dept:
                    value = max(agent.Q_dept[category], key=agent.Q_dept[category].get)
                else:
                    value = DEPARTMENTS[0]

            reward = step_reward(action, value, correct_steps, step_idx)
            ep_reward += reward
            if reward < 5.0:
                all_ok = False
            step_dict = {"action": action}
            if value:
                step_dict["value"] = value
            pred_summary.append(step_dict)

        total_reward += ep_reward
        if all_ok:
            perfect_episodes += 1

        mark = "✓" if all_ok else "✗"
        print(
            f"{mark} ep_reward={ep_reward:+5.1f}  "
            f"cat={category:12s}  "
            f"email={email_text[:42]!r}"
        )
        print(f"    expected : {json.dumps(correct_steps, ensure_ascii=False)[:80]}")
        print(f"    got      : {json.dumps(pred_summary,  ensure_ascii=False)[:80]}")

    n = len(emails)
    print(f"\n{'='*60}")
    print(f"  Emails tested         : {n}")
    print(f"  Perfect sequences     : {perfect_episodes}/{n} ({100*perfect_episodes/n:.1f}%)")
    print(f"  Avg reward / episode  : {total_reward/n:+.2f}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Email Triage Multi-step Q-Learning Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--server", type=str, default="",
                        help="Run against live server URL (e.g. http://localhost:8000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="models/qtable.json")
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--test", action="store_true",
                        help="Evaluate saved model on test set (requires --load)")
    args = parser.parse_args()

    if args.server:
        run_server_episodes(args.server.rstrip("/"), args.episodes)
    elif args.test:
        if not args.load:
            print("ERROR: --test requires --load <path>")
            sys.exit(1)
        run_test(args.load, seed=args.seed)
    else:
        agent = run_local_training(
            episodes=args.episodes,
            seed=args.seed,
            preload_path=args.load or None,
        )
        if args.save:
            agent.save(args.save)


if __name__ == "__main__":
    main()

