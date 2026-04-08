#!/usr/bin/env python3
"""
Email Triage — Inference Script
================================
There are TWO separate services:
  1. LLM API  — controlled by API_BASE_URL / MODEL_NAME / HF_TOKEN
  2. Env server — the OpenEnv FastAPI server (HF Space or local Docker)
                  controlled by ENV_SERVER_URL

Environment variables:
    HF_TOKEN          HuggingFace token (for LLM API calls)
    API_BASE_URL      LLM API endpoint
                        HF router : https://router.huggingface.co/v1  (default)
                        Local vLLM: http://localhost:8080/v1
    MODEL_NAME        LLM model ID (default: Qwen/Qwen2.5-0.5B-Instruct)

    ENV_SERVER_URL    OpenEnv environment server URL; when set, emails are
                        fetched live from GET /reset instead of the local JSONL.
                        HF Space : https://chanchal11-email-triage.hf.space
                        Local    : http://localhost:8000

    GRPO_MODEL_PATH   Path to local fine-tuned model; overrides the API backend.
    EMAIL_TRIAGE_TEST Path to test JSONL (default: docs/test_data.jsonl)
    EPISODES          Number of episodes to run (default: all test emails)

STDOUT FORMAT
  [START] task=<task> env=email_triage model=<model>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Usage:
    # Against the deployed HF Space (recommended):
    HF_TOKEN=hf_... \
    ENV_SERVER_URL=https://chanchal11-email-triage.hf.space \
    MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct \
    python inference.py

    # Local Docker + HF LLM:
    HF_TOKEN=hf_... \
    ENV_SERVER_URL=http://localhost:8000 \
    python inference.py

    # Local JSONL (no env server needed):
    HF_TOKEN=hf_... python inference.py

    # Local fine-tuned GRPO model:
    GRPO_MODEL_PATH=models/grpo_email_triage python inference.py

    # Limit to 10 episodes:
    EPISODES=10 HF_TOKEN=hf_... ENV_SERVER_URL=https://chanchal11-email-triage.hf.space python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

# ── Force PyTorch backend (prevents broken TF from intercepting transformers) ─
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")

# ─────────────────────────── Configuration ────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-0.5B-Instruct")
GRPO_MODEL_PATH = os.getenv("GRPO_MODEL_PATH", "")   # local fine-tuned model

# OpenEnv environment server (HF Space or local Docker)
# When set, emails are fetched live via GET /reset instead of the local JSONL.
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "").rstrip("/")
# e.g. https://chanchal11-email-triage.hf.space  OR  http://localhost:8000

TASK_NAME = "email_triage"
BENCHMARK = "email_triage_env"

DOCS = Path(__file__).parent / "docs"
TEST_JSONL = Path(os.environ.get("EMAIL_TRIAGE_TEST", str(DOCS / "test_data.jsonl")))

# Max steps per episode and reward per step
MAX_STEPS          = 5
MAX_REWARD_PER_STEP = 10.0
# Max possible reward for one episode (worst case: all 5 steps perfect)
MAX_EPISODE_REWARD  = MAX_STEPS * MAX_REWARD_PER_STEP   # 50.0

TEMPERATURE = 0.3   # low temperature → deterministic, better for triage
MAX_TOKENS  = 256   # enough for a JSON array with 3-4 steps

# An episode is "successful" if it achieves at least this fraction of max reward
SUCCESS_THRESHOLD = 0.5

# ─────────────────────────── Action constants ──────────────────────────────────
CRISIS_ACTION = 'high-priority "crisis"'
ALL_ACTIONS   = [
    "mark_spam", "ignore", "reply",
    "mark_important", "route_to_department",
    CRISIS_ACTION,
]
DEPARTMENTS = [
    "HR", "Management", "Tech Support", "Billing & Finance",
    "Legal", "Business Team", "Customer Support", "Sales",
    "Operations", "Security",
]
_VALUE_ACTIONS = {"reply", "route_to_department"}

# ─────────────────────────── Logging helpers ──────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action!r} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: list[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────── Dataset helpers ──────────────────────────────────

def fetch_emails_from_server(n: int) -> list[dict]:
    """Fetch *n* email episodes from the live environment server via GET /reset.

    Each call to /reset samples a new random email from the server's dataset
    and returns {email, category, correct_steps}.
    """
    import urllib.error
    import urllib.request

    url = f"{ENV_SERVER_URL}/reset"
    records: list[dict] = []
    print(f"[INFO] Fetching {n} emails from {url}", file=sys.stderr)
    for i in range(n):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
                data = json.loads(resp.read())
            records.append({
                "email_text": data["email"],
                "category":   data["category"],
                "steps":      data["correct_steps"],
            })
        except urllib.error.URLError as exc:
            print(f"[WARN] /reset call {i+1} failed: {exc}", file=sys.stderr)
    if not records:
        print(
            f"[ERROR] Could not fetch any emails from {ENV_SERVER_URL}/reset",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[INFO] Fetched {len(records)} emails from {ENV_SERVER_URL}", file=sys.stderr)
    return records


def load_test_emails() -> list[dict]:
    if not TEST_JSONL.exists():
        print(
            f"[ERROR] Test dataset not found: {TEST_JSONL}\n"
            "Run: python scripts/generate_data.py",
            file=sys.stderr,
        )
        sys.exit(1)
    records = []
    with open(TEST_JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[INFO] Loaded {len(records)} test emails from {TEST_JSONL}", file=sys.stderr)
    return records


# ─────────────────────────── Step extraction ──────────────────────────────────

def _normalize_step(action: str, value: Any) -> dict:
    """Strip value for actions that must not carry one."""
    if action not in _VALUE_ACTIONS:
        return {"action": action, "value": None}
    return {"action": action, "value": value if value else None}


def _preprocess_crisis(text: str) -> str:
    """Escape the inner quotes of crisis action so the JSON parser can handle it.

    Models frequently output:  {"action": "high-priority "crisis""}
    We need:                   {"action": "high-priority \"crisis\""}
    """
    return re.sub(
        r'"high-priority "crisis""',
        r'"high-priority \\"crisis\\""',
        text,
    )


def extract_steps(text: str) -> list[dict]:
    """Parse LLM output into a list of triage step dicts.

    Strategy (in order):
      1. Fix the common malformed crisis-action quoting, then JSON-parse.
      2. Try regex extraction of individual {"action": ...} objects.
      3. Fall back to keyword scan (first found action only).
    """
    text = text.strip()

    # 1. Try JSON array extraction after preprocessing
    candidate = _preprocess_crisis(text)
    match = re.search(r"\[.*\]", candidate, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                steps = []
                for item in parsed:
                    if isinstance(item, dict) and "action" in item:
                        steps.append(_normalize_step(item["action"], item.get("value")))
                if steps:
                    return steps
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. Regex extraction of individual action objects (resilient to bad JSON)
    #    Handles: {"action": "foo"} and {"action": "foo", "value": "bar"}
    obj_pattern = re.compile(
        r'\{\s*"action"\s*:\s*"([^"]*(?:"crisis"[^"]*)?)"\s*'
        r'(?:,\s*"value"\s*:\s*"([^"]*)")?\s*\}',
        re.DOTALL,
    )
    matches = obj_pattern.findall(text)
    if matches:
        steps = []
        for action_raw, value_raw in matches:
            # Restore the canonical crisis action string
            action = action_raw.strip()
            if "crisis" in action.lower():
                action = CRISIS_ACTION
            value = value_raw.strip() if value_raw else None
            steps.append(_normalize_step(action, value))
        if steps:
            return steps

    # 3. Keyword fallback — first found action only
    fallback_order = [
        "route_to_department", "mark_important", "mark_spam",
        CRISIS_ACTION, "ignore", "reply",
    ]
    text_lower = text.lower()
    for action in fallback_order:
        if action.lower() in text_lower:
            step: dict[str, Any] = {"action": action, "value": None}
            if action == "route_to_department":
                for dept in DEPARTMENTS:
                    if dept.lower() in text_lower:
                        step["value"] = dept
                        break
            return [step]

    return [{"action": "ignore", "value": None}]


# ─────────────────────────── Reward helper ────────────────────────────────────

def compute_step_reward(
    action: str,
    value: Optional[str],
    correct_steps: list[dict],
    step_idx: int,
) -> float:
    """Mirror the RL-environment's per-step reward logic."""
    if step_idx >= len(correct_steps):
        return -3.0   # extra step beyond expected

    exp_action = correct_steps[step_idx].get("action", "")
    exp_value  = correct_steps[step_idx].get("value")

    if action == exp_action:
        if action == "route_to_department":
            if value and exp_value and value.lower() == exp_value.lower():
                return 10.0
            return 2.0
        return 10.0

    if any(s.get("action") == action for s in correct_steps[step_idx + 1:]):
        return 2.0

    return -5.0


# ────────────────────── System prompt (reused from grpo_train.py) ──────────────
SYSTEM_PROMPT = (
    "You are an email triage assistant. "
    "Return a JSON array of the ordered triage steps for the email.\n\n"
    "Actions (choose the minimum necessary steps):\n"
    '  {"action": "mark_spam"}                              '
    "— spam / scam / phishing / unsolicited bulk email\n"
    '  {"action": "ignore"}                                '
    "— low-priority newsletter, promotion, or marketing\n"
    '  {"action": "reply", "value": "<short reply>"}       '
    "— professional response needed\n"
    '  {"action": "mark_important"}                        '
    "— flag as important (NOT a crisis)\n"
    '  {"action": "route_to_department", "value": "<dept>"}'
    " — forward to: " + " | ".join(DEPARTMENTS) + "\n"
    '  {"action": "high-priority \\"crisis\\""}            '
    "— ONLY for genuine production outages, security breaches, "
    "financial fraud, physical emergencies, or legal crises. "
    "ALWAYS the FIRST step, followed by mark_important and route_to_department.\n\n"
    "Rules:\n"
    "  • Output ONLY the JSON array — no explanation, no extra text.\n"
    "  • Spam/scam → one step: mark_spam.\n"
    "  • Promotions/newsletters → one step: ignore.\n"
    "  • Most routine work → mark_important then route_to_department.\n"
    "  • Crisis is RARE — only real emergencies qualify.\n\n"
    "Examples:\n"
    '  Promotion → [{"action": "ignore"}]\n'
    '  Phishing  → [{"action": "mark_spam"}]\n'
    '  HR issue  → [{"action": "mark_important"}, '
    '{"action": "route_to_department", "value": "HR"}]\n'
    '  DB breach → [{"action": "high-priority \\"crisis\\""}, '
    '{"action": "mark_important"}, '
    '{"action": "route_to_department", "value": "Tech Support"}]\n'
)


# ─────────────────────────── LLM backends ────────────────────────────────────

def _call_openai_api(client, email_text: str, history: list[str]) -> str:
    """Call the remote LLM via OpenAI-compatible API."""
    history_block = "\n".join(history[-4:]) if history else "None"
    user_msg = textwrap.dedent(f"""
        Email:
        {email_text}

        Previous triage decisions:
        {history_block}

        Respond with the JSON action array only.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[WARN] API call failed: {exc}", file=sys.stderr)
        return '[{"action": "ignore"}]'


def _build_local_model():
    """Load the fine-tuned GRPO model for local inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = GRPO_MODEL_PATH
    print(f"[INFO] Loading local model from {path}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    try:
        from peft import PeftModel
        base_id = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        base = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, path)
        model = model.merge_and_unload()
        print("[INFO] LoRA adapter loaded and merged.", file=sys.stderr)
    except Exception:
        import torch as _t
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
        )
        print("[INFO] Full model loaded.", file=sys.stderr)

    model.eval()
    return model, tokenizer


def _call_local_model(model, tokenizer, email_text: str) -> str:
    """Run greedy decode on the local fine-tuned model."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Email: {email_text}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────── Episode runner ───────────────────────────────────

def run_episode(
    episode_idx: int,
    email_record: dict,
    api_client,          # OpenAI client or None
    local_model_pair,    # (model, tokenizer) or None
) -> tuple[list[float], int, bool]:
    """Run one email triage episode.

    Returns (rewards_per_step, steps_taken, success).
    """
    email_text    = email_record["email_text"]
    correct_steps = email_record.get("steps", [])
    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    done = False

    # ── Ask the model for the full action sequence ──
    if local_model_pair:
        raw_output = _call_local_model(local_model_pair[0], local_model_pair[1], email_text)
    else:
        raw_output = _call_openai_api(api_client, email_text, history)

    predicted_steps = extract_steps(raw_output)

    # ── Execute each predicted step in the environment ──
    for step_idx, step_dict in enumerate(predicted_steps):
        if done or step_idx >= MAX_STEPS:
            break

        action = step_dict.get("action", "ignore")
        value  = step_dict.get("value")

        # Compute reward via the same logic as the RL environment
        reward = compute_step_reward(action, value, correct_steps, step_idx)

        steps_taken = step_idx + 1
        rewards.append(reward)

        # Format action string for logging (mirrors the env's action representation)
        if value:
            action_str = f"{action}:{value}"
        else:
            action_str = action

        # Episode is done if wrong action, extra step, or all expected steps consumed
        all_done = steps_taken >= len(correct_steps)
        wrong    = reward < 0
        over     = steps_taken >= MAX_STEPS
        done     = all_done or wrong or over

        log_step(
            step=steps_taken,
            action=action_str,
            reward=reward,
            done=done,
            error=None,
        )

        history.append(f"step {steps_taken}: {action_str!r} → reward {reward:+.1f}")

        if done:
            break

    episode_reward = sum(rewards)
    score_fraction = max(0.0, episode_reward) / MAX_EPISODE_REWARD
    success = score_fraction >= SUCCESS_THRESHOLD

    return rewards, steps_taken, success


# ─────────────────────────── Main ────────────────────────────────────────────

def main() -> None:
    # ── Choose email source: live server or local JSONL ──
    max_episodes_env = os.environ.get("EPISODES", "")

    if ENV_SERVER_URL:
        n = int(max_episodes_env) if max_episodes_env else 20
        print(f"[INFO] ENV_SERVER_URL={ENV_SERVER_URL} — fetching {n} emails live", file=sys.stderr)
        emails = fetch_emails_from_server(n)
    else:
        emails = load_test_emails()
        max_episodes = int(max_episodes_env) if max_episodes_env else len(emails)
        emails = emails[:max_episodes]

    # ── Build the inference backend ──
    api_client       = None
    local_model_pair = None

    if GRPO_MODEL_PATH and Path(GRPO_MODEL_PATH).exists():
        local_model_pair = _build_local_model()
        active_model = f"local:{GRPO_MODEL_PATH}"
    else:
        if not API_KEY:
            print(
                "[WARN] HF_TOKEN / API_KEY not set. Requests may fail for gated models.",
                file=sys.stderr,
            )
        from openai import OpenAI
        api_client   = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "none")
        active_model = MODEL_NAME

    log_start(task=TASK_NAME, env=BENCHMARK, model=active_model)

    all_rewards: list[float] = []
    total_steps  = 0
    total_success = 0

    for ep_idx, email_record in enumerate(emails):
        rewards, steps, success = run_episode(
            ep_idx, email_record, api_client, local_model_pair
        )
        all_rewards.extend(rewards)
        total_steps  += steps
        total_success += int(success)

    # ── Compute overall episode-level score ──
    # score = average fraction of max reward achieved per episode
    total_max    = len(emails) * MAX_EPISODE_REWARD
    total_earned = sum(r for r in all_rewards if r > 0)  # positive rewards only
    overall_score = min(max(total_earned / total_max, 0.0), 1.0)

    overall_success = (total_success / len(emails)) >= SUCCESS_THRESHOLD

    log_end(
        success=overall_success,
        steps=total_steps,
        score=overall_score,
        rewards=all_rewards,
    )

    # ── Human-readable summary to stderr ──
    n = len(emails)
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Episodes     : {n}", file=sys.stderr)
    print(f"  Successful   : {total_success}/{n} ({100*total_success/n:.1f}%)", file=sys.stderr)
    print(f"  Overall score: {overall_score:.3f}", file=sys.stderr)
    print(f"  Total reward : {sum(all_rewards):+.1f} / {total_max:.0f} max", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
