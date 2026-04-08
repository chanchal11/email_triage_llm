#!/usr/bin/env python3
"""
Email Triage — GRPO LLM Fine-tuning (multi-step edition)
=========================================================

Fine-tunes a small causal LM using Group Relative Policy Optimization (GRPO)
to predict the *ordered sequence of triage actions* for each email, covering
the full expanded action set:

  mark_spam | mark_important | ignore | reply | route_to_department | high-priority "crisis"

How GRPO works here:
  1. For each email prompt, the model generates N completions.
  2. Each completion (a JSON array of action steps) is scored against ground truth.
  3. Relative advantages within the group guide the policy gradient update.
  4. No separate critic / value network is needed (unlike PPO).

Data format (train_data.jsonl / test_data.jsonl):
  {"id": 1, "email_text": "...", "category": "...",
   "steps": [{"action": "...", "value": "..."}, ...]}

Default model : Qwen/Qwen2.5-0.5B-Instruct  (~1 GB download)
LoRA applied automatically when peft is installed.

Usage:
    python grpo_train.py                        # full training
    python grpo_train.py --steps 30             # quick smoke-test
    python grpo_train.py --model Qwen/Qwen2.5-1.5B-Instruct
    python grpo_train.py --test                 # evaluate saved model on test set
    python grpo_train.py --test --model-path models/grpo_email_triage
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DOCS = Path(__file__).parent / "docs"
TRAIN_JSONL = Path(os.environ.get("EMAIL_TRIAGE_TRAIN", str(DOCS / "train_data.jsonl")))
TEST_JSONL  = Path(os.environ.get("EMAIL_TRIAGE_TEST",  str(DOCS / "test_data.jsonl")))

# ---------------------------------------------------------------------------
# Action / department constants
# ---------------------------------------------------------------------------
CRISIS_ACTION = 'high-priority "crisis"'

ALL_ACTIONS = [
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

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an email triage assistant. "
    "Return a JSON array of the ordered triage steps for the email.\n\n"
    "Actions (choose the minimum necessary steps):\n"
    '  {"action": "mark_spam"}                             '
    "— spam / scam / phishing / unsolicited bulk email\n"
    '  {"action": "ignore"}                               '
    "— low-priority newsletter, promotion, or marketing\n"
    '  {"action": "reply", "value": "<short reply>"}      '
    "— professional response needed\n"
    '  {"action": "mark_important"}                       '
    "— flag as important (NOT a crisis)\n"
    '  {"action": "route_to_department", "value": "<dept>"}'
    " — forward to: "
    + " | ".join(DEPARTMENTS)
    + "\n"
    '  {"action": "high-priority \\"crisis\\""}           '
    "— ONLY for genuine production outages, security breaches, "
    "financial fraud, physical emergencies, or legal crises. "
    "ALWAYS the FIRST step, followed by mark_important and route_to_department.\n\n"
    "Rules:\n"
    "  • Output ONLY the JSON array. No explanation or extra text.\n"
    "  • Spam → one step only: mark_spam.\n"
    "  • Promotions / newsletters / marketing → one step only: ignore.\n"
    "  • Most routine work emails → mark_important then route_to_department.\n"
    "  • Crisis is RARE — only real emergencies qualify.\n\n"
    "Examples:\n"
    '  Promotion/newsletter → [{"action": "ignore"}]\n'
    '  Phishing/lottery scam → [{"action": "mark_spam"}]\n'
    '  Meeting request → [{"action": "reply", "value": "Happy to connect! '
    'Let me check my calendar."}]\n'
    '  HR complaint → [{"action": "mark_important"}, '
    '{"action": "route_to_department", "value": "HR"}]\n'
    '  Production database breach → '
    '[{"action": "high-priority \\"crisis\\""}, '
    '{"action": "mark_important"}, '
    '{"action": "route_to_department", "value": "Tech Support"}]\n'
)


# ---------------------------------------------------------------------------
# Data loading
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


# ---------------------------------------------------------------------------
# Step extraction from model output
# ---------------------------------------------------------------------------

# Actions that legitimately carry a value payload
_VALUE_ACTIONS = {"reply", "route_to_department"}


def _normalize_step(action: str, value: Any) -> dict:
    """Normalise one extracted step.

    • Strip value for actions that must not have one (crisis, mark_*, ignore).
    • Treat empty-string value as missing.
    """
    if action not in _VALUE_ACTIONS:
        return {"action": action, "value": None}
    return {"action": action, "value": value if value else None}


def extract_steps(text: str) -> list[dict]:
    """Extract triage steps from model output.

    Strategy:
      1. Find the first JSON array in the output and parse it.
      2. Fall back to scanning for the FIRST recognised action keyword only
         (avoids accumulating all keywords that appear anywhere in the text).

    Returns list of {"action": str, "value": str | None} dicts.
    """
    text = text.strip()

    # 1. Try to extract a JSON array (greedy match to get the full array)
    match = re.search(r"\[.*\]", text, re.DOTALL)
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

    # 2. Keyword fallback — return ONLY the FIRST found action, not all of them.
    #    Order: specific multi-token tokens first to avoid substring collisions.
    fallback_order = [
        "route_to_department",
        "mark_important",
        "mark_spam",
        CRISIS_ACTION,
        "ignore",
        "reply",
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


# ---------------------------------------------------------------------------
# Step-sequence reward scorer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step-sequence reward scorer
# ---------------------------------------------------------------------------

# Asymmetric penalty table: (predicted_first_action, correct_first_action) → extra penalty
# Applied ON TOP of the per-step -5 for a wrong answer.
# Captures domain knowledge: calling something a crisis when it isn't is
# extremely costly; conversely, missing a real crisis is also costly.
_FALSE_FIRST_STEP_PENALTY: dict[tuple[str, str], float] = {
    # false crisis alarms — wastes emergency resources
    (CRISIS_ACTION, "mark_spam"):      -10.0,
    (CRISIS_ACTION, "ignore"):         -10.0,
    (CRISIS_ACTION, "reply"):           -8.0,
    (CRISIS_ACTION, "mark_important"): -5.0,
    # missing a real crisis
    ("ignore",        CRISIS_ACTION):  -10.0,
    ("mark_spam",     CRISIS_ACTION):  -10.0,
    ("reply",         CRISIS_ACTION):   -8.0,
    ("mark_important",CRISIS_ACTION):   -5.0,
    # treating spam as something actionable
    ("mark_important", "mark_spam"):   -5.0,
    ("reply",          "mark_spam"):   -5.0,
}


def score_steps(predicted: list[dict], correct: list[dict]) -> float:
    """Score a predicted action sequence against the ground truth.

    Per-step scoring (averaged over len(correct)):
      Correct action in correct position     → +10
      Correct action in wrong position       → +2  (partial credit)
      Wrong action                           → -5
      Missing expected step                  → -3

    Asymmetric first-step penalty:
      Predicting crisis when email is not a crisis → extra -10 / -8 / -5
      Missing a real crisis                        → extra -10 / -8 / -5
    """
    n = max(len(correct), 1)
    total = 0.0

    for i, exp in enumerate(correct):
        if i < len(predicted):
            pred_action = predicted[i].get("action", "")
            exp_action  = exp.get("action", "")
            if pred_action == exp_action:
                if exp_action == "route_to_department":
                    pred_val = (predicted[i].get("value") or "").lower()
                    exp_val  = (exp.get("value") or "").lower()
                    total += 10.0 if pred_val == exp_val else 2.0
                else:
                    total += 10.0
            elif any(p.get("action") == exp_action for p in predicted):
                total += 2.0               # right action, wrong position
            else:
                total += -5.0              # completely wrong
        else:
            total += -3.0                  # step missing

    # Penalise extra steps beyond the correct sequence
    extra = len(predicted) - len(correct)
    if extra > 0:
        total -= extra * 2.0

    # Apply asymmetric first-step penalty (on the raw total, not per-step)
    if predicted and correct:
        pred0 = predicted[0].get("action", "")
        exp0  = correct[0].get("action", "")
        if pred0 != exp0:
            total += _FALSE_FIRST_STEP_PENALTY.get((pred0, exp0), 0.0)

    return total / n


# ---------------------------------------------------------------------------
# Chat-template prompt builder
# ---------------------------------------------------------------------------

def make_prompt(email_text: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Email: {email_text}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# GRPO reward function
# ---------------------------------------------------------------------------

def reward_fn(
    completions: list[str],
    correct_steps_json: list[str],
    **kwargs,
) -> list[float]:
    """Called by GRPOTrainer for each batch.

    `correct_steps_json` is a JSON-encoded string of the ground-truth steps list,
    forwarded from the dataset column of the same name.

    Also applies a diversity penalty when ALL completions in the group predict
    the same first action — this discourages mode collapse during GRPO training.
    """
    rewards = []
    predicted_list = []
    for completion, steps_json in zip(completions, correct_steps_json):
        correct = json.loads(steps_json)
        predicted = extract_steps(completion)
        predicted_list.append(predicted)
        rewards.append(score_steps(predicted, correct))

    # Diversity check: if every completion in the group predicts the same first
    # action, subtract a collapse penalty from every reward in the group.
    first_actions = [
        p[0].get("action", "") if p else "" for p in predicted_list
    ]
    if len(set(first_actions)) == 1 and len(first_actions) > 1:
        collapse_penalty = -3.0
        rewards = [r + collapse_penalty for r in rewards]

    return rewards


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    try:
        from peft import LoraConfig
        _use_lora = True
    except ImportError:
        _use_lora = False
        print("[WARN] peft not installed — training without LoRA (higher VRAM usage).")

    print(f"\n[GRPO] Base model : {args.model}")
    print(f"[GRPO] Output dir : {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    emails = load_emails(TRAIN_JSONL)

    # --- Balanced sampling: upsample minority first-action classes -------
    from collections import defaultdict
    import random as _random
    _random.seed(42)
    by_first_action: dict[str, list] = defaultdict(list)
    for e in emails:
        key = e["steps"][0]["action"] if e.get("steps") else "ignore"
        by_first_action[key].append(e)
    max_count = max(len(v) for v in by_first_action.values())
    balanced: list = []
    for action_emails in by_first_action.values():
        oversampled = (action_emails * ((max_count // len(action_emails)) + 1))[:max_count]
        balanced.extend(oversampled)
    _random.shuffle(balanced)
    emails = balanced
    print(f"[GRPO] Balanced dataset: {len(emails)} examples "
          f"({len(by_first_action)} action classes × {max_count} each)")
    # ---------------------------------------------------------------------

    # Encode ground-truth steps as JSON string so GRPOTrainer can pass them to reward_fn
    dataset = Dataset.from_dict(
        {
            "prompt": [make_prompt(e["email_text"], tokenizer) for e in emails],
            "correct_steps_json": [json.dumps(e["steps"], ensure_ascii=False) for e in emails],
            "email_text": [e["email_text"] for e in emails],
            "category":   [e.get("category", "") for e in emails],
        }
    )

    lora_config = None
    if _use_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.steps if args.steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        max_completion_length=128,  # JSON arrays of steps need more tokens than single words
        temperature=args.temperature,
        logging_steps=5,
        save_steps=50,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(
        f"\n[GRPO] Training  epochs={args.epochs}  "
        f"steps={'auto' if args.steps <= 0 else args.steps}  "
        f"batch={args.batch_size}  generations={args.num_generations}  lr={args.lr}"
    )
    trainer.train()

    print(f"\n[GRPO] Saving fine-tuned model → {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[GRPO] Done! Evaluate with: python grpo_train.py --test")


# ---------------------------------------------------------------------------
# Evaluation / inference  (uses test_data.jsonl — no overlap with training)
# ---------------------------------------------------------------------------

def test(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model_path
    print(f"\n[TEST] Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    try:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        print("[TEST] LoRA adapter loaded and merged into base model.")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
        )
        print("[TEST] Loaded full model.")

    model.eval()
    emails = load_emails(TEST_JSONL)

    perfect = 0       # all steps correct
    partial = 0       # at least first step correct
    total_reward = 0.0

    print(f"\n{'='*72}")
    print("Evaluation — GRPO model on held-out test set (greedy decoding)")
    print(f"{'='*72}\n")

    for email in emails:
        email_text = email["email_text"]
        correct_steps = email["steps"]

        prompt = make_prompt(email_text, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        predicted_steps = extract_steps(completion)
        reward = score_steps(predicted_steps, correct_steps)
        total_reward += reward

        is_perfect = (reward >= 10.0 and len(predicted_steps) == len(correct_steps))
        first_ok = bool(predicted_steps and correct_steps and
                        predicted_steps[0].get("action") == correct_steps[0].get("action"))
        if is_perfect:
            perfect += 1
        if first_ok:
            partial += 1

        mark = "✓" if is_perfect else ("~" if first_ok else "✗")
        correct_str = json.dumps(correct_steps, ensure_ascii=False)[:60]
        pred_str    = json.dumps(predicted_steps, ensure_ascii=False)[:60]
        print(f"{mark} reward={reward:+5.1f}")
        print(f"  email    : {email_text[:64]!r}")
        print(f"  expected : {correct_str}")
        print(f"  got      : {pred_str}")
        print()

    n = len(emails)
    print(f"{'='*72}")
    print(f"  Emails tested           : {n}")
    print(f"  Perfect sequence match  : {perfect}/{n} ({100*perfect/n:.1f}%)")
    print(f"  First-step correct      : {partial}/{n} ({100*partial/n:.1f}%)")
    print(f"  Avg reward per email    : {total_reward/n:+.2f}")
    print(f"{'='*72}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Email Triage — GRPO LLM Fine-tuning (multi-step)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--output-dir", default="models/grpo_email_triage",
        help="Where to save the fine-tuned model",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--steps", type=int, default=0,
        help="Max training steps (0 = use --epochs). Use 30 for a quick smoke-test.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--test", action="store_true", help="Evaluate saved model on test set")
    parser.add_argument(
        "--model-path", default="models/grpo_email_triage",
        help="Path to fine-tuned model weights for --test",
    )
    args = parser.parse_args()

    if args.test:
        test(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

