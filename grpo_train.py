#!/usr/bin/env python3
"""
Email Triage — GRPO LLM Fine-tuning
=====================================

Fine-tunes a small causal LM using Group Relative Policy Optimization (GRPO)
so the model learns to classify emails into triage actions directly, replacing
the Q-table with a language model policy.

How GRPO works here:
  1. For each email prompt, the model generates N completions (the "group").
  2. Each completion is scored by compute_reward() against the true_label.
  3. Relative advantages within the group guide the policy gradient update.
  4. No separate critic / value network is needed (unlike PPO).

Default model: Qwen/Qwen2.5-0.5B-Instruct  (~1 GB download, runs on CPU/GPU)
LoRA is applied automatically when `peft` is installed, drastically reducing
GPU memory usage.

Usage:
    # Fine-tune (downloads model on first run):
    python grpo_train.py

    # Fewer steps for a quick smoke-test:
    python grpo_train.py --steps 30

    # Larger / different base model:
    python grpo_train.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 200

    # Evaluate the saved fine-tuned model:
    python grpo_train.py --test

    # Evaluate against a different checkpoint:
    python grpo_train.py --test --model-path models/grpo_email_triage
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import torch

# Force PyTorch backend — prevents broken TF installations from interfering.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH = Path(
    os.environ.get(
        "EMAIL_TRIAGE_DATA",
        str(Path(__file__).parent / "docs" / "email_test_data.csv"),
    )
)

ACTIONS = ["reply", "mark_spam", "mark_important", "ignore"]

# Shaped reward table — same as agent.py / llm.py so results are comparable.
_REWARD_TABLE: dict[tuple[str, str], float] = {
    ("mark_spam", "mark_important"): -10.0,
    ("ignore", "mark_important"): -8.0,
    ("reply", "mark_important"): -2.0,
    ("mark_important", "mark_spam"): -7.0,
    ("reply", "mark_spam"): -7.0,
}

SYSTEM_PROMPT = (
    "You are an email triage assistant. "
    "Read the email and respond with EXACTLY ONE action word from this list:\n"
    "  reply          — the email needs a professional response\n"
    "  mark_spam      — unsolicited / scam / bulk mail\n"
    "  mark_important — urgent, critical, or time-sensitive issue\n"
    "  ignore         — low-priority promotion or newsletter\n\n"
    "Output only the action word. No explanation, no punctuation."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_emails() -> list[dict]:
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


def extract_action(text: str) -> str:
    """Extract the first recognised action keyword from model output."""
    text = text.lower().strip()
    # Check longer tokens first to avoid substring collisions (mark_important before mark_spam)
    for action in ["mark_important", "mark_spam", "reply", "ignore"]:
        if re.search(rf"\b{re.escape(action)}\b", text):
            return action
    return "ignore"  # safe default


def compute_reward(action: str, correct_action: str) -> float:
    if action == correct_action:
        return 10.0
    return _REWARD_TABLE.get((action, correct_action), -5.0)


def make_prompt(email_text: str, tokenizer) -> str:
    """Format an email as a chat-template prompt ready for the model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Email: {email_text}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# GRPO reward function
# Called by GRPOTrainer for each batch of completions.
# `true_label` is a list[str] automatically forwarded from the dataset column.
# ---------------------------------------------------------------------------

def reward_fn(completions: list[str], true_label: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion, label in zip(completions, true_label):
        predicted = extract_action(completion)
        rewards.append(compute_reward(predicted, label))
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

    # Load tokenizer first so we can build prompts.
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    emails = load_emails()

    # Build HF dataset — extra columns beyond "prompt" are forwarded to reward_fn.
    dataset = Dataset.from_dict(
        {
            "prompt":      [make_prompt(e["email_text"], tokenizer) for e in emails],
            "true_label":  [e["true_label"] for e in emails],
            "email_text":  [e["email_text"] for e in emails],
        }
    )

    # LoRA config — reduces trainable params by ~95 %, fits on <8 GB VRAM.
    lora_config = None
    if _use_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            # Common attention projection names; adjust if using a different arch.
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
        # GRPO-specific: number of completions sampled per prompt per step.
        num_generations=args.num_generations,
        max_completion_length=16,   # one action word is enough
        temperature=args.temperature,
        logging_steps=5,
        save_steps=50,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",           # disable wandb / tensorboard by default
        remove_unused_columns=False, # keep true_label column for reward_fn
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"\n[GRPO] Training  epochs={args.epochs}  steps={'auto' if args.steps <= 0 else args.steps}"
          f"  batch={args.batch_size}  generations={args.num_generations}  lr={args.lr}")
    trainer.train()

    print(f"\n[GRPO] Saving fine-tuned model → {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[GRPO] Done! Run evaluation with: python grpo_train.py --test")


# ---------------------------------------------------------------------------
# Evaluation / inference
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

    # Try to load as a PEFT/LoRA adapter first, then fall back to full model.
    try:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        print("[TEST] Loaded LoRA adapter and merged into base model.")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
        )
        print("[TEST] Loaded full model.")

    model.eval()
    emails = load_emails()

    correct = 0
    total_reward = 0.0

    print(f"\n{'='*70}")
    print("Evaluation — GRPO Fine-tuned Model (greedy decoding)")
    print(f"{'='*70}\n")

    for email in emails:
        email_text = email["email_text"]
        true_label = email["true_label"]

        prompt = make_prompt(email_text, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,        # greedy — deterministic at eval time
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        predicted = extract_action(completion)
        reward = compute_reward(predicted, true_label)
        total_reward += reward
        if reward > 0:
            correct += 1

        mark = "✓" if reward > 0 else "✗"
        print(
            f"{mark} pred={predicted:15s}  true={true_label:15s}  "
            f"reward={reward:+5.1f}  email={email_text[:42]!r}"
        )

    n = len(emails)
    print(f"\n{'='*70}")
    print(f"  Emails tested  : {n}")
    print(f"  Correct actions: {correct}/{n} ({100*correct/n:.1f}%)")
    print(f"  Total reward   : {total_reward:+.1f}")
    print(f"  Avg reward     : {total_reward/n:+.2f}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Email Triage — GRPO LLM Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID for the base model (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--output-dir", default="models/grpo_email_triage",
        help="Where to save the fine-tuned model (default: models/grpo_email_triage)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Training epochs over the dataset (default: 3)",
    )
    parser.add_argument(
        "--steps", type=int, default=0,
        help="Max training steps; overrides --epochs when > 0. Use 30-50 for a quick test.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Per-device train batch size (default: 2; lower if OOM)",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-6,
        help="Learning rate (default: 5e-6)",
    )
    parser.add_argument(
        "--num-generations", type=int, default=8,
        help="GRPO completions per prompt per step (default: 8; lower if OOM)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9,
        help="Sampling temperature for GRPO completions (default: 0.9)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Evaluate a saved fine-tuned model instead of training",
    )
    parser.add_argument(
        "--model-path", default="models/grpo_email_triage",
        help="Path to fine-tuned model for --test (default: models/grpo_email_triage)",
    )
    args = parser.parse_args()

    if args.test:
        test(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
