---
title: Email Triage RL Environment
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - rl
  - llm
---

# Email Triage Environment

An OpenEnv-compatible RL environment that simulates an enterprise email triage workflow. The agent classifies incoming emails and decides the correct ordered sequence of triage actions — from marking spam to escalating genuine crises.

Built for the [OpenEnv Hackathon SF](https://cerebralvalley.ai/e/openenv-hackathon-sf/details).

### Key Results

> **GRPO fine-tuning on Qwen2.5-0.5B-Instruct achieves 47.1% perfect sequence match and +4.15 avg reward on 87 held-out test emails.**

| Metric | Value |
|--------|-------|
| Emails tested | 87 |
| Perfect sequence match | 41/87 **(47.1%)** |
| First-step correct | 51/87 (58.6%) |
| Avg reward per email | **+4.15** |

### Example: Crisis Escalation

> **Email:** "URGENT: Our production database is under active attack. Customer data may be compromised. We need immediate action."

The agent must output the correct ordered action sequence:

```json
[
  {"action": "high-priority \"crisis\""},
  {"action": "mark_important"},
  {"action": "route_to_department", "value": "Tech Support"}
]
```

**Reward:** +10 per correct step in correct position, with asymmetric penalties for false crisis alarms (up to -10).

## Table of Contents

- [Quick Start](#quick-start)
- [Actions (6)](#actions-6)
- [Email Categories (5)](#email-categories-5)
- [Dataset](#dataset)
- [Reward / Scoring](#reward--scoring)
- [Environment API](#environment-api)
- [Project Structure](#project-structure)
- [Training](#training)
- [Inference](#inference)
- [Q-Learning Agent](#q-learning-agent)
- [Installation](#installation)
- [Building & Running](#building--running)
- [Live Demo](#live-demo)

## Quick Start

```python
from client import EmailTriageEnv, EmailTriageAction

# Connect to the environment server
with EmailTriageEnv(base_url="http://localhost:8000") as env:
    obs = env.reset()
    print("Email:", obs.email_text)      # the email to triage
    print("Category:", obs.category)    # classified category

    # Submit an action
    result = env.step(EmailTriageAction(action="mark_spam"))
    print("Reward:", result.reward)
```

Or use the REST demo endpoints directly:

```bash
# Sample a new email
curl http://localhost:8000/reset

# Submit an action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": "mark_important"}'

# Full LLM pipeline: classify → decide → reward → reply
curl http://localhost:8000/auto
```

## Actions (6)

The agent selects an **ordered sequence** of actions for each email (up to 5 steps).

| Action | Value required | Description |
|--------|---------------|-------------|
| `mark_spam` | No | Spam, scam, phishing, or unsolicited bulk email |
| `ignore` | No | Low-priority newsletter, promotion, or marketing |
| `reply` | Yes — reply text | Professional response required |
| `mark_important` | No | Flag as important (not a crisis) |
| `route_to_department` | Yes — department name | Forward to a specific team |
| `high-priority "crisis"` | No | Genuine emergency — **always the first step**, followed by `mark_important` + `route_to_department` |

### Routing Departments (10)

HR · Management · Tech Support · Billing & Finance · Legal · Business Team · Customer Support · Sales · Operations · Security

### Action Rules

- **Spam/scam** → single step: `mark_spam`
- **Promotions/newsletters** → single step: `ignore`
- **Routine work emails** → `mark_important` then `route_to_department`
- **Crisis** (production outage, security breach, financial fraud, physical emergency, legal crisis) → `high-priority "crisis"` first, then `mark_important`, then `route_to_department`
- Crisis is **rare** — only genuine emergencies qualify

## Email Categories (5)

| Category | Default action |
|----------|---------------|
| `work` | `reply` |
| `spam` | `mark_spam` |
| `personal` | `ignore` |
| `promotion` | `ignore` |
| `urgent` | `mark_important` |

## Dataset

| Split | File | Count |
|-------|------|-------|
| Train | `docs/train_data.jsonl` | 348 emails |
| Test | `docs/test_data.jsonl` | 87 emails |

Each record:
```json
{
  "id": 1,
  "email_text": "Hi, could we schedule a quick sync?",
  "category": "work",
  "steps": [
    {"action": "reply", "value": "Happy to connect! Let me check my calendar."}
  ]
}
```

Generate fresh data:
```bash
python scripts/generate_data.py
```

## Reward / Scoring

### Per-step scoring (averaged over expected steps)

| Outcome | Reward |
|---------|--------|
| Correct action, correct position | **+10** |
| Correct action, wrong position | +2 (partial credit) |
| Wrong action | -5 |
| Missing expected step | -3 |
| Extra step beyond sequence | -2 each |

### Asymmetric first-step penalties

Applied on top of the -5 wrong-action penalty to capture domain risk:

| Prediction → Truth | Extra penalty |
|-------------------|--------------|
| crisis → spam/ignore | **-10** |
| crisis → reply | -8 |
| crisis → mark_important | -5 |
| ignore/spam → crisis | **-10** |
| reply → crisis | -8 |
| mark_important → crisis | -5 |
| mark_important/reply → spam | -5 |

### Diversity penalty (GRPO training only)

If all completions in a generation group predict the same first action, a **-3 collapse penalty** is applied to every reward to discourage mode collapse.

## Environment API

### OpenEnv Interface

```
reset()  → EmailTriageObservation   # Sample a new email, return text + category
step()   → StepResult               # Submit one action, receive reward
state    → State                    # Current step count, episode ID
```

### Episode Flow

```
1. env.reset()
   → {"email_text": "Production outage...", "category": "urgent"}

2. env.step(EmailTriageAction(action='high-priority "crisis"'))
   → reward: +10.0

3. env.step(EmailTriageAction(action="mark_important"))
   → reward: +10.0

4. env.step(EmailTriageAction(action="route_to_department", value="Tech Support"))
   → reward: +10.0  (done)
```

### REST Demo Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/reset` | Sample a new email; returns `{email, category, correct_steps}` |
| `POST` | `/step` | Submit `{action, value?}`; returns `{reward, reply?}` |
| `GET` | `/auto` | Full pipeline: classify → decide → reward → optional reply |

## Project Structure

```
email_triage/
├── README.md                          # This file
├── openenv.yaml                       # OpenEnv manifest
├── pyproject.toml                     # Project metadata + dependencies
├── uv.lock                            # Locked dependency versions
├── __init__.py                        # Module exports (EmailTriageEnv, models)
├── client.py                          # EmailTriageEnv WebSocket client
├── models.py                          # Action / Observation Pydantic models
├── llm.py                             # classify_email, decide_action, generate_reply
├── agent.py                           # Q-learning agent (tabular, multi-step)
├── grpo_train.py                      # GRPO LLM fine-tuning + evaluation
├── inference.py                       # LLM inference runner (API or local model)
├── Dockerfile                         # Container image
├── requirements.txt                   # Pip-compatible dependency list
├── docs/
│   ├── train_data.jsonl               # 348 training emails
│   └── test_data.jsonl                # 87 test emails (no overlap)
├── models/
│   ├── qtable.json                    # Saved Q-table (Q-learning agent)
│   ├── qtable_v2.json                 # Improved Q-table
│   └── grpo_email_triage/             # Fine-tuned GRPO LoRA adapter
├── scripts/
│   └── generate_data.py              # Dataset generation script
└── server/
    ├── __init__.py
    ├── app.py                         # FastAPI application
    ├── email_triage_env_environment.py # Core environment (Environment subclass)
    ├── Dockerfile                     # Server-only container
    └── requirements.txt               # Server dependencies
```

## Training

Fine-tune `Qwen/Qwen2.5-0.5B-Instruct` with GRPO using `grpo_train.py`.

### Setup

- **Model**: Qwen/Qwen2.5-0.5B-Instruct (~1 GB download); swap via `--model`
- **Algorithm**: GRPO (Group Relative Policy Optimization) — no separate critic needed
- **LoRA**: applied automatically when `peft` is installed (rank 16, alpha 32, targets q/k/v/o projections)
- **Data balancing**: minority action classes are upsampled so all 6 action types are equally represented
- **Key defaults**: 3 epochs, batch size 2, grad accum 4, lr 5e-6, 8 generations per prompt, temperature 0.9

### Run

```bash
# Full training (saves to models/grpo_email_triage)
python grpo_train.py

# Quick smoke-test (30 steps)
python grpo_train.py --steps 30

# Larger model
python grpo_train.py --model Qwen/Qwen2.5-1.5B-Instruct

# Evaluate saved model on test set
python grpo_train.py --test

# Evaluate a specific checkpoint
python grpo_train.py --test --model-path models/grpo_email_triage
```

### Results

```
========================================================================
  Emails tested           : 87
  Perfect sequence match  : 41/87 (47.1%)
  First-step correct      : 51/87 (58.6%)
  Avg reward per email    : +4.15
========================================================================
```

## Inference

`inference.py` runs the trained model against the test set or a live environment server.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace token (for API calls to gated models) |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | Remote model ID |
| `GRPO_MODEL_PATH` | `models/grpo_email_triage` | Local fine-tuned model path (takes priority over API) |
| `ENV_SERVER_URL` | _(unset)_ | OpenEnv server URL; when set, emails are fetched live from `/reset` |
| `EMAIL_TRIAGE_TEST` | `docs/test_data.jsonl` | Path to test JSONL |
| `EPISODES` | all | Number of episodes to run |

### Run

```bash
# Local fine-tuned model against local JSONL
python inference.py

# Against the deployed HF Space
HF_TOKEN=hf_... \
ENV_SERVER_URL=https://chanchal11-email-triage.hf.space \
python inference.py

# Remote HF LLM via API
HF_TOKEN=hf_... \
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct \
python inference.py

# Limit to 10 episodes
EPISODES=10 HF_TOKEN=hf_... python inference.py
```

### Output format

```
[START] task=email_triage env=email_triage_env model=local:models/grpo_email_triage
[STEP]  step=1 action='mark_spam' reward=10.00 done=true error=null
[END]   success=true steps=1 score=0.200 rewards=10.00
```

## Q-Learning Agent

`agent.py` implements a tabular Q-learning agent for comparison and rapid prototyping.

- **State**: `"<category>:<step_idx>"` — e.g. `"urgent:0"`, `"spam:0"`
- **Actions**: all 6 triage actions
- **Update**: TD(0) with shaped per-step rewards
- **Exploration**: epsilon-greedy with decay

```bash
# Fast demo (200 episodes, no LLM)
python agent.py --no-llm --episodes 200

# Train and save Q-table
python agent.py --save models/qtable.json

# Evaluate saved Q-table
python agent.py --test --load models/qtable.json

# Run against live environment server
python agent.py --server http://localhost:8000
```

## Installation

```bash
# Install core dependencies
uv pip install -e .

# Or with pip
pip install -r requirements.txt
```

## Building & Running

```bash
# Run locally (development, with auto-reload)
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Build Docker image
docker build -t email-triage-env:latest .

# Run Docker container
docker run -p 8000:8000 email-triage-env:latest

# Validate OpenEnv compliance
openenv validate

# Deploy to HF Spaces
openenv push
```

## Live Demo

Try the environment on Hugging Face Spaces: https://chanchal11-email-triage.hf.space

## System Flow

```
Email → LLM (classify) → Category → Policy → Action → Reward + Reply
```

## State & Actions

| Concept  | Values |
|----------|--------|
| **Categories** | `work`, `spam`, `personal`, `promotion`, `urgent` |
| **Actions**    | `reply`, `mark_spam`, `mark_important`, `ignore` |

## Reward Shape

| Situation | Reward |
|-----------|--------|
| Correct action | +10 |
| Reply to urgent (close but not best) | -2 |
| Ignore urgent | -8 |
| Mark urgent as spam | -10 |
| Most other mismatches | -5 |

## Quick Start — Python Client

```python
from email_triage_env import EmailTriageAction, EmailTriageEnv

with EmailTriageEnv.from_docker_image("email_triage_env-env:latest") as env:
    # Start a new episode (get a random email)
    obs = env.reset()
    print("Email:", obs.email_text)

    # Take an action
    result = env.step(EmailTriageAction(action="reply"))
    print("Reward:", result.reward)
    print("Reply:", result.observation.reply)
```

## REST API Demo

Three simple REST endpoints for demos and hackathons:

```bash
# 1. Load a new random email
curl http://localhost:8000/reset

# 2. Submit your action (reply / mark_spam / mark_important / ignore)
curl -X POST http://localhost:8000/step \
     -H "Content-Type: application/json" \
     -d '{"action": "reply"}'

# 3. Full auto-triage pipeline (LLM classify → decide → reward → reply)
curl http://localhost:8000/auto
```

Example `/auto` response:
```json
{
  "email": "Urgent meeting at 5 PM today",
  "category": "urgent",
  "action": "mark_important",
  "correct_action": "mark_important",
  "reward": 10.0,
  "reply": null
}
```

## Q-Learning Agent

Run the standalone RL agent that trains a Q-table over email categories:

```bash
# Fast demo (no LLM, uses ground-truth labels as state)
python agent.py --no-llm --episodes 100

# With LLM classification (requires transformers + torch)
python agent.py --episodes 50

# Run against a live server
python agent.py --server http://localhost:8000 --episodes 20
```

## Building & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run server locally
uvicorn email_triage_env.server.app:app --reload --port 8000

# Or via Docker
docker build -t email_triage_env-env:latest -f server/Dockerfile .
docker run -p 8000:8000 email_triage_env-env:latest
```

## Architecture

| Module | File | Role |
|--------|------|------|
| Models | `email_triage_env/models.py` | Pydantic action/observation types |
| LLM | `email_triage_env/llm.py` | classify_email, decide_action, generate_reply, compute_reward |
| Environment | `email_triage_env/server/email_triage_env_environment.py` | OpenEnv RL environment |
| Server | `email_triage_env/server/app.py` | FastAPI + OpenEnv WebSocket server |
| Client | `email_triage_env/client.py` | Python WebSocket client |
| RL Agent | `agent.py` | Q-learning demo agent |

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**EmailTriageAction**: Contains a single field
- `message` (str) - The message to echo back

### Observation
**EmailTriageObservation**: Contains the echo response and metadata
- `echoed_message` (str) - The message echoed back
- `message_length` (int) - Length of the message
- `reward` (float) - Reward based on message length (length × 0.1)
- `done` (bool) - Always False for echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have a Email Triage Env environment server running, you can connect directly:

```python
from email_triage_env import EmailTriageEnv

# Connect to existing server
email_triage_envenv = EmailTriageEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = email_triage_envenv.reset()
result = email_triage_envenv.step(EmailTriageAction(message="Hello!"))
```

Note: When connecting to an existing server, `email_triage_envenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from email_triage_env import EmailTriageAction, EmailTriageEnv

# Connect with context manager (auto-connects and closes)
with EmailTriageEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(EmailTriageAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    EmailTriageEnvironment,  # Pass class, not instance
    EmailTriageAction,
    EmailTriageObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from email_triage_env import EmailTriageAction, EmailTriageEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with EmailTriageEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(EmailTriageAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/email_triage_env_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
email_triage_env/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # EmailTriageEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── email_triage_env_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
