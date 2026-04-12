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

# Email Triage RL Environment (LLM + RL + OpenEnv)

An AI agent that reads emails, classifies them with an LLM, decides actions via a learned policy,
and generates professional replies — all with a reinforcement-learning reward signal.

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
