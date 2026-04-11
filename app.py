from fastapi import FastAPI, Request
from server.pkt_schd_rl_environment import PacketSchedEnv
from models import PacketAction
import subprocess
import math
import re

app = FastAPI()

# deterministic env instance
env = PacketSchedEnv(task="easy")


# -----------------------------
# RESET
# -----------------------------
@app.post("/reset")
async def reset(request: Request):
    global env

    try:
        body = await request.json()
        task = body.get("task", "easy")
    except Exception:
        task = "easy"

    # reinitialize env with task
    env = PacketSchedEnv(task=task)

    result = env.reset()

    return {
        "observation": result.observation.model_dump(),
        "reward": float(result.reward),
        "done": bool(result.done),
        "info": {}
    }


# -----------------------------
# STEP
# -----------------------------
@app.post("/step")
def step(action: dict):
    act = PacketAction(**action["action"])
    result = env.step(act)

    return {
        "observation": result.observation.model_dump(),
        "reward": float(result.reward),
        "done": bool(result.done),
        "info": {}
    }


# -----------------------------
# TASKS (STRICT SCHEMA)
# -----------------------------
@app.get("/tasks")
def tasks():
    schema = {
        "type": "object",
        "properties": {
            "priority_ratio": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            }
        },
        "required": ["priority_ratio"]
    }

    return {
        "tasks": [
            {
                "name": "easy",
                "description": "Low traffic, stable conditions",
                "action_schema": schema
            },
            {
                "name": "moderate",
                "description": "Bursty traffic with regime shifts",
                "action_schema": schema
            },
            {
                "name": "hard",
                "description": "Adversarial congestion + fairness constraints",
                "action_schema": schema
            }
        ]
    }


# -----------------------------
# GRADER (STRICT (0,1))
# -----------------------------
@app.post("/grader")
async def grader(request: Request):
    data = await request.json()
    rewards = data.get("rewards", [])

    if not rewards:
        return {"score": 0.001}  # NEVER 0

    total = sum(rewards)

    # smooth deterministic normalization
    score = 1 / (1 + math.exp(-total / 20))

    # enforce STRICT bounds
    score = max(0.001, min(0.999, score))

    return {"score": float(score)}


# -----------------------------
# BASELINE
# -----------------------------
@app.post("/baseline")
def baseline():
    tasks = ["easy", "moderate", "hard"]
    results = {}

    for task in tasks:
        env = {"TASK_NAME": task}

        result = subprocess.run(
            ["python", "inference.py"],
            capture_output=True,
            text=True,
            env={**os.environ, **env}
        )

        output = result.stdout

        # extract score from [END]
        match = re.search(r"score=([0-9\.]+)", output)
        if match:
            score = float(match.group(1))
        else:
            score = 0.001  # fallback safe

        # enforce strict bounds
        score = max(0.001, min(0.999, score))

        results[task] = score
