from fastapi import Request
from openenv.core.env_server.http_server import create_app

from models import PacketAction, PacketObservation
from server.pkt_schd_rl_environment import PacketSchedEnv


# -----------------------------
# CREATE BASE APP
# -----------------------------
app = create_app(
    PacketSchedEnv,
    PacketAction,
    PacketObservation,
    env_name="packet_scheduling",
    max_concurrent_envs=1,
)


# -----------------------------
# TASKS ENDPOINT (REQUIRED)
# -----------------------------
@app.get("/tasks")
async def tasks():
    return {
        "tasks": [
            {
                "name": "easy",
                "action_schema": {
                    "priority_ratio": "float (0 to 1)"
                }
            },
            {
                "name": "moderate",
                "action_schema": {
                    "priority_ratio": "float (0 to 1)"
                }
            },
            {
                "name": "hard",
                "action_schema": {
                    "priority_ratio": "float (0 to 1)"
                }
            }
        ]
    }


# -----------------------------
# SIMPLE GRADER (DETERMINISTIC)
# -----------------------------
@app.post("/grader")
async def grader(request: Request):
    data = await request.json()

    rewards = data.get("rewards", [])

    if not rewards:
        return {"score": 0.0}

    min_r = min(rewards)
    max_r = max(rewards)

    if max_r - min_r > 1e-6:
        normalized = [(r - min_r) / (max_r - min_r) for r in rewards]
        score = sum(normalized) / len(normalized)
    else:
        score = 0.5

    score = max(0.0, min(1.0, score))

    return {"score": score}


# -----------------------------
# BASELINE ENDPOINT
# -----------------------------
@app.get("/baseline")
async def baseline():
    return {
        "status": "ok",
        "message": "Baseline endpoint available"
    }