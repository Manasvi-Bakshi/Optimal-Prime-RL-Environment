from fastapi import FastAPI
from server.pkt_schd_rl_environment import PacketSchedEnv
from models import PacketAction
import uvicorn

app = FastAPI()
env = PacketSchedEnv(task="easy")


@app.post("/reset")
def reset():
    result = env.reset()
    return {
        "observation": {"observation": result.observation.model_dump()},
        "reward": result.reward,
        "done": result.done
    }


@app.post("/step")
def step(action: dict):
    act = PacketAction(**action["action"])
    result = env.step(act)

    return {
        "observation": {"observation": result.observation.model_dump()},
        "reward": result.reward,
        "done": result.done
    }


# REQUIRED for OpenEnv validator
def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


# REQUIRED entrypoint
if __name__ == "__main__":
    main()
