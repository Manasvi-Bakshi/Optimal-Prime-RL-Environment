from fastapi import FastAPI
from app.env import Environment
from app.models import Action

app = FastAPI()
env = Environment()

@app.get("/reset")
def reset():
    state = env.reset()
    return {"state": state}

@app.post("/step")
def step(action: Action):
    result = env.step(action)
    return result

@app.get("/state")
def state():
    return env.get_state()

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"id": "task1"},
            {"id": "task2"},
            {"id": "task3"}
        ]
    }

@app.post("/grader")
def grader():
    return {"score": 0.5}

@app.get("/baseline")
def baseline():
    return {"scores": [0.5, 0.6, 0.7]}