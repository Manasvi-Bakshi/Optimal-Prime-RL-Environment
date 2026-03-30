from pydantic import BaseModel
from typing import Any, Dict

class Observation(BaseModel):
    state: Dict[str, Any]

class Action(BaseModel):
    action_type: str
    payload: Dict[str, Any]

class Reward(BaseModel):
    score: float