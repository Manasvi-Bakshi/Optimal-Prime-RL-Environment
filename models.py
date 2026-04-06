from pydantic import BaseModel

class PacketObservation(BaseModel):
    q_priority: float
    q_regular: float
    incoming: float
    step: int

class PacketAction(BaseModel):
    priority_ratio: float  # 0 → 1

class StepResult(BaseModel):
    observation: PacketObservation
    reward: float
    done: bool