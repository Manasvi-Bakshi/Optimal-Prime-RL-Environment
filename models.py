from pydantic import BaseModel


class _FastBaseModel(BaseModel):
    model_config = {
        "extra": "forbid",
        "validate_assignment": False,
        "arbitrary_types_allowed": True,
    }


class PacketObservation(_FastBaseModel):
    q_priority: float
    q_regular: float
    incoming: float
    step: int

    p_lost: int
    r_lost: int
    loss_rate: float
    avg_latency: float
    throughput: float
    fairness_index: float


class PacketAction(_FastBaseModel):
    priority_ratio: float


class StepResult(_FastBaseModel):
    observation: PacketObservation
    reward: float
    done: bool