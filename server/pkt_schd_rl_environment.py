import numpy as np
from openenv.core.env_server.interfaces import Environment
from models import PacketObservation, PacketAction, StepResult


class PacketSchedEnv(Environment):

    def __init__(self, task: str = "easy", max_steps: int = 50, seed: int = 42):
        self.task = task
        self.max_steps = max_steps
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # state
        self.step_count = 0
        self.q_priority = 2.0
        self.q_regular = 2.0

    # RESET
    def reset(self) -> StepResult:
        self.step_count = 0
        self.q_priority = 2.0
        self.q_regular = 2.0
        self.rng = np.random.RandomState(self.seed)

        obs = PacketObservation(
            q_priority=self.q_priority,
            q_regular=self.q_regular,
            incoming=0.0,
            step=self.step_count
        )

        return StepResult(observation=obs, reward=0.0, done=False)

    # STEP
    def step(self, action: PacketAction) -> StepResult:
        self.step_count += 1

        p_ratio = float(np.clip(action.priority_ratio, 0, 1))
        r_ratio = 1.0 - p_ratio

        # ---------------------------
        # REGIME SWITCHING (time-based difficulty)
        # ---------------------------
        if self.step_count < 15:
            base_load = 2
        elif self.step_count < 30:
            base_load = 5
        else:
            base_load = 10

        # ---------------------------
        # TASK CONFIG
        # ---------------------------
        if self.task == "easy":
            p_arrival = base_load
            r_arrival = base_load
            max_bw, p_penalty = 5.0, 2

        elif self.task == "moderate":
            p_arrival = base_load + self.rng.poisson(1)
            r_arrival = base_load + self.rng.choice([1, 5], p=[0.8, 0.2])
            max_bw, p_penalty = 6.0, 5

        else:  # hard
            p_arrival = base_load + self.rng.poisson(3)
            r_arrival = base_load + self.rng.choice([5, 20], p=[0.7, 0.3])
            max_bw, p_penalty = 8.0, 10

        # ---------------------------
        # SERVE
        # ---------------------------
        p_served = min(self.q_priority, max_bw * p_ratio)
        r_served = min(self.q_regular, max_bw * r_ratio)

        self.q_priority = min(100, self.q_priority + p_arrival - p_served)
        self.q_regular = min(100, self.q_regular + r_arrival - r_served)

        # ---------------------------
        # QUEUE COUPLING (long-term effect)
        # ---------------------------
        if self.q_priority > 40:
            self.q_regular = min(100, self.q_regular + 3)

        if self.q_regular > 60:
            self.q_priority = min(100, self.q_priority + 2)

        # ---------------------------
        # NON-LINEAR REWARD (strong signal)
        # ---------------------------
        reward = -(self.q_priority ** 2 * p_penalty * 0.1) - (self.q_regular ** 1.5)

        # catastrophic overflow
        if self.q_priority >= 100 or self.q_regular >= 100:
            reward -= 1000

        # ---------------------------
        # ANTI-TRIVIAL POLICY PENALTY
        # ---------------------------
        if abs(p_ratio - 0.5) < 0.1:
            reward -= 10

        done = self.step_count >= self.max_steps

        obs = PacketObservation(
            q_priority=self.q_priority,
            q_regular=self.q_regular,
            incoming=p_arrival + r_arrival,
            step=self.step_count
        )

        return StepResult(observation=obs, reward=reward, done=done)

    def close(self):
        pass

    def state(self):
        return {
            "q_priority": self.q_priority,
            "q_regular": self.q_regular,
            "step": self.step_count,
            "task": self.task
        }