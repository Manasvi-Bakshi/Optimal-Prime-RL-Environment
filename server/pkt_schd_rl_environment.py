import numpy as np
import math
from openenv.core.env_server.interfaces import Environment
from models import PacketObservation, PacketAction, StepResult


REGIMES = {
    "balanced": dict(
        p_low=2, p_high=5,
        r_low=2, r_high=5,
        burst_p=False, burst_r=False,
        max_bw=6.0,
        rw_loss=2.0, rw_latency=1.5, rw_throughput=1.5, rw_fairness=1.0,
        qos_loss=0.05, qos_latency=5.0, qos_fairness=0.7,
    ),
    "priority_flood": dict(
        p_low=7, p_high=12,
        r_low=1, r_high=3,
        burst_p=True, burst_r=False,
        max_bw=7.0,
        rw_loss=3.0, rw_latency=2.0, rw_throughput=1.0, rw_fairness=0.5,
        qos_loss=0.03, qos_latency=4.0, qos_fairness=0.5,
    ),
    "regular_surge": dict(
        p_low=1, p_high=3,
        r_low=7, r_high=12,
        burst_p=False, burst_r=True,
        max_bw=7.0,
        rw_loss=2.0, rw_latency=1.0, rw_throughput=1.5, rw_fairness=2.5,
        qos_loss=0.05, qos_latency=6.0, qos_fairness=0.8,
    ),
    "fairness_stress": dict(
        p_low=3, p_high=6,
        r_low=4, r_high=7,
        burst_p=False, burst_r=False,
        max_bw=6.0,
        rw_loss=1.0, rw_latency=1.0, rw_throughput=1.0, rw_fairness=4.0,
        qos_loss=0.08, qos_latency=8.0, qos_fairness=0.85,
    ),
    "throughput_race": dict(
        p_low=5, p_high=9,
        r_low=5, r_high=9,
        burst_p=True, burst_r=True,
        max_bw=8.0,
        rw_loss=1.5, rw_latency=0.5, rw_throughput=3.5, rw_fairness=0.5,
        qos_loss=0.10, qos_latency=10.0, qos_fairness=0.5,
    ),
}


TASK_PHASES = {
    "easy": ["balanced", "balanced"],
    "moderate": ["balanced", "priority_flood", "regular_surge"],
    "hard": ["priority_flood", "regular_surge", "fairness_stress",
             "throughput_race", "priority_flood"],
}


class PacketSchedEnv(Environment):

    def __init__(self, task="easy", max_steps=50, seed=42):
        self.task = task
        self.max_steps = max_steps
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.max_queue_capacity = 40

        self._HISTORY_LEN = 8

        self.reset()

    def reset(self) -> StepResult:
        self.rng = np.random.RandomState(self.seed)

        self.step_count = 0
        self.q_priority = 2.0
        self.q_regular = 2.0
        self.total_latency = 0.0
        self.total_served = 0.0

        self._action_history = []
        self._sum_actions = 0.0
        self._sum_sq_actions = 0.0

        self._qos_streak = 0
        self._prev_loss_rate = 0.0

        phases = TASK_PHASES.get(self.task, TASK_PHASES["easy"])
        n = len(phases)

        boundaries = [int(round(self.max_steps * i / n)) for i in range(n + 1)]

        self._schedule = []
        for i in range(n):
            name = phases[i]
            r = REGIMES[name]
            self._schedule.append((
                boundaries[i],
                boundaries[i + 1],
                name,
                r
            ))

        self._phase_idx = 0

        obs = PacketObservation(
            q_priority=self.q_priority,
            q_regular=self.q_regular,
            incoming=0.0,
            step=self.step_count,
            p_lost=0,
            r_lost=0,
            loss_rate=0.0,
            avg_latency=0.0,
            throughput=0.0,
            fairness_index=1.0,
        )
        return StepResult(observation=obs, reward=0.0, done=False)

    def _current_phase(self):
        start, end, name, regime = self._schedule[self._phase_idx]
        if self.step_count >= end and self._phase_idx < len(self._schedule) - 1:
            self._phase_idx += 1
            start, end, name, regime = self._schedule[self._phase_idx]
        return name, regime

    def _sigmoid_loss(self, utilization):
        return 1.0 / (1.0 + math.exp(-14.0 * (utilization - 0.70)))

    def _update_action_stats(self, val):
        if len(self._action_history) == self._HISTORY_LEN:
            old = self._action_history.pop(0)
            self._sum_actions -= old
            self._sum_sq_actions -= old * old

        self._action_history.append(val)
        self._sum_actions += val
        self._sum_sq_actions += val * val

    def _action_variance(self):
        n = len(self._action_history)
        if n < 2:
            return 1.0
        mean = self._sum_actions / n
        return (self._sum_sq_actions / n) - (mean * mean)

    def step(self, action: PacketAction) -> StepResult:
        self.step_count += 1

        p_ratio = action.priority_ratio
        if p_ratio < 0.0:
            p_ratio = 0.0
        elif p_ratio > 1.0:
            p_ratio = 1.0

        r_ratio = 1.0 - p_ratio

        self._update_action_stats(p_ratio)

        phase_name, regime = self._current_phase()

        max_bw = regime["max_bw"]

        p_arrival = self.rng.randint(regime["p_low"], regime["p_high"] + 1)
        r_arrival = self.rng.randint(regime["r_low"], regime["r_high"] + 1)

        if regime["burst_p"] and self.step_count % 5 == 0:
            p_arrival += self.rng.randint(3, 7)
        if regime["burst_r"] and self.step_count % 7 == 0:
            r_arrival += self.rng.randint(3, 7)

        if self.task == "hard":
            p_arrival += self.rng.randint(0, 3)
            r_arrival += self.rng.randint(0, 3)

        p_served = min(self.q_priority, max_bw * p_ratio)
        r_served = min(self.q_regular, max_bw * r_ratio)
        total_served = p_served + r_served
        self.total_served += total_served

        utilization = (self.q_priority + self.q_regular) / self.max_queue_capacity
        loss_prob = self._sigmoid_loss(utilization)

        p_lost = self.rng.binomial(p_arrival, loss_prob)
        r_lost = self.rng.binomial(r_arrival, loss_prob)

        self.q_priority = self.q_priority + p_arrival - p_served - p_lost
        self.q_regular = self.q_regular + r_arrival - r_served - r_lost

        if self.q_priority < 0.0:
            self.q_priority = 0.0
        elif self.q_priority > self.max_queue_capacity:
            self.q_priority = self.max_queue_capacity

        if self.q_regular < 0.0:
            self.q_regular = 0.0
        elif self.q_regular > self.max_queue_capacity:
            self.q_regular = self.max_queue_capacity

        avg_latency = (self.q_priority + self.q_regular) / (total_served + 1e-6)
        self.total_latency += avg_latency

        total_arrival = p_arrival + r_arrival
        loss_rate = (p_lost + r_lost) / (total_arrival if total_arrival > 0 else 1)

        pw = 0.7
        priority_loss = p_lost / (p_arrival if p_arrival > 0 else 1)
        regular_loss = r_lost / (r_arrival if r_arrival > 0 else 1)
        weighted_loss = pw * priority_loss + (1 - pw) * regular_loss

        priority_latency = self.q_priority / (p_served + 1e-6)
        regular_latency = self.q_regular / (r_served + 1e-6)
        weighted_latency = pw * priority_latency + (1 - pw) * regular_latency

        weighted_throughput = total_served / max_bw

        sum_x = p_served + r_served
        sum_x2 = p_served * p_served + r_served * r_served
        fairness_index = (sum_x * sum_x) / (2.0 * sum_x2 + 1e-9)

        loss_score = max(0.0, 1.0 - weighted_loss / 0.15)
        latency_score = max(0.0, 1.0 - weighted_latency / 6.0)
        throughput_score = weighted_throughput if weighted_throughput < 1.0 else 1.0

        base_reward = (
            regime["rw_loss"] * loss_score +
            regime["rw_latency"] * latency_score +
            regime["rw_throughput"] * throughput_score +
            regime["rw_fairness"] * fairness_index
        )

        qos_pass = (
            weighted_loss < regime["qos_loss"] and
            weighted_latency < regime["qos_latency"] and
            fairness_index > regime["qos_fairness"]
        )

        self._qos_streak = self._qos_streak + 1 if qos_pass else 0
        qos_bonus = min(3.0, 0.5 * self._qos_streak) if self._qos_streak >= 2 else 0.0

        loss_cliff = 0.0
        if weighted_loss > 0.10:
            x = (weighted_loss - 0.10) / 0.10
            loss_cliff = 3.0 * x * x

        latency_cliff = 2.0 if weighted_latency > regime["qos_latency"] else 0.0
        fairness_cliff = 2.5 if fairness_index < regime["qos_fairness"] - 0.1 else 0.0

        overflow_penalty = 0.0
        if self.q_priority >= self.max_queue_capacity:
            overflow_penalty += 3.0
        if self.q_regular >= self.max_queue_capacity:
            overflow_penalty += 3.0

        delta_loss = loss_rate - self._prev_loss_rate
        spike_penalty = 2.0 * (delta_loss - 0.05) if delta_loss > 0.05 else 0.0
        self._prev_loss_rate = loss_rate

        action_var = self._action_variance()
        if len(self._action_history) >= self._HISTORY_LEN:
            if action_var < 0.001:
                memory_penalty = 1.5
            elif action_var < 0.005:
                memory_penalty = 0.5
            else:
                memory_penalty = 0.0
        else:
            memory_penalty = 0.0

        regime_penalty = 0.0
        if phase_name == "priority_flood" and p_ratio < 0.55:
            regime_penalty = 1.5 * (0.55 - p_ratio)
        elif phase_name == "regular_surge" and p_ratio > 0.50:
            regime_penalty = 1.5 * (p_ratio - 0.50)
        elif phase_name == "fairness_stress":
            if p_ratio < 0.38 or p_ratio > 0.62:
                regime_penalty = abs(p_ratio - 0.50)

        reward = (
            base_reward + qos_bonus
            - loss_cliff - latency_cliff - fairness_cliff
            - overflow_penalty - spike_penalty
            - memory_penalty - regime_penalty
        )

        done = self.step_count >= self.max_steps

        obs = PacketObservation(
            q_priority=self.q_priority,
            q_regular=self.q_regular,
            incoming=float(total_arrival),
            step=self.step_count,
            p_lost=p_lost,
            r_lost=r_lost,
            loss_rate=loss_rate,
            avg_latency=avg_latency,
            throughput=total_served,
            fairness_index=fairness_index,
        )

        return StepResult(observation=obs, reward=reward, done=done)

    def close(self):
        pass

    def state(self):
        phase_name, _ = self._current_phase()
        return {
            "q_priority": self.q_priority,
            "q_regular": self.q_regular,
            "step": self.step_count,
            "task": self.task,
            "avg_latency": self.total_latency / max(1, self.step_count),
            "throughput": self.total_served,
            "regime": phase_name,
            "qos_streak": self._qos_streak,
        }