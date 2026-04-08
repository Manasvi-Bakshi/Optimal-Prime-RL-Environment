import os
import requests
from typing import List, Optional

BASE_ENV_URL = os.getenv("BASE_ENV_URL", "http://localhost:8000")
MAX_STEPS = int(os.getenv("MAX_STEPS", 50))
SUCCESS_THRESHOLD = float(os.getenv("SUCCESS_THRESHOLD", 0.6))

TASK_NAME = "packet_scheduling"
BENCHMARK = "openenv_packet_env"


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def detect_regime(obs, history):
    q_p = obs["q_priority"]
    q_r = obs["q_regular"]
    incoming = obs["incoming"]
    fairness = obs["fairness_index"]
    loss = obs["loss_rate"]

    n = len(history)
    total_p = q_p
    total_r = q_r

    for i in range(max(0, n - 4), n):
        h = history[i]
        total_p += h["q_priority"]
        total_r += h["q_regular"]

    denom = min(5, n + 1)
    avg_q_p = total_p / denom
    avg_q_r = total_r / denom

    total_q = avg_q_p + avg_q_r + 1e-6
    p_pressure = avg_q_p / total_q

    if incoming > 14 and loss > 0.05:
        return "throughput_race"
    if p_pressure > 0.58 and avg_q_p > avg_q_r * 1.4:
        return "priority_flood"
    if p_pressure < 0.42 and avg_q_r > avg_q_p * 1.4:
        return "regular_surge"
    if fairness < 0.72 and total_q > 4.0:
        return "fairness_stress"

    return "balanced"


def heuristic_action(obs, history, prev_ratio):
    q_p = obs["q_priority"]
    q_r = obs["q_regular"]
    loss = obs["loss_rate"]
    latency = obs["avg_latency"]
    fairness = obs["fairness_index"]

    regime = detect_regime(obs, history)

    if regime == "priority_flood":
        target = 0.72
        if q_p > 20:
            target = min(0.90, target + 0.08)
        if q_r > 15:
            target -= 0.06

    elif regime == "regular_surge":
        target = 0.38
        if q_r > 20:
            target = max(0.15, target - 0.08)
        if q_p > 15:
            target += 0.06

    elif regime == "fairness_stress":
        target = 0.50
        total = q_p + q_r + 1e-6
        imbalance = (q_p - q_r) / total
        target -= 0.3 * imbalance
        target = max(0.38, min(0.62, target))

    elif regime == "throughput_race":
        total = q_p + q_r + 1e-6
        target = 0.50 if total < 1.0 else max(0.30, min(0.70, q_p / total))

    else:
        total = q_p + q_r + 1e-6
        target = max(0.40, min(0.65, q_p / total if total > 0 else 0.50))

    if loss > 0.10:
        target = min(target + 0.05, 0.85)

    if latency > 8.0:
        target = min(target + 0.05, 0.90) if q_p > q_r else max(target - 0.05, 0.10)

    if fairness < 0.55:
        target = 0.50

    if q_p >= 36:
        target = min(0.92, target + 0.15)
    if q_r >= 36:
        target = max(0.08, target - 0.15)

    return max(0.0, min(1.0, 0.8 * target + 0.2 * prev_ratio))


def safe_post(session, url, payload=None):
    try:
        res = session.post(url, json=payload, timeout=10)
        res.raise_for_status()
        return res.json(), None
    except Exception as e:
        return None, str(e)


def main():
    rewards = []
    total_reward = 0.0
    steps_taken = 0
    success = False
    score = 0.0

    obs_history = []
    prev_ratio = 0.5

    log_start(TASK_NAME, BENCHMARK, "heuristic-baseline")

    session = requests.Session()

    try:
        data, err = safe_post(session, f"{BASE_ENV_URL}/reset")
        if err:
            log_step(0, "error", 0.0, True, err)
            log_end(False, 0, 0.0, [])
            return

        obs = data["observation"]["observation"]

        for step in range(1, MAX_STEPS + 1):
            action_val = heuristic_action(obs, obs_history, prev_ratio)
            prev_ratio = action_val

            data, err = safe_post(
                session,
                f"{BASE_ENV_URL}/step",
                {"action": {"priority_ratio": action_val}}
            )

            if err:
                log_step(step, "error", 0.0, True, err)
                break

            obs = data["observation"]["observation"]
            reward = float(data["reward"])
            done = data["done"]

            obs_history.append(obs)
            if len(obs_history) > 8:
                obs_history.pop(0)

            rewards.append(reward)
            total_reward += reward
            steps_taken = step

            log_step(step, f"{action_val:.2f}", reward, done, None)

            if done:
                break

        max_total_reward = MAX_STEPS * 5.0
        score = total_reward / max_total_reward
        score = max(0.0, min(1.0, score))

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        log_step(steps_taken, "error", 0.0, True, str(e))

    finally:
        session.close()
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    main()
