import os
import requests
import numpy as np
from typing import List, Optional

# ---------------- CONFIG ---------------- #

BASE_ENV_URL = os.getenv("BASE_ENV_URL", "http://localhost:8000")
MAX_STEPS = int(os.getenv("MAX_STEPS", 50))
SUCCESS_THRESHOLD = float(os.getenv("SUCCESS_THRESHOLD", 0.6))

BENCHMARK = "openenv_packet_env"

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

LLM_RETRIES = 3

# ---------------- LOGGING ---------------- #

def log_start(task: str):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


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

# ---------------- LLM ---------------- #

def call_llm(messages):
    if not API_KEY:
        return None

    for _ in range(LLM_RETRIES):
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "temperature": 0.2,
                },
                timeout=10,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception:
            continue

    return None


def warmup_llm():
    try:
        call_llm([{"role": "user", "content": "ping"}])
    except Exception:
        pass

# ---------------- HEURISTIC ---------------- #

def heuristic_action(obs, prev_ratio):
    q_p = obs["q_priority"]
    q_r = obs["q_regular"]

    imbalance = q_r - q_p

    if imbalance > 5:
        base = 0.4
    else:
        total = q_p + q_r + 1e-6
        base = q_p / total

    # 🔥 reference smoothing added
    return max(0.2, min(0.8, 0.6 * base + 0.4 * prev_ratio))

# ---------------- LLM POLICY ---------------- #

def get_llm_action(obs, prev_ratio, history):

    history_text = "\n".join([
        f"p={h['q_priority']:.2f}, r={h['q_regular']:.2f}, loss={h['loss_rate']:.3f}"
        for h in history[-3:]
    ])

    prompt = f"""
State:
Priority queue: {obs['q_priority']}
Regular queue: {obs['q_regular']}
Incoming: {obs['incoming']}
Loss rate: {obs['loss_rate']}
Latency: {obs['avg_latency']}
Throughput: {obs['throughput']}
Fairness: {obs['fairness_index']}
Previous ratio: {prev_ratio}

Recent:
{history_text}

Return ONLY a float between 0.0 and 1.0.
"""

    messages = [
        {"role": "system", "content": "Return ONLY a float between 0.0 and 1.0."},
        {"role": "user", "content": prompt},
    ]

    output = call_llm(messages)

    try:
        val = float(output.strip())
        return max(0.0, min(1.0, val))
    except:
        return None

# ---------------- SAFE POST ---------------- #

def safe_post(session, url, payload=None):
    try:
        res = session.post(url, json=payload, timeout=10)
        res.raise_for_status()
        return res.json(), None
    except Exception as e:
        return None, str(e)

# ---------------- CORE TASK ---------------- #

def run_task(task_name: str):
    rewards = []
    total_reward = 0.0
    steps_taken = 0
    prev_ratio = 0.5

    success = False
    score = 0.01

    obs_history = []
    metrics_history = []

    log_start(task_name)
    warmup_llm()

    session = requests.Session()

    try:
        data, err = safe_post(session, f"{BASE_ENV_URL}/reset", {"task": task_name})

        if err or data is None:
            log_step(0, "error", 0.0, True, err)
        else:
            obs = data["observation"]["observation"]

            for step in range(1, MAX_STEPS + 1):

                heuristic = heuristic_action(obs, prev_ratio)
                llm = get_llm_action(obs, prev_ratio, obs_history)

                # ✅ BLENDING (CRITICAL FIX)
                if llm is not None:
                    action_val = 0.7 * heuristic + 0.3 * llm
                else:
                    action_val = heuristic

                action_val += np.random.uniform(-0.03, 0.03)
                action_val = max(0.0, min(1.0, action_val))

                delta_ratio = abs(action_val - prev_ratio)

                data, err = safe_post(
                    session,
                    f"{BASE_ENV_URL}/step",
                    {"action": {"priority_ratio": round(action_val, 4)}},
                )

                if err or data is None:
                    log_step(step, "error", 0.0, True, err)
                    break

                obs = data["observation"]["observation"]
                reward = float(data["reward"])
                done = bool(data["done"])

                rewards.append(reward)
                total_reward += reward
                steps_taken = step

                metrics_history.append({
                    "loss_rate": obs["loss_rate"],
                    "latency": obs["avg_latency"],
                    "throughput": obs["throughput"],
                    "fairness": obs["fairness_index"],
                    "delta_ratio": delta_ratio,
                    "reward": reward
                })

                prev_ratio = action_val

                obs_history.append(obs)
                if len(obs_history) > 8:
                    obs_history.pop(0)

                log_step(step, f"{action_val:.2f}", reward, done, None)

                if done:
                    break

        # ---------------- SCORING ---------------- #

        grader_response, err = safe_post(
            session,
            f"{BASE_ENV_URL}/grader",
            {"history": metrics_history}
        )

        if err or grader_response is None:
            # fallback to reference normalization
            max_possible = max(1.0, sum(abs(r) for r in rewards) + 1e-6)
            score = total_reward / max_possible
            score = max(0.01, min(0.98, score))
        else:
            score = float(grader_response.get("score", 0.0))

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        log_step(steps_taken, "error", 0.0, True, str(e))
        success = False
        score = 0.01

    finally:
        session.close()
        log_end(success, steps_taken, score, rewards)

# ---------------- MAIN ---------------- #

def main():
    for task in ["easy", "moderate", "hard"]:
        run_task(task)


if __name__ == "__main__":
    main()
