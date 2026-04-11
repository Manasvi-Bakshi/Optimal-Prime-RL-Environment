import os
import requests
from typing import List, Optional
from openai import OpenAI

BASE_ENV_URL = os.getenv("BASE_ENV_URL", "http://localhost:8000")
MAX_STEPS = int(os.getenv("MAX_STEPS", 50))
SUCCESS_THRESHOLD = float(os.getenv("SUCCESS_THRESHOLD", 0.6))

BENCHMARK = "openenv_packet_env"

API_KEY = os.environ.get("API_KEY")
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

client = None
if API_KEY and API_BASE_URL:
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


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


def call_llm():
    if client is None:
        return None
    try:
        return client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0.0,
        )
    except Exception:
        return None


def heuristic_action(obs, prev_ratio):
    q_p = obs["q_priority"]
    q_r = obs["q_regular"]

    total = q_p + q_r + 1e-6
    base = q_p / total

    return max(0.0, min(1.0, 0.8 * base + 0.2 * prev_ratio))


def safe_post(session, url, payload=None):
    try:
        res = session.post(url, json=payload, timeout=10)
        res.raise_for_status()
        return res.json(), None
    except Exception as e:
        return None, str(e)


def run_task(task_name: str):
    rewards = []
    total_reward = 0.0
    steps_taken = 0
    prev_ratio = 0.5

    # ✅ CRITICAL: initialize defaults
    success = False
    score = 0.01

    log_start(task_name)

    _ = call_llm()

    session = requests.Session()

    try:
        data, err = safe_post(
            session,
            f"{BASE_ENV_URL}/reset",
            {"task": task_name}
        )

        if err:
            log_step(0, "error", 0.0, True, err)
        else:
            obs = data["observation"]

            for step in range(1, MAX_STEPS + 1):
                action_val = heuristic_action(obs, prev_ratio)
                prev_ratio = action_val

                data, err = safe_post(
                    session,
                    f"{BASE_ENV_URL}/step",
                    {"action": {"priority_ratio": round(action_val, 4)}},
                )

                if err:
                    log_step(step, "error", 0.0, True, err)
                    break

                obs = data["observation"]
                reward = float(data["reward"])
                done = bool(data["done"])

                rewards.append(reward)
                total_reward += reward
                steps_taken = step

                log_step(step, f"{action_val:.2f}", reward, done, None)

                if done:
                    break

        # normalization (even if partial run)
        max_possible = max(1.0, sum(abs(r) for r in rewards) + 1e-6)
        score = total_reward / max_possible

        # strict bounds
        score = max(0.01, min(0.98, score))

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        log_step(steps_taken, "error", 0.0, True, str(e))
        success = False
        score = 0.01

    finally:
        session.close()
        log_end(success, steps_taken, score, rewards)


def main():
    for task in ["easy", "moderate", "hard"]:
        run_task(task)


if __name__ == "__main__":
    main()
