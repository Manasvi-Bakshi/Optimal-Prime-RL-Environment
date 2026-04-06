import asyncio
import os
import re
from typing import List, Optional

from openai import OpenAI
import requests

from dotenv import load_dotenv
load_dotenv()


# -----------------------------
# CONFIG 
# -----------------------------
API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

BASE_ENV_URL = os.getenv("BASE_ENV_URL", "http://localhost:8000")

MAX_STEPS = int(os.getenv("MAX_STEPS", 50))
SUCCESS_THRESHOLD = float(os.getenv("SUCCESS_THRESHOLD", 0.6))
TASK_NAME = "packet_scheduling"
BENCHMARK = "openenv_packet_env"
TEMPERATURE = 0.3
MAX_TOKENS = 100



# -----------------------------
# LOGGING (STRICT FORMAT)
# -----------------------------
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# -----------------------------
# PROMPT
# -----------------------------
SYSTEM_PROMPT = """
You are controlling a network packet scheduler.

State:
- q_priority: packets in priority queue
- q_regular: packets in regular queue
- incoming: new packets arriving
- step: current time step

Action:
- Output ONLY a number between 0 and 1 (priority_ratio)

Goal:
- Minimize total queue sizes
- Avoid overflow
- Balance priority vs regular traffic

Rules:
- If priority queue is larger → increase ratio
- If regular queue is larger → decrease ratio
- Avoid constant 0.5
- Think ahead (future congestion matters)

Output ONLY a float like: 0.73
No explanation.
"""


def build_prompt(obs, step, last_reward):
    return f"""
Step: {step}

State:
q_priority = {obs['q_priority']}
q_regular = {obs['q_regular']}
incoming = {obs['incoming']}

Last reward: {last_reward:.2f}

Choose priority_ratio:
"""


def get_action_from_llm(client: OpenAI, obs, step, last_reward) -> float:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(obs, step, last_reward)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        text = (completion.choices[0].message.content or "").strip()
        print("[LLM RAW OUTPUT]", text, flush=True)

        #value = float(text)
        match = re.search(r"\d*\.?\d+", text)
        if match:
            value = float(match.group())
        else:
            raise ValueError("No valid float found")
        return max(0.0, min(1.0, value))

    except Exception as e:
        print(f"[LLM ERROR] {str(e)}", flush=True)
        return 0.5


# -----------------------------
# MAIN LOOP
# -----------------------------
async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards = []
    steps_taken = 0
    total_reward = 0.0
    success = False
    score = 0.0

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        # RESET
        res = requests.post(f"{BASE_ENV_URL}/reset")
        data = res.json()

        obs = data["observation"]["observation"]
        last_reward = 0.0

        # 🔥 NEW: memory of last action
        last_action = 0.5

        for step in range(1, MAX_STEPS + 1):

            # 🔥 CALL LLM ONLY EVERY 5 STEPS
            if step % 5 == 1:
                last_action = get_action_from_llm(client, obs, step, last_reward)

            action_val = last_action

            # 🔥 SMART FALLBACK (if LLM failed earlier)
            if action_val == 0.5:
                if obs["q_priority"] > obs["q_regular"]:
                    action_val = 0.7
                else:
                    action_val = 0.3

            payload = {
                "action": {
                    "priority_ratio": action_val
                }
            }

            res = requests.post(f"{BASE_ENV_URL}/step", json=payload)
            data = res.json()

            obs = data["observation"]["observation"]
            reward = float(data["reward"])
            done = data["done"]

            rewards.append(reward)
            total_reward += reward
            steps_taken = step
            last_reward = reward

            log_step(step, f"{action_val:.2f}", reward, done, None)

            if done:
                break

        # -----------------------------
        # GRADER
        # -----------------------------
        MIN_REWARD = -5000.0
        MAX_REWARD = 0.0

        score = (total_reward - MIN_REWARD) / (MAX_REWARD - MIN_REWARD)
        score = max(0.0, min(1.0, score))

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        log_step(steps_taken, "error", 0.0, True, str(e))

    finally:
        log_end(success, steps_taken, score, rewards)

        
if __name__ == "__main__":
    asyncio.run(main())