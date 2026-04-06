import asyncio
import json
import os
import numpy as np

from openai import OpenAI

from env import PacketSchedEnv
from models import PacketAction
from grader import compute_score


# 🔧 CONFIG
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")

MAX_STEPS = 50


# 🤖 PROMPT BUILDER
def build_prompt(obs, history):
    return f"""
You are managing a network router.

Current State:
- Priority Queue: {obs.q_priority:.2f}
- Regular Queue: {obs.q_regular:.2f}
- Incoming Traffic: {obs.incoming:.2f}

Goal:
Minimize total queue size, especially priority queue.

Rules:
- Choose a number between 0 and 1
- This is the fraction of bandwidth for PRIORITY traffic

Previous actions:
{history[-3:] if history else "None"}

Respond ONLY in JSON:

{{
  "priority_ratio": <float between 0 and 1>
}}
"""


#LLM ACTION
def get_action(client, prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50,
        )

        text = response.choices[0].message.content.strip()

        data = json.loads(text)
        val = float(data["priority_ratio"])

        return PacketAction(priority_ratio=np.clip(val, 0, 1))

    except Exception as e:
        print(f"[WARN] LLM failed, fallback used: {e}")
        return PacketAction(priority_ratio=0.5)


#RUN SINGLE EPISODE
async def run_episode(env, client):
    result = await env.reset()

    obs = result.observation
    done = result.done

    total_reward = 0
    history = []

    step = 0

    while not done and step < MAX_STEPS:
        step += 1

        prompt = build_prompt(obs, history)
        action = get_action(client, prompt)

        result = await env.step(action)

        obs = result.observation
        reward = result.reward
        done = result.done

        total_reward += reward
        history.append(f"step={step}, action={action.priority_ratio:.2f}, reward={reward:.2f}")

    return total_reward


#RUN ALL TASKS
async def evaluate():
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    results = {}

    for task in ["easy", "moderate", "hard"]:
        print(f"\n[RUNNING TASK] {task}")

        env = PacketSchedEnv(task=task)

        total_reward = await run_episode(env, client)
        score = compute_score(total_reward, task)

        results[task] = {
            "reward": total_reward,
            "score": score
        }

        print(f"Reward: {total_reward:.2f} | Score: {score:.3f}")

    final_score = np.mean([r["score"] for r in results.values()])

    print("\n========== FINAL RESULT ==========")
    print(results)
    print(f"FINAL SCORE: {final_score:.3f}")


#ENTRYPOINT
if __name__ == "__main__":
    asyncio.run(evaluate())