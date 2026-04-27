# Dynamic Bandwidth Allocation using Reinforcement Learning

A simple reinforcement learning project that learns how to manage network traffic using a hybrid approach (rules + AI).

This project simulates a network where two types of packets (priority and regular) compete for limited bandwidth. The goal is to dynamically adjust how much bandwidth each gets in order to reduce packet loss, improve latency, and maintain fairness.


## What this project does

In real-world networks, you often need to decide:
- Who gets priority?
- How to avoid congestion?
- How to balance fairness vs performance?

This project builds a small environment where an agent learns to:
- Allocate bandwidth between priority and regular traffic
- Adapt to changing traffic patterns
- Maximize overall network performance


## Key Idea

Instead of using pure reinforcement learning, this project combines:

- A **heuristic policy** (rule-based logic)
- A **language model (LLM)** for decision support
- A **blending strategy** to combine both

Final action:
action = 0.7 * heuristic + 0.3 * LLM

## Environment Overview

The environment simulates packet scheduling with:

- Two queues:
  - Priority queue
  - Regular queue

- Limited bandwidth per step

- Changing traffic patterns (called *regimes*):
  - Balanced traffic
  - Priority-heavy traffic
  - Regular-heavy traffic
  - Fairness stress scenarios
  - Throughput-focused scenarios

These regimes change over time, making the problem dynamic and harder.


## Observations (State)

At each step, the agent receives:

- Queue sizes (priority + regular)
- Incoming packets
- Packet loss rate
- Latency
- Throughput
- Fairness score

## Actions

The agent outputs:

- `priority_ratio` → a value between 0 and 1

This controls how much bandwidth goes to priority traffic vs regular traffic.

## Reward Function

The agent is rewarded based on:

- Lower packet loss
- Lower latency
- Higher throughput
- Better fairness

It is penalized for:
- Congestion
- Unstable decisions
- Ignoring traffic patterns


## How the Agent Works

The decision pipeline:

1. Compute a **heuristic action** (fast, stable)
2. Query the **LLM** (optional, if API key provided)
3. Blend both outputs
4. Add small randomness for exploration
5. Send action to environment

