# Architecture Overview

This project implements a minimal LLM-driven agent simulation.

## Core Concepts

- **Agent** – Represents an individual simulated entity (e.g., a low-income household) with state variables like wealth and health. An agent decides its next state via an LLM.
- **Environment** – Container that holds multiple agents, tracks time steps, and coordinates the simulation by invoking each agent's `step` method.
- **LLM Interface** – Thin wrapper around a local OpenAI-compatible API (e.g., vLLM), used to query a model with a structured prompt and parse JSON responses.

## Data Flow

1. `main.py` creates an `Environment` instance.
2. `main.py` instantiates one or more `Agent` objects with initial state and adds them to the `Environment`.
3. On each simulation step, `Environment.run_step()` calls `Agent.step(environment)` for every agent.
4. Each `Agent` builds a prompt from its current state and history, calls the LLM via `get_llm_response`, and updates its internal state from the JSON returned by the model.
5. The new state is appended to the agent's history, and `Environment` advances its `time_step` counter.

## Execution Model

- The simulation is **synchronous**: each time step iterates agents sequentially.
- Each agent step currently results in **one LLM call**.
- There is no inter-agent interaction yet; agents only read and update their own state.

## Extensibility Ideas

- Add more state variables (e.g., employment status, expenses, dependents).
- Model interactions between agents (e.g., trade, social support).
- Introduce policies or external shocks via environment-level parameters.
- Swap in different models or prompts via configuration.
