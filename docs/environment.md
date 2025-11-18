# `environment.py`

Defines the `Environment` class that manages agents and simulation time.

## Environment State

- `agents` – List of `Agent` instances participating in the simulation.
- `time_step` – Integer counter for the current simulation step (starts at 0).

## Core Methods

- `__init__()`
  - Initializes an empty environment with no agents and `time_step = 0`.

- `add_agent(agent: Agent)`
  - Appends the provided `Agent` instance to the environment's `agents` list.

- `run_step()`
  - Prints a header with the current `time_step`.
  - Iterates over `self.agents` and calls `agent.step(self)` on each.
  - After each agent's step, prints `agent.to_json()` to show the new state.
  - Increments `time_step` by 1.

- `get_agents()`
  - Returns the current list of agents.

## Characteristics

- Steps are run sequentially; there is no concurrency in `run_step`.
- Agents don't explicitly interact with one another yet; they only update their internal state based on LLM output.

## Extension Points

- Add global environment variables that agents can read (e.g. prices, policies).
- Implement inter-agent interactions, markets, or events triggered per step.
- Support different scheduling strategies (e.g. random order, subsets of agents).
