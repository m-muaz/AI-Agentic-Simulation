# `agent.py`

Defines the `Agent` class, which models an LLM-driven entity in the simulation.

## Agent State

Each agent tracks:

- `agent_id` – Unique identifier (provided or auto-generated via UUID).
- `wealth` – Numeric value representing financial resources.
- `health` – Float between 0.0 and 1.0 representing health level.
- `history` – List of past state snapshots (each entry is the result of `to_dict()`).

## Core Methods

- `__init__(initial_wealth, initial_health, agent_id=None)`
  - Initializes the agent's state and seeds `history` with the initial snapshot.

- `to_dict()`
  - Returns a dict with `agent_id`, `wealth`, and `health` (current state only).

- `to_json()`
  - Convenience wrapper that returns the current state as pretty-printed JSON.

- `_build_prompt()`
  - Private method that constructs a natural-language prompt embedding:
    - Current state (`to_dict()`)
    - Full `history` list so far
  - The prompt instructs the LLM to propose a new state as JSON with `wealth` and `health` fields, with constraints on value ranges.

- `step(environment)`
  - Core behavior for one time step.
  - Builds a prompt and calls `get_llm_response(prompt)`.
  - On a valid response containing `wealth` and `health`:
    - Updates `self.wealth` and clamped `self.health` (between 0.0 and 1.0).
    - Appends the new `to_dict()` snapshot to `history`.
  - On invalid response or error, logs an error and keeps the previous state.

- `__repr__()`
  - Returns a concise text representation for debugging/logging.

## Error Handling

- `get_llm_response` can return `{}` on failure; `step` checks keys defensively and prints a message if the update is skipped.

## Extension Points

- Add more fields to the agent state and include them in `to_dict`, `history`, and `_build_prompt`.
- Incorporate environment-level signals into the prompt (e.g., policy changes, resource availability).
- Add policy rules to post-process the LLM suggestion (e.g., cap max wealth changes per step).
