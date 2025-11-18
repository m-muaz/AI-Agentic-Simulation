# `main.py`

Entry point for running the simulation from the command line.

## Responsibilities

- Instantiate the `Environment`.
- Create and configure `Agent` instances with fixed IDs and initial state.
- Register agents with the environment.
- Run a small number of simulation steps.

## Key Flow

1. `Environment()` – creates an empty simulation environment with `time_step = 0`.
2. Two `Agent` instances are created with different initial wealth/health and stable IDs (`agent_1`, `agent_2`).
3. Agents are added to the environment via `env.add_agent(...)`.
4. A loop over `num_steps` calls `env.run_step()`.

## Notes

- Each call to `env.run_step()` will trigger *one LLM request per agent* via `Agent.step`.
- Adjust `num_steps` and the initial agent parameters here when experimenting.
