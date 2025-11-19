# `main.py`

Entry point for running the simulation from the command line.

## Responsibilities

- Instantiate the `Environment`.
- Create and configure `Agent` instances with fixed IDs and initial state.
- Register agents with the environment.
- Run a small number of simulation steps.

## Key Flow

1. `Environment()` – creates an empty simulation environment with `time_step = 0`.
2. Agent definitions are loaded from `configs/agents.yaml` (override with `AGENT_CONFIG_PATH` if desired).
3. Agents are added to the environment via `env.add_agent(...)`.
4. A loop over `num_steps` calls `env.run_step()`.

## Notes

- Each call to `env.run_step()` will trigger *one LLM request per agent* via `Agent.step`.
- Adjust `num_steps` and the initial agent parameters here when experimenting.
- Use `python scenarios/dummy_simulation.py --steps 2 5 10` to run varying-length dummy episodes that exercise the new memory system without modifying `main.py`.
