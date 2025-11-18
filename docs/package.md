# Package Layout (`ai_agent_simulation`)

The `ai_agent_simulation` package (under `src/ai_agent_simulation`) contains the core simulation code.

## Files

- `__init__.py`
  - Marks the directory as a Python package.
  - Currently does not expose additional symbols.

- `agent.py`
  - Defines the `Agent` class for LLM-driven simulated entities.

- `environment.py`
  - Defines the `Environment` class that manages agents and simulation steps.

- `llm_interface.py`
  - Contains functions to interact with the underlying LLM via an OpenAI-compatible API.

## How It Fits Together

- `main.py` imports `Environment` and `Agent` from this package.
- The package is configured as the installable project module via `pyproject.toml` (`[tool.setuptools.packages.find]` pointing to `src`).

When extending the project, add new modules under `src/ai_agent_simulation` and document them in this folder.
