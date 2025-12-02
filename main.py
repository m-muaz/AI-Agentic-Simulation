import os
from typing import Optional

from ai_agent_simulation.agent import Agent
from ai_agent_simulation.config import load_agent_configs
from ai_agent_simulation.environment import Environment


def main(config_path: Optional[str] = None, steps: Optional[int] = None):
    """
    Entrypoint that wires the LLM-driven agents into the environment.
    """
    env = Environment()

    agent_configs = load_agent_configs(config_path or os.getenv("AGENT_CONFIG_PATH"))
    for config in agent_configs:
        env.add_agent(Agent.from_config(config))

    num_steps = steps or int(os.getenv("SIM_STEPS", "2"))
    for _ in range(num_steps):
        env.run_step()


if __name__ == "__main__":
    main()
