import os
from typing import Optional

from ai_agent_simulation.agent import Agent
from ai_agent_simulation.config import load_agent_configs
from ai_agent_simulation.environment import Environment


def main(config_path: Optional[str] = None):
    """
    Main function to set up and run the simulation.
    """
    # Create the environment
    env = Environment()

    # Load agent definitions from configuration
    agent_configs = load_agent_configs(config_path or os.getenv("AGENT_CONFIG_PATH"))
    for config in agent_configs:
        env.add_agent(Agent.from_config(config))

    # Run the simulation for a few steps
    # Note: Each step involves an API call for each agent.
    num_steps = 2
    for i in range(num_steps):
        env.run_step()


if __name__ == "__main__":
    main()
