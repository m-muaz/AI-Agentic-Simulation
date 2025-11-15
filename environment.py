from typing import List
from agent import Agent


class Environment:
    """
    Manages the simulation world, its agents, and their interactions.
    """

    def __init__(self):
        """
        Initializes the environment.
        """
        self.agents: List[Agent] = []
        self.time_step = 0

    def add_agent(self, agent: Agent):
        """
        Adds an agent to the environment.

        Args:
            agent: The agent to add.
        """
        self.agents.append(agent)

    def run_step(self):
        """
        Executes a single time step of the simulation.

        This involves calling the 'step' method for each agent.
        """
        print(f"--- Time Step {self.time_step} ---")
        for agent in self.agents:
            agent.step(self)
            print(agent.to_json())
        self.time_step += 1

    def get_agents(self) -> List[Agent]:
        """
        Returns the list of all agents in the environment.

        Returns:
            A list of agents.
        """
        return self.agents
