from typing import List

from .agent import Agent


class Environment:
    """
    Manages the simulation world, its agents, and their interactions.
    """

    def __init__(
        self,
        economic_model=None,
        experiment_phase: int = 1,
        structural_data_path: str | None = None,
    ):
        """
        Initializes the environment.
        """
        self.agents: List[Agent] = []
        self.time_step = 0
        self.economic_model = economic_model
        self.experiment_phase = int(experiment_phase or 1)
        # Optional: path to structural data used by phase 3.
        self.structural_data_path = structural_data_path

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
        Returns the list of all agents.

        Returns:
            A list of agents.
        """
        return self.agents
