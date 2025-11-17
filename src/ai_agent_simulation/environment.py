from typing import List
from enum import Enum, auto
import random
from .agent import Agent


class Policy(Enum):
    """
    Represents the policy interventions available in the environment.
    """
    BASELINE = auto()
    CASH_TRANSFER = auto()
    MICROLOAN_ACCESS = auto()


class Environment:
    """
    Manages the simulation world, its agents, and their interactions.
    Represents the macroeconomic setting within which all agents operate.
    """

    def __init__(self, wage_rate: float, loan_interest: float, price_level: float,
                 initial_policy: Policy = Policy.BASELINE):
        """
        Initializes the environment with macroeconomic settings.
        """
        self.agents: List[Agent] = []
        self.time_step = 0
        self.wage_rate = wage_rate
        self.loan_interest = loan_interest
        self.price_level = price_level
        self.policy = initial_policy

    def add_agent(self, agent: Agent):
        """
        Adds an agent to the environment.
        """
        self.agents.append(agent)

    def set_policy(self, policy: Policy):
        """
        Switches the active policy.
        """
        print(f"--- Changing policy to {policy.name} ---")
        self.policy = policy

    def _apply_policy(self):
        """
        Applies the current policy to the agents.
        """
        if self.policy == Policy.CASH_TRANSFER:
            # All low-income agents receive a fixed wealth amount.
            # (Here, we apply it to all agents for simplicity)
            for agent in self.agents:
                agent.wealth += 20
        elif self.policy == Policy.MICROLOAN_ACCESS:
            # A subset of agents get loan access.
            # (Here, we give it to all agents for simplicity)
            for agent in self.agents:
                agent.loan_access = True
        elif self.policy == Policy.BASELINE:
            # Reset any policy effects if necessary
            # For now, we'll reset loan_access for demonstration
            for agent in self.agents:
                agent.loan_access = False


    def run_step(self):
        """
        Executes a single time step of the simulation.
        """
        print(f"--- Time Step {self.time_step} (Policy: {self.policy.name}) ---")

        # Apply the current policy at the beginning of the step
        self._apply_policy()

        # Agents take their actions
        for agent in self.agents:
            agent.step(self)
            print(agent.to_json())
        self.time_step += 1

    def get_agents(self) -> List[Agent]:
        """
        Returns the list of all agents in the environment.
        """
        return self.agents
