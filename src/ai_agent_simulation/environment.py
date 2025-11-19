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
                 initial_policy: Policy = Policy.BASELINE, microloan_access_percentage: float = 0.5,
                 cash_transfer_amount: float = 20.0):
        """
        Initializes the environment with macroeconomic settings.
        """
        self.agents: List[Agent] = []
        self.time_step = 0
        self.wage_rate = wage_rate
        self.loan_interest = loan_interest
        self.price_level = price_level
        self.policy = initial_policy
        self.microloan_access_percentage = microloan_access_percentage
        self.cash_transfer_amount = cash_transfer_amount

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
        # Reset loan_access for all agents before applying new policy
        for agent in self.agents:
            agent.loan_access = False

        if self.policy == Policy.CASH_TRANSFER:
            # All low-income agents receive a fixed wealth amount.
            for agent in self.agents:
                agent.wealth += self.cash_transfer_amount
        elif self.policy == Policy.MICROLOAN_ACCESS:
            # A subset of agents get loan access.
            num_agents_to_grant_access = int(len(self.agents) * self.microloan_access_percentage)
            agents_to_grant_access = random.sample(self.agents, num_agents_to_grant_access)
            for agent in agents_to_grant_access:
                agent.loan_access = True
        # For BASELINE, loan_access remains False for all agents (due to reset above)


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
