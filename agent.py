import json
import uuid


class Agent:
    """
    Represents an individual agent in the simulation.

    Each agent has a state consisting of various attributes like wealth and health.
    """

    def __init__(self, initial_wealth: float, initial_health: float):
        """
        Initializes an agent with a unique ID and starting attributes.

        Args:
            initial_wealth: The starting wealth of the agent.
            initial_health: The starting health of the agent (e.g., on a scale of 0 to 1).
        """
        self.agent_id = str(uuid.uuid4())
        self.wealth = initial_wealth
        self.health = initial_health

    def to_dict(self) -> dict:
        """
        Returns the agent's current state as a dictionary.

        Returns:
            A dictionary representing the agent's state.
        """
        return {
            "agent_id": self.agent_id,
            "wealth": self.wealth,
            "health": self.health,
        }

    def to_json(self) -> str:
        """
        Returns the agent's current state as a JSON string.

        Returns:
            A JSON string representing the agent's state.
        """
        return json.dumps(self.to_dict(), indent=4)

    def step(self, environment):
        """
        Defines the agent's behavior for a single time step.

        This method will be called by the environment at each step of the simulation.
        For now, it's a placeholder.

        Args:
            environment: The environment in which the agent exists.
        """
        # In the future, agent logic will go here.
        # For example, the agent might interact with the environment or other agents.
        pass

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, wealth={self.wealth}, health={self.health})"
