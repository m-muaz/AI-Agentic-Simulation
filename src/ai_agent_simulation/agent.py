import json
import uuid
from typing import Any, Dict, List, Optional

from .config import AgentScenario
from .llm_interface import get_llm_response
from .memory import AgentMemory, MemoryConfig


class Agent:
    """
    Represents an individual agent in the simulation, driven by an LLM.

    Each agent has a state consisting of various attributes like wealth and health.
    The agent's behavior and state transitions are determined by an LLM.
    """

    def __init__(
        self,
        initial_wealth: float,
        initial_health: float,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        household_history: Optional[str] = None,
        starting_parameters: Optional[Dict[str, Any]] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        memory: Optional[AgentMemory] = None,
    ):
        """
        Initializes an agent with a unique ID and starting attributes.

        Args:
            initial_wealth: The starting wealth of the agent.
            initial_health: The starting health of the agent (e.g., on a scale of 0 to 1).
        """
        self.agent_id = agent_id if agent_id else str(uuid.uuid4())
        self.name = name or self.agent_id
        self.wealth = initial_wealth
        self.health = initial_health
        self.history = [self.to_dict()]
        self.household_history = (household_history or "").strip()
        self.starting_parameters = starting_parameters or {}
        memory_cfg = memory_config if isinstance(memory_config, MemoryConfig) else MemoryConfig.from_dict(memory_config)
        self.memory = memory or AgentMemory(agent_id=self.agent_id, config=memory_cfg)
        self.memory.bootstrap(self.household_history)

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

    def _recent_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        return self.history[-limit:]

    def _build_prompt(self) -> str:
        """
        Builds the prompt for the LLM based on the agent's current state and history.
        """
        memory_context = self.memory.render_context()
        recent_history = json.dumps(self._recent_history(), indent=2)
        parameters = json.dumps(self.starting_parameters, indent=2) if self.starting_parameters else "None provided."

        prompt = f"""
You are an agent in a simulation of a low-income household.
Your goal is to make decisions that improve your wealth and health without referencing other agents.
You are currently roleplaying: {self.name} ({self.agent_id})
Household background: {self.household_history or "Not specified."}

Key scenario parameters:
{parameters}

Long-term memory summary:
{memory_context['summary']}

Recent notable events:
{memory_context['recent_events']}

This is your current state:
{json.dumps(self.to_dict(), indent=2)}

Recent state snapshots:
{recent_history}

Based on your current state and history, decide on your new state for the next time step.
Your health should be a value between 0.0 and 1.0.
Your wealth can be any non-negative number.

Please respond with a JSON object containing your updated "wealth" and "health".
For example: {{"wealth": 105.0, "health": 0.85}}
"""
        return prompt

    def step(self, environment):
        """
        Defines the agent's behavior for a single time step using an LLM.
        """
        prompt = self._build_prompt()
        llm_response = get_llm_response(prompt, agent_id=self.agent_id)
        response_payload = llm_response if llm_response else {"error": "empty_response"}
        self.memory.record_interaction(prompt, response_payload)

        if llm_response and "wealth" in llm_response and "health" in llm_response:
            # Update state based on LLM response
            self.wealth = float(llm_response["wealth"])
            self.health = max(0.0, min(1.0, float(llm_response["health"]))) # Clamp health between 0 and 1

            # Record new state in history
            self.history.append(self.to_dict())
        else:
            print(f"Agent {self.agent_id}: Could not update state due to invalid LLM response: {llm_response}")
            # Summarize and clear the buffer to prevent compounding errors.
            self.memory.force_summarize()

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, wealth={self.wealth}, health={self.health})"

    @classmethod
    def from_config(cls, scenario: AgentScenario) -> "Agent":
        """
        Convenience constructor that builds an Agent from a YAML scenario config.
        """
        return cls(
            initial_wealth=scenario.initial_wealth,
            initial_health=scenario.initial_health,
            agent_id=scenario.agent_id,
            name=scenario.name,
            household_history=scenario.household_history,
            starting_parameters=scenario.starting_parameters,
            memory_config=scenario.memory,
        )
