import json
import uuid
from .llm_interface import get_llm_response
from .economics_model import calculate_poverty_trap_projection 

class Agent:
    """
    Represents an individual agent in the simulation, driven by an LLM.
    Each agent has a state consisting of various attributes like wealth and health.
    """

    def __init__(self, initial_wealth: float, initial_health: float, agent_id=None):
        self.agent_id = agent_id if agent_id else str(uuid.uuid4())
        self.wealth = initial_wealth
        self.health = initial_health
        self.history = [self.to_dict()]

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "wealth": self.wealth,
            "health": self.health,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def _build_prompt(self) -> str:
        """
        Builds the prompt for the LLM based on the agent's current state and history.
        """
        
        # Calculate the mathematical reality of the simulation
        economic_context = calculate_poverty_trap_projection(self.wealth, self.health)

        prompt = f"""
You are an agent in a simulation of a low-income household based on the Banerjee and Duflo economic model.
Your goal is to make decisions that improve your wealth and health, trying to break out of the poverty trap.


This is your current state:
{json.dumps(self.to_dict(), indent=2)}

**Economic Reality:**
{economic_context['description']}
(If your wealth is below 80.0, you face diminishing returns. Above 80.0, your capital compounds efficiently.)

This is the history of your past states:
{json.dumps(self.history, indent=2)}

Based on your current state, history, and the Economic Reality provided above, decide on your new state.
- If you invest in health, your productivity increases, but it costs wealth.
- If you save wealth, you might cross the threshold, but your health might suffer.

Please respond with a JSON object containing your updated "wealth" and "health".
For example: {{"wealth": 105.0, "health": 0.85}}
"""
        return prompt

    def step(self, environment):
        """
        Defines the agent's behavior for a single time step using an LLM.
        """
        prompt = self._build_prompt()
        llm_response = get_llm_response(prompt) #

        if llm_response and "wealth" in llm_response and "health" in llm_response:
            self.wealth = float(llm_response["wealth"])
            self.health = max(0.0, min(1.0, float(llm_response["health"])))
            self.history.append(self.to_dict())
        else:
            print(f"Agent {self.agent_id}: Could not update state due to invalid LLM response: {llm_response}")

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, wealth={self.wealth}, health={self.health})"