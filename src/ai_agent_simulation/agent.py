import json
import uuid
from .llm_interface import get_llm_response
from .economics_model import calculate_poverty_trap_projection # NEW IMPORT

class Agent:
    """
    Represents an individual agent in the simulation, driven by an LLM.
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
        Builds the prompt for the LLM, including the economic projection.
        """
        
        # --- NEW LOGIC START ---
        econ_projection = calculate_poverty_trap_projection(self.wealth, self.health)
        # --- NEW LOGIC END ---

        prompt = f"""
You are an agent in a poverty simulation based on the Banerjee & Duflo S-Curve model.
Your primary goal is to maximize your wealth and health, aiming to break out of the poverty trap (wealth threshold is 80.0).

CURRENT STATE:
Wealth: {self.wealth:.2f}
Health: {self.health:.2f}

--- ECONOMIC REALITY (The laws of the simulation) ---
{econ_projection['description']}

The mathematical model says if you take no action, your wealth will automatically become: {econ_projection['projected_wealth']:.2f}.
----------------------------------------------

DECISION:
You can choose to **deviate** from this default outcome by changing your new wealth and health values.
For instance:
1.  **Work Harder (Sacrifice Health for Wealth):** Set New Health lower than your current health to generate more income (New Wealth > Projected Wealth).
2.  **Invest in Health (Sacrifice Wealth for Health):** Set New Health higher than your current health, costing you more wealth (New Wealth < Projected Wealth).
3.  **Accept Default:** Set new values close to the projected values.

Respond with a JSON object containing your NEW "wealth" and "health".
Example: {{"wealth": {econ_projection['projected_wealth']:.2f}, "health": {self.health}}}
"""
        return prompt

    def step(self, environment):
        """
        Defines the agent's behavior for a single time step using an LLM.
        """
        prompt = self._build_prompt()
        llm_response = get_llm_response(prompt)

        if llm_response and "wealth" in llm_response and "health" in llm_response:
            self.wealth = float(llm_response["wealth"])
            self.health = max(0.0, min(1.0, float(llm_response["health"]))) # Clamp health
            self.history.append(self.to_dict())
        else:
            print(f"Agent {self.agent_id}: Could not update state due to invalid LLM response.")

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, wealth={self.wealth}, health={self.health})"