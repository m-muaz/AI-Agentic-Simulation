import json
import uuid
# NEW IMPORT: Get the threshold and projection function from the economics model
from .economics_model import calculate_poverty_trap_projection, get_poverty_trap_threshold 
from .llm_interface import get_llm_response


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

    def _build_prompt(self, economic_projection: float, threshold: float) -> str:
        """
        Builds the prompt to send to the LLM, including economic context.
        """
        
        # Determine the agent's economic status for strategic advice
        if economic_projection < 0 and self.wealth < threshold:
            status_advice = "CRITICAL: The economic model projects your wealth will **DECLINE**. You are trapped. Your decision must be strategic to counter this loss, likely by increasing health or making a risky investment."
        elif economic_projection > 0 and self.wealth > threshold:
            status_advice = "STABLE: The economic model projects your wealth will **GROW**. You should maintain or slightly adjust your current trajectory to maximize growth."
        elif economic_projection < 0:
            status_advice = "WARNING: The economic model projects your wealth will **DECLINE**. Your current health may be too low for your capital level."
        else:
            status_advice = "CAUTION: The economic model projects slight wealth **GROWTH**, but you are near the threshold. You must be careful."

        prompt = f"""
You are an intelligent agent in a poverty trap simulation.
Your goal is to make decisions that maximize your **wealth** and **health** over the long term.

**ECONOMIC CONTEXT**
- The Poverty Trap Threshold (the minimum capital required for growth) is **{threshold:.2f}**.
- Your current wealth is {self.wealth:.2f}.

**MODEL PROJECTION (If you do nothing):**
- The current economic model projects a change in wealth of **{economic_projection:+.2f}** for the next step.

**STRATEGIC ADVICE:**
{status_advice}

**CURRENT STATE:**
{json.dumps(self.to_dict(), indent=2)}

Please respond with a JSON object containing your updated "wealth" and "health".
For example: {{"wealth": 105.0, "health": 0.85}}
"""
        return prompt

    def step(self, environment):
        """
        Defines the agent's behavior for a single time step.
        """
        # 1. Get the economic model's projection
        economic_projection = calculate_poverty_trap_projection(self.wealth, self.health)
        threshold = get_poverty_trap_threshold()
        
        # 2. Build the LLM prompt with the projection
        prompt = self._build_prompt(economic_projection, threshold)
        llm_response = get_llm_response(prompt)

        if llm_response and "wealth" in llm_response and "health" in llm_response:
            
            # Agent decides on the new state (LLM output)
            llm_decided_wealth = float(llm_response.get("wealth", self.wealth))
            llm_decided_health = float(llm_response.get("health", self.health))
            
            # 1. Update state based on LLM decision (the "action")
            # We apply the agent's desired change first.
            self.wealth = max(0.0, llm_decided_wealth)
            self.health = max(0.0, min(1.0, llm_decided_health))
            
            # 2. Apply the economic change from the S-Curve model
            # The environment's physics (dk/dt) apply to the resulting wealth level.
            self.wealth += economic_projection
            
            # Final state must be non-negative
            self.wealth = max(0.0, self.wealth)
            
            # Record new state in history
            self.history.append(self.to_dict())
        else:
            print(f"Agent {self.agent_id}: Could not update state due to invalid LLM response. Applying economic projection only.")
            # If the LLM fails, apply the economic projection to continue the simulation
            self.wealth += economic_projection
            self.wealth = max(0.0, self.wealth)
            self.history.append(self.to_dict())