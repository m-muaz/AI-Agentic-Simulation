import json
import uuid
from .llm_interface import get_llm_response


class Agent:
    """
    Represents an individual agent in the simulation, driven by an LLM.
    Each agent represents a low-income household deciding how to use their money.
    """

    def __init__(self, agent_id: str, initial_wealth: float, initial_health: float,
                 income: float, loan_access: bool, education: float, consumption_preference: float):
        """
        Initializes an agent with a unique ID and starting attributes.
        """
        self.agent_id = agent_id
        self.wealth = initial_wealth
        self.health = initial_health
        self.income = income
        self.loan_access = loan_access
        self.education = education
        self.consumption_preference = consumption_preference
        self.history = [self.to_dict()]

    def to_dict(self) -> dict:
        """
        Returns the agent's current state as a dictionary.
        """
        return {
            "agent_id": self.agent_id,
            "wealth": self.wealth,
            "health": self.health,
            "income": self.income,
            "loan_access": self.loan_access,
            "education": self.education,
            "consumption_preference": self.consumption_preference,
        }

    def to_json(self) -> str:
        """
        Returns the agent's current state as a JSON string.

        Returns:
            A JSON string representing the agent's state.
        """
        return json.dumps(self.to_dict(), indent=4)

    def _build_prompt(self, environment) -> str:
        """
        Builds the prompt for the LLM based on the agent's and environment's current state.
        """
        env_state = {
            "wage_rate": environment.wage_rate,
            "loan_interest": environment.loan_interest,
            "price_level": environment.price_level,
            "policy": environment.policy.name,
        }

        prompt = f"""
You are an agent in a simulation of a low-income household.
Your goal is to make decisions that improve your long-term well-being, primarily your wealth and health.

This is the current state of the macroeconomic environment you live in:
{json.dumps(env_state, indent=2)}

'price_level' is a general cost index for consumption and investment.
'wage_rate' is a baseline income multiplier.
The current policy is '{env_state["policy"]}'.

This is your personal current state:
{json.dumps(self.to_dict(), indent=2)}

This is the history of your past states:
{json.dumps(self.history, indent=2)}

Your 'education' and 'health' affect your productivity and future income.
'loan_access' determines if you can borrow money.
'consumption_preference' is the fraction of disposable income you prefer to consume.

Given the environment and your personal state, decide on your new state for the next time step.
Your decisions should reflect a rational attempt to improve your situation.
For example, you might choose to spend money on something that improves your health or education,
or you might save money to increase your wealth.

Please respond with a JSON object containing your updated "wealth" and "health".
The change in your wealth should be realistic based on your income, consumption preferences, and the environment's price level.
Your health should be a value between 0.0 and 1.0.

Example response: {{"wealth": 1050.0, "health": 0.85}}
"""
        return prompt

    def step(self, environment):
        """
        Defines the agent's behavior for a single time step using an LLM.
        """
        prompt = self._build_prompt(environment)
        llm_response = get_llm_response(prompt)

        if llm_response and "wealth" in llm_response and "health" in llm_response:
            # Update state based on LLM response
            self.wealth = float(llm_response["wealth"])
            self.health = max(0.0, min(1.0, float(llm_response["health"]))) # Clamp health between 0 and 1

            # Record new state in history
            self.history.append(self.to_dict())
        else:
            print(f"Agent {self.agent_id}: Could not update state due to invalid LLM response: {llm_response}")


    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, wealth={self.wealth}, health={self.health})"