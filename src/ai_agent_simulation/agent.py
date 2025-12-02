import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AgentScenario
from .llm_interface import get_llm_response
from .memory import AgentMemory, MemoryConfig
from .tokenization import count_tokens


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
        self.last_rationale: Optional[str] = None
        self.history = [self.to_dict()]
        self.household_history = (household_history or "").strip()
        self.starting_parameters = starting_parameters or {}
        memory_cfg = (
            memory_config
            if isinstance(memory_config, MemoryConfig)
            else MemoryConfig.from_dict(memory_config)
        )
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
            "rationale": self.last_rationale,
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
        household_background = " ".join(self.household_history.split())
        wealth_change = ", ".join(str(snap["wealth"]) for snap in self.history)
        health_change = ", ".join(str(snap["health"]) for snap in self.history)
        parameters = (
            "; ".join(f"{k}: {v}" for k, v in self.starting_parameters.items())
            if self.starting_parameters
            else "None"
        )

        prompt = f"""
        You are an agent in a simulation of a low-income household.
        You are currently roleplaying: {self.name} ({self.agent_id})
        Household background: {household_background or "Not specified."}
        You care about the following factors when making decisions:
        {parameters}
        Wealth trajectory over time: {wealth_change}
        Health trajectory over time: {health_change}
        This is your current state:
        {json.dumps(self.to_dict(), indent=2)}
        Based on your current state and history, decide on your new state for the next time step. Take the change in wealth and health into careful consideration and think how your current decision will impact your goals in the future.
        Your health should be a value between 0.0 and 1.0.
        Your wealth can be any non-negative number.
        Please respond with a JSON object containing the amount of change for "wealth", "health", and a short "rationale" for each variable's change (1-3 sentences).         """
        return prompt

    def step(self, environment):
        """
        Defines the agent's behavior for a single time step using an LLM.
        """
        prompt = self._build_prompt()
        context_usage = self.memory.context_usage()
        prompt_tokens = count_tokens(prompt)
        limit = context_usage["context_token_limit"]
        context_usage["prompt_tokens"] = prompt_tokens
        context_usage["prompt_percent_of_limit"] = min(
            100.0, prompt_tokens / limit * 100.0
        )

        llm_response = get_llm_response(prompt, agent_id=self.agent_id)
        response_payload = llm_response if llm_response else {"error": "empty_response"}
        self.memory.record_interaction(prompt, response_payload)

        if llm_response and "wealth" in llm_response and "health" in llm_response:
            # Update state based on LLM response
            self.wealth = float(llm_response["wealth"])
            self.health = max(
                0.0, min(1.0, float(llm_response["health"]))
            )  # Clamp health between 0 and 1
            rationale = llm_response.get("rationale")
            self.last_rationale = (
                rationale.strip() if isinstance(rationale, str) else None
            )

            # Record new state in history
            self.history.append(self.to_dict())
        else:
            print(
                f"Agent {self.agent_id}: Could not update state due to invalid LLM response: {llm_response}"
            )
            # Summarize and clear the buffer to prevent compounding errors.
            self.memory.force_summarize()

        self._log_step(
            step_idx=environment.time_step,
            prompt=prompt,
            llm_response=llm_response,
            context_usage=context_usage,
        )

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

    def _log_step(
        self,
        step_idx: int,
        prompt: str,
        llm_response: Dict[str, Any],
        context_usage: Dict[str, float],
    ):
        """
        Persist a JSONL record of each step with rationale, context stats, and memory snapshots.
        """
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "step": step_idx,
            "agent_id": self.agent_id,
            "prompt": prompt,
            "response": llm_response,
            "rationale": self.last_rationale,
            "state": self.to_dict(),
            "context_usage": context_usage,
            "memory": self.memory.export_state(),
        }
        log_path = log_dir / f"agent_{self.agent_id}.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

        # Also log state trajectories and rationale to CSV for quick audit.
        csv_path = log_dir / f"agent_{self.agent_id}.csv"
        file_exists = csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(
                    ["timestamp", "step", "agent_id", "wealth", "health", "rationale"]
                )
            writer.writerow(
                [
                    entry["timestamp"],
                    step_idx,
                    self.agent_id,
                    self.wealth,
                    self.health,
                    self.last_rationale or "",
                ]
            )
