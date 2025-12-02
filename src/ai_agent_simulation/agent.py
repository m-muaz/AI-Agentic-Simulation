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
        hhid: Optional[int] = None,
        start_wave: int = 1,
    ):
        """
        Initializes an agent with a unique ID and starting attributes.

        Args:
            initial_wealth: The starting wealth of the agent.
            initial_health: The starting health of the agent (e.g., on a scale of 0 to 1).
        """
        self.agent_id = agent_id if agent_id else str(uuid.uuid4())
        self.name = name or self.agent_id
        self.hhid = hhid
        self.start_wave = start_wave
        self.wealth = initial_wealth
        self.health = initial_health
        self.baseline_wealth = initial_wealth
        # Additional state used by Banerjee/Balboni-style simulations.
        self.investment: Optional[float] = None
        self.total_gain: Optional[float] = None
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
            "investment": self.investment,
            "total_gain": self.total_gain,
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

    def _build_prompt(self, environment=None) -> str:
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
        last_state = self.history[-1] if self.history else self.to_dict()
        prev_wealth = last_state.get("wealth", 0.0)
        prev_health = last_state.get("health", 0.0)
        step_index = getattr(environment, "time_step", len(self.history) - 1)
        asset_threshold = None
        econ_model = getattr(environment, "economic_model", None) if environment else None
        if econ_model and getattr(econ_model, "params", None):
            asset_threshold = getattr(econ_model.params, "asset_threshold", None)

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
        Economic environment notes:
          - There is a poverty threshold at asset level: {asset_threshold if asset_threshold is not None else "not specified"}.
          - Below the threshold, returns are low; above, returns are higher.
          - Health increases or decreases returns.
          - Shocks can positively or negatively affect future wealth.
        Current simulation step: {step_index}
        Your goal: choose actions that help the household grow wealth and avoid slipping below the threshold.
        Decide on your actions for the next time step. The economic model will compute wealth based on your decisions.
        Keep values realistic and avoid leaving variables unchanged without justification.
        Your output MUST be valid JSON with the following keys:
          - "savings_rate": fraction of income/returns you save (0.0-1.0).
          - "health_effort": effort toward health (0.0-1.0); >0.5 improves health modestly, <0.5 reduces it.
          - Optional: "risk_profile": "low" | "medium" | "high" (controls exposure to shocks). If omitted, use neutral risk.
          - Optional: "risk_tolerance": numeric shock scale (0.1-3.0). If both risk_profile and risk_tolerance are provided, risk_tolerance wins.
          - "rationale": 1-3 sentences explaining your choices and expected impact.
        Use prior changes to guide variation; avoid flat trajectories unless justified.
        """
        return prompt

    def step(self, environment):
        """
        Defines the agent's behavior for a single time step using an LLM.
        If an economic_model is present on the environment, the LLM outputs decisions
        and the model computes the next state.
        """
        prompt = self._build_prompt(environment)
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

        if llm_response:
            econ_model = getattr(environment, "economic_model", None)
            if econ_model is not None:
                self._apply_decision_with_model(llm_response, econ_model)
            elif "wealth" in llm_response and "health" in llm_response:
                # Fallback to direct-state mode if no economic model is provided.
                self._apply_direct_state(llm_response)
            else:
                print(
                    f"Agent {self.agent_id}: Could not update state due to invalid LLM response: {llm_response}"
                )
                self.memory.force_summarize()
        else:
            print(
                f"Agent {self.agent_id}: Could not update state due to invalid LLM response: {llm_response}"
            )
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
            hhid=getattr(scenario, "hhid", None),
            start_wave=getattr(scenario, "start_wave", 1),
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

    def _apply_decision_with_model(self, llm_response: Dict[str, Any], econ_model):
        """
        Apply LLM decisions via the economic model to compute next state.
        """
        prev_wealth = float(self.wealth)
        prev_health = float(self.health)
        savings_rate = llm_response.get("savings_rate", None)
        risk_tol = llm_response.get("risk_tolerance", 1.0)
        risk_profile = llm_response.get("risk_profile")
        health_effort = llm_response.get("health_effort", 0.5)

        # Clamp inputs
        try:
            savings_rate = None if savings_rate is None else max(0.0, min(1.0, float(savings_rate)))
        except (TypeError, ValueError):
            savings_rate = None
        # Map risk_profile to numeric tolerance if provided.
        profile_map = {"low": 0.6, "medium": 1.0, "high": 1.6}
        if isinstance(risk_profile, str) and risk_profile.lower() in profile_map:
            risk_tol = profile_map[risk_profile.lower()]
        try:
            risk_tol = max(0.1, min(3.0, float(risk_tol)))
        except (TypeError, ValueError):
            risk_tol = 1.0
        try:
            health_effort = max(0.0, min(1.0, float(health_effort)))
        except (TypeError, ValueError):
            health_effort = 0.5

        # Health adjustment: nudge toward/away from baseline based on effort
        health_delta = 0.1 * (health_effort - 0.5)
        new_health = max(0.0, min(1.0, prev_health + health_delta))

        next_wealth = econ_model.simulate_next_wealth(
            wealth_t=prev_wealth,
            health_t=new_health,
            savings_rate_override=savings_rate,
            shock_scale=risk_tol,
        )

        self.wealth = float(next_wealth)
        self.health = new_health
        self.investment = self.wealth - prev_wealth
        self.total_gain = self.wealth - self.baseline_wealth

        rationale = llm_response.get("rationale")
        self.last_rationale = rationale.strip() if isinstance(rationale, str) else None
        self.history.append(self.to_dict())

    def _apply_direct_state(self, llm_response: Dict[str, Any]):
        """
        Fallback for environments without an economic model; apply state directly.
        """
        self.wealth = float(llm_response["wealth"])
        self.health = max(0.0, min(1.0, float(llm_response["health"])))
        if "investment" in llm_response:
            try:
                self.investment = float(llm_response["investment"])
            except (TypeError, ValueError):
                self.investment = None
        if "total_gain" in llm_response:
            try:
                self.total_gain = float(llm_response["total_gain"])
            except (TypeError, ValueError):
                self.total_gain = None
        rationale = llm_response.get("rationale")
        self.last_rationale = rationale.strip() if isinstance(rationale, str) else None
        self.history.append(self.to_dict())
