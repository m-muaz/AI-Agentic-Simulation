"""
Experiment Phase 1 helpers:
- LLM chooses savings_rate_t and risk_profile_t; model handles returns, threshold,
  depreciation, shocks, and investment (delta wealth).
"""

from __future__ import annotations

import json
from typing import Any, Dict


def build_prompt(agent, environment) -> str:
    household_background = " ".join(agent.household_history.split())
    wealth_change = ", ".join(str(snap["wealth"]) for snap in agent.history)
    health_change = ", ".join(str(snap["health"]) for snap in agent.history)
    last_state = agent.history[-1] if agent.history else agent.to_dict()
    step_index = getattr(environment, "time_step", len(agent.history) - 1)
    econ_model = getattr(environment, "economic_model", None) if environment else None
    params = getattr(econ_model, "params", None) if econ_model else None
    threshold = getattr(params, "asset_threshold", "not specified") if params else "not specified"
    low_return = getattr(params, "low_return_rate", None) if params else None
    high_return = getattr(params, "high_return_rate", None) if params else None
    depreciation = getattr(params, "depreciation_rate", None) if params else None
    shock_std = getattr(params, "shock_std_dev", None) if params else None

    prompt = f"""
    You are an agent in a simulation of a low-income household.
    You are currently roleplaying: {agent.name} ({agent.agent_id})
    Household background: {household_background or "Not specified."}
    Wealth trajectory over time: {wealth_change}
    Health trajectory over time: {health_change}
    This is your current state:
    {json.dumps(last_state, indent=2)}
    Economic environment notes:
      - Threshold regime at asset level: {threshold}.
      - Returns: low-regime={low_return} vs high-regime={high_return}; depreciation={depreciation}.
      - Shocks have std dev={shock_std}; higher risk exposure increases variability.
    Current simulation step: {step_index}
    Experiment Phase 1: you only decide savings and risk. The environment/model applies returns, threshold regime, depreciation, and shocks to update wealth. Investment is tracked as the change in wealth (delta wealth) after the update.
    Your output MUST be valid JSON with the following keys:
      - "savings_rate_t": fraction of income/returns to save (0.0-1.0).
      - "risk_profile_t": "low" | "medium" | "high" (controls exposure to shocks).
      - Optional: "rationale": 1-3 sentences explaining your choices and expected impact.
    Do NOT include wealth, health, or investment in the output; those are computed by the model.
    Example: {{"savings_rate_t": 0.55, "risk_profile_t": "medium", "rationale": "..." }}
    """
    return prompt


def apply_decision(agent, llm_response: Dict[str, Any], econ_model):
    """
    Phase 1: LLM provides savings rate and risk profile; model handles wealth update.
    """
    prev_wealth = float(agent.wealth)
    prev_health = float(agent.health)
    savings_rate = llm_response.get("savings_rate_t", llm_response.get("savings_rate"))
    risk_profile = llm_response.get("risk_profile_t", llm_response.get("risk_profile"))
    rationale = llm_response.get("rationale")

    try:
        savings_rate = None if savings_rate is None else max(0.0, min(1.0, float(savings_rate)))
    except (TypeError, ValueError):
        savings_rate = None
    profile_map = {"low": 0.6, "medium": 1.0, "high": 1.6}
    risk_scale = profile_map.get(str(risk_profile).lower(), 1.0) if risk_profile is not None else 1.0

    next_wealth = econ_model.simulate_next_wealth(
        wealth_t=prev_wealth,
        health_t=prev_health,
        savings_rate_override=savings_rate,
        shock_scale=risk_scale,
    )

    agent.wealth = float(next_wealth)
    agent.health = prev_health  # Health dynamics are held constant in phase 1.
    agent.investment = agent.wealth - prev_wealth
    agent.total_gain = agent.wealth - agent.baseline_wealth
    agent.last_rationale = rationale.strip() if isinstance(rationale, str) else None
    agent.history.append(agent.to_dict())
