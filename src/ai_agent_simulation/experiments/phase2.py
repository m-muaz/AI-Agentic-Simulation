"""
Experiment Phase 2 helpers:
- LLM picks discrete coping actions (sell assets / borrow / insurance).
- Optional health effort input.
"""

from __future__ import annotations

from typing import Any, Dict


def build_prompt(agent, environment) -> str:
    """
    Build the phase 2 prompt emphasizing discrete coping actions.
    """
    household_background = " ".join(agent.household_history.split())
    wealth_change = ", ".join(str(snap["wealth"]) for snap in agent.history)
    health_change = ", ".join(str(snap["health"]) for snap in agent.history)
    last_state = agent.history[-1] if agent.history else agent.to_dict()
    step_index = getattr(environment, "time_step", len(agent.history) - 1)
    econ_model = getattr(environment, "economic_model", None) if environment else None
    params = getattr(econ_model, "params", None) if econ_model else None
    threshold = getattr(params, "asset_threshold", "not specified") if params else "not specified"

    prompt = f"""
    You are an agent in a simulation of a low-income household.
    You are currently roleplaying: {agent.name} ({agent.agent_id})
    Household background: {household_background or "Not specified."}
    Wealth trajectory over time: {wealth_change}
    Health trajectory over time: {health_change}
    This is your current state:
    {last_state}
    Economic environment notes:
      - Threshold regime at asset level: {threshold}.
      - Returns follow a low/high regime with depreciation; shocks are applied each step.
    Current simulation step: {step_index}
    Experiment Phase 2: choose a discrete coping action and optional health effort.
    Your output MUST be valid JSON with:
      - "coping_action": "sell_assets" | "borrow" | "insurance"
      - "coping_intensity": 0.0-1.0 (fractional intensity of that action)
      - Optional: "health_effort": 0.0-1.0 (nudge health up/down; >0.5 improves health slightly)
      - Optional: "savings_rate_t": 0.0-1.0 (saved share of returns/income)
      - Optional: "risk_profile_t": "low" | "medium" | "high" (risk appetite for shocks)
      - Optional: "rationale": 1-3 sentences explaining the coping choice and expected impact.
    Do NOT include wealth or investment updates; the model computes those.
    Example: {{"coping_action": "borrow", "coping_intensity": 0.3, "health_effort": 0.6, "rationale": "..."}}
    """
    return prompt


def apply_decision(agent, llm_response: Dict[str, Any], econ_model):
    """
    Apply phase 2 coping logic before invoking the economic model.
    """
    prev_wealth = float(agent.wealth)
    prev_health = float(agent.health)
    action = (llm_response.get("coping_action") or "").lower()
    intensity = llm_response.get("coping_intensity", 0.0)
    savings_rate = llm_response.get("savings_rate_t", llm_response.get("savings_rate"))
    health_effort = llm_response.get("health_effort", 0.5)
    risk_profile = llm_response.get("risk_profile_t", llm_response.get("risk_profile"))
    rationale = llm_response.get("rationale")

    # Clamp inputs
    try:
        intensity = max(0.0, min(1.0, float(intensity)))
    except (TypeError, ValueError):
        intensity = 0.0
    try:
        savings_rate = None if savings_rate is None else max(0.0, min(1.0, float(savings_rate)))
    except (TypeError, ValueError):
        savings_rate = None
    try:
        health_effort = max(0.0, min(1.0, float(health_effort)))
    except (TypeError, ValueError):
        health_effort = 0.5

    # Map risk profile to shock scale
    profile_map = {"low": 0.7, "medium": 1.0, "high": 1.4}
    shock_scale = profile_map.get(risk_profile.lower(), 1.0) if isinstance(risk_profile, str) else 1.0

    working_wealth = prev_wealth
    debt_service = 0.0

    if action == "sell_assets":
        # Distress sale reduces productive capital; apply a discount on sold portion.
        sale_loss = intensity * prev_wealth * 0.5
        working_wealth = max(0.0, prev_wealth - sale_loss)
    elif action == "borrow":
        # Borrowing adds capital now but adds a servicing cost.
        borrowed = intensity * prev_wealth
        debt_service = 0.05 * borrowed
        working_wealth = max(0.0, prev_wealth + borrowed)
        shock_scale = max(shock_scale, 1.1)  # leverage raises exposure to shocks
    elif action == "insurance":
        # Insurance reduces shock exposure but costs a premium.
        premium = 0.02 * max(0.0, prev_wealth) * max(intensity, 0.1)
        working_wealth = max(0.0, prev_wealth - premium)
        shock_scale = min(shock_scale, 0.6)

    # Health adjustment: small nudge based on effort.
    health_delta = 0.1 * (health_effort - 0.5)
    new_health = max(0.0, min(1.0, prev_health + health_delta))

    next_wealth = econ_model.simulate_next_wealth(
        wealth_t=working_wealth,
        health_t=new_health,
        savings_rate_override=savings_rate,
        shock_scale=shock_scale,
    )

    # Deduct any debt servicing after returns.
    next_wealth = max(0.0, next_wealth - debt_service)

    agent.wealth = float(next_wealth)
    agent.health = new_health
    agent.investment = agent.wealth - prev_wealth
    agent.total_gain = agent.wealth - agent.baseline_wealth
    agent.last_rationale = rationale.strip() if isinstance(rationale, str) else None
    agent.history.append(agent.to_dict())
