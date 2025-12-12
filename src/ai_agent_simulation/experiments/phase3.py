"""
Experiment Phase 3 helpers:
- Assumes access to PovertyTraps_structural.dta (if present).
- LLM chooses labor allocation / sector choice.
- Model computes income flows and savings before applying the economic model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _load_structural_data(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    data_path = Path(path)
    if data_path.exists():
        try:
            return pd.read_stata(data_path)
        except Exception:
            return None
    return None


def build_prompt(agent, environment, data_path: Optional[str] = None) -> str:
    """
    Build the phase 3 prompt emphasizing labor allocation and sector choice.
    """
    # Pick data path: env override > arg > default.
    data_path = data_path or getattr(environment, "structural_data_path", None) or "data/PovertyTraps_structural.dta"
    household_background = " ".join(agent.household_history.split())
    wealth_change = ", ".join(str(snap["wealth"]) for snap in agent.history)
    health_change = ", ".join(str(snap["health"]) for snap in agent.history)
    last_state = agent.history[-1] if agent.history else agent.to_dict()
    step_index = getattr(environment, "time_step", len(agent.history) - 1)
    econ_model = getattr(environment, "economic_model", None) if environment else None
    params = getattr(econ_model, "params", None) if econ_model else None
    threshold = getattr(params, "asset_threshold", "not specified") if params else "not specified"

    has_data = _load_structural_data(data_path) is not None
    data_note = (
        f"Structural data available at {data_path}; sector choices can leverage observed patterns."
        if has_data
        else f"Structural data not found at {data_path}; use reasonable assumptions."
    )

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
      - Returns follow a low/high regime with depreciation and shocks.
    Current simulation step: {step_index}
    Data note: {data_note}
    Experiment Phase 3: choose labor allocation and sector; the model computes income flows and savings.
    Your output MUST be valid JSON with:
      - "labor_allocation": an object with shares that sum to 1.0, keys may include "farm", "wage_labor", "self_employment".
      - "sector_choice": "agriculture" | "manufacturing" | "services" | "construction" | "trade"
      - Optional: "savings_rate_t": 0.0-1.0 (share of income to save)
      - Optional: "health_effort": 0.0-1.0 (small nudge to health; >0.5 improves health slightly)
      - Optional: "rationale": 1-3 sentences explaining your allocation and sector choice.
    Do NOT include wealth or investment updates; the model computes those.
    Example: {{"labor_allocation": {{"farm": 0.5, "wage_labor": 0.3, "self_employment": 0.2}}, "sector_choice": "agriculture", "savings_rate_t": 0.4, "rationale": "..." }}
    """
    return prompt


def _normalize_allocation(raw_alloc: Dict[str, Any]) -> Dict[str, float]:
    cleaned = {}
    total = 0.0
    for key, val in (raw_alloc or {}).items():
        try:
            v = max(0.0, float(val))
        except (TypeError, ValueError):
            continue
        cleaned[key] = v
        total += v
    if total == 0.0:
        return {"farm": 0.5, "wage_labor": 0.25, "self_employment": 0.25}
    return {k: v / total for k, v in cleaned.items()}


def apply_decision(
    agent,
    llm_response: Dict[str, Any],
    econ_model,
    data_path: Optional[str] = None,
):
    """
    Apply phase 3 logic: compute income from labor/sector, then feed through economic model.
    """
    data_path = data_path or getattr(agent, "structural_data_path", None) or getattr(
        econ_model, "structural_data_path", None
    ) or "data/PovertyTraps_structural.dta"
    prev_wealth = float(agent.wealth)
    prev_health = float(agent.health)
    allocation = _normalize_allocation(llm_response.get("labor_allocation") or {})
    sector = (llm_response.get("sector_choice") or "agriculture").lower()
    savings_rate = llm_response.get("savings_rate_t", llm_response.get("savings_rate", 0.5))
    health_effort = llm_response.get("health_effort", 0.5)
    rationale = llm_response.get("rationale")

    try:
        savings_rate = max(0.0, min(1.0, float(savings_rate)))
    except (TypeError, ValueError):
        savings_rate = 0.5
    try:
        health_effort = max(0.0, min(1.0, float(health_effort)))
    except (TypeError, ValueError):
        health_effort = 0.5

    sector_multipliers = {
        "agriculture": 1.0,
        "manufacturing": 1.2,
        "services": 1.1,
        "construction": 1.15,
        "trade": 1.05,
    }
    income_baseline = {"farm": 8.0, "wage_labor": 10.0, "self_employment": 9.0}
    sector_mult = sector_multipliers.get(sector, 1.0)

    health_income_bump = 1.0 + 0.1 * prev_health
    income_flow = 0.0
    for k, share in allocation.items():
        base = income_baseline.get(k, 8.0)
        income_flow += share * base * sector_mult * health_income_bump

    # Optional: attempt to adjust income using structural data (placeholder hook).
    structural = _load_structural_data(data_path)
    if structural is not None and not structural.empty:
        # Placeholder: scale income modestly if data exists to signal availability.
        income_flow *= 1.05

    # Savings from income are added to wealth before returns.
    savings_from_income = income_flow * savings_rate
    working_wealth = prev_wealth + savings_from_income

    health_delta = 0.1 * (health_effort - 0.5)
    new_health = max(0.0, min(1.0, prev_health + health_delta))

    next_wealth = econ_model.simulate_next_wealth(
        wealth_t=working_wealth,
        health_t=new_health,
        savings_rate_override=savings_rate,
        shock_scale=1.0,
    )

    agent.wealth = float(max(0.0, next_wealth))
    agent.health = new_health
    agent.investment = agent.wealth - prev_wealth
    agent.total_gain = agent.wealth - agent.baseline_wealth
    agent.last_rationale = rationale.strip() if isinstance(rationale, str) else None
    agent.history.append(agent.to_dict())
