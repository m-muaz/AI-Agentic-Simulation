"""
    This file contains the "Physics" of the simulation. It enforces the S-shape curve where:
    Below the inflection point ($80): Money generates slowly (manual labor), often not covering the cost of living.
    Above the inflection point ($80): Money generates fast (capital investment), easily covering costs and allowing growth.
"""

import math

def calculate_poverty_trap_projection(current_wealth: float, current_health: float) -> dict:
    """
    Calculates the 'natural' economic progression based on the Banerjee-Duflo S-curve.

    Args:
        current_wealth: The agent's current wealth ($W_t$).
        current_health: The agent's health ($H_t$), acting as a productivity multiplier (0.0 to 1.0).

    Returns:
        A dictionary containing the projected wealth and a narrative description for the LLM.
    """

    # --- Model Parameters ---
    # Cost to survive one time step (food, rent)
    survival_cost = 10.0
    # The 'Trap' Threshold: Below this, wealth likely decreases. Above this, it compounds.
    inflection_point = 80.0 
    # Maximum productivity output per step (before health multiplier)
    growth_potential = 20.0 
    # How sharp the transition is from 'poor' to 'rich' returns
    steepness = 0.2

    # 1. Calculate Productivity (P) using a Logistic Function (S-Curve)
    # This models that capital yields low returns when small, and high returns when large.
    productivity = growth_potential / (1 + math.exp(-steepness * (current_wealth - inflection_point)))

    # 2. Adjust for Health
    # Sick agents cannot utilize their capital/labor efficiently.
    effective_income = productivity * current_health

    # 3. Calculate Projected Next Wealth ($W_{t+1}$)
    # Formula: W_{t+1} = W_t + Income - Consumption
    projected_wealth = current_wealth + effective_income - survival_cost

    # 4. Generate Narrative for the LLM
    gap = projected_wealth - current_wealth
    
    if projected_wealth < current_wealth:
        status = "CRITICAL: DEGRADING"
        advice = "You are in the poverty trap. Your costs exceed your income."
    elif gap < 2.0:
        status = "STAGNANT"
        advice = "You are treading water. You need more capital or better health to grow."
    else:
        status = "GROWING"
        advice = "You have crossed the threshold. Your capital is working for you."

    description = (
        f"ECONOMIC FORECAST: {status}. "
        f"Based on current wealth ({current_wealth:.2f}) and health ({current_health:.2f}), "
        f"the economic model predicts your wealth will move to {projected_wealth:.2f} (Change: {gap:+.2f}). "
        f"{advice}"
    )

    return {
        "projected_wealth": projected_wealth,
        "income": effective_income,
        "cost": survival_cost,
        "description": description
    }