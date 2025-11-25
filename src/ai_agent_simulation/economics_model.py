import math

def calculate_poverty_trap_projection(current_wealth: float, current_health: float) -> dict:
    """
    Calculates the 'natural' economic progression based on the Banerjee-Duflo S-curve.

    If the agent is below the threshold, wealth tends to degrade or grow slowly.
    If the agent is above the threshold, wealth grows exponentially up to a cap.

    Args:
        current_wealth: The agent's current wealth ($W_t$).
        current_health: The agent's health ($H_t$), acting as a productivity multiplier (0.0 to 1.0).

    Returns:
        A dictionary containing the projected wealth and a narrative description for the LLM.
    """

    # --- Tunable Model Parameters (Set here after visualization) ---
    survival_cost = 10.0      # Cost to survive one time step
    inflection_point = 80.0   # The 'Trap' Threshold (The inflection point of the S-curve)
    growth_potential = 20.0   # Maximum productivity output per step
    steepness = 0.2           # How sharp the transition is from low to high returns

    # 1. Calculate Productivity (P) using a Logistic Function (S-Curve)
    productivity = growth_potential / (1 + math.exp(-steepness * (current_wealth - inflection_point)))

    # 2. Adjust for Health (Productivity is constrained by the agent's capacity/health)
    effective_income = productivity * current_health

    # 3. Calculate Projected Next Wealth ($W_{t+1}$)
    projected_wealth = current_wealth + effective_income - survival_cost

    # 4. Generate Narrative for the LLM
    gap = projected_wealth - current_wealth
    
    if projected_wealth < current_wealth:
        status = "CRITICAL: DEGRADING"
        advice = "Your costs exceed your income. You are in the poverty trap."
    elif gap < 2.0:
        status = "STAGNANT"
        advice = "You are treading water. You need to invest or save strategically."
    else:
        status = "GROWING"
        advice = "You have crossed the threshold. Your capital is working for you."

    description = (
        f"ECONOMIC FORECAST: {status}. "
        f"The model predicts your wealth will move to {projected_wealth:.2f} (Change: {gap:+.2f}). "
        f"Income: {effective_income:.2f} | Survival Cost: {survival_cost:.2f}. "
        f"Recommendation: {advice}"
    )

    return {
        "projected_wealth": projected_wealth,
        "income": effective_income,
        "cost": survival_cost,
        "description": description
    }