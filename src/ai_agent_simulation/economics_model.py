import numpy as np

# --- EMPIRICALLY INFORMED S-CURVE PARAMETERS ---
# Based on the uploaded analysis tables, we set parameters to reflect a clear poverty trap:
THRESHOLD = 80.0              # The critical wealth level (inflection point, k-hat)
STEEPNESS = 0.15              # Controls the sharpness of the curve (how fast returns change)
GROWTH_POTENTIAL = 0.25       # Maximum wealth growth rate (upper asymptote)
DECLINE_POTENTIAL = -0.05     # Maximum wealth decline rate (lower asymptote)


def calculate_poverty_trap_projection(current_wealth: float, current_health: float) -> float:
    """
    Calculates the change in wealth (dk/dt) based on the Banerjee-Duflo S-curve model.

    Args:
        current_wealth: The agent's current wealth (k).
        current_health: The agent's current health (human capital, h) (0.0 to 1.0).

    Returns:
        The projected change in wealth for the next step (dk).
    """
    
    # 1. Calculate the base growth rate from the S-curve
    growth_rate = DECLINE_POTENTIAL + \
                  (GROWTH_POTENTIAL - DECLINE_POTENTIAL) / \
                  (1 + np.exp(-STEEPNESS * (current_wealth - THRESHOLD)))

    # 2. Scale the growth rate by the agent's Health (human capital)
    # We use a squared health factor to reflect the high dependence of returns on health/capacity.
    effective_growth = growth_rate * (current_health ** 2)

    # 3. Project the change in wealth
    change_in_wealth = current_wealth * effective_growth
    
    return change_in_wealth

def get_poverty_trap_threshold() -> float:
    """Returns the central threshold used in the model for reference."""
    return THRESHOLD