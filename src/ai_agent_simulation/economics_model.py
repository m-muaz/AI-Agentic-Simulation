import numpy as np

# --- EMPIRICALLY INFORMED S-CURVE PARAMETERS ---
# Based on the uploaded analysis tables, we set parameters to reflect a clear poverty trap:
# THRESHOLD (k-hat): The unstable equilibrium point where returns accelerate.
# STEEPNESS: Controls the sharpness of the transition.
# GROWTH_POTENTIAL/DECLINE_POTENTIAL: Define the upper and lower bounds of growth rate.
THRESHOLD = 80.0              
STEEPNESS = 0.15              
GROWTH_POTENTIAL = 0.25       
DECLINE_POTENTIAL = -0.05     


def calculate_poverty_trap_projection(current_wealth: float, current_health: float) -> float:
    """
    Calculates the change in wealth (dk/dt) based on the Banerjee-Duflo S-curve model.

    The formula is: dk/dt = k * [Decay + (Growth - Decay) / (1 + exp(-Steepness * (k - THRESHOLD)))] * h^2

    Args:
        current_wealth: The agent's current wealth (k).
        current_health: The agent's current health (human capital, h) (0.0 to 1.0).

    Returns:
        The projected change in wealth for the next step (dk).
    """
    
    # 1. Calculate the base growth rate from the S-curve (the non-linear function)
    growth_rate = DECLINE_POTENTIAL + \
                  (GROWTH_POTENTIAL - DECLINE_POTENTIAL) / \
                  (1 + np.exp(-STEEPNESS * (current_wealth - THRESHOLD)))

    # 2. Scale the growth rate by the agent's Health (human capital)
    # The h^2 term reflects the empirical finding that human capital heavily conditions returns.
    effective_growth = growth_rate * (current_health ** 2)

    # 3. Project the change in wealth (dk = k * growth_rate)
    change_in_wealth = current_wealth * effective_growth
    
    return change_in_wealth

def get_poverty_trap_threshold() -> float:
    """Returns the central threshold used in the model for reference."""
    return THRESHOLD