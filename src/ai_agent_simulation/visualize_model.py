# ONLY FOR VISUALIZATION PURPOSES - NOT PART OF THE SIMULATION

import matplotlib.pyplot as plt
import numpy as np
import math

def calculate_next_wealth(current_wealth, health=1.0):
    """
    Replicating the logic from the proposed economics_model.py
    for visualization purposes.
    """
    # --- Tunable Parameters ---
    survival_cost = 10.0      
    inflection_point = 60.0    # The "hump" required to escape
    growth_potential = 150.0   
    steepness = 0.1            
    # --------------------------

    # 1. S-Curve Productivity Calculation
    # Logistic function: output increases slowly, then rapidly, then plateaus
    productivity = growth_potential / (1 + math.exp(-steepness * (current_wealth - inflection_point)))
    
    # 2. Effective Income (constrained by health)
    effective_income = productivity * health
    
    # 3. Next Wealth Calculation
    next_wealth = current_wealth + effective_income - survival_cost
    
    return next_wealth

def plot_poverty_trap():
    # Generate wealth range from 0 to 150
    current_wealth_values = np.linspace(0, 150, 300)
    
    # Calculate next wealth for full health (1.0) and poor health (0.5)
    next_wealth_healthy = [calculate_next_wealth(w, 1.0) for w in current_wealth_values]
    next_wealth_sick = [calculate_next_wealth(w, 0.5) for w in current_wealth_values]

    # Create the Plot
    plt.figure(figsize=(10, 6))
    
    # 1. Plot the 45-degree line (Equilibrium Line)
    # If the curve is ON this line, wealth is stable.
    plt.plot(current_wealth_values, current_wealth_values, 
             linestyle='--', color='gray', label='45-Degree Line (Maintenance)')

    # 2. Plot the Economic Reality Curves
    plt.plot(current_wealth_values, next_wealth_healthy, 
             color='blue', linewidth=2, label='Healthy Agent (Health=1.0)')
    
    plt.plot(current_wealth_values, next_wealth_sick, 
             color='red', linewidth=2, label='Sick Agent (Health=0.5)')

    # Formatting and Annotation
    plt.title('Banerjee & Duflo "S-Curve" Poverty Trap Model', fontsize=14)
    plt.xlabel('Current Wealth ($W_t$)', fontsize=12)
    plt.ylabel('Next Step Wealth ($W_{t+1}$)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Shade the "Trap" area for the Healthy Agent
    # Find where the blue line crosses the gray line
    # This is a rough estimation for visualization
    idx = np.argwhere(np.diff(np.sign(np.array(next_wealth_healthy) - current_wealth_values))).flatten()
    if len(idx) > 0:
        threshold = current_wealth_values[idx[0]]
        plt.axvline(x=threshold, color='green', linestyle=':', label='Poverty Threshold')
        plt.text(threshold + 2, 10, f'Threshold ~ {threshold:.1f}', color='green')
        
        # Shade the trap
        plt.fill_between(current_wealth_values, 0, current_wealth_values, 
                         where=(current_wealth_values < threshold), 
                         color='red', alpha=0.1, label='The Poverty Trap')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_poverty_trap()