import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from .agent import Agent
from .economics_model import get_poverty_trap_threshold

def plot_wealth_distribution(agents: List[Agent], empirical_image_name: str):
    """
    Plots the final wealth distribution (KDE) of the agents and provides context 
    for the empirical target.
    
    Args:
        agents: The list of Agent objects after the simulation.
        empirical_image_name: The name of the empirical data image for context.
    """
    if not agents:
        print("No agents to visualize.")
        return

    # 1. Extract final wealth data
    final_wealths = [agent.wealth for agent in agents]
    df = pd.DataFrame({'Final Wealth (k)': final_wealths})
    
    # 2. Plot the Kernel Density Estimate (KDE)
    plt.figure(figsize=(10, 6))
    # Use Kernel Density Estimation (KDE) to smooth the distribution and check for bimodality
    sns.kdeplot(df['Final Wealth (k)'], fill=True, label='Simulated Final Wealth Distribution')
    
    # 3. Add Context (Poverty Trap Threshold)
    threshold = get_poverty_trap_threshold()
    plt.axvline(threshold, color='r', linestyle='--', label=f'Poverty Trap Threshold ($\\hat{{k}}$ = {threshold:.2f})')
    
    plt.title('Simulated Agent Wealth Distribution (KDE)')
    plt.xlabel('Productive Assets (Wealth, k)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Display the plot
    plt.show()
    print("--- Visualization Complete (Distribution) ---")
    print("The simulation is aiming to reproduce the bimodal distribution seen in the empirical data.")
    print(f"Empirical Target (Bimodality):")
    # Reference the key empirical image for comparison
    print(f"") 
    
    # Plot the agent trajectories as a secondary visualization
    plot_agent_trajectories(agents)


def plot_agent_trajectories(agents: List[Agent]):
    """
    Plots the wealth trajectory of each agent over the simulation steps.
    """
    plt.figure(figsize=(10, 6))
    
    for agent in agents:
        # Extract wealth history
        wealth_history = [state['wealth'] for state in agent.history]
        steps = range(len(wealth_history))
        plt.plot(steps, wealth_history, marker='.', alpha=0.6) # Plotting without label for cleaner look

    threshold = get_poverty_trap_threshold()
    plt.axhline(threshold, color='r', linestyle='--', label=f'Poverty Trap Threshold ($\\hat{{k}}$)')
    
    plt.title('Agent Wealth Trajectories Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Productive Assets (Wealth, k)')
    # Note: Removed individual legend for clarity when running many agents
    plt.legend(loc='upper left')
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    print("--- Visualization Complete (Trajectories) ---")


    def plot_wealth_trajectory(states):

        wealth = [s.wealth for s in states]

        plt.figure(figsize=(7,4))
        plt.plot(wealth, marker='o')
        plt.xlabel("Time")
        plt.ylabel("Wealth")
        plt.title("Wealth trajectory (Historical + Simulated)")
        plt.show()

