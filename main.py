from ai_agent_simulation.environment import Environment
from ai_agent_simulation.agent import Agent
from ai_agent_simulation.data_loader import load_and_prepare_agents
from .visualizer import plot_wealth_distribution

def print_agent_summary(env: Environment):
    """Prints a clear summary of the agent's current state and economic status."""
    
    print("\n====================================")
    print("SIMULATION STEP SUMMARY")
    print("====================================")
    for agent in env.get_agents():
        # The poverty trap threshold is set at 80.0 in the economics_model
        status = "Poverty Trap (Critical)" if agent.wealth < 80.0 else "Stable/Growing"
        
        print(f"Agent ID: {agent.agent_id}")
        print(f"  Status: {status}")
        print(f"  Wealth: {agent.wealth:.2f}")
        print(f"  Health: {agent.health:.2f}")
        print("-" * 30)

def main():
    """
    Main function to set up and run the simulation.
    """
    # Create the environment
    env = Environment()

    # --- Step 2: Initialize agents using empirical data distribution ---
    # We use 50 agents and 10 steps to allow the S-curve to influence the distribution
    N_AGENTS = 50  
    initial_states = load_initial_agent_states(N_AGENTS)

    # Create and add agents to the environment
    for i, (wealth, health) in enumerate(initial_states):
        env.add_agent(Agent(initial_wealth=wealth, initial_health=health, agent_id=f"agent_{i+1}"))

    # Run the simulation for a few steps
    num_steps = 10 
    print(f"\nStarting simulation for {N_AGENTS} agents over {num_steps} steps...")
    for i in range(num_steps):
        env.run_step()

    # --- Step 3: Visualization ---
    print("\nSimulation complete. Proceeding to visualization.")
    # We target the key empirical density plot (Bangladesh 2011) for comparison
    plot_wealth_distribution(env.get_agents(), "idhs_kdensity_pca_wealth_index_Bangladesh_2011_agassets.tif")


if __name__ == "__main__":
    main()