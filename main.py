from ai_agent_simulation.environment import Environment
from ai_agent_simulation.agent import Agent
from ai_agent_simulation.economics_model import calculate_poverty_trap_projection # NEW IMPORT for logging

def print_agent_summary(agent: Agent, step: int):
    """Prints a clear summary of the agent's current state and economic status."""
    
    # Calculate the economic reality for the summary
    # We use the NEW state (agent.wealth/health) to project the NEXT step's outcome
    econ_data = calculate_poverty_trap_projection(agent.wealth, agent.health)
    
    # Determine the current status relative to the threshold (80.0)
    current_status = "ABOVE THRESHOLD (Potential for compounding growth)" if agent.wealth >= 80.0 else "BELOW THRESHOLD (Risk of falling into the Trap)"

    print(f"--- Agent {agent.agent_id} State (End of Step {step}) ---")
    print(f"  Current Economic Position: {current_status}")
    print(f"  FINAL WEALTH: {agent.wealth:.2f} | FINAL HEALTH: {agent.health:.2f}")
    print(f"  --> NEXT STEP FORECAST: {econ_data['description']}")
    print("-" * 60)


def main():
    """
    Main function to set up and run the simulation.
    """
    # Create the environment
    env = Environment()

    # Create and add agents to the environment
    # Agent A: ABOVE threshold (100.0 > 80.0) -> Expected to GROW easily.
    env.add_agent(Agent(initial_wealth=100.0, initial_health=0.8, agent_id="Agent_A_Rich"))
    # Agent B: BELOW threshold (50.0 < 80.0) -> Expected to be TRAPPED (wealth degrades).
    env.add_agent(Agent(initial_wealth=50.0, initial_health=0.6, agent_id="Agent_B_Poor"))

    # Run the simulation for a few steps
    num_steps = 3
    for i in range(1, num_steps + 1):
        print("\n" + "=" * 60)
        print(f"=========== SIMULATION STEP {i} ===========")
        print("=" * 60)
        
        # Run the environment step (calls agent.step() for all agents, updating their state)
        env.run_step()
        
        # Log the outcomes clearly using the new function
        for agent in env.get_agents():
            print_agent_summary(agent, i)

    # Final summary
    print("\n" + "#" * 60)
    print("############ SIMULATION END #############")
    print("#" * 60)
    for agent in env.get_agents():
        print(f"Agent {agent.agent_id} Final State: Wealth={agent.wealth:.2f}, Health={agent.health:.2f}")


if __name__ == "__main__":
    main()