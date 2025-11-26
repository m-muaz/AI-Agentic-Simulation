from ai_agent_simulation.environment import Environment
from ai_agent_simulation.agent import Agent
from ai_agent_simulation.data_loader import load_and_prepare_agents # NEW IMPORT

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

    # Load data and create agents using the dedicated function
    load_and_prepare_agents(env) 
    
    if not env.get_agents():
        print("No agents were created. Simulation cannot proceed.")
        return

    # Run the simulation for a few steps
    # Note: Each step involves an API call for each agent.
    num_steps = 2
    for i in range(1, num_steps + 1):
        print("\n" + "=" * 60)
        print(f"=========== SIMULATION STEP {i} ===========")
        print("=" * 60)
        env.run_step()
        print_agent_summary(env)

    # Final summary
    print("\n" + "#" * 60)
    print("############ SIMULATION END #############")
    print("#" * 60)
    for agent in env.get_agents():
        print(f"Agent {agent.agent_id} Final State: Wealth={agent.wealth:.2f}, Health={agent.health:.2f}")


if __name__ == "__main__":
    main()