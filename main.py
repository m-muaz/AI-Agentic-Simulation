from ai_agent_simulation.environment import Environment
from ai_agent_simulation.agent import Agent


def main():
    """
    Main function to set up and run the simulation.
    """
    # Create the environment
    env = Environment()

    # Create and add agents to the environment
    # Using fixed agent IDs for more predictable history between runs
    env.add_agent(Agent(initial_wealth=100.0, initial_health=0.8, agent_id="agent_1"))
    env.add_agent(Agent(initial_wealth=50.0, initial_health=0.6, agent_id="agent_2"))

    # Run the simulation for a few steps
    # Note: Each step involves an API call for each agent.
    num_steps = 2
    for i in range(num_steps):
        env.run_step()


if __name__ == "__main__":
    main()
