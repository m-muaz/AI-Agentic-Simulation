from environment import Environment
from agent import Agent


def main():
    """
    Main function to set up and run the simulation.
    """
    # Create the environment
    env = Environment()

    # Create and add agents to the environment
    # These represent low-income households with different starting conditions.
    env.add_agent(Agent(initial_wealth=100.0, initial_health=0.8))
    env.add_agent(Agent(initial_wealth=50.0, initial_health=0.6))
    env.add_agent(Agent(initial_wealth=120.0, initial_health=0.9))

    # Run the simulation for a few steps
    num_steps = 5
    for i in range(num_steps):
        env.run_step()


if __name__ == "__main__":
    main()