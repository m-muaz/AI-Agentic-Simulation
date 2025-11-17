from ai_agent_simulation.environment import Environment, Policy
from ai_agent_simulation.agent import Agent


def main():
    """
    Main function to set up and run the simulation.
    """
    # 1. Create the environment with macroeconomic settings and a policy.
    # You can change the 'initial_policy' to test different modes.
    # e.g., Policy.BASELINE, Policy.CASH_TRANSFER, Policy.MICROLOAN_ACCESS
    env = Environment(
        wage_rate=1.0,
        loan_interest=0.05,
        price_level=1.0,
        initial_policy=Policy.BASELINE
    )

    # 2. Create and add agents to the environment
    env.add_agent(Agent(
        agent_id="agent_1",
        initial_wealth=1000.0,
        initial_health=0.8,
        income=500.0,
        loan_access=False,
        education=0.5,
        consumption_preference=0.7
    ))
    env.add_agent(Agent(
        agent_id="agent_2",
        initial_wealth=400.0,
        initial_health=0.6,
        income=250.0,
        loan_access=False,
        education=0.2,
        consumption_preference=0.9
    ))

    # 3. Run the simulation for a few steps
    num_steps = 3
    for i in range(num_steps):
        env.run_step()


if __name__ == "__main__":
    main()
