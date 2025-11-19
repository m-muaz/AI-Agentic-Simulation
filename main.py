import os
import pandas as pd
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
        initial_policy=Policy.BASELINE,
        microloan_access_percentage=0.5, # 50% of agents will get microloan access under that policy
        cash_transfer_amount=20.0 # Amount of cash transfer under that policy
    )

    # 2. Load processed agent data and create agents
    processed_data_path = os.path.join(os.getcwd(), "data", "processed", "processed_agent_data.csv")
    if not os.path.exists(processed_data_path):
        print(f"Error: Processed data file not found at {processed_data_path}")
        print("Please run data_explorer.py first to generate the processed data.")
        return

    try:
        df_agents = pd.read_csv(processed_data_path)
        print(f"\nLoaded {len(df_agents)} agents from processed data.")

        # Limit the number of agents for testing purposes to avoid excessive LLM calls
        num_agents_to_simulate = 5
        for index, row in df_agents.head(num_agents_to_simulate).iterrows():
            agent = Agent(
                agent_id=row['agent_id'],
                initial_wealth=row['initial_wealth'],
                initial_health=row['initial_health'],
                income=row['income'],
                loan_access=row['loan_access'],
                education=row['education'],
                consumption_preference=row['consumption_preference']
            )
            env.add_agent(agent)
        print(f"Initialized {len(env.get_agents())} agents in the environment.")

    except Exception as e:
        print(f"An error occurred while loading or creating agents from data: {e}")
        return

    # 3. Run the simulation for a few steps
    num_steps = 3
    for i in range(num_steps):
        env.run_step()


if __name__ == "__main__":
    main()
