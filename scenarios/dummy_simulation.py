import argparse
from typing import List, Optional

from ai_agent_simulation.agent import Agent
from ai_agent_simulation.config import load_agent_configs
from ai_agent_simulation.environment import Environment


def run_episode(num_steps: int, config_path: Optional[str] = None):
    """
    Runs a single simulation episode and prints memory summaries.
    """
    env = Environment()
    for scenario in load_agent_configs(config_path):
        env.add_agent(Agent.from_config(scenario))

    for _ in range(num_steps):
        env.run_step()

    print("Episode completed. Agent snapshots:")
    for agent in env.get_agents():
        memory_state = agent.memory.export_state()
        print(f"- {agent.agent_id} wealth={agent.wealth:.2f} health={agent.health:.2f}")
        print(f"  Summary: {memory_state['summary']}")


def main(step_options: List[int], config_path: Optional[str]):
    for steps in step_options:
        print("=" * 60)
        print(f"Running dummy episode with {steps} steps")
        run_episode(steps, config_path=config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dummy simulations with varying episode lengths.")
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[2, 5, 10],
        help="Episode lengths to run sequentially.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional override path for the agent config YAML.",
    )
    args = parser.parse_args()
    main(step_options=args.steps, config_path=args.config)
