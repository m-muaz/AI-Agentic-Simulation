from env.balboni_env import BalboniEnv
from ai_agent_simulation.economics_model import EconomicParameters

def run_balboni_simulation(household_id=None, steps=15):
    params = EconomicParameters(
        asset_threshold=5,
        low_return_rate=0.04,
        high_return_rate=0.12,
        depreciation_rate=0.06,
        savings_rate=0.5,
        shock_std_dev=0.3,
        health_productivity_sensitivity=0.1
    )

    env = BalboniEnv(params)

    state = env.reset(household_id)

    states = [state]
    for _ in range(steps):
        next_state = env.step()
        states.append(next_state)

    return states

if __name__ == "__main__":
    trajectory = run_balboni_simulation()
    for s in trajectory:
        print(s)
