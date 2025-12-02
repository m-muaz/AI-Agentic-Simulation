# src/ai_agent_simulation/economics_model.py
import numpy as np

class EconomicParameters:
    """
    Container for exogenous structural parameters
    used in the simplified Balboni-style poverty trap model.
    """

    def __init__(
        self,
        asset_threshold=5.0,
        low_return_rate=0.04,
        high_return_rate=0.12,
        depreciation_rate=0.05,
        savings_rate=0.60,
        shock_std_dev=0.5,
        health_productivity_sensitivity=0.10,
    ):
        self.asset_threshold = asset_threshold
        self.low_return_rate = low_return_rate
        self.high_return_rate = high_return_rate
        self.depreciation_rate = depreciation_rate
        self.savings_rate = savings_rate
        self.shock_std_dev = shock_std_dev
        self.health_productivity_sensitivity = health_productivity_sensitivity


class EconomicModel:
    """
    Simplified economic model of household asset dynamics
    following a threshold-based nonlinear return function,
    with health-modulated productivity and shocks.
    """

    def __init__(self, params: EconomicParameters):
        self.params = params

    def compute_return_rate(self, wealth, health):
        """
        Computes the return rate given the asset level and health.
        """
        # threshold logic
        base_rate = (
            self.params.high_return_rate
            if wealth >= self.params.asset_threshold
            else self.params.low_return_rate
        )

        # health modification
        health_factor = 1.0 + self.params.health_productivity_sensitivity * health
        health_factor = max(0.0, health_factor)

        return base_rate * health_factor

    def simulate_next_wealth(self, wealth_t, health_t):
        """
        Produces next-period wealth using the nonlinear transition law.

        k_{t+1} = (1 - delta)*k_t + s * r_t(k_t,h_t) * k_t + shock_t
        """
        p = self.params

        # return rate
        r_t = self.compute_return_rate(wealth_t, health_t)

        # deterministic capital evolution
        k_det = (1 - p.depreciation_rate) * wealth_t
        k_det += p.savings_rate * r_t * wealth_t

        # add stochastic component
        shock = np.random.normal(0, p.shock_std_dev)

        return max(0.0, k_det + shock)

    def simulate_path(self, initial_wealth, health_sequence, T=10):
        """
        Simulates a time path of wealth over T steps.
        health_sequence: list of health_t values of length T.
        """
        path = [initial_wealth]
        current = initial_wealth

        for t in range(T):
            next_k = self.simulate_next_wealth(
                wealth_t=current,
                health_t=health_sequence[t]
            )
            path.append(next_k)
            current = next_k

        return path
