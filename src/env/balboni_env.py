import numpy as np
from ai_agent_simulation.data.balboni_data import BalboniDataset
from ai_agent_simulation.agents.agent_state import AgentState
from ai_agent_simulation.economics_model import EconomicModel, EconomicParameters


class BalboniEnv:
    """
    Environment for simulating household dynamics based on:
      - Balboni dataset (historical)
      - Simplified poverty-trap economic model (future evolution)
    """

    def __init__(self, params: EconomicParameters):
        self.params = params
        self.model = EconomicModel(params)

        # Load dataset
        self.dataset = BalboniDataset()
        self.panel = self.dataset.panel
        self.households = self.panel['hhid5'].unique()

        self.current_hhid = None
        self.current_wave = None

    def reset(self, household_id=None):
        """Initialize simulation at wave 1 for a given household."""
        if household_id is None:
            household_id = int(np.random.choice(self.households))

        self.current_hhid = household_id
        self.current_wave = 1

        return self._extract_state_from_data()

    def step(self, action=None):
        """
        One step:
          - If real data exists for this wave → use it
          - If beyond data → simulate using the economic model
        """
        self.current_wave += 1

        if self.current_wave <= 5:
            return self._extract_state_from_data()
        else:
            return self._simulate_future_step()

    def _extract_state_from_data(self):
        """Extract observed historical data for waves 1–5."""
        row = self.panel[
            (self.panel.hhid5 == self.current_hhid) &
            (self.panel.survey_wave == self.current_wave)
        ]

        if row.empty:
            # safety fallback
            return self._simulate_future_step()

        r = row.iloc[0]
        return AgentState(
            wealth=r['wealth'],
            health=r['health_index'],
            investment=r['investment_amount'],
            total_gain=r['total_gain_from_baseline']
        )

    def _simulate_future_step(self):
        """Simulate future waves using the EconomicModel."""

        # pull last state
        prev_state = self._extract_state_from_data() if self.current_wave <= 5 else self.last_state

        new_wealth = self.model.simulate_next_wealth(
            wealth_t=prev_state.wealth,
            health_t=prev_state.health
        )

        next_state = AgentState(
            wealth=new_wealth,
            health=prev_state.health,         # health static, but can be modeled later
            investment=new_wealth - prev_state.wealth,
            total_gain=new_wealth - 0         # baseline Qk0 could be added here
        )

        self.last_state = next_state
        return next_state
