import numpy as np
import pandas as pd

from data_loaders.dataset_manager import DatasetManager
from agents.agent_state import AgentState
from ai_agent_simulation.economics_model import EconomicModel, EconomicParameters


class BalboniEnv:
    """
    Environment for simulating household dynamics based on:
      - Balboni dataset (historical)
      - Simplified poverty-trap economic model (future evolution)
    """

    def __init__(self, params: EconomicParameters, data_zip="data/balboni/dataverse_files.zip"):
        self.params = params
        self.model = EconomicModel(params)

        self.dataset = DatasetManager(data_zip)
        self.panel = self.dataset.get_panel()
        self.households = self.panel["hhid5"].unique()

        self.current_hhid = None
        self.current_wave = None
        self.last_state: AgentState | None = None
        self.baseline_wealth = 0.0

    def reset(self, household_id=None):
        """Initialize simulation at wave 1 for a given household."""
        if household_id is None:
            household_id = int(np.random.choice(self.households))

        self.current_hhid = household_id
        self.current_wave = 1

        state = self._extract_state_from_data()
        if state:
            self.baseline_wealth = state.wealth or 0.0
            self.last_state = state
        return state

    def step(self, action=None):
        """
        One step:
          - If real data exists for this wave → use it
          - If beyond data → simulate using the economic model
        """
        self.current_wave += 1

        max_wave = int(self.panel["survey_wave"].max()) if not self.panel.empty else 5
        if self.current_wave <= max_wave:
            state = self._extract_state_from_data()
        else:
            state = self._simulate_future_step()

        if state:
            self.last_state = state
        return state

    def _extract_state_from_data(self):
        """Extract observed historical data for waves 1–5."""
        row = self.panel[
            (self.panel.hhid5 == self.current_hhid)
            & (self.panel.survey_wave == self.current_wave)
        ]

        if row.empty:
            return None

        r = row.iloc[0]
        wealth = float(r["wealth"]) if not pd.isna(r.get("wealth")) else 0.0
        health = float(r["health_index"]) if not pd.isna(r.get("health_index")) else 0.0
        investment = (
            float(r["investment_amount"]) if not pd.isna(r.get("investment_amount")) else 0.0
        )
        total_gain = (
            float(r["total_gain_from_baseline"]) if not pd.isna(r.get("total_gain_from_baseline")) else 0.0
        )
        return AgentState(
            wealth=wealth,
            health=health,
            investment=investment,
            total_gain=total_gain,
        )

    def _simulate_future_step(self):
        """Simulate future waves using the EconomicModel."""

        # pull last state
        prev_state = self.last_state
        if prev_state is None:
            # If somehow called without a previous state, start from wave 1 data.
            prev_state = self._extract_state_from_data()
        if prev_state is None:
            prev_state = AgentState(wealth=0.0, health=0.0, investment=0.0, total_gain=0.0)

        new_wealth = self.model.simulate_next_wealth(
            wealth_t=prev_state.wealth,
            health_t=prev_state.health,
        )

        next_state = AgentState(
            wealth=new_wealth,
            health=prev_state.health,  # health static, but can be modeled later
            investment=new_wealth - prev_state.wealth,
            total_gain=new_wealth - self.baseline_wealth,
        )

        self.last_state = next_state
        return next_state
