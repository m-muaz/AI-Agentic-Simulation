# src/envs/banerjee_duflo_env.py
import numpy as np
from src.data_loaders.dataset_manager import DatasetManager


class BanerjeeDufloEnv:
    """
    Simple environment wrapper exposing household states from the Balboni panel.
    Not an OpenAI Gym environment—just a convenience wrapper to get state dicts.
    """

    def __init__(self, data_zip="data/balboni/dataverse_files.zip"):
        self.dm = DatasetManager(data_zip)
        self.panel = self.dm.get_panel()
        self.households = self.panel["hhid5"].unique()
        self.current_hhid = None
        self.current_wave = None

    def reset(self, household_id=None, wave=1):
        if household_id is None:
            household_id = int(np.random.choice(self.households))
        self.current_hhid = int(household_id)
        self.current_wave = int(wave)
        return self._get_state()

    def step(self, action=None):
        # action is not applied to dataset; dataset is observational.
        self.current_wave += 1
        return self._get_state()

    def _get_state(self):
        df = self.panel
        row = df[(df["hhid5"] == self.current_hhid) & (df["survey_wave"] == self.current_wave)]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "hhid": int(self.current_hhid),
            "wave": int(self.current_wave),
            "wealth": float(r["wealth"]) if not pd.isna(r["wealth"]) else None,
            "investment": float(r["investment_amount"]) if not pd.isna(r["investment_amount"]) else 0.0,
            "health": float(r["health_index"]) if not pd.isna(r["health_index"]) else None,
            "total_gain": float(r["total_gain_from_baseline"]) if not pd.isna(r["total_gain_from_baseline"]) else None,
        }
