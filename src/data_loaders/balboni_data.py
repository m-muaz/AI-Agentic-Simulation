# src/data_loaders/balboni_data.py
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np


DATA_ZIP_DEFAULT = Path("data/balboni/dataverse_files.zip")


def _load_analysis_dta(zip_path: Path) -> pd.DataFrame:
    if not zip_path.exists():
        raise FileNotFoundError(f"dataverse zip not found at {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        inner = "replication files/Data/PovertyTraps_analysis.dta"
        if inner not in z.namelist():
            raise FileNotFoundError(f"{inner} missing inside {zip_path}")
        with z.open(inner) as f:
            df = pd.read_stata(f, convert_categoricals=False)
    return df


def build_balboni_panel(zip_path: Path | str = None) -> pd.DataFrame:
    """
    Produce panel with the fields:
      - hhid5
      - survey_wave
      - wealth             (Qk_t)
      - investment_amount  (delta QpAssets)
      - health_index       (-(sum physical limitations + anxiety))
      - total_gain_full    (Qk5 - Qk0)
      - total_gain_from_baseline (wealth - Qk0)
    """
    zip_path = Path(zip_path) if zip_path is not None else DATA_ZIP_DEFAULT
    df = _load_analysis_dta(zip_path)

    activity_cols = [
        "H_activity_walk",
        "H_activity_carry",
        "H_activity_water",
        "H_activity_stand",
        "H_activity_ladder",
        "H_anxietyD",
    ]

    required = [
        "hhid5",
        "survey_wave",
        "QpAssets",
        "Qk0", "Qk1", "Qk2", "Qk3", "Qk4", "Qk5",
        *activity_cols,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in analysis file: {missing}")

    df_small = df[required].copy()

    agg = df_small.groupby(["hhid5", "survey_wave"], as_index=False).agg(
        {
            "QpAssets": "mean",
            "Qk0": "max", "Qk1": "max", "Qk2": "max", "Qk3": "max", "Qk4": "max", "Qk5": "max",
            **{c: "mean" for c in activity_cols},
        }
    )

    # wealth per wave
    wave_to_Qk = {1: "Qk1", 2: "Qk2", 3: "Qk3", 4: "Qk4", 5: "Qk5"}

    def _wave_wealth(row):
        w = row["survey_wave"]
        try:
            return row[wave_to_Qk[int(w)]]
        except Exception:
            return np.nan

    agg["wealth"] = agg.apply(_wave_wealth, axis=1)

    # health index: negative sum of activity limitations + anxiety
    agg["health_limitations"] = agg[activity_cols].sum(axis=1, min_count=1)
    agg["health_index"] = -agg["health_limitations"]

    # investment = change in QpAssets
    agg = agg.sort_values(["hhid5", "survey_wave"])
    agg["investment_amount"] = agg.groupby("hhid5")["QpAssets"].diff()

    # total gain full and from baseline
    hh_levels = (
        agg.groupby("hhid5")
        .agg({"Qk0": "max", "Qk5": "max"})
        .rename(columns={"Qk0": "Qk0_hh", "Qk5": "Qk5_hh"})
    )
    agg = agg.merge(hh_levels, on="hhid5", how="left")
    agg["total_gain_full"] = agg["Qk5_hh"] - agg["Qk0_hh"]
    agg["total_gain_from_baseline"] = agg["wealth"] - agg["Qk0_hh"]

    out = agg[["hhid5", "survey_wave", "wealth", "investment_amount", "health_index", "total_gain_full", "total_gain_from_baseline"]].copy()
    return out
