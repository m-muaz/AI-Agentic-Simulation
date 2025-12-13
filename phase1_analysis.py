import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ai_agent_simulation.economics_model import EconomicParameters

# -----------------------------
# Configuration
# -----------------------------
ASSET_THRESHOLD = EconomicParameters().asset_threshold

# Update these paths as needed
SIM_ZIP_PATH = Path("logs_run_phase1_larger.zip")

# Your project originally expects this path; keep it, but allow override
DEFAULT_BALBONI_PANEL_PATH = Path("data/balboni/processed/balboni_panel.csv")
BALBONI_PANEL_PATH = DEFAULT_BALBONI_PANEL_PATH


# -----------------------------
# Helpers
# -----------------------------
def load_balboni_panel(path: Path) -> pd.DataFrame | None:
    """Load observed/empirical panel. Expected columns: hhid5, survey_wave, wealth."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"Warning: Balboni panel not found at {path}. Empirical comparison will be skipped.")
        return None
    except Exception as exc:
        print(f"Warning: Failed to load Balboni panel ({exc}). Empirical comparison will be skipped.")
        return None


def load_sim_logs(zip_path: Path) -> pd.DataFrame:
    """
    Load simulation CSV logs and merge with JSONL decision logs (if present).
    Expected CSV columns: step, wealth, health, investment, total_gain, ...
    JSONL records expected to contain: step and response.{savings_rate_t or savings_rate}.
    """
    dfs = []
    decision_rows = []

    def _read_csv_robust(fileobj) -> pd.DataFrame:
        """
        Try UTF-8 first; fall back to latin-1 with replacement on decode errors.
        """
        try:
            return pd.read_csv(fileobj)
        except UnicodeDecodeError:
            fileobj.seek(0)
            return pd.read_csv(fileobj, encoding="latin-1", on_bad_lines="skip")

    with zipfile.ZipFile(zip_path) as z:
        for fname in z.namelist():
            if fname.endswith(".csv"):
                with z.open(fname) as f:
                    df = _read_csv_robust(f)
                    df["household_id"] = Path(fname).stem
                    dfs.append(df)

            elif fname.endswith(".jsonl"):
                with z.open(fname) as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        response = rec.get("response", {}) or {}
                        decision_rows.append(
                            {
                                "household_id": Path(fname).stem,
                                "step": rec.get("step"),
                                "savings_rate": response.get("savings_rate_t", response.get("savings_rate")),
                                "risk_profile": response.get("risk_profile_t", response.get("risk_profile")),
                            }
                        )

    if not dfs:
        raise RuntimeError(f"No CSV files found inside {zip_path}")

    sim = pd.concat(dfs, ignore_index=True)

    # numeric hygiene
    if "step" in sim.columns:
        sim["step"] = pd.to_numeric(sim["step"], errors="coerce")
    if "wealth" in sim.columns:
        sim["wealth"] = pd.to_numeric(sim["wealth"], errors="coerce")

    # merge decisions (optional)
    decisions = pd.DataFrame(decision_rows)
    if not decisions.empty:
        decisions["step"] = pd.to_numeric(decisions["step"], errors="coerce")
        decisions["savings_rate"] = pd.to_numeric(decisions["savings_rate"], errors="coerce")
        sim = sim.merge(decisions, on=["household_id", "step"], how="left")

    return sim


# -----------------------------
# Plotting (most important only)
# -----------------------------
def plot_wealth_trajectories(sim: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    for h, g in sim.groupby("household_id"):
        plt.plot(g["step"], g["wealth"], alpha=0.8)

    plt.axhline(y=ASSET_THRESHOLD, linestyle="--", color="black", label="Threshold")
    plt.xlabel("Simulation Step")
    plt.ylabel("Wealth (Qk)")
    plt.title("Phase 1: Wealth Trajectories by Household")
    plt.legend()
    plt.show()


def plot_mean_wealth_dispersion(sim: pd.DataFrame) -> None:
    summary = sim.groupby("step")["wealth"].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(summary["step"], summary["mean"], label="Mean Wealth")
    plt.fill_between(
        summary["step"],
        summary["mean"] - summary["std"],
        summary["mean"] + summary["std"],
        alpha=0.3
    )
    plt.xlabel("Simulation Step")
    plt.ylabel("Wealth")
    plt.title("Mean Wealth ± 1 SD (Simulation)")
    plt.show()


def plot_savings_vs_wealth_if_available(sim: pd.DataFrame) -> None:
    if "savings_rate" in sim.columns and sim["savings_rate"].notna().any():
        data = sim[sim["savings_rate"].notna() & sim["wealth"].notna()].copy()
        if data.empty:
            print("Skipping savings-rate plot: no valid wealth/savings pairs.")
            return

        # Bin wealth into deciles and plot average savings rate per bin.
        data["wealth_bin"] = pd.qcut(data["wealth"], q=10, duplicates="drop")
        bin_means = (
            data.groupby("wealth_bin")["savings_rate"]
            .mean()
            .reset_index()
            .sort_values("wealth_bin")
        )

        plt.figure(figsize=(9, 5))
        plt.bar(bin_means["wealth_bin"].astype(str), bin_means["savings_rate"], color="tab:green", alpha=0.7)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Wealth Decile (binned)")
        plt.ylabel("Average Savings Rate")
        plt.title("Average Savings Rate by Wealth Bin (Phase 1)")
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping savings-rate plot: no savings_rate values found in logs.")


def plot_wealth_distribution_observed_vs_simulated(sim: pd.DataFrame, analysis: pd.DataFrame | None) -> None:
    if analysis is None:
        print("Empirical comparison skipped: `analysis` DataFrame not loaded.")
        return

    # expected columns: hhid5, survey_wave, wealth
    empirical = analysis[["hhid5", "survey_wave", "wealth"]].copy()
    empirical.rename(columns={"survey_wave": "step"}, inplace=True)

    # Compare end-of-window distribution:
    # - Observed: wave 5 (or max observed wave)
    # - Simulated: last simulation step
    obs_last_wave = int(pd.to_numeric(empirical["step"], errors="coerce").max())
    sim_last_step = int(pd.to_numeric(sim["step"], errors="coerce").max())

    emp_end = empirical[empirical["step"] == obs_last_wave]["wealth"].dropna()
    sim_end = sim[sim["step"] == sim_last_step]["wealth"].dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(emp_end, bins=30, alpha=0.5, label=f"Observed (wave {obs_last_wave})", density=True)
    plt.hist(sim_end, bins=30, alpha=0.5, label=f"Simulated (step {sim_last_step})", density=True)

    def _plot_gaussian(data: pd.Series, label: str, color: str) -> None:
        if data.empty:
            return
        mu = data.mean()
        sigma = data.std()
        if sigma == 0 or pd.isna(sigma):
            return
        x_min, x_max = data.min(), data.max()
        if x_min == x_max:
            return
        x_vals = np.linspace(x_min, x_max, 200)
        pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mu) / sigma) ** 2)
        plt.plot(x_vals, pdf, color=color, linewidth=2, label=f"{label} Gaussian")

    _plot_gaussian(emp_end, "Observed", "tab:blue")
    _plot_gaussian(sim_end, "Simulated", "tab:orange")
    plt.xlabel("Wealth")
    plt.ylabel("Density")
    plt.title("Wealth Distribution: Observed vs Simulated")
    plt.legend()
    plt.show()


def plot_poverty_transition(sim: pd.DataFrame) -> None:
    sim = sim.sort_values(["household_id", "step"]).copy()
    sim["poor"] = sim["wealth"] < ASSET_THRESHOLD
    sim["poor_next"] = sim.groupby("household_id")["poor"].shift(-1)

    # Escape probability: P(non-poor at t+1 | poor at t)
    escape_prob = 1.0 - sim.loc[sim["poor"] == True, "poor_next"].mean()

    print("P(Poor → Non-poor):", escape_prob)

    plt.figure(figsize=(6, 4))
    plt.bar(["Simulated"], [escape_prob])
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.title("Probability of Escaping Poverty (Phase 1)")
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # Load observed data (optional). If you want to use the uploaded file directly,
    # set BALBONI_PANEL_PATH = Path("balboni_panel.csv") or adjust accordingly.
    analysis = load_balboni_panel(BALBONI_PANEL_PATH)

    # Load simulation logs
    sim = load_sim_logs(SIM_ZIP_PATH)

    # 5 key plots
    plot_wealth_trajectories(sim)
    plot_mean_wealth_dispersion(sim)
    plot_savings_vs_wealth_if_available(sim)
    plot_wealth_distribution_observed_vs_simulated(sim, analysis)
    plot_poverty_transition(sim)


if __name__ == "__main__":
    main()
