"""
Quick plots for Phase 2 simulation logs.
- Expects a zip of per-agent CSV/JSONL logs (like logs_run_phase2.zip).
- CSV columns: timestamp, step, agent_id, wealth, health, investment, total_gain, rationale
- JSONL keys (response): coping_action, coping_intensity, savings_rate_t/savings_rate, health_effort, risk_profile_t/risk_profile
- Optional: overlay observed wealth distribution (Balboni panel) if available.
"""

import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ZIP_PATH = Path("logs_run_phase2_larger.zip")
DEFAULT_OBS_PATH = Path("data/balboni/processed/balboni_panel.csv")


def _read_csv_robust(fileobj) -> pd.DataFrame:
    """Try UTF-8 first; fall back to latin-1 with line skipping on decode errors."""
    try:
        return pd.read_csv(fileobj)
    except UnicodeDecodeError:
        fileobj.seek(0)
        return pd.read_csv(fileobj, encoding="latin-1", on_bad_lines="skip")


def load_sim_logs(zip_path: Path) -> pd.DataFrame:
    dfs = []
    decision_rows = []

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
                        resp = rec.get("response", {}) or {}
                        decision_rows.append(
                            {
                                "household_id": Path(fname).stem,
                                "step": rec.get("step"),
                                "coping_action": resp.get("coping_action"),
                                "coping_intensity": resp.get("coping_intensity"),
                                "savings_rate": resp.get("savings_rate_t", resp.get("savings_rate")),
                                "risk_profile": resp.get("risk_profile_t", resp.get("risk_profile")),
                                "health_effort": resp.get("health_effort"),
                            }
                        )

    if not dfs:
        raise RuntimeError(f"No CSV files found inside {zip_path}")

    sim = pd.concat(dfs, ignore_index=True)

    # numeric hygiene
    for col in ["step", "wealth", "health", "investment", "total_gain"]:
        if col in sim.columns:
            sim[col] = pd.to_numeric(sim[col], errors="coerce")

    decisions = pd.DataFrame(decision_rows)
    if not decisions.empty:
        for col in ["step", "coping_intensity", "savings_rate", "health_effort"]:
            if col in decisions.columns:
                decisions[col] = pd.to_numeric(decisions[col], errors="coerce")
        sim = sim.merge(decisions, on=["household_id", "step"], how="left")

    return sim


def load_observed_panel(path: Path) -> pd.DataFrame | None:
    """
    Load observed Balboni panel (hhid5, survey_wave, wealth). Returns None if missing.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Observed panel not found at {path}; skipping observed comparison.")
    except Exception as exc:
        print(f"Failed to load observed panel ({exc}); skipping observed comparison.")
    return None


def plot_wealth_trajectories(sim: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    for h, g in sim.groupby("household_id"):
        plt.plot(g["step"], g["wealth"], alpha=0.7)
    plt.xlabel("Simulation Step")
    plt.ylabel("Wealth")
    plt.title("Phase 2: Wealth Trajectories by Household")
    plt.show()


def plot_mean_wealth_and_health(sim: pd.DataFrame) -> None:
    summary = sim.groupby("step").agg({"wealth": ["mean", "std"], "health": "mean"})
    summary.columns = ["wealth_mean", "wealth_std", "health_mean"]
    summary = summary.reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(summary["step"], summary["wealth_mean"], label="Mean Wealth")
    plt.fill_between(
        summary["step"],
        summary["wealth_mean"] - summary["wealth_std"],
        summary["wealth_mean"] + summary["wealth_std"],
        alpha=0.3,
        label="Wealth ±1 SD",
    )
    plt.xlabel("Simulation Step")
    plt.ylabel("Wealth")
    plt.title("Mean Wealth ± 1 SD (Phase 2)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(summary["step"], summary["health_mean"], color="tab:green", label="Mean Health")
    plt.xlabel("Simulation Step")
    plt.ylabel("Health")
    plt.title("Mean Health (Phase 2)")
    plt.legend()
    plt.show()


def plot_coping_actions(sim: pd.DataFrame) -> None:
    if "coping_action" not in sim.columns:
        print("Skipping coping-action plot: no decisions merged.")
        return
    action_counts = (
        sim.groupby(["step", "coping_action"])
        .size()
        .reset_index(name="count")
        .pivot(index="step", columns="coping_action", values="count")
        .fillna(0)
    )
    action_share = action_counts.div(action_counts.sum(axis=1), axis=0)

    action_share.plot(kind="area", stacked=True, figsize=(9, 5), colormap="tab20")
    plt.xlabel("Simulation Step")
    plt.ylabel("Share of Actions")
    plt.title("Coping Action Mix Over Time (Phase 2)")
    plt.tight_layout()
    plt.show()


def plot_savings_vs_wealth(sim: pd.DataFrame) -> None:
    if "savings_rate" not in sim.columns or sim["savings_rate"].notna().sum() == 0:
        print("Skipping savings plot: no savings_rate values.")
        return
    data = sim[sim["savings_rate"].notna() & sim["wealth"].notna()].copy()
    if data.empty:
        print("Skipping savings plot: no valid wealth/savings pairs.")
        return
    data["wealth_bin"] = pd.qcut(data["wealth"], q=10, duplicates="drop")
    bin_means = (
        data.groupby("wealth_bin")["savings_rate"]
        .mean()
        .reset_index()
        .sort_values("wealth_bin")
    )
    plt.figure(figsize=(9, 5))
    plt.bar(bin_means["wealth_bin"].astype(str), bin_means["savings_rate"], color="tab:purple", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Wealth Decile (binned)")
    plt.ylabel("Average Savings Rate")
    plt.title("Average Savings Rate by Wealth Bin (Phase 2)")
    plt.tight_layout()
    plt.show()


def plot_wealth_distribution(sim: pd.DataFrame, observed: pd.DataFrame | None = None) -> None:
    sim_last = int(pd.to_numeric(sim["step"], errors="coerce").max())
    wealth = sim.loc[sim["step"] == sim_last, "wealth"].dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(wealth, bins=30, density=True, alpha=0.6, color="tab:blue", label=f"Simulated (step {sim_last})")

    if observed is not None and {"survey_wave", "wealth"}.issubset(observed.columns):
        obs_last = int(pd.to_numeric(observed["survey_wave"], errors="coerce").max())
        obs_wealth = observed.loc[observed["survey_wave"] == obs_last, "wealth"].dropna()
        if not obs_wealth.empty:
            plt.hist(obs_wealth, bins=30, density=True, alpha=0.4, color="tab:green", label=f"Observed (wave {obs_last})")

    def _plot_gaussian(series: pd.Series, label: str, color: str) -> None:
        if len(series) <= 1:
            return
        mu, sigma = series.mean(), series.std()
        if sigma == 0 or pd.isna(sigma):
            return
        x_vals = np.linspace(series.min(), series.max(), 200)
        pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mu) / sigma) ** 2)
        plt.plot(x_vals, pdf, color=color, linewidth=2, label=label)

    _plot_gaussian(wealth, "Simulated Gaussian", "tab:blue")
    if observed is not None and {"survey_wave", "wealth"}.issubset(observed.columns):
        obs_last = int(pd.to_numeric(observed["survey_wave"], errors="coerce").max())
        obs_wealth = observed.loc[observed["survey_wave"] == obs_last, "wealth"].dropna()
        _plot_gaussian(obs_wealth, "Observed Gaussian", "tab:green")

    plt.xlabel("Wealth")
    plt.ylabel("Density")
    plt.title("Wealth Distribution (Final Step, Phase 2)")
    plt.legend()
    plt.show()


def main() -> None:
    zip_path = DEFAULT_ZIP_PATH
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected zip at {zip_path}")

    sim = load_sim_logs(zip_path)
    observed = load_observed_panel(DEFAULT_OBS_PATH)

    plot_wealth_trajectories(sim)
    plot_mean_wealth_and_health(sim)
    plot_coping_actions(sim)
    plot_savings_vs_wealth(sim)
    plot_wealth_distribution(sim, observed)


if __name__ == "__main__":
    main()
