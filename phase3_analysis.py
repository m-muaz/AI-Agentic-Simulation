"""
Quick plots for Phase 3 simulation logs.
- Expects a zip of per-agent CSV/JSONL logs (like logs_run_phase3.zip).
- CSV columns: timestamp, step, agent_id, wealth, health, investment, total_gain, rationale
- JSONL keys (response): labor_allocation, sector_choice, savings_rate_t/savings_rate, health_effort
- Optional: overlay observed wealth distribution (Balboni panel) if available.
"""

import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ZIP_PATH = Path("logs_run_phase3_larger.zip")
DEFAULT_OBS_PATH = Path("data/balboni/processed/balboni_panel.csv")


def _read_csv_robust(fileobj) -> pd.DataFrame:
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
                        labor_alloc = resp.get("labor_allocation") or {}
                        decision_rows.append(
                            {
                                "household_id": Path(fname).stem,
                                "step": rec.get("step"),
                                "sector_choice": resp.get("sector_choice"),
                                "savings_rate": resp.get("savings_rate_t", resp.get("savings_rate")),
                                "health_effort": resp.get("health_effort"),
                                "labor_farm": labor_alloc.get("farm"),
                                "labor_wage_labor": labor_alloc.get("wage_labor"),
                                "labor_self_employment": labor_alloc.get("self_employment"),
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
        for col in ["step", "savings_rate", "health_effort", "labor_farm", "labor_wage_labor", "labor_self_employment"]:
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
    plt.title("Phase 3: Wealth Trajectories by Household")
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
    plt.title("Mean Wealth ± 1 SD (Phase 3)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(summary["step"], summary["health_mean"], color="tab:green", label="Mean Health")
    plt.xlabel("Simulation Step")
    plt.ylabel("Health")
    plt.title("Mean Health (Phase 3)")
    plt.legend()
    plt.show()


def plot_sector_choice(sim: pd.DataFrame) -> None:
    if "sector_choice" not in sim.columns:
        print("Skipping sector plot: no decisions merged.")
        return
    sector_counts = (
        sim.groupby(["step", "sector_choice"])
        .size()
        .reset_index(name="count")
        .pivot(index="step", columns="sector_choice", values="count")
        .fillna(0)
    )
    sector_share = sector_counts.div(sector_counts.sum(axis=1), axis=0)
    sector_share.plot(kind="area", stacked=True, figsize=(9, 5), colormap="tab20")
    plt.xlabel("Simulation Step")
    plt.ylabel("Share")
    plt.title("Sector Choice Mix Over Time (Phase 3)")
    plt.tight_layout()
    plt.show()


def plot_labor_allocation(sim: pd.DataFrame) -> None:
    cols = ["labor_farm", "labor_wage_labor", "labor_self_employment"]
    if not set(cols).issubset(sim.columns):
        print("Skipping labor allocation plot: no labor allocation data.")
        return
    data = sim[["step", *cols]].dropna()
    if data.empty:
        print("Skipping labor allocation plot: no labor allocation data.")
        return
    alloc_mean = data.groupby("step")[cols].mean().reset_index()
    plt.figure(figsize=(9, 5))
    plt.stackplot(
        alloc_mean["step"],
        alloc_mean["labor_farm"],
        alloc_mean["labor_wage_labor"],
        alloc_mean["labor_self_employment"],
        labels=["Farm", "Wage Labor", "Self Employment"],
        alpha=0.8,
    )
    plt.xlabel("Simulation Step")
    plt.ylabel("Average Labor Share")
    plt.title("Average Labor Allocation Over Time (Phase 3)")
    plt.legend(loc="upper right")
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
    if observed is not None and "survey_wave" in observed and "wealth" in observed:
        obs_last = int(pd.to_numeric(observed["survey_wave"], errors="coerce").max())
        obs_wealth = observed.loc[observed["survey_wave"] == obs_last, "wealth"].dropna()
        _plot_gaussian(obs_wealth, "Observed Gaussian", "tab:green")

    plt.xlabel("Wealth")
    plt.ylabel("Density")
    plt.title("Wealth Distribution (Final Step, Phase 3)")
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
    plot_sector_choice(sim)
    plot_labor_allocation(sim)
    plot_wealth_distribution(sim, observed)


if __name__ == "__main__":
    main()
