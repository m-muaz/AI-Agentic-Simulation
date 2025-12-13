import json
import zipfile
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from src.ai_agent_simulation.economics_model import EconomicParameters, EconomicModel

ASSET_THRESHOLD = EconomicParameters().asset_threshold
savings_rate = EconomicParameters().savings_rate

# Load empirical Balboni panel for comparison.
BALBONI_PANEL_PATH = Path("data/balboni/processed/balboni_panel.csv")
try:
    analysis = pd.read_csv(BALBONI_PANEL_PATH)
except FileNotFoundError:
    analysis = None
    print(f"Warning: Balboni panel not found at {BALBONI_PANEL_PATH}. Empirical comparison will be skipped.")
except Exception as exc:
    analysis = None
    print(f"Warning: Failed to load Balboni panel ({exc}). Empirical comparison will be skipped.")



# Load all household logs into one DataFrame
# sim is our simulation panel with columns

zip_path = "logs_run_phase1.zip"

dfs = []
decision_rows = []
with zipfile.ZipFile(zip_path) as z:
    for fname in z.namelist():
        if fname.endswith(".csv"):
            with z.open(fname) as f:
                df = pd.read_csv(f)
                df["household_id"] = Path(fname).stem
                dfs.append(df)
        elif fname.endswith(".jsonl"):
            # JSONL logs contain the LLM's decisions (savings_rate_t, risk_profile_t).
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
                            "savings_rate": response.get(
                                "savings_rate_t", response.get("savings_rate")
                            ),
                            "risk_profile": response.get(
                                "risk_profile_t", response.get("risk_profile")
                            ),
                        }
                    )

sim = pd.concat(dfs, ignore_index=True)
decisions = pd.DataFrame(decision_rows)
if not decisions.empty:
    sim = sim.merge(decisions, on=["household_id", "step"], how="left")

plt.figure(figsize=(8,5))
for h, g in sim.groupby("household_id"):
    plt.plot(g["step"], g["wealth"], alpha=0.8)

plt.axhline(y=ASSET_THRESHOLD, linestyle="--", color="black", label="Threshold")
plt.xlabel("Simulation Step")
plt.ylabel("Wealth (Qk)")
plt.title("Phase 1: Wealth Trajectories by Household")
plt.legend()
plt.show()

# Phase 1 Standalone Evaluation
# Wealth Trajectories (Key Figure)

# Wealth trajectories diverge sharply across households, 
# with strong persistence below the asset threshold and sustained growth above it.

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
for h, g in sim.groupby("household_id"):
    plt.plot(g["step"], g["wealth"], alpha=0.8)

plt.axhline(y=ASSET_THRESHOLD, linestyle="--", color="black", label="Threshold")
plt.xlabel("Simulation Step")
plt.ylabel("Wealth (Qk)")
plt.title("Phase 1: Wealth Trajectories by Household")
plt.legend()
plt.show()

# Mean Wealth and Dispersion

summary = sim.groupby("step")["wealth"].agg(["mean", "std"]).reset_index()

plt.figure(figsize=(8,5))
plt.plot(summary["step"], summary["mean"], label="Mean Wealth")
plt.fill_between(
    summary["step"],
    summary["mean"] - summary["std"],
    summary["mean"] + summary["std"],
    alpha=0.3
)
plt.xlabel("Simulation Step")
plt.ylabel("Wealth")
plt.title("Mean Wealth ± 1 SD")
plt.show()

# LLM Savings Behavior Sanity Check

if "savings_rate" in sim.columns and sim["savings_rate"].notna().any():
    plt.figure(figsize=(7,5))
    plt.scatter(sim["wealth"], sim["savings_rate"], alpha=0.4)
    plt.xlabel("Wealth")
    plt.ylabel("Savings Rate")
    plt.title("LLM Savings Rate vs Wealth (Phase 1)")
    plt.show()
else:
    print("Skipping savings-rate plot: no savings_rate values found in logs.")

# Investment Dynamics

plt.figure(figsize=(7,5))
plt.scatter(sim["wealth"], sim["investment"], alpha=0.4)
plt.axhline(0, linestyle="--", color="black")
plt.xlabel("Wealth")
plt.ylabel("Investment (Δ Wealth)")
plt.title("Investment vs Wealth")
plt.show()

# Comparison to Original Data (Balboni)

if analysis is not None:
    empirical = analysis[["hhid5", "survey_wave", "wealth"]].copy()
    empirical.rename(columns={"survey_wave": "step"}, inplace=True)

    # Mean Wealth: Observed vs Simulated
    emp_mean = empirical.groupby("step")["wealth"].mean()
    sim_mean = sim.groupby("step")["wealth"].mean()

    plt.figure(figsize=(8,5))
    plt.plot(emp_mean.index, emp_mean.values, label="Observed", marker="o")
    plt.plot(sim_mean.index, sim_mean.values, label="Simulated", marker="x")
    plt.xlabel("Time")
    plt.ylabel("Mean Wealth")
    plt.title("Mean Wealth: Observed vs Simulated (Phase 1)")
    plt.legend()
    plt.show()

    # Distributional Comparison (End of Window)
    plt.figure(figsize=(8,5))
    sim_end = sim[sim["step"] == sim["step"].max()]["wealth"]
    emp_end = empirical[empirical["step"] == 5]["wealth"]

    plt.hist(emp_end, bins=30, alpha=0.5, label="Observed", density=True)
    plt.hist(sim_end, bins=30, alpha=0.5, label="Simulated", density=True)
    plt.xlabel("Wealth")
    plt.ylabel("Density")
    plt.title("Wealth Distribution: Observed vs Simulated")
    plt.legend()
    plt.show()
else:
    print("Empirical comparison skipped: `analysis` DataFrame not loaded.")

# Strong Validation: Poverty Trap Transitions

sim["poor"] = sim["wealth"] < ASSET_THRESHOLD
sim["poor_next"] = sim.groupby("household_id")["poor"].shift(-1)

transition_rate = (
    sim[(sim["poor"] == True)]
    .groupby("step")["poor_next"]
    .mean()
)

print("P(Poor → Non-poor):", 1 - transition_rate.mean())

