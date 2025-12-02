"""
Evaluate simulation logs and (optionally) compare against Balboni observations.

Outputs (CSV only, no plotting dependency):
- per_agent_metrics.csv: final wealth/health, deltas, volatility.
- aggregate_metrics.csv: aggregates across agents.
- wealth_distribution.csv: histogram bins for simulated vs observed wealth.
- health_by_wave.csv: mean simulated vs observed (scaled) health by wave (matched pairs).
- mae_vs_observed.csv: MAE of wealth/health vs Balboni (if matching data exists).

Usage example:
  python scripts/evaluate_sim.py --log-dir logs --save-dir eval --balboni-zip data/balboni/dataverse_files.zip --start-wave 1
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from data_loaders.dataset_manager import DatasetManager
except Exception:
    DatasetManager = None


def load_sim_logs(log_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(log_dir.glob("agent_*.jsonl")):
        with path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                state = rec.get("state", {}) or {}
                agent_id = rec.get("agent_id") or state.get("agent_id")
                hhid = None
                if agent_id:
                    m = re.search(r"hh_(\\d+)", agent_id)
                    if m:
                        try:
                            hhid = int(m.group(1))
                        except ValueError:
                            hhid = None
                rows.append(
                    {
                        "agent_id": agent_id,
                        "hhid": hhid,
                        "step": rec.get("step"),
                        "wealth": state.get("wealth"),
                        "health": state.get("health"),
                        "investment": state.get("investment"),
                        "total_gain": state.get("total_gain"),
                    }
                )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Normalize types
    if not df.empty:
        df["hhid"] = pd.to_numeric(df["hhid"], errors="coerce").astype("Int64")
        df["step"] = pd.to_numeric(df["step"], errors="coerce").fillna(0).astype(int)
        df["wealth"] = pd.to_numeric(df["wealth"], errors="coerce")
        df["health"] = pd.to_numeric(df["health"], errors="coerce")
    return df.sort_values(["agent_id", "step"]).reset_index(drop=True)


def load_panel(panel_path: Optional[str | Path], zip_path: Optional[str | Path]) -> Optional[pd.DataFrame]:
    """
    Prefer a processed CSV panel if provided; otherwise fall back to DatasetManager with the zip.
    """
    if panel_path:
        p = Path(panel_path)
        if p.exists():
            try:
                df = pd.read_csv(p)
                return df
            except Exception as exc:
                print(f"Could not load Balboni panel CSV ({panel_path}): {exc}")
    if not zip_path or DatasetManager is None:
        return None
    try:
        dm = DatasetManager(zip_path)
        return dm.get_panel()
    except Exception as exc:
        print(f"Could not load Balboni data ({zip_path}): {exc}")
        return None


def per_agent_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    for agent_id, grp in df.groupby("agent_id"):
        first = grp.iloc[0]
        last = grp.iloc[-1]
        wealth_delta = (last["wealth"] or 0) - (first["wealth"] or 0)
        health_delta = (last["health"] or 0) - (first["health"] or 0)
        metrics.append(
            {
                "agent_id": agent_id,
                "hhid": grp["hhid"].iloc[0],
                "steps": len(grp),
                "wealth_start": first["wealth"],
                "wealth_end": last["wealth"],
                "wealth_delta": wealth_delta,
                "wealth_std": grp["wealth"].std(),
                "health_start": first["health"],
                "health_end": last["health"],
                "health_delta": health_delta,
                "health_std": grp["health"].std(),
            }
        )
    return pd.DataFrame(metrics)


def aggregate_metrics(per_agent_df: pd.DataFrame) -> pd.DataFrame:
    if per_agent_df.empty:
        return pd.DataFrame()
    agg = {
        "agents": len(per_agent_df),
        "mean_final_wealth": per_agent_df["wealth_end"].mean(),
        "median_final_wealth": per_agent_df["wealth_end"].median(),
        "mean_wealth_delta": per_agent_df["wealth_delta"].mean(),
        "mean_wealth_std": per_agent_df["wealth_std"].mean(),
        "mean_final_health": per_agent_df["health_end"].mean(),
        "mean_health_delta": per_agent_df["health_delta"].mean(),
        "mean_health_std": per_agent_df["health_std"].mean(),
    }
    return pd.DataFrame([agg])


def compute_mae_vs_panel(sim_df: pd.DataFrame, panel: pd.DataFrame, start_wave: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Approximate wave alignment: wave = start_wave + step.
    This assumes step 0 corresponds to the starting wave.
    Returns:
      - mae_df: single-row MAE summary
      - merged: merged simulated/observed rows for plotting
    """
    if sim_df.empty or panel is None:
        return pd.DataFrame(), pd.DataFrame()
    sim = sim_df.copy()
    sim["wave"] = sim["step"].apply(lambda s: start_wave + int(s or 0))
    # Scale observed health to 0-1
    health_min = panel["health_index"].min(skipna=True)
    health_max = panel["health_index"].max(skipna=True)
    panel_scaled = panel.copy()
    if health_max != health_min:
        panel_scaled["health_scaled"] = ((panel_scaled["health_index"] - health_min) / (health_max - health_min)).clip(0.0, 1.0)
    else:
        panel_scaled["health_scaled"] = 0.5
    panel_scaled = panel_scaled.copy()
    panel_scaled["hhid5"] = pd.to_numeric(panel_scaled["hhid5"], errors="coerce").astype("Int64")
    merged = sim.merge(
        panel_scaled[["hhid5", "survey_wave", "wealth", "health_scaled"]],
        left_on=["hhid", "wave"],
        right_on=["hhid5", "survey_wave"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged["wealth_abs_err"] = (merged["wealth_x"] - merged["wealth_y"]).abs()
    merged["health_abs_err"] = (merged["health"] - merged["health_scaled"]).abs()
    mae = {
        "pairs": len(merged),
        "wealth_mae": merged["wealth_abs_err"].mean(),
        "health_mae": merged["health_abs_err"].mean(),
    }
    return pd.DataFrame([mae]), merged


def wealth_histogram(sim_df: pd.DataFrame, panel: Optional[pd.DataFrame], bins: int = 20) -> pd.DataFrame:
    """
    Build histogram bins for simulated and observed wealth (densities and counts).
    """
    if sim_df.empty:
        return pd.DataFrame()
    sim_wealth = sim_df["wealth"].dropna().values
    obs_wealth = panel["wealth"].dropna().values if panel is not None else np.array([])
    all_vals = np.concatenate([sim_wealth, obs_wealth]) if obs_wealth.size else sim_wealth
    if all_vals.size == 0:
        return pd.DataFrame()
    hist_range = (float(all_vals.min()), float(all_vals.max()))
    bin_counts_sim, bin_edges = np.histogram(sim_wealth, bins=bins, range=hist_range)
    bin_counts_obs, _ = np.histogram(obs_wealth, bins=bin_edges) if obs_wealth.size else (np.zeros_like(bin_counts_sim), bin_edges)
    bin_widths = np.diff(bin_edges)
    densities_sim = bin_counts_sim / bin_widths / max(1, sim_wealth.size)
    densities_obs = bin_counts_obs / bin_widths / max(1, obs_wealth.size if obs_wealth.size else 1)
    rows = []
    for i in range(len(bin_counts_sim)):
        rows.append(
            {
                "bin_lower": bin_edges[i],
                "bin_upper": bin_edges[i + 1],
                "sim_count": int(bin_counts_sim[i]),
                "sim_density": float(densities_sim[i]),
                "obs_count": int(bin_counts_obs[i]),
                "obs_density": float(densities_obs[i]),
            }
        )
    return pd.DataFrame(rows)


def health_by_wave(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Mean simulated vs observed (scaled) health by wave for matched pairs.
    """
    if merged.empty:
        return pd.DataFrame()
    df = merged.copy()
    df["wave"] = df["survey_wave"]
    grp = (
        df.groupby("wave")
        .agg(sim_mean_health=("health", "mean"), obs_mean_health=("health_scaled", "mean"), pairs=("health", "count"))
        .reset_index()
    )
    return grp


def plot_vs_observed_means(merged: pd.DataFrame, save_dir: Path):
    """
    Plot mean simulated vs observed wealth/health by wave (matched pairs).
    """
    if merged.empty or plt is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    df = merged.copy()
    df["wave"] = df["survey_wave"]
    wealth_group = df.groupby("wave")[["wealth_x", "wealth_y"]].mean()
    health_group = df.groupby("wave")[["health", "health_scaled"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(wealth_group.index, wealth_group["wealth_x"], marker="o", label="Sim wealth (mean)")
    axes[0].plot(wealth_group.index, wealth_group["wealth_y"], marker="x", label="Observed wealth (mean)")
    axes[0].set_title("Wealth: simulated vs observed (mean by wave)")
    axes[0].set_xlabel("Wave")
    axes[0].set_ylabel("Wealth")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(health_group.index, health_group["health"], marker="o", label="Sim health (mean)")
    axes[1].plot(health_group.index, health_group["health_scaled"], marker="x", label="Observed health (mean, scaled)")
    axes[1].set_title("Health: simulated vs observed (mean by wave)")
    axes[1].set_xlabel("Wave")
    axes[1].set_ylabel("Health (0-1)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_dir / "eval_vs_observed_means.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vs_observed_scatter(merged: pd.DataFrame, save_dir: Path):
    """
    Scatter plots of simulated vs observed wealth/health for matched pairs.
    """
    if merged.empty or plt is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(merged["wealth_y"], merged["wealth_x"], alpha=0.5)
    axes[0].plot([merged["wealth_y"].min(), merged["wealth_y"].max()],
                 [merged["wealth_y"].min(), merged["wealth_y"].max()],
                 color="red", linestyle="--", label="y=x")
    axes[0].set_title("Simulated vs Observed Wealth (pairs)")
    axes[0].set_xlabel("Observed wealth")
    axes[0].set_ylabel("Simulated wealth")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(merged["health_scaled"], merged["health"], alpha=0.5)
    axes[1].plot([0, 1], [0, 1], color="red", linestyle="--", label="y=x")
    axes[1].set_title("Simulated vs Observed Health (scaled, pairs)")
    axes[1].set_xlabel("Observed health (scaled)")
    axes[1].set_ylabel("Simulated health")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_dir / "eval_vs_observed_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate simulation logs.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--save-dir", type=Path, default=Path("eval"))
    parser.add_argument("--balboni-panel", type=Path, default=Path("data/balboni/processed/balboni_panel.csv"), help="Path to processed Balboni panel CSV.")
    parser.add_argument("--balboni-zip", type=Path, default=None, help="Optional path to dataverse_files.zip (fallback if panel CSV is missing).")
    parser.add_argument("--start-wave", type=int, default=1, help="Assumed starting survey wave for step alignment to panel.")
    args = parser.parse_args()

    sim_df = load_sim_logs(args.log_dir)
    if sim_df.empty:
        print(f"No simulation logs found in {args.log_dir}")
        return

    per_agent_df = per_agent_metrics(sim_df)
    agg_df = aggregate_metrics(per_agent_df)
    panel = load_panel(args.balboni_panel, args.balboni_zip)
    mae_df, merged = compute_mae_vs_panel(sim_df, panel, args.start_wave) if panel is not None else (pd.DataFrame(), pd.DataFrame())

    args.save_dir.mkdir(parents=True, exist_ok=True)
    per_agent_df.to_csv(args.save_dir / "per_agent_metrics.csv", index=False)
    agg_df.to_csv(args.save_dir / "aggregate_metrics.csv", index=False)
    if not mae_df.empty:
        mae_df.to_csv(args.save_dir / "mae_vs_observed.csv", index=False)

    wealth_hist_df = wealth_histogram(sim_df, panel)
    if not wealth_hist_df.empty:
        wealth_hist_df.to_csv(args.save_dir / "wealth_distribution.csv", index=False)
    health_wave_df = health_by_wave(merged)
    if not health_wave_df.empty:
        health_wave_df.to_csv(args.save_dir / "health_by_wave.csv", index=False)

    # Plots (require matplotlib; will raise if missing)
    plot_vs_observed_means(merged, args.save_dir)
    plot_vs_observed_scatter(merged, args.save_dir)

    print(f"Saved metrics to {args.save_dir}")
    if not mae_df.empty:
        print(f"MAE vs observed: {mae_df.to_dict(orient='records')[0]}")


if __name__ == "__main__":
    main()
