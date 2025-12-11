"""
Plot simulation trajectories against the Balboni observational data.

Usage example:
  python scripts/plot_sim_vs_data.py --log-dir logs --save-dir plots --agent-limit 3 --obs-window 3
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_loaders.dataset_manager import DatasetManager


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
                        "rationale": rec.get("rationale"),
                    }
                )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["agent_id", "step"])


def load_panel(zip_path: str | Path) -> pd.DataFrame:
    dm = DatasetManager(zip_path)
    return dm.get_panel()


def parse_hhids(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            print(f"Skipping non-numeric household id '{part}'")
    return out or None


def choose_households(panel: pd.DataFrame, sim_df: pd.DataFrame, hhids_arg: Optional[List[int]], limit: int) -> List[int]:
    if hhids_arg:
        hhids = hhids_arg
    elif not sim_df.empty and sim_df["hhid"].notna().any():
        hhids = [int(h) for h in sim_df["hhid"].dropna().unique()]
    else:
        hhids = [int(h) for h in panel["hhid5"].dropna().unique()]
    if limit and len(hhids) > limit:
        hhids = hhids[:limit]
    return hhids


def scale_health(series: pd.Series, hmin: float, hmax: float) -> pd.Series:
    if hmax == hmin:
        return pd.Series([0.5] * len(series), index=series.index)
    return ((series - hmin) / (hmax - hmin)).clip(0.0, 1.0)


def plot_household(hhid: int, panel: pd.DataFrame, sim_df: pd.DataFrame, health_min: float, health_max: float, save_dir: Path):
    obs = panel[panel["hhid5"] == hhid].sort_values("survey_wave")
    sim = sim_df[sim_df["hhid"] == hhid].sort_values("step")
    if obs.empty and sim.empty:
        print(f"No data for hhid={hhid}; skipping.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=False)
    # Wealth (bars + lines)
    if not obs.empty:
        axes[0].bar(
            obs["survey_wave"] - 0.15,
            obs["wealth"],
            width=0.18,
            alpha=0.15,
            label="Observed wealth (bar)",
            color="tab:blue",
            )
    if not obs.empty:
        axes[0].plot(obs["survey_wave"], obs["wealth"], marker="o", label="Observed wealth (Balboni)")
    if not sim.empty:
        axes[0].bar(
            sim["step"] + 0.15,
            sim["wealth"],
            width=0.18,
            alpha=0.15,
            label="Sim wealth (bar)",
            color="tab:orange",
            )
    if not sim.empty:
        axes[0].plot(sim["step"], sim["wealth"], marker="x", label="Simulated wealth")
    axes[0].set_title(f"HHID {hhid} Wealth")
    axes[0].set_xlabel("Wave / Step")
    axes[0].set_ylabel("Wealth")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Health (scaled observed to 0-1 to match sim; bars + lines)
    if not obs.empty:
        axes[1].bar(
            obs["survey_wave"] - 0.15,
            scale_health(obs["health_index"], health_min, health_max),
            width=0.18,
            alpha=0.15,
            label="Observed health (bar)",
            color="tab:green",
            )
    if not obs.empty:
        axes[1].plot(
            obs["survey_wave"],
            scale_health(obs["health_index"], health_min, health_max),
            marker="o",
            label="Observed health (scaled)",
        )
    if not sim.empty:
        axes[1].bar(
            sim["step"] + 0.15,
            sim["health"],
            width=0.18,
            alpha=0.15,
            label="Sim health (bar)",
            color="tab:red",
            )
    if not sim.empty:
        axes[1].plot(sim["step"], sim["health"], marker="x", label="Simulated health")
    axes[1].set_title(f"HHID {hhid} Health")
    axes[1].set_xlabel("Wave / Step")
    axes[1].set_ylabel("Health (0-1)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"sim_vs_data_hh_{hhid}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_all_sim(sim_df: pd.DataFrame, save_dir: Path):
    if sim_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for agent_id, group in sim_df.groupby("agent_id"):
        axes[0].plot(group["step"], group["wealth"], marker="o", label=str(agent_id))
        axes[1].plot(group["step"], group["health"], marker="o", label=str(agent_id))
    axes[0].set_title("Simulated wealth per agent")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Wealth")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Simulated health per agent")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Health (0-1)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(title="Agent", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "sim_all_agents.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_aggregate(panel: pd.DataFrame, sim_df: pd.DataFrame, health_min: float, health_max: float, save_dir: Path):
    if sim_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    wealth_mean = sim_df.groupby("step")["wealth"].mean()
    health_mean = sim_df.groupby("step")["health"].mean()

    axes[0].bar(wealth_mean.index, wealth_mean.values, width=0.4, alpha=0.15, label="Mean wealth (bar)", color="tab:orange")
    axes[0].plot(wealth_mean.index, wealth_mean.values, marker="o", label="Mean wealth (line)", color="tab:blue")
    axes[0].set_title("Mean simulated wealth over steps")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Wealth")

    axes[1].bar(health_mean.index, health_mean.values, width=0.4, alpha=0.15, label="Mean health (bar)", color="tab:red")
    axes[1].plot(health_mean.index, health_mean.values, marker="o", label="Mean health (line)", color="tab:green")
    axes[1].set_title("Mean simulated health over steps")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Health (0-1)")

    for ax in axes:
        ax.legend()

    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "sim_aggregate.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot simulation logs against Balboni data.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--balboni-zip", type=Path, default=Path("data/balboni/dataverse_files.zip"))
    parser.add_argument("--save-dir", type=Path, default=Path("plots"))
    parser.add_argument("--agent-limit", type=int, default=3)
    parser.add_argument("--agent-hhids", type=str, default="")
    args = parser.parse_args()

    sim_df = load_sim_logs(args.log_dir)
    panel = load_panel(args.balboni_zip)
    hhids_arg = parse_hhids(args.agent_hhids)
    hhids = choose_households(panel, sim_df, hhids_arg, args.agent_limit)
    health_min = panel["health_index"].min(skipna=True)
    health_max = panel["health_index"].max(skipna=True)

    if not sim_df.empty:
        plot_aggregate(panel, sim_df, health_min, health_max, args.save_dir)
        plot_all_sim(sim_df, args.save_dir)

    for hhid in hhids:
        plot_household(hhid, panel, sim_df, health_min, health_max, args.save_dir)


if __name__ == "__main__":
    main()
