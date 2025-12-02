import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback if tqdm missing
    tqdm = None

# Ensure src/ is on the path when running without installation
REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_loaders.dataset_manager import DatasetManager

from ai_agent_simulation.agent import Agent
from ai_agent_simulation.config import AgentScenario
from ai_agent_simulation.environment import Environment
from ai_agent_simulation.economics_model import EconomicModel, EconomicParameters


BALBONI_DEFAULT_ZIP = "data/balboni/dataverse_files.zip"


def _maybe_load_balboni_panel(zip_path: Optional[str] = None):
    """
    Load the Balboni panel if available and not explicitly disabled.
    """
    if os.getenv("USE_BALBONI_DATA", "1") == "0":
        return None
    zip_path = zip_path or os.getenv("BALBONI_ZIP_PATH", BALBONI_DEFAULT_ZIP)
    try:
        dm = DatasetManager(zip_path)
        panel = dm.get_panel()
        print(f"Loaded Balboni panel from {zip_path} with {len(panel)} rows.")
        return panel
    except Exception as exc:
        print(f"Could not load Balboni data ({zip_path}): {exc}")
        return None


def _assign_households(agent_configs, panel):
    """
    If configs omit hhids, assign them sequentially from the panel list.
    """
    if panel is None:
        return agent_configs
    hhids = list(panel["hhid5"].dropna().unique())
    for idx, scenario in enumerate(agent_configs):
        if getattr(scenario, "hhid", None) is None and idx < len(hhids):
            scenario.hhid = int(hhids[idx])
    return agent_configs


def _scale_health(value: float, min_val: float, max_val: float) -> Optional[float]:
    if pd.isna(value):
        return None
    if max_val == min_val:
        return 0.5
    scaled = (float(value) - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, scaled))


def _update_from_balboni(agent_configs, panel):
    """
    Replace starting wealth/health with observed Balboni values when available.
    """
    if panel is None or panel.empty:
        return agent_configs

    health_min = panel["health_index"].min(skipna=True)
    health_max = panel["health_index"].max(skipna=True)

    for scenario in agent_configs:
        hhid = getattr(scenario, "hhid", None)
        if hhid is None:
            continue
        wave = getattr(scenario, "start_wave", 1)
        row = panel[(panel["hhid5"] == hhid) & (panel["survey_wave"] == wave)]
        if row.empty:
            print(
                f"Balboni panel: no row for hhid={hhid} wave={wave}; keeping config defaults."
            )
            continue
        r = row.iloc[0]
        wealth = r.get("wealth")
        health_idx = r.get("health_index")
        investment = r.get("investment_amount")
        total_gain = r.get("total_gain_from_baseline")

        if not pd.isna(wealth):
            scenario.initial_wealth = float(wealth)

        scaled_health = _scale_health(health_idx, health_min, health_max)
        if scaled_health is not None:
            scenario.initial_health = scaled_health

        # Carry the raw observation into the household history for LLM context.
        auto_history = (
            f"Balboni data | hhid={hhid} wave={wave} | "
            f"wealth={wealth} health_index={health_idx} "
            f"investment={investment} total_gain_from_baseline={total_gain}"
        )
        scenario.household_history = (
            f"{scenario.household_history} {auto_history}".strip()
        )
    return agent_configs


def _select_households(panel, limit: Optional[int] = None) -> list[int]:
    hhids = list(panel["hhid5"].dropna().unique())
    if limit is not None:
        hhids = hhids[:limit]
    return hhids


def _build_agents_from_data(
    panel,
    start_wave: int,
    agent_limit: Optional[int],
    hhids_override: Optional[List[int]] = None,
) -> list[AgentScenario]:
    """
    Build agent scenarios directly from the Balboni panel instead of configs/agents.yaml.
    """
    if panel is None or panel.empty:
        raise RuntimeError("Balboni panel is empty; cannot build agents from data.")

    hhids = hhids_override if hhids_override is not None else _select_households(panel, agent_limit)
    health_min = panel["health_index"].min(skipna=True)
    health_max = panel["health_index"].max(skipna=True)

    scenarios: list[AgentScenario] = []
    for hhid in hhids:
        row = panel[(panel["hhid5"] == hhid) & (panel["survey_wave"] == start_wave)]
        if row.empty:
            print(f"No data for hhid={hhid} at wave={start_wave}; skipping.")
            continue
        r = row.iloc[0]
        wealth = float(r["wealth"]) if not pd.isna(r.get("wealth")) else 0.0
        health_raw = r.get("health_index")
        health = _scale_health(health_raw, health_min, health_max) or 0.0
        investment = (
            float(r["investment_amount"]) if not pd.isna(r.get("investment_amount")) else 0.0
        )
        total_gain = (
            float(r["total_gain_from_baseline"])
            if not pd.isna(r.get("total_gain_from_baseline"))
            else wealth  # fallback to wealth as gain
        )

        scenarios.append(
            AgentScenario(
                agent_id=f"hh_{hhid}",
                initial_wealth=wealth,
                initial_health=health,
                name=f"Household {hhid}",
                household_history=(
                    f"Balboni data hhid={hhid} wave={start_wave} "
                    f"wealth={wealth} health_index={health_raw} "
                    f"investment={investment} total_gain_from_baseline={total_gain}"
                ),
                starting_parameters={
                    "hhid": hhid,
                    "start_wave": start_wave,
                    "investment": investment,
                    "total_gain": total_gain,
                },
                memory={"context_token_limit": 4096, "summary_trigger_ratio": 0.6},
                hhid=hhid,
                start_wave=start_wave,
            )
        )
    return scenarios


def _seed_observed_history(agent: Agent, panel, health_min, health_max, obs_window: int):
    """
    Append observed waves to the agent history before LLM prediction begins.
    """
    if panel is None or panel.empty or agent.hhid is None:
        return 0

    start_wave = getattr(agent, "start_wave", 1)
    max_wave = start_wave + obs_window - 1
    rows = (
        panel[
            (panel["hhid5"] == agent.hhid)
            & (panel["survey_wave"] >= start_wave)
            & (panel["survey_wave"] <= max_wave)
        ]
        .sort_values("survey_wave")
        .to_dict("records")
    )

    added = 0
    for row in rows:
        wave = int(row.get("survey_wave", start_wave))
        wealth = float(row["wealth"]) if not pd.isna(row.get("wealth")) else 0.0
        health_raw = row.get("health_index")
        health = _scale_health(health_raw, health_min, health_max) or 0.0
        investment = float(row["investment_amount"]) if not pd.isna(row.get("investment_amount")) else 0.0
        total_gain = float(row["total_gain_from_baseline"]) if not pd.isna(row.get("total_gain_from_baseline")) else wealth

        agent.wealth = wealth
        agent.health = health
        agent.investment = investment
        agent.total_gain = total_gain

        # Refresh the first history entry and append subsequent waves.
        if added == 0 and wave == start_wave and agent.history:
            agent.history[0] = agent.to_dict()
        else:
            agent.history.append(agent.to_dict())
            added += 1
    return added


def _parse_hhid_list(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    hhids = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            hhids.append(int(part))
        except ValueError:
            print(f"Skipping non-numeric household id '{part}' from CLI.")
    return hhids or None


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the Balboni LLM simulation.")
    parser.add_argument("--sim-mode", choices=["llm", "mixed"], default=os.getenv("SIM_MODE", "llm"))
    parser.add_argument("--sim-steps", type=int, default=int(os.getenv("SIM_STEPS", "2")))
    parser.add_argument("--start-wave", type=int, default=int(os.getenv("START_WAVE", "1")))
    parser.add_argument("--obs-window", type=int, default=int(os.getenv("OBS_WINDOW", "2")))
    parser.add_argument("--agent-limit", type=int, default=int(os.getenv("AGENT_LIMIT", "0") or "0"))
    parser.add_argument(
        "--agent-hhids",
        type=str,
        default=os.getenv("AGENT_HHIDS", ""),
        help="Comma-separated list of household ids to simulate.",
    )
    parser.add_argument(
        "--balboni-zip",
        type=str,
        default=os.getenv("BALBONI_ZIP_PATH", BALBONI_DEFAULT_ZIP),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=os.getenv("PROGRESS", "1") != "0",
        help="Show tqdm progress bar for simulation steps.",
    )
    return parser.parse_args()


def _run_balboni_mixed(agent_configs, steps: int, data_zip: str):
    """
    Runs the mixed observed + simulated future method for each configured household.

    - Waves present in the Balboni panel are used as-is.
    - Beyond the last observed wave, the economic model generates future wealth paths.
    """
    from env.balboni_env import BalboniEnv

    params = EconomicParameters()
    for scenario in agent_configs:
        env = BalboniEnv(params=params, data_zip=data_zip)
        hhid = getattr(scenario, "hhid", None)
        state = env.reset(hhid)
        print(f"Household {env.current_hhid} | wave={env.current_wave} | state={state}")
        for _ in range(steps):
            next_state = env.step()
            if next_state is None:
                print(f"No further data for household {env.current_hhid} at wave {env.current_wave}.")
                break
            print(f"Household {env.current_hhid} | wave={env.current_wave} | state={next_state}")


def main(config_path: Optional[str] = None, steps: Optional[int] = None):
    """
    Entrypoint that wires the LLM-driven agents into the environment.
    """
    args = _parse_args()
    econ_model = EconomicModel(EconomicParameters())
    panel = _maybe_load_balboni_panel(args.balboni_zip)
    hhids_override = _parse_hhid_list(args.agent_hhids)
    agent_limit = args.agent_limit if args.agent_limit > 0 else None
    num_steps = steps or args.sim_steps
    start_wave = args.start_wave

    agent_configs = _build_agents_from_data(
        panel,
        start_wave=start_wave,
        agent_limit=agent_limit,
        hhids_override=hhids_override,
    )

    sim_mode = args.sim_mode.lower()

    # Mixed mode: consume observed window, then let the LLM predict future states.
    if sim_mode == "mixed":
        if panel is None:
            raise RuntimeError("Balboni data required for mixed mode; panel is empty.")
        obs_window = args.obs_window
        env = Environment(economic_model=econ_model)
        health_min = panel["health_index"].min(skipna=True)
        health_max = panel["health_index"].max(skipna=True)
        for config in agent_configs:
            agent = Agent.from_config(config)
            # Carry over observed investment/gain if present.
            agent.investment = config.starting_parameters.get("investment")
            agent.total_gain = config.starting_parameters.get("total_gain")
            _seed_observed_history(agent, panel, health_min, health_max, obs_window)
            env.add_agent(agent)
        env.time_step = obs_window  # start LLM predictions after observed window
        step_iter = range(num_steps)
        if args.progress and tqdm is not None:
            step_iter = tqdm(step_iter, total=num_steps, desc="Sim steps (mixed)")
        for _ in step_iter:
            env.run_step()
        return

    env = Environment(economic_model=econ_model)
    for config in agent_configs:
        agent = Agent.from_config(config)
        agent.investment = config.starting_parameters.get("investment")
        agent.total_gain = config.starting_parameters.get("total_gain")
        env.add_agent(agent)

    step_iter = range(num_steps)
    if args.progress and tqdm is not None:
        step_iter = tqdm(step_iter, total=num_steps, desc="Sim steps")
    for _ in step_iter:
        env.run_step()


if __name__ == "__main__":
    main()
