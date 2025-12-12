# AI-Agentic-Simulation
Empty repository for AI Agentic Sumulation

Data:

Download the dataverse_files.zip into data/balboni directory.
`data/balboni/dataverse_files.zip`

From the top of the repo, run the following:
``` python experiments/run_balboni_simulation.py ```

If imports fail for your models, open experiments/run_balboni_simulation.py and change the import lines:

`
# e.g. replace:
from ai_agent_simulation.economic_model import EconomicModel
# with the actual module path you have, e.g.:
from src.ai_agent_simulation.economic_model import EconomicModel

`

## Run the LLM simulation seeded by Balboni data
- Place the Balboni Dataverse zip at `data/balboni/dataverse_files.zip` (already expected by default).
- Install the project (`pip install -e .`) and run `python main.py`. The runner will load the Balboni panel, auto-assign households to each agent if `hhid` is not set, and seed starting wealth/health from wave 1.
- To target specific households, add `hhid: <number>` and optional `start_wave: <1-5>` to each entry in `configs/agents.yaml`.
- Environment variables:
  - `BALBONI_ZIP_PATH`: custom path to the Dataverse zip.
  - `USE_BALBONI_DATA=0`: disable Balboni seeding and fall back to manual initial values.
  - `SIM_STEPS`: number of simulation steps (default 2).
  - `SIM_MODE=mixed`: use the mixed observed + simulated future runner (Balboni panel for an observed window, then LLM predicts future states). Default `SIM_MODE=llm` keeps the LLM-driven agents.
  - `OBS_WINDOW`: how many observed waves to feed to the LLM before it starts predicting (default 2).
- Experiment phases (toggle the experiment design):
  - Phase 1 (savings/risk only): `python main.py --experiment-phase 1`
  - Phase 2 (coping actions + health effort): `python main.py --experiment-phase 2`
  - Phase 3 (labor/sector income; optional structural data): `python main.py --experiment-phase 3 --structural-data "path/to/PovertyTraps_structural.dta"`
  - You can also set env vars: `EXPERIMENT_PHASE=1|2|3` and `STRUCTURAL_DATA_PATH=...` (for phase 3).

## Visualize simulation vs original data
- Install plotting dependency if needed: `pip install matplotlib` (already listed in `pyproject.toml`).
- After running simulations (logs in `logs/agent_*.jsonl`), generate plots comparing simulated trajectories to Balboni observations:
  ```bash
  python scripts/plot_sim_vs_data.py --log-dir logs --save-dir plots --agent-limit 3
  ```
  Options:
  - `--agent-hhids 101,202`: plot specific households if present in logs/data.
  - `--balboni-zip path/to/dataverse_files.zip`: custom data source.
  - `--agent-limit N`: cap number of households plotted (defaults to 3).
  Outputs go to `plots/`, with per-household overlays and an aggregate simulation summary.
  - `sim_all_agents.png`: all simulated agents plotted together (wealth and health) with a legend.
  - `sim_aggregate.png`: mean simulated wealth/health over steps (aggregating across agents).

## Evaluate simulation performance
- Produce quick metrics/plots from the simulation logs (and optional MAE vs Balboni observations):
  ```bash
  python scripts/evaluate_sim.py --log-dir logs --save-dir eval --balboni-panel data/balboni/processed/balboni_panel.csv --start-wave 1
  ```
  Outputs go to `eval/`:
  - `per_agent_metrics.csv`: final wealth/health, deltas, volatility per agent.
  - `aggregate_metrics.csv`: mean/median wealth, deltas, variability across agents.
  - `mae_vs_observed.csv` (if Balboni zip provided and matches exist): MAE of wealth/health vs observed waves (aligned as wave = start_wave + step).
  - `wealth_distribution.csv`: histogram bins with simulated vs observed wealth counts/densities.
  - `health_by_wave.csv`: mean simulated vs observed (scaled) health by wave for matched pairs.
  - Plots: `eval_vs_observed_means.png` (mean simulated vs observed wealth/health by wave) and `eval_vs_observed_scatter.png` (paired scatter plots). Requires matplotlib (now a hard dependency for evaluation).
