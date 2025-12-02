# experiments/run_balboni_simulation.py
from pathlib import Path
import sys

# Make sure top-level src is on path
repo_root = Path(__file__).resolve().parents[1]
src_root = repo_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

# Import the new env & dataset manager
from data_loaders.dataset_manager import DatasetManager
from envs.banerjee_duflo_env import BanerjeeDufloEnv

# Try to import user's economic model & visualizer from ai_agent_simulation package.
EconomicModel = None
Visualizer = None
try:
    # try likely package path: ai_agent_simulation.economic_model
    from ai_agent_simulation.economic_model import EconomicModel as _EM
    EconomicModel = _EM
except Exception:
    try:
        from ai_agent_simulation import economic_model as em
        EconomicModel = getattr(em, "EconomicModel", None)
    except Exception:
        pass

try:
    from ai_agent_simulation.visualizer import Visualizer as _V
    Visualizer = _V
except Exception:
    try:
        from ai_agent_simulation.visualize_model import Visualizer as _V2
        Visualizer = _V2
    except Exception:
        pass

def main():
    print("Building dataset (this will cache to data/balboni/processed/)...")
    dm = DatasetManager()
    panel = dm.get_panel()
    print("Panel shape:", panel.shape)

    print("Creating environment...")
    env = BanerjeeDufloEnv()

    # pick an example household
    hhids = panel["hhid5"].unique()
    sample_hhid = int(hhids[0])
    state = env.reset(household_id=sample_hhid, wave=1)
    print("Sample initial state:", state)

    if EconomicModel is None:
        print("EconomicModel class not found in ai_agent_simulation.*. Skipping model run.")
    else:
        print("Instantiating EconomicModel...")
        model = EconomicModel()
        # call a standard method if available (best-effort)
        if hasattr(model, "simulate_from_state"):
            print("Running simulate_from_state(...)")
            sim = model.simulate_from_state(state)
            print("Simulation output (sample):", sim)
        else:
            print("EconomicModel found but no 'simulate_from_state' method detected; adjust runner to call your API.")

    if Visualizer is None:
        print("Visualizer not found. If you'd like, I can adjust this runner to your visualizer API.")
    else:
        print("Creating visualizer and drawing sample plot...")
        viz = Visualizer()
        if hasattr(viz, "plot_panel"):
            viz.plot_panel(panel)
        elif hasattr(viz, "visualize"):
            viz.visualize(panel)
        else:
            print("Visualizer found but has no recognized plotting API. Please adapt.")

if __name__ == "__main__":
    main()
