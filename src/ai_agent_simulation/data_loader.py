'''
This module uses common variable name conventions from poverty 
trap studies: A_w1 for Productive Assets (k) and bmi_w1 for 
Health (h) (Body Mass Index). It loads the data, cleans it, 
and samples initial states for the agents.
'''


import pandas as pd
import numpy as np
import pyreadstat
from typing import Tuple, List

# Define the data source file and assumed variable names
DTA_FILE = "PovertyTraps_replication_data.dta"
# Assumed variables based on common conventions and paper context:
ASSET_VAR = 'A_w1'      # Productive Assets (k) at Wave 1 (Baseline)
HEALTH_VAR = 'bmi_w1'   # Health/Human Capital Proxy (h) at Wave 1

def load_initial_agent_states(n_agents: int) -> List[Tuple[float, float]]:
    """
    Loads initial wealth (k) and health (h) distributions from the empirical data.

    Args:
        n_agents: The number of agents to sample for the simulation.

    Returns:
        A list of tuples [(initial_wealth, initial_health), ...] or fallback states on error.
    """
    try:
        # Requires pyreadstat to be installed
        df, _ = pyreadstat.read_dta(DTA_FILE)
        
        # 1. Select the assumed columns for wealth and health
        df = df[[ASSET_VAR, HEALTH_VAR]].copy()

        # 2. Clean and prepare data
        df.dropna(inplace=True)
        # Filter out extreme values (e.g., zero or negative assets)
        df = df[df[ASSET_VAR] > 0] 

        # 3. Normalize Health (BMI) to a 0 to 1 range
        # Assumed typical BMI range for low-income populations: 15.0 to 35.0
        min_bmi = 15.0
        max_bmi = 35.0
        df['normalized_health'] = np.clip(
            (df[HEALTH_VAR] - min_bmi) / (max_bmi - min_bmi), 
            0.05, # Set a small minimum health floor
            1.0
        )
        
        # Sample the requested number of agents (with replacement if needed)
        sampled_df = df.sample(n=n_agents, replace=True)

        initial_states = list(zip(
            sampled_df[ASSET_VAR].values, 
            sampled_df['normalized_health'].values
        ))
        
        print(f"Successfully loaded and sampled {n_agents} initial states from {DTA_FILE}")
        return initial_states

    except FileNotFoundError:
        print(f"Error: The data file {DTA_FILE} was not found.")
    except KeyError:
        print(f"Error: Could not find one or both columns ('{ASSET_VAR}', '{HEALTH_VAR}') in {DTA_FILE}.")
        print("Falling back to default fixed initial states.")
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        
    # Fallback to fixed initial states if loading fails
    return [
        (100.0, 0.8), (50.0, 0.6), (80.0, 0.7), (120.0, 0.9), (30.0, 0.5), 
        (75.0, 0.75), (55.0, 0.65), (95.0, 0.85), (25.0, 0.4), (110.0, 0.8)
    ][:n_agents]