import pandas as pd
import numpy as np
import uuid
from typing import TYPE_CHECKING
# Import the Agent class using a relative import
from .agent import Agent 

# The Environment type hint needs a special import to avoid circular dependency
if TYPE_CHECKING:
    from .environment import Environment 

# --- CONFIGURATION ---
DATA_FILE_NAME = "PovertyTraps_replication_data.dta"
WEALTH_COLUMN = "pce_total" # Using Per Capita Total Consumption Expenditure
DEFAULT_INITIAL_HEALTH = 0.75 
MIN_AGENTS = 10  # Number of agents to create from the dataset sample
MAX_WEALTH_SIM_SCALE = 200.0 # Max simulated wealth for normalization

def load_and_prepare_agents(env: 'Environment'):
    """
    Loads data from the file, normalizes the wealth column, and creates agents 
    in the provided Environment object.
    """
    print(f"Loading data from {DATA_FILE_NAME}...")
    
    # 1. Load Data
    try:
        # Load the Stata file. 
        data = pd.read_stata(DATA_FILE_NAME)
    except Exception as e:
        print(f"ERROR: Could not load data file ({DATA_FILE_NAME}).")
        print("Please ensure the file is in the same directory as main.py and you have 'pandas' and 'pystata' or 'openpyxl' installed.")
        print(f"Error details: {e}")
        return

    # 2. Clean and Prepare Data
    data = data.dropna(subset=[WEALTH_COLUMN])
    data = data[pd.to_numeric(data[WEALTH_COLUMN], errors='coerce').notnull()]
    data = data[data[WEALTH_COLUMN] > 0] 

    # 3. Normalize Wealth Data
    # Normalize the raw wealth data to the simulation scale (0 to MAX_WEALTH_SIM_SCALE)
    max_raw_wealth = data[WEALTH_COLUMN].max()
    
    if max_raw_wealth == 0 or len(data) == 0:
        print("Error: No valid data found after cleaning. Cannot normalize.")
        return

    data['sim_wealth'] = (data[WEALTH_COLUMN] / max_raw_wealth) * MAX_WEALTH_SIM_SCALE

    # Use a sample of the data 
    num_rows = min(len(data), MIN_AGENTS)
    agent_data = data.head(num_rows) 
    
    print(f"Successfully created a sample of {num_rows} agent profiles.")

    # 4. Create Agents
    for index, row in agent_data.iterrows():
        initial_wealth = row['sim_wealth']
        
        env.add_agent(Agent(
            initial_wealth=initial_wealth,
            initial_health=DEFAULT_INITIAL_HEALTH,
            agent_id=f"Agent_{index}" 
        ))