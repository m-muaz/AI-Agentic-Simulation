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

