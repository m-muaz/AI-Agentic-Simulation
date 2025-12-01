# AI-Agentic-Simulation

LLM-driven multi-agent simulation with configurable households, per-agent memory (buffer + summaries), and a vLLM/OpenAI-compatible client.

## Prerequisites
- Python 3.12+
- vLLM (or another OpenAI-compatible server) serving your model, e.g. `Qwen/Qwen3-0.6B`
- (Optional) GPU with enough VRAM for the chosen model

## Setup
Firstly, make sure you have `uv` already installed on your system. If not install it via the following command:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone this repository and navigate into it:
```bash
git clone <this-repo>
cd AI-Agentic-Simulation
uv venv --python 3.12
source .venv/bin/activate && uv pip install -e .
```

## Serve the model with vLLM
Adjust `$MODEL_NAME$` to match what you host (defaults in code expect Qwen):
```bash
vllm serve $MODEL_NAME --port 8000 --host 0.0.0.0 > vllm.log 2>&1 & # Serve model in the background

```

## Configure agents
Edit `configs/agents.yaml` to define households:
- `id`, `name` – stable identifiers/persona labels
- `initial_wealth`, `initial_health` – starting state
- `household_history` – background blurb seeded into memory
- `starting_parameters` – arbitrary scenario metadata (e.g., dependents, priorities)
- `memory` – per-agent memory settings (`context_token_limit`, `summary_trigger_ratio`, etc.)

Override the config path via `AGENT_CONFIG_PATH=/path/to/custom.yaml`.

## Run the simulation
Default entry point (uses `configs/agents.yaml` and runs 2 steps):
```bash
python main.py
```

Dummy episodes of varying lengths to exercise memory summarization:
```bash
python scenarios/dummy_simulation.py --steps 2 5 10
```

## How it works (quick map)
- `main.py` – loads agent configs, builds `Environment`, runs steps.
- `src/ai_agent_simulation/agent.py` – per-agent prompt construction, state updates, memory integration.
- `src/ai_agent_simulation/memory.py` – LangChain buffer + summary memory with token-based triggers.
- `src/ai_agent_simulation/config.py` – YAML loader for agent scenarios.
- `src/ai_agent_simulation/llm_interface.py` – OpenAI-compatible client targeting vLLM.
- `src/ai_agent_simulation/tokenization.py` – token counting via HF tokenizers for context budgeting.
- `scenarios/dummy_simulation.py` – quick runner for multi-length episodes and memory inspection.

## Troubleshooting
- `ModuleNotFoundError: langchain...` – ensure dependencies are installed via `pip install -e .` (we pin to pre-1.0 layout).
- Repeated/identical LLM outputs – increase `temperature`/`top_p`, add penalties, or tweak prompts; ensure the server isn’t caching responses.
- Context overruns – lower `summary_trigger_ratio` or `context_token_limit` in agent memory config.
