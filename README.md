# AI-Agentic-Simulation

LLM-driven multi-agent simulation with configurable households, per-agent memory (buffer + summaries), and a vLLM/OpenAI-compatible client.

## Prerequisites
- Python 3.12+
- vLLM (or another OpenAI-compatible server) serving your model, e.g. `Qwen/Qwen3-0.6B`
- (Optional) GPU with enough VRAM for the chosen model

## Setup
```bash
git clone <this-repo>
cd AI-Agentic-Simulation
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Serve the model with vLLM
Adjust `--model` to match what you host (defaults in code expect Qwen):
```bash
CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000
```
If you use a different model/base URL, set environment variables:
- `LLM_MODEL` / `LLM_SUMMARY_MODEL` – model ids vLLM serves
- `LLM_BASE_URL` – OpenAI-compatible endpoint (default `http://localhost:8000/v1`)
- `LLM_API_KEY` – if your server requires one (default `not-needed`)

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
