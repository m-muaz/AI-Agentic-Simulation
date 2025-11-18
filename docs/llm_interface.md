# `llm_interface.py`

Defines helper functions to communicate with a local OpenAI-compatible LLM server.

## Design

- Uses the official `openai` Python client, configured with a custom `base_url` pointing to a local vLLM server.
- `api_key` is a placeholder string (`"not-needed"`) because most local setups don't require real authentication.

## Core Functions

- `get_client() -> OpenAI`
  - Lazily creates a singleton `OpenAI` client and caches it in a module-level `_client` variable.
  - Uses:
    - `base_url="http://localhost:8000/v1"`
    - `api_key="not-needed"`

- `get_llm_response(prompt: str) -> dict`
  - Sends a chat completion request to the model with:
    - Model name: `"Qwen/Qwen3-0.6B"` (adjust this to match your vLLM server).
    - A system message instructing the model to respond with JSON only.
    - The user message containing the constructed prompt.
    - `response_format={"type": "json_object"}` to request structured output (if supported).
    - `temperature=0.7` for some variability.
  - Prints the raw LLM output string.
  - Attempts to parse the content as JSON and returns a Python `dict`.
  - On error (connection issues, invalid JSON, etc.), logs a warning and returns an empty dict (`{}`).

## Configuration Notes

- Before running the simulation, ensure a vLLM or OpenAI-compatible server is running at `http://localhost:8000/v1` and serving a model named `"Qwen/Qwen3-0.6B"` (or update the code accordingly).
- For remote APIs, you would typically change `base_url`, provide a real `api_key`, and possibly tweak rate limiting and retries.

## Extension Ideas

- Parameterize model name, base URL, and temperature via environment variables or a config file.
- Add retry/backoff logic around the API call.
- Support non-JSON models by removing `response_format` and manually extracting JSON from text.
