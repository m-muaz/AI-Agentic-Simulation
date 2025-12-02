# Brainstorm Log

## Objectives
- Validate each agent's LLM response path and guard against regressions.
- Introduce a reusable memory module so agents can recall and summarize their trajectories for downstream analysis.

## 1. LLM Response Testing Strategies
1. **Structured Logging & Telemetry**
   - Capture every prompt/response pair with timestamps, agent IDs, latency, token counts, and whether the JSON schema was valid.
   - Store logs locally (JSONL/SQLite) so we can replay or audit specific runs.
2. **Schema & Constraint Validation**
   - Add a JSON Schema (wealth ≥ 0, 0 ≤ health ≤ 1, optional metadata) and validate responses before mutating state; surface metrics on schema failures.
3. **Deterministic Contract Tests (Mocked LLM)**
   - Inject a fake `llm_interface` during tests to return canned responses for multiple scenarios (happy path, invalid JSON, missing keys, out-of-range values).
   - Enables CI to cover branching logic without hitting real GPUs.
4. **Replay/Regression Harness**
   - Record real responses from a “golden” run, then feed them back in offline mode to verify we produce identical histories.
5. **Load & Chaos Testing**
   - Run many agents/steps with rate-limit simulators to ensure we handle throttling, high latency, or partial failures gracefully.
6. **Evaluation Metrics**
   - Define KPIs per agent (wealth delta, health variance, JSON validity rate) and alert when they deviate from expected bands.

## 2. Memory Module Brainstorm
### Requirements
- Maintain per-agent memory beyond the raw history list.
- Provide fast summarization of actions to feed future prompts + analytics.
- Prefer leveraging a maintained library that already powers production LLM apps.

### Candidate Approaches
1. **LangChain Memory Components**
   - Use `ConversationBufferMemory` + `ConversationSummaryMemory` (or the structured `DynamicStructuredToolMemory`).
   - Pros: battle-tested, pluggable, easy JSON serialization, integrates with OpenAI-compatible backends.
   - Cons: pulls in LangChain dependency; may need adapters to fit our simple agent loop.
2. **LlamaIndex Chat Memory**
   - `ChatMemoryBuffer` + `SummaryMemory` to maintain event logs plus periodic TL;DRs using lightweight prompts.
   - Pros: strong summarization utilities; plays well with vector stores if we later need retrieval.
   - Cons: Heavier dependency; more opinionated architecture.
3. **MemGPT / AutoGen Memories**
   - Provide advanced hierarchical “short/long-term” memory with eviction policies.
   - Pros: closer to production multi-agent systems; built-in summarization & persistence.
   - Cons: Overkill for small sim; larger learning curve.
4. **Custom Minimal Module (inspired by above)**
   - Implement a simple `MemoryStore` that keeps structured events, rolling summaries, and stats; optionally call LangChain summarizers under the hood.
   - Pros: keeps footprint low, tailored to our state schema.
   - Cons: more maintenance unless we leverage existing library functions.

### Recommended Direction
- Adopt LangChain’s memory primitives per agent:
  1. Store raw events in `ConversationBufferMemory` keyed by agent_id.
  2. Periodically summarize with `ConversationSummaryMemory` to keep prompt context short; persist both buffer + summary (JSON) for analysis.
  3. Expose a thin adapter class (e.g., `AgentMemory`) wrapping LangChain objects so the rest of the codebase stays library-agnostic.
- For analytics, dump both the structured history and generated summaries to disk after each simulation run.

## 3. Proposed Next Actions
1. Extend `llm_interface` to support dependency injection (pass fake client) and add schema validation helpers.
2. Build a `tests/test_agent_responses.py` module that mocks the LLM to cover success/failure/edge cases.
3. Add runtime logging hooks (JSONL) capturing prompt/response per agent for replay.
4. Introduce a new `memory.py` with an `AgentMemory` adapter that internally uses LangChain’s buffer + summary memories.
5. Update `Agent` to read/write via `AgentMemory` (append state transitions, fetch summarized context for prompts, expose summary for reports).
6. Document the workflow in `docs/memory.md` and update `README` with the new testing & memory tooling.

## 4. Detailed Plan for Current Implementation Tasks

### 4.1 LangChain Conversation Buffer + Summary Triggers
- **Per-agent memory stack**: wrap LangChain’s `ConversationBufferMemory` and `ConversationSummaryMemory` inside `AgentMemory`.
- **Trigger logic**:
  - Track approximate token usage via LangChain’s `get_buffer_string(return_messages=True)` + tiktoken counter, or rely on vLLM usage metadata if exposed.
  - When buffer token count exceeds configurable thresholds (e.g., 70% of target model context window), invoke `ConversationSummaryMemory.predict_new_summary` to condense old exchanges and truncate the buffer.
  - Also trigger summarization when the LLM response quality degrades (schema failure spike) to simulate “LLM forgetting”.
- **Outputs**: store both rolling summaries and raw snippets in structured form so analytics can reconstruct narratives.

### 4.2 Dummy Simulation Test Harness
- Build a `scenarios/dummy_runs.py` (or pytest parametrized test) that:
  1. Seeds a deterministic fake LLM returning scripted responses.
  2. Runs simulations of varying episode lengths (short, medium, long) to ensure memory resets/fire at the right thresholds.
  3. Asserts that summaries are produced only when expected and that health/wealth updates survive long runs without cross-contamination.
- Later swap the fake LLM for the real vLLM server to spot-check behavior end-to-end.

### 4.3 Configurable Agent Initial State
- Introduce a config file (`configs/agents.yaml`) describing per-agent metadata: household history, demographic fields, starting wealth/health, scenario tags.
- Extend `Agent` (or a factory) to accept this structured config, so `main.py` can load scenarios without code edits.
- Persist memories alongside config metadata to correlate later analysis with original assumptions.

### 4.4 Context Isolation Between Agents
- Always instantiate independent `AgentMemory` objects per agent; never reuse LangChain memory instances across agents.
- Before calling the LLM for an agent, construct the chat payload exclusively from that agent’s buffer + summary, ensuring no shared context.
- Optionally reset the underlying LangChain buffer after each response while retaining the persisted summary, preventing bleed-over even if the LLM client caches conversations.
- Consider tagging each request with `agent_id` in the system prompt as a further safeguard.

### 4.5 Configuration & Monitoring Hooks
- Add settings to `pyproject`/`config.toml` (or environment variables) for: context-window threshold, summary prompt template, dummy-run lengths.
- Expose instrumentation counters (e.g., `memory_summaries_total`, `context_resets_total`) so we can verify in logs/metrics that the safeguards fire during long simulations.

## 5. Changes Implemented (current state)
- Hugging Face tokenizer-based token counting (`tokenization.py`) wired into memory context usage.
- Prompts trimmed to use only recent notable events (long-term summaries removed from prompts to reduce context size).
- Importance-first memory: each turn distills a concise event (wealth/health/rationale) instead of logging full prompt/response; summaries roll up these events and clear buffers.
- Per-step logging with rationale, context usage, and memory snapshots to `logs/agent_<id>.jsonl` plus separate notable-event audits in `logs/events_agent_<id>.jsonl`.
- YAML agent configs remain the scenario source; environment variables still control LLM endpoint/model; vLLM/OpenAI-compatible client unchanged.
