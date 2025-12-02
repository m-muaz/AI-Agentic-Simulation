from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage

from .llm_interface import get_langchain_llm
from .tokenization import count_tokens


@dataclass
class MemoryConfig:
    """
    Configuration for per-agent memory handling.
    """

    context_token_limit: int = 2048
    summary_trigger_ratio: float = 0.7
    summary_preamble: str = (
        "You maintain operational notes for a single household. "
        "Summarize the following exchange between the agent and the simulator "
        "highlighting key financial and health decisions."
    )
    max_recent_events: int = 8
    event_extraction_preamble: str = (
        "You will distill a simulation turn into a single concise event."
        " Focus on what changed (wealth/health/priorities/actions) and why."
        " Keep it under 250 characters."
    )

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "MemoryConfig":
        if not data:
            return cls()
        return cls(
            context_token_limit=int(data.get("context_token_limit", cls.context_token_limit)),
            summary_trigger_ratio=float(data.get("summary_trigger_ratio", cls.summary_trigger_ratio)),
            summary_preamble=data.get("summary_preamble", cls.summary_preamble),
            max_recent_events=int(data.get("max_recent_events", cls.max_recent_events)),
            event_extraction_preamble=data.get("event_extraction_preamble", cls.event_extraction_preamble),
        )


class AgentMemory:
    """
    Wraps LangChain buffer & summary memories to keep per-agent context bounded.
    """

    def __init__(self, agent_id: str, config: Optional[MemoryConfig] = None):
        self.agent_id = agent_id
        self.config = config or MemoryConfig()
        self.buffer_memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self.summary_llm = get_langchain_llm()
        prompt_template = PromptTemplate(
            input_variables=["summary", "new_lines"],
            template=(
                f"{self.config.summary_preamble}\n\n"
                "Current running summary:\n{summary}\n\n"
                "New observations:\n{new_lines}\n\n"
                "Consolidated summary:"
            ),
        )
        self.summary_memory = ConversationSummaryMemory(
            llm=self.summary_llm,
            memory_key="summary",
            return_messages=False,
            prompt=prompt_template,
        )
        self.summary_text: str = ""
        self.events: List[str] = []

    def bootstrap(self, initial_summary: Optional[str]):
        """
        Seed the summary store with prior household information.
        """
        if initial_summary:
            self.summary_text = initial_summary.strip()

    def record_interaction(self, prompt: str, response: Dict):
        event_text = self._extract_event(prompt, response)
        if event_text:
            self.events.append(event_text)
            # Keep buffer memory aligned with events so summarizer can use it.
            self.buffer_memory.chat_memory.add_ai_message(event_text)
        # Clip recent events to avoid unbounded growth between summaries.
        if len(self.events) > self.config.max_recent_events:
            self.events = self.events[-self.config.max_recent_events :]
        self._maybe_summarize()

    def reset_buffer(self):
        self.buffer_memory.chat_memory.clear()

    def _messages_to_string(self, messages: List[BaseMessage]) -> str:
        rendered = []
        for message in messages:
            role = getattr(message, "type", "user")
            rendered.append(f"{role.upper()}: {message.content}")
        return "\n".join(rendered).strip()

    def _buffer_token_count(self) -> int:
        if not self.events:
            return 0
        return count_tokens(self.events)

    def _maybe_summarize(self):
        token_threshold = int(self.config.context_token_limit * self.config.summary_trigger_ratio)
        current_tokens = self._buffer_token_count()
        if current_tokens < token_threshold or not self.buffer_memory.chat_memory.messages:
            return

        messages = self.buffer_memory.chat_memory.messages
        self._summarize_messages(messages)

    def render_context(self) -> Dict[str, str]:
        """
        Returns text snippets for prompt construction.
        """
        buffer_messages = self.buffer_memory.chat_memory.messages
        buffer_section = "\n".join(self.events) if self.events else "No recent exchanges."
        summary_section = self.summary_text or "No summary yet."
        return {
            "summary": summary_section,
            "recent_events": buffer_section,
        }

    def export_state(self) -> Dict[str, str]:
        """
        Expose memory state for analytics.
        """
        ctx = self.render_context()
        return {
            "summary": ctx["summary"],
            "recent_events": ctx["recent_events"],
            "events": list(self.events),
        }

    def context_usage(self) -> Dict[str, float]:
        """
        Returns token usage statistics for buffer/summary relative to the configured limit.
        """
        buffer_tokens = self._buffer_token_count()
        summary_tokens = count_tokens(self.summary_text) if self.summary_text else 0
        limit = max(1, self.config.context_token_limit)
        percent = min(100.0, (buffer_tokens + summary_tokens) / limit * 100.0)
        return {
            "buffer_tokens": buffer_tokens,
            "summary_tokens": summary_tokens,
            "context_token_limit": limit,
            "percent_of_limit": percent,
        }

    def force_summarize(self):
        """
        Immediately summarizes current buffer regardless of token size.
        """
        messages = self.buffer_memory.chat_memory.messages
        if not messages:
            return
        self._summarize_messages(messages)

    def _summarize_messages(self, messages: List[BaseMessage]):
        new_summary = self.summary_memory.predict_new_summary(messages, self.summary_text or "")
        self.summary_text = new_summary.strip()
        self.reset_buffer()
        self.events = []

    def _extract_event(self, prompt: str, response: Dict) -> Optional[str]:
        """
        Use the LLM to distill the prompt/response into a single important event line.
        """
        # If the response contains wealth/health/rationale, prefer those.
        if response and "wealth" in response and "health" in response:
            rationale = response.get("rationale", "")
            return (
                f"Wealth {response.get('wealth')} Health {response.get('health')} "
                f"Rationale: {rationale}"
            ).strip()

        try:
            extraction_prompt = (
                f"{self.config.event_extraction_preamble}\n\n"
                f"Prompt:\n{prompt}\n\n"
                f"Response:\n{json.dumps(response)}\n\n"
                "Return a single sentence capturing the most important decision/change."
            )
            return self.summary_llm.predict(extraction_prompt).strip()
        except Exception:
            return None
