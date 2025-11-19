from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import tiktoken
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

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "MemoryConfig":
        if not data:
            return cls()
        return cls(
            context_token_limit=int(data.get("context_token_limit", cls.context_token_limit)),
            summary_trigger_ratio=float(data.get("summary_trigger_ratio", cls.summary_trigger_ratio)),
            summary_preamble=data.get("summary_preamble", cls.summary_preamble),
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
        # self.encoder = tiktoken.get_encoding("cl100k_base")

    def bootstrap(self, initial_summary: Optional[str]):
        """
        Seed the summary store with prior household information.
        """
        if initial_summary:
            self.summary_text = initial_summary.strip()

    def record_interaction(self, prompt: str, response: Dict):
        self.buffer_memory.chat_memory.add_user_message(prompt)
        self.buffer_memory.chat_memory.add_ai_message(json.dumps(response))
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
        messages = self.buffer_memory.chat_memory.messages
        if not messages:
            return 0
        segments = [f"{getattr(msg, 'type', 'user')}: {msg.content}" for msg in messages]
        return count_tokens(segments)
        # content = self._messages_to_string(messages)
        # return len(self.encoder.encode(content))
        

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
        buffer_section = self._messages_to_string(buffer_messages) if buffer_messages else "No recent exchanges."
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
