"""
Utility helpers for estimating token counts using Hugging Face tokenizers.

This is useful when serving open-source models through vLLM: call `count_tokens`
before constructing a prompt to make sure you remain within the model's context
window. The default model name follows the `LLM_MODEL` environment variable so it
stays aligned with whatever checkpoint is currently hosted.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List, Union

from transformers import AutoTokenizer

DEFAULT_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-0.6B")


@lru_cache(maxsize=8)
def get_tokenizer(model_name: str = DEFAULT_MODEL):
    """
    Lazily load and cache a Hugging Face tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _normalize_inputs(text_or_messages: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(text_or_messages, str):
        return [text_or_messages]
    return [segment for segment in text_or_messages if segment]


def count_tokens(
    text_or_messages: Union[str, Iterable[str]],
    model_name: str = DEFAULT_MODEL,
) -> int:
    """
    Estimate token usage for arbitrary text using the specified tokenizer.

    Args:
        text_or_messages: Plain text or a list of message snippets to encode.
        model_name: Hugging Face model identifier, defaults to `LLM_MODEL`.

    Returns:
        Integer token count.
    """
    tokenizer = get_tokenizer(model_name)
    segments = _normalize_inputs(text_or_messages)
    total_ids = 0
    for chunk in segments:
        encoded = tokenizer(
            chunk,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        total_ids += len(encoded["input_ids"])
    return total_ids
