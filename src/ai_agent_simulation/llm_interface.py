import json
import os
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from openai import OpenAI

_client: Optional[OpenAI] = None
_langchain_clients: Dict[str, ChatOpenAI] = {}

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("LLM_API_KEY", "not-needed")
MODEL_NAME = os.getenv("LLM_MODEL", "Qwen3-0.6B")
SUMMARY_MODEL_NAME = os.getenv("LLM_SUMMARY_MODEL", MODEL_NAME)


def get_client() -> OpenAI:
    """
    Initializes and returns the OpenAI client, configured for a local vLLM server.
    """
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY  # Placeholder for local deployments
        )
    return _client


def get_langchain_llm(model: Optional[str] = None, temperature: float = 0.0) -> ChatOpenAI:
    """
    Returns a cached LangChain ChatOpenAI client for memory summarization.
    """
    cache_key = f"{model or SUMMARY_MODEL_NAME}:{temperature}"
    if cache_key not in _langchain_clients:
        _langchain_clients[cache_key] = ChatOpenAI(
            model=model or SUMMARY_MODEL_NAME,
            temperature=temperature,
            base_url=BASE_URL,
            api_key=API_KEY,
            max_retries=2,
        )
    return _langchain_clients[cache_key]


def get_llm_response(
    prompt: str,
    agent_id: Optional[str] = None,
    extra_messages: Optional[List[Dict[str, str]]] = None,
) -> dict:
    """
    Gets a response from the local LLM.

    Args:
        prompt: The prompt to send to the LLM.
        agent_id: Optional agent identifier to scope the system prompt.
        extra_messages: Optional message list to prepend (each with role/content).

    Returns:
        A dictionary parsed from the LLM's JSON response.
    """
    try:
        client = get_client()
        system_content = (
            "You simulate a single household in isolation. Always produce valid JSON "
            "and never reference other agents in the simulation."
        )
        if agent_id:
            system_content += f" You are currently role-playing agent {agent_id}."

        messages = extra_messages[:] if extra_messages else []
        messages.extend(
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        llm_output = response.choices[0].message.content
        print(llm_output)
        return json.loads(llm_output)
    except Exception as e:
        print(f"An error occurred while contacting the local LLM: {e}")
        print(f"Please ensure your vLLM server is running and accessible at {BASE_URL}")
        return {}
