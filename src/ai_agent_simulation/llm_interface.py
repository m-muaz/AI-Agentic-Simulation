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


def _strip_code_fences(text: str) -> str:
    """
    Remove common Markdown fences around JSON blobs (```json ... ```).
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    # Drop first and last fence lines if present.
    if lines:
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
    cleaned = "\n".join(lines).strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned


def _parse_llm_json(llm_output: str) -> Optional[dict]:
    """
    Try multiple strategies to recover a JSON object from the LLM output.
    """
    if not llm_output or not isinstance(llm_output, str):
        return None

    candidates = []
    stripped = llm_output.strip()
    candidates.append(stripped)

    fenced = _strip_code_fences(stripped)
    if fenced != stripped:
        candidates.append(fenced)

    brace_start = llm_output.find("{")
    brace_end = llm_output.rfind("}")
    if 0 <= brace_start < brace_end:
        candidates.append(llm_output[brace_start : brace_end + 1].strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def get_client() -> OpenAI:
    """
    Initializes and returns the OpenAI client, configured for a local vLLM server.
    """
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,  # Placeholder for local deployments
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
        system_content = """You are a household economic decision agent. 
                Your goal is to help the household increase its wealth over time,
                avoid falling into the low-return poverty trap, and manage risk.

                You will receive the household's current state each step.
                Based on this, you must choose an action that will influence 
                the household's economic future.Always produce valid JSON"""
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
            temperature=0.85,
        )
        message = response.choices[0].message
        # Prefer parsed content if the client provides it (newer OpenAI SDKs).
        llm_output = getattr(message, "parsed", None) or message.content
        parsed = _parse_llm_json(llm_output) if isinstance(llm_output, str) else llm_output
        if parsed is None:
            raw_text = "" if llm_output is None else str(llm_output)
            snippet = raw_text[:400] + ("..." if len(raw_text) > 400 else "")
            print("LLM returned unparsable JSON. Snippet:")
            print(snippet)
            return {}
        print(parsed)
        return parsed
    except Exception as e:
        print(f"An error occurred while contacting the local LLM: {e}")
        print(f"Please ensure your vLLM server is running and accessible at {BASE_URL}")
        return {}
