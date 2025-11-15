import os
import json
from openai import OpenAI

_client = None

def get_client() -> OpenAI:
    """
    Initializes and returns the OpenAI client, configured for a local vLLM server.
    """
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="not-needed" # Placeholder
        )
    return _client


def get_llm_response(prompt: str) -> dict:
    """
    Gets a response from the local LLM.

    Args:
        prompt: The prompt to send to the LLM.

    Returns:
        A dictionary parsed from the LLM's JSON response.
    """
    try:
        client = get_client()
        response = client.chat.completions.create(
            # Note: Change the model name to match what our vLLM server is serving.
            model="Qwen/Qwen3-0.6B",
            messages=[
                {"role": "system", "content": "You are a component of a simulation. Respond with only JSON."},
                {"role": "user", "content": prompt}
            ],
            # Note: Not all local models support JSON mode.
            # may need to remove this and parse the JSON from a string response.
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        llm_output = response.choices[0].message.content
        print(llm_output)
        return json.loads(llm_output)
    except Exception as e:
        print(f"An error occurred while contacting the local LLM: {e}")
        print("Please ensure your vLLM server is running and accessible at http://localhost:8000/v1")
        return {}
