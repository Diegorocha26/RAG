"""
llm_client.py
-------------
Thin wrapper around LiteLLM so the rest of the codebase is provider-agnostic.
Any model string supported by LiteLLM works (openai/gpt-4o, anthropic/claude-opus-4, etc.)
"""

# TODO: check structured outputs implementation with pydantic
# TODO: max_tokens controls the number of output tokens of the model, this should scale with the number of questions that are looking to be generated.
#       change this in config.yaml, generator.py and in this file. also leave a note for the user because each model has different limits

import re
import json
import litellm
from typing import Any


def call_llm(
    prompt: str,
    system: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    **kwargs,
) -> str:
    """
    Send a prompt to the LLM and return the raw text response.
    Raises RuntimeError on failure.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]

    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    return response.choices[0].message.content


def extract_json(text: str) -> Any:
    """
    Robustly extract a JSON object or array from LLM output.
    Handles markdown code fences and extra prose around the JSON.
    """
    # TODO: this is good, but check if it can be replaced for pydantic structured outputs
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the first [...] or {...} block
    for pattern in (r"(\[.*\])", r"(\{.*\})"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract valid JSON from LLM output:\n{text[:500]}")


# Test LLM
if __name__ == "__main__":
    prompt = "Hello"
    system = "You're a helpfull and funny assistant."
    model = "openai/gpt-4.1-nano"
    temperature = 0.7
    max_tokens = 8192
    response = call_llm(prompt, system, model, temperature, max_tokens)
    print(response)
