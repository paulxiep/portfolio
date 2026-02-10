"""Response parsing utilities for extracting code and JSON from LLM responses.

Shared across patcher, explorer, and charts modules.
Pure string manipulation — no external dependencies.
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_code_from_response(raw_response: str) -> str:
    """Extract a Python code block from an LLM response.

    Handles:
    - Markdown fenced code blocks (```python ... ```)
    - Generic fenced blocks (``` ... ```)
    - Plain code responses (no fences)
    - Multiple code blocks (takes the last one)

    Raises ValueError if no code block is found.
    """
    # Try fenced code blocks first (```python or ```)
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, raw_response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # If the entire response looks like code (has Python syntax indicators),
    # treat it as code
    stripped = raw_response.strip()
    code_indicators = ["import ", "def ", "class ", "for ", "if ", "print(", "plt.", "pd.", "="]
    if any(stripped.startswith(ind) or f"\n{ind}" in stripped for ind in code_indicators):
        return stripped

    raise ValueError("No Python code block found in LLM response")


def parse_json_from_response(raw_response: str) -> Any:
    """Extract and parse JSON from an LLM response.

    Handles:
    - Markdown fenced blocks (```json ... ```)
    - Plain JSON responses
    - JSON embedded in prose (extracts first valid JSON object/array)

    Raises ValueError if no valid JSON is found.
    """
    # Try fenced JSON blocks first
    pattern = r"```(?:json)?\s*\n(.*?)```"
    matches = re.findall(pattern, raw_response, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try parsing the entire response as JSON
    stripped = raw_response.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object or array embedded in prose
    for pattern in [r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", r"\[.*?\]"]:
        matches = re.findall(pattern, stripped, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    raise ValueError("No valid JSON found in LLM response")
