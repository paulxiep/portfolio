"""Foundation configuration dataclasses.

LLMConfig and SandboxConfig are shared across plotlint and autodash.
All behavior-controlling parameters live here, not as magic numbers in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMConfig:
    """Vendor-neutral configuration for LLM API calls.

    Controls call parameters only. Auth (API keys) is handled
    by each client implementation, not here.
    """

    max_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for subprocess code execution."""

    timeout_seconds: int = 30
    max_memory_mb: int = 512
    allowed_imports: Optional[frozenset[str]] = None
