"""LLM client protocol and vendor implementations.

All LLM-calling modules depend on the LLMClient protocol via dependency
injection. No module directly instantiates any vendor SDK.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Optional, Protocol, runtime_checkable

from plotlint.core.config import LLMConfig
from plotlint.core.errors import LLMError


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM API calls.

    All modules depend on this, not on anthropic directly.
    Enables testing with MockLLMClient.
    """

    async def complete(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Return the text content of the LLM response."""
        ...

    async def complete_with_image(
        self,
        system: str,
        user: str,
        image_bytes: bytes,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Return the text content of the LLM response (vision)."""
        ...


class AnthropicClient:
    """LLMClient implementation using the Anthropic API.

    Owns its own auth and model defaults. LLMConfig provides
    vendor-neutral call parameters (retries, temperature, max_tokens).
    """

    def __init__(
        self,
        config: LLMConfig,
        api_key: str = "",
        default_model: str = "claude-sonnet-4-5-20250929",
        vision_model: str = "claude-sonnet-4-5-20250929",
    ) -> None:
        self.config = config
        self.default_model = default_model
        self.vision_model = vision_model
        # Lazy import: anthropic is an optional dependency
        try:
            import anthropic
        except ImportError as e:
            raise LLMError(
                "anthropic package not installed. "
                "Install with: pip install auto-dash[llm]"
            ) from e
        self._client = anthropic.AsyncAnthropic(api_key=api_key or None)

    async def complete(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.messages.create(
                    model=model or self.default_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return response.content[0].text
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2**attempt)
        raise LLMError(f"LLM call failed after {self.config.max_retries} retries: {last_error}")

    async def complete_with_image(
        self,
        system: str,
        user: str,
        image_bytes: bytes,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.messages.create(
                    model=model or self.vision_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": b64,
                                    },
                                },
                                {"type": "text", "text": user},
                            ],
                        }
                    ],
                )
                return response.content[0].text
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2**attempt)
        raise LLMError(f"LLM vision call failed after {self.config.max_retries} retries: {last_error}")


class GeminiClient:
    """LLMClient implementation using the Google Gemini API.

    Uses the google-genai SDK. Free tier available for testing.
    """

    def __init__(
        self,
        config: LLMConfig,
        api_key: str = "",
        default_model: str = "gemini-2.0-flash",
        vision_model: str = "gemini-2.0-flash",
    ) -> None:
        self.config = config
        self.default_model = default_model
        self.vision_model = vision_model
        try:
            from google import genai
        except ImportError as e:
            raise LLMError(
                "google-genai package not installed. "
                "Install with: pip install auto-dash[gemini]"
            ) from e
        self._client = genai.Client(api_key=api_key or None)

    async def complete(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        from google.genai import types

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.aio.models.generate_content(
                    model=model or self.default_model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2**attempt)
        raise LLMError(f"Gemini call failed after {self.config.max_retries} retries: {last_error}")

    async def complete_with_image(
        self,
        system: str,
        user: str,
        image_bytes: bytes,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        from google.genai import types

        last_error: Optional[Exception] = None
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.aio.models.generate_content(
                    model=model or self.vision_model,
                    contents=[image_part, user],
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2**attempt)
        raise LLMError(f"Gemini vision call failed after {self.config.max_retries} retries: {last_error}")
