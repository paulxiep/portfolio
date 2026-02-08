"""Tests for plotlint.core.config — config defaults and immutability."""

import pytest

from plotlint.core.config import LLMConfig, SandboxConfig


def test_llm_config_defaults():
    config = LLMConfig()
    assert config.max_retries == 3
    assert config.temperature == 0.0
    assert config.max_tokens == 4096


def test_llm_config_custom():
    config = LLMConfig(max_retries=5, temperature=0.5)
    assert config.max_retries == 5
    assert config.temperature == 0.5


def test_llm_config_frozen():
    config = LLMConfig()
    with pytest.raises(AttributeError):
        config.api_key = "changed"


def test_sandbox_config_defaults():
    config = SandboxConfig()
    assert config.timeout_seconds == 30
    assert config.max_memory_mb == 512
    assert config.allowed_imports is None


def test_sandbox_config_frozen():
    config = SandboxConfig()
    with pytest.raises(AttributeError):
        config.timeout_seconds = 60
