"""Tests for plotlint.core.errors — error hierarchy and isinstance checks."""

from plotlint.core.errors import (
    AutoDashError,
    ChartGenerationError,
    ConfigError,
    DataError,
    ExplorationError,
    ExtractionError,
    LLMError,
    PatchError,
    PipelineError,
    PlotlintError,
    RenderError,
    SandboxError,
)


def test_autodash_error_is_base():
    assert issubclass(AutoDashError, Exception)


def test_shared_errors_inherit_from_root():
    for cls in [ConfigError, LLMError, SandboxError]:
        assert issubclass(cls, AutoDashError)


def test_plotlint_errors_inherit_from_plotlint_error():
    for cls in [RenderError, ExtractionError, PatchError]:
        assert issubclass(cls, PlotlintError)
        assert issubclass(cls, AutoDashError)


def test_pipeline_errors_inherit_from_pipeline_error():
    for cls in [DataError, ExplorationError, ChartGenerationError]:
        assert issubclass(cls, PipelineError)
        assert issubclass(cls, AutoDashError)


def test_catch_broad_category():
    """Catching AutoDashError catches all specific errors."""
    try:
        raise RenderError("test")
    except AutoDashError:
        pass  # should be caught


def test_error_message():
    err = LLMError("API timeout")
    assert str(err) == "API timeout"
