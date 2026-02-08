"""Exception hierarchy for the AutoDash/plotlint project.

AutoDashError is the root. All exceptions inherit from it so callers
can catch broad categories or specific types.
"""


class AutoDashError(Exception):
    """Root exception for the entire project."""


# --- Shared errors ---


class ConfigError(AutoDashError):
    """Configuration errors: missing API key, invalid config values."""


class LLMError(AutoDashError):
    """LLM API call failures: network errors, rate limits, malformed responses."""


class SandboxError(AutoDashError):
    """Code execution failures in the subprocess sandbox."""


# --- plotlint-specific ---


class PlotlintError(AutoDashError):
    """Base for all plotlint errors."""


class RenderError(PlotlintError):
    """Rendering failures: syntax errors in chart code, timeout, no figure produced."""


class ExtractionError(PlotlintError):
    """Element extraction failures: unpickling failed, artist tree inaccessible."""


class PatchError(PlotlintError):
    """Patching failures: LLM returned invalid code, all retries exhausted."""


# --- autodash-specific ---


class PipelineError(AutoDashError):
    """Base for all autodash pipeline errors."""


class DataError(PipelineError):
    """Data loading or profiling failed."""


class ExplorationError(PipelineError):
    """Data exploration step failed after retries."""


class ChartGenerationError(PipelineError):
    """Chart code generation failures: invalid spec, code gen failed."""
