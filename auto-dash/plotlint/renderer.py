"""Renderer protocol and matplotlib implementation.

Renderer protocol defines the interface for chart renderers.
MatplotlibRenderer executes matplotlib code in a subprocess sandbox,
captures the Figure object (pickled) and PNG bytes.

RendererBundle pairs a Renderer with its matching Extractor to prevent
cross-framework mismatches.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from plotlint.core.sandbox import ExecutionResult, ExecutionStatus, execute_code
from plotlint.models import RenderResult, RenderStatus


# --- Status mapping (ExecutionStatus → RenderStatus) ---

_STATUS_MAP: dict[ExecutionStatus, RenderStatus] = {
    ExecutionStatus.SYNTAX_ERROR: RenderStatus.SYNTAX_ERROR,
    ExecutionStatus.RUNTIME_ERROR: RenderStatus.RUNTIME_ERROR,
    ExecutionStatus.TIMEOUT: RenderStatus.TIMEOUT,
    ExecutionStatus.IMPORT_ERROR: RenderStatus.IMPORT_ERROR,
}


# --- Renderer protocol ---


@runtime_checkable
class Renderer(Protocol):
    """Protocol for chart renderers.

    OCP: New renderers (Plotly, Altair) implement this protocol.
    The convergence loop depends on this interface, not concrete classes.
    """

    @property
    def renderer_type(self) -> str:
        """Identifier for this renderer (e.g. 'matplotlib', 'plotly')."""
        ...

    def render(self, code: str) -> RenderResult:
        """Execute chart code and return the render result.

        Args:
            code: Complete, self-contained Python script that produces a chart.

        Returns:
            RenderResult with PNG bytes and optionally figure data.
        """
        ...


# --- Matplotlib renderer ---


@dataclass
class MatplotlibRenderer:
    """Render matplotlib charts in a subprocess with Agg backend.

    Wraps user code with Agg backend setup and figure capture,
    delegates subprocess execution to plotlint.core.sandbox.execute_code().
    """

    dpi: int = 100
    timeout_seconds: int = 30

    @property
    def renderer_type(self) -> str:
        return "matplotlib"

    def render(self, code: str) -> RenderResult:
        """Execute a matplotlib script in a subprocess.

        The wrapped code:
        1. Sets matplotlib backend to Agg
        2. Monkey-patches plt.show/plt.close to prevent figure loss
        3. Executes the user code
        4. Captures plt.gcf(), verifies it has axes
        5. Pickles the Figure object and saves PNG to buffer
        6. Returns both via sandbox __result__ mechanism
        """
        wrapped = self._prepare_worker_code(code)
        exec_result = execute_code(wrapped, timeout_seconds=self.timeout_seconds)
        return self._to_render_result(exec_result)

    def _prepare_worker_code(self, user_code: str) -> str:
        """Wrap user code with Agg backend setup and figure capture.

        Injects matplotlib instrumentation before and after user code.
        Does NOT modify the user's plotting logic.

        Preamble and postamble are dedented separately from user code
        to avoid indentation conflicts with multi-line user code.
        """
        preamble = textwrap.dedent("""\
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import pickle
            import io

            # Prevent figure loss from user code calling show/close
            plt.show = lambda *_args, **_kwargs: None
            plt.close = lambda *_args, **_kwargs: None

        """)

        postamble = textwrap.dedent(f"""\

            # Capture the current figure
            _fig = plt.gcf()
            if not _fig.get_axes():
                __result__ = {{"status": "no_figure"}}
            else:
                _fig.set_dpi({self.dpi})
                _figure_bytes = pickle.dumps(_fig)
                _buf = io.BytesIO()
                _fig.savefig(_buf, format='png', dpi={self.dpi})
                _png_bytes = _buf.getvalue()
                __result__ = {{
                    "status": "success",
                    "figure": _figure_bytes,
                    "png": _png_bytes,
                    "figsize": tuple(_fig.get_size_inches().tolist()),
                }}
        """)

        return preamble + user_code + postamble

    def _to_render_result(self, exec_result: ExecutionResult) -> RenderResult:
        """Map sandbox ExecutionResult to plotlint RenderResult."""
        # Error path: sandbox reported a non-success status
        if exec_result.status != ExecutionStatus.SUCCESS:
            return RenderResult(
                status=_STATUS_MAP.get(
                    exec_result.status, RenderStatus.RUNTIME_ERROR
                ),
                error_message=exec_result.error_message,
                error_type=exec_result.error_type,
                execution_time_ms=exec_result.execution_time_ms,
                dpi=self.dpi,
            )

        # Success but no __result__ captured (shouldn't happen with correct wrapper)
        rv = exec_result.return_value
        if rv is None:
            return RenderResult(
                status=RenderStatus.RUNTIME_ERROR,
                error_message="Wrapper did not set __result__",
                execution_time_ms=exec_result.execution_time_ms,
                dpi=self.dpi,
            )

        # Code ran but produced no figure
        if rv.get("status") == "no_figure":
            return RenderResult(
                status=RenderStatus.NO_FIGURE,
                error_message="Code executed but produced no figure",
                execution_time_ms=exec_result.execution_time_ms,
                dpi=self.dpi,
            )

        # Full success
        return RenderResult(
            status=RenderStatus.SUCCESS,
            png_bytes=rv["png"],
            figure_data=rv["figure"],
            execution_time_ms=exec_result.execution_time_ms,
            dpi=self.dpi,
            figsize=rv.get("figsize"),
        )


# --- Renderer-Extractor pairing ---


@dataclass(frozen=True)
class RendererBundle:
    """Pairs a Renderer with its matching Extractor.

    Prevents mismatches (e.g., matplotlib renderer with Plotly extractor).
    The convergence loop receives a bundle, not loose components.
    """

    renderer: Renderer
    extractor: Any  # Extractor protocol (MVP.7)
    renderer_type: str  # "matplotlib" or "plotly"


def matplotlib_bundle(
    dpi: int = 100,
    timeout_seconds: int = 30,
) -> RendererBundle:
    """Create a matplotlib renderer+extractor bundle.

    Factory function. Called by build_convergence_graph() when
    no explicit bundle is provided.

    extractor is None until MVP.7 provides MatplotlibExtractor.
    """
    return RendererBundle(
        renderer=MatplotlibRenderer(dpi=dpi, timeout_seconds=timeout_seconds),
        extractor=None,
        renderer_type="matplotlib",
    )
