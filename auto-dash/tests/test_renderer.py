"""Tests for plotlint.renderer — MatplotlibRenderer and Renderer protocol."""

import matplotlib
matplotlib.use('Agg')

import pickle
import struct

import pytest

from plotlint.models import RenderStatus
from plotlint.renderer import (
    MatplotlibRenderer,
    Renderer,
    RendererBundle,
    matplotlib_bundle,
)


# --- Fixture: shared renderer instance ---


@pytest.fixture
def renderer():
    return MatplotlibRenderer(dpi=100, timeout_seconds=15)


# --- Simple chart code snippets ---

SIMPLE_BAR = """\
import matplotlib.pyplot as plt
plt.bar(['a', 'b', 'c'], [1, 2, 3])
plt.title('Test Bar Chart')
"""

SIMPLE_LINE = """\
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
plt.xlabel('X')
plt.ylabel('Y')
"""

SYNTAX_ERROR_CODE = "def bad("

RUNTIME_ERROR_CODE = "x = 1 / 0"

IMPORT_ERROR_CODE = "import nonexistent_module_xyz_123"

NO_FIGURE_CODE = "x = 1 + 1"

SLOW_CODE = "import time; time.sleep(30)"

CODE_WITH_SHOW = """\
import matplotlib.pyplot as plt
plt.bar([1, 2], [3, 4])
plt.show()
"""

CODE_WITH_CLOSE = """\
import matplotlib.pyplot as plt
plt.bar([1, 2], [3, 4])
plt.close()
"""


# --- Protocol conformance ---


class TestProtocol:
    def test_renderer_is_protocol_instance(self, renderer):
        assert isinstance(renderer, Renderer)

    def test_renderer_type(self, renderer):
        assert renderer.renderer_type == "matplotlib"


# --- Successful rendering ---


class TestRenderSuccess:
    def test_simple_bar_succeeds(self, renderer):
        result = renderer.render(SIMPLE_BAR)
        assert result.succeeded
        assert result.status == RenderStatus.SUCCESS
        assert result.png_bytes is not None
        assert len(result.png_bytes) > 0
        assert result.figure_data is not None
        assert len(result.figure_data) > 0

    def test_simple_line_succeeds(self, renderer):
        result = renderer.render(SIMPLE_LINE)
        assert result.succeeded

    def test_png_is_valid_png(self, renderer):
        """PNG bytes start with the PNG magic number."""
        result = renderer.render(SIMPLE_BAR)
        assert result.png_bytes[:8] == b'\x89PNG\r\n\x1a\n'

    def test_dpi_recorded(self, renderer):
        result = renderer.render(SIMPLE_BAR)
        assert result.dpi == 100

    def test_figsize_captured(self, renderer):
        result = renderer.render(SIMPLE_BAR)
        assert result.figsize is not None
        assert len(result.figsize) == 2
        assert all(dim > 0 for dim in result.figsize)

    def test_execution_time_tracked(self, renderer):
        result = renderer.render(SIMPLE_BAR)
        assert result.execution_time_ms >= 0


# --- Figure integrity (pickled figure is usable) ---


class TestFigureIntegrity:
    def test_figure_unpicklable(self, renderer):
        result = renderer.render(SIMPLE_BAR)
        fig = pickle.loads(result.figure_data)
        assert fig is not None

    def test_figure_has_axes(self, renderer):
        result = renderer.render(SIMPLE_BAR)
        fig = pickle.loads(result.figure_data)
        axes = fig.get_axes()
        assert len(axes) > 0

    def test_figure_children_traversable(self, renderer):
        """Validate that the artist tree is intact after pickling."""
        result = renderer.render(SIMPLE_BAR)
        fig = pickle.loads(result.figure_data)
        children = fig.get_children()
        assert len(children) > 0

    def test_bbox_thesis(self, renderer):
        """Core plotlint thesis: after render + unpickle, we can extract
        bounding boxes from tick labels via get_window_extent(renderer).

        If this test fails, the fundamental plotlint approach needs rethinking.
        """
        result = renderer.render(SIMPLE_BAR)
        fig = pickle.loads(result.figure_data)

        # Finalize layout (required for bbox calculation)
        fig.canvas.draw()
        canvas_renderer = fig.canvas.get_renderer()

        ax = fig.get_axes()[0]
        tick_labels = ax.get_xticklabels()
        assert len(tick_labels) > 0

        for label in tick_labels:
            bbox = label.get_window_extent(canvas_renderer)
            # Bbox should be non-degenerate (has width and height)
            assert bbox.width >= 0
            assert bbox.height >= 0


# --- Error handling ---


class TestRenderErrors:
    def test_syntax_error(self, renderer):
        result = renderer.render(SYNTAX_ERROR_CODE)
        assert not result.succeeded
        assert result.status == RenderStatus.SYNTAX_ERROR
        assert result.error_message is not None

    def test_runtime_error(self, renderer):
        result = renderer.render(RUNTIME_ERROR_CODE)
        assert not result.succeeded
        assert result.status == RenderStatus.RUNTIME_ERROR
        assert result.error_type == "ZeroDivisionError"

    def test_import_error(self, renderer):
        result = renderer.render(IMPORT_ERROR_CODE)
        assert not result.succeeded
        assert result.status == RenderStatus.IMPORT_ERROR

    def test_timeout(self):
        renderer = MatplotlibRenderer(timeout_seconds=2)
        result = renderer.render(SLOW_CODE)
        assert not result.succeeded
        assert result.status == RenderStatus.TIMEOUT

    def test_no_figure(self, renderer):
        result = renderer.render(NO_FIGURE_CODE)
        assert not result.succeeded
        assert result.status == RenderStatus.NO_FIGURE


# --- Defensive monkey-patching ---


class TestMonkeyPatching:
    def test_show_does_not_break(self, renderer):
        """plt.show() in user code should not block or cause errors."""
        result = renderer.render(CODE_WITH_SHOW)
        assert result.succeeded

    def test_close_does_not_lose_figure(self, renderer):
        """plt.close() in user code should not prevent figure capture."""
        result = renderer.render(CODE_WITH_CLOSE)
        assert result.succeeded
        assert result.png_bytes is not None


# --- DPI validation ---


class TestDPI:
    def test_png_dimensions_match_dpi(self):
        """PNG pixel dimensions should equal figsize_inches * dpi."""
        code = """\
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar([1, 2], [3, 4])
"""
        renderer = MatplotlibRenderer(dpi=100)
        result = renderer.render(code)
        assert result.succeeded

        # Parse PNG IHDR chunk for width/height
        # PNG structure: 8-byte signature, then chunks
        # IHDR chunk: 4-byte length, 4-byte type, 4-byte width, 4-byte height
        png = result.png_bytes
        width = struct.unpack('>I', png[16:20])[0]
        height = struct.unpack('>I', png[20:24])[0]

        expected_width = int(4 * 100)   # figsize_w * dpi
        expected_height = int(3 * 100)  # figsize_h * dpi
        assert width == expected_width
        assert height == expected_height


# --- RendererBundle ---


class TestRendererBundle:
    def test_matplotlib_bundle_creates_bundle(self):
        bundle = matplotlib_bundle()
        assert isinstance(bundle, RendererBundle)
        assert bundle.renderer_type == "matplotlib"
        assert isinstance(bundle.renderer, MatplotlibRenderer)
        assert bundle.extractor is None  # Until MVP.7

    def test_matplotlib_bundle_passes_config(self):
        bundle = matplotlib_bundle(dpi=150, timeout_seconds=60)
        assert bundle.renderer.dpi == 150
        assert bundle.renderer.timeout_seconds == 60
