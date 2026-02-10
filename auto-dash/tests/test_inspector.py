"""Tests for inspector orchestration."""

import pickle
from unittest.mock import Mock
from plotlint.inspector import inspect, inspect_from_figure
from plotlint.elements import ElementMap, ElementInfo, ElementCategory
from plotlint.geometry import BoundingBox
from plotlint.models import Issue, DefectType, Severity


def test_inspect_calls_all_checks(monkeypatch):
    """Verify inspector queries registry and runs all checks."""
    from plotlint import inspector as inspector_module

    # Mock checks
    check1_called = False
    check2_called = False

    class TestCheck1:
        name = "test_check1"
        def __call__(self, elements):
            nonlocal check1_called
            check1_called = True
            return []

    class TestCheck2:
        name = "test_check2"
        def __call__(self, elements):
            nonlocal check2_called
            check2_called = True
            return []

    # Mock get_registered_checks in the inspector module
    def mock_get_checks():
        return {
            "test_check1": TestCheck1(),
            "test_check2": TestCheck2(),
        }

    monkeypatch.setattr(inspector_module, "get_registered_checks", mock_get_checks)

    elements = ElementMap([], BoundingBox(0, 0, 100, 100))
    result = inspect(elements)

    assert check1_called
    assert check2_called
    assert result.score == 1.0  # No issues


def test_inspect_aggregates_issues(monkeypatch):
    """Verify inspector aggregates issues from multiple checks."""
    from plotlint import inspector as inspector_module

    class TestCheck1:
        name = "test_check1"
        def __call__(self, elements):
            return [Issue(DefectType.LABEL_OVERLAP, Severity.HIGH, "test1", "fix1")]

    class TestCheck2:
        name = "test_check2"
        def __call__(self, elements):
            return [Issue(DefectType.ELEMENT_CUTOFF, Severity.MEDIUM, "test2", "fix2")]

    def mock_get_checks():
        return {
            "test_check1": TestCheck1(),
            "test_check2": TestCheck2(),
        }

    monkeypatch.setattr(inspector_module, "get_registered_checks", mock_get_checks)

    elements = ElementMap([], BoundingBox(0, 0, 100, 100))
    result = inspect(elements)

    assert len(result.issues) == 2
    assert result.score == 0.70  # 1.0 - (1.0 + 0.5)/5.0


def test_inspect_from_figure_integration():
    """End-to-end: pickled figure → extraction → inspection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plotlint.extractors.matplotlib import MatplotlibExtractor

    # Create simple chart
    fig, ax = plt.subplots()
    ax.bar(['A', 'B', 'C'], [1, 2, 3])
    ax.set_title('Test')
    fig.canvas.draw()
    figure_data = pickle.dumps(fig)

    # Extract and inspect
    extractor = MatplotlibExtractor()
    inspection = inspect_from_figure(figure_data, extractor)

    # Verify
    assert 0.0 <= inspection.score <= 1.0
    assert inspection.element_count > 0
