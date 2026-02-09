"""Tests for matplotlib extractor - THE CRITICAL TEST."""

import pickle
import pytest


@pytest.fixture
def simple_bar_figure():
    """Fixture that returns pickled Figure bytes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.bar(['A', 'B', 'C'], [1, 2, 3])
    ax.set_xlabel('Category')
    ax.set_ylabel('Value')
    ax.set_title('Test Chart')
    fig.canvas.draw()
    return pickle.dumps(fig)


def test_extract_tick_labels(simple_bar_figure):
    from plotlint.extractors.matplotlib import MatplotlibExtractor
    from plotlint.elements import ElementCategory

    extractor = MatplotlibExtractor()
    elements = extractor.extract(simple_bar_figure)

    tick_labels = elements.by_category(ElementCategory.TICK_LABEL)
    assert len(tick_labels) >= 3  # At least A, B, C on x-axis
    assert all(e.bbox.area > 0 for e in tick_labels)


def test_coordinate_system_flip(simple_bar_figure):
    from plotlint.extractors.matplotlib import MatplotlibExtractor

    extractor = MatplotlibExtractor()
    elements = extractor.extract(simple_bar_figure)

    # All y0 values should be less than y1 (screen coords)
    assert all(e.bbox.y0 < e.bbox.y1 for e in elements.elements)


def test_figure_bbox_size(simple_bar_figure):
    from plotlint.extractors.matplotlib import MatplotlibExtractor

    # 10x6 inches at 100 DPI = 1000x600 pixels
    extractor = MatplotlibExtractor()
    elements = extractor.extract(simple_bar_figure)
    assert elements.figure_bbox.width == 1000
    assert elements.figure_bbox.height == 600


def test_extract_title(simple_bar_figure):
    from plotlint.extractors.matplotlib import MatplotlibExtractor
    from plotlint.elements import ElementCategory

    extractor = MatplotlibExtractor()
    elements = extractor.extract(simple_bar_figure)

    titles = elements.by_category(ElementCategory.TITLE)
    assert len(titles) == 1
    assert titles[0].text == 'Test Chart'
    assert titles[0].bbox.area > 0


def test_extract_axis_labels(simple_bar_figure):
    from plotlint.extractors.matplotlib import MatplotlibExtractor
    from plotlint.elements import ElementCategory

    extractor = MatplotlibExtractor()
    elements = extractor.extract(simple_bar_figure)

    axis_labels = elements.by_category(ElementCategory.AXIS_LABEL)
    assert len(axis_labels) == 2  # xlabel and ylabel

    texts = [e.text for e in axis_labels]
    assert 'Category' in texts
    assert 'Value' in texts


def test_overlapping_labels_detected():
    """THE CRITICAL TEST - validates core plotlint thesis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plotlint.extractors.matplotlib import MatplotlibExtractor
    from plotlint.checks.overlap import LabelOverlapCheck
    from plotlint.models import DefectType

    # Create chart with many x-axis labels that WILL overlap
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = [f'Category_{i}' for i in range(20)]
    ax.bar(categories, range(20))
    fig.canvas.draw()
    figure_data = pickle.dumps(fig)

    # Extract
    extractor = MatplotlibExtractor()
    elements = extractor.extract(figure_data)

    # Inspect
    check = LabelOverlapCheck()
    issues = check(elements)

    # VERIFY: overlap detected
    assert len(issues) > 0
    assert issues[0].defect_type == DefectType.LABEL_OVERLAP
    assert "X-axis labels overlap" in issues[0].details


def test_extraction_error_handling():
    from plotlint.extractors.matplotlib import MatplotlibExtractor
    from plotlint.core.errors import ExtractionError

    extractor = MatplotlibExtractor()

    # Invalid pickle data
    with pytest.raises(ExtractionError):
        extractor.extract(b"not a pickle")
