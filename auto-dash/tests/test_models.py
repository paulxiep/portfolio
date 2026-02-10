"""Tests for data models — serialization, properties, construction."""

import pytest

from plotlint.models import (
    ConvergenceState,
    DefectType,
    FixAttempt,
    InspectionResult,
    Issue,
    RenderResult,
    RenderStatus,
    Severity,
)
from autodash.models import (
    AggregationType,
    AnalysisStep,
    ChartPlan,
    ChartSpec,
    ChartType,
    ColumnProfile,
    DataMapping,
    DataProfile,
    RendererType,
    SemanticType,
    ChartPriority,
)


class TestIssue:
    def test_to_prompt_context(self):
        issue = Issue(
            defect_type=DefectType.LABEL_OVERLAP,
            severity=Severity.HIGH,
            details="X-axis labels overlap",
            suggestion="Rotate labels 45 degrees",
        )
        ctx = issue.to_prompt_context()
        assert "[HIGH]" in ctx
        assert "label_overlap" in ctx
        assert "Rotate" in ctx

    def test_frozen(self):
        issue = Issue(
            defect_type=DefectType.LABEL_OVERLAP,
            severity=Severity.HIGH,
            details="test",
            suggestion="fix",
        )
        with pytest.raises(AttributeError):
            issue.details = "changed"


class TestInspectionResult:
    def test_has_issues(self):
        result = InspectionResult(issues=[], score=1.0)
        assert not result.has_issues

    def test_highest_severity_issue(self):
        issues = [
            Issue(DefectType.LABEL_OVERLAP, Severity.LOW, "low", "fix"),
            Issue(DefectType.ELEMENT_CUTOFF, Severity.HIGH, "high", "fix"),
        ]
        result = InspectionResult(issues=issues, score=0.5)
        assert result.highest_severity_issue.severity == Severity.HIGH

    def test_no_issues_returns_none(self):
        result = InspectionResult(issues=[], score=1.0)
        assert result.highest_severity_issue is None


class TestFixAttempt:
    def test_improved(self):
        fix = FixAttempt(
            iteration=1,
            target_issue=DefectType.LABEL_OVERLAP,
            description="rotated labels",
            code_hash="abc123",
            score_before=0.5,
            score_after=0.8,
        )
        assert fix.improved

    def test_not_improved(self):
        fix = FixAttempt(
            iteration=1,
            target_issue=DefectType.LABEL_OVERLAP,
            description="bad fix",
            code_hash="def456",
            score_before=0.5,
            score_after=0.3,
        )
        assert not fix.improved


class TestRenderResult:
    def test_succeeded(self):
        result = RenderResult(status=RenderStatus.SUCCESS, png_bytes=b"png")
        assert result.succeeded

    def test_not_succeeded_on_error(self):
        result = RenderResult(status=RenderStatus.SYNTAX_ERROR)
        assert not result.succeeded

    def test_not_succeeded_without_png(self):
        result = RenderResult(status=RenderStatus.SUCCESS, png_bytes=None)
        assert not result.succeeded


class TestDataProfile:
    def _make_profile(self):
        col = ColumnProfile(
            name="revenue",
            pandas_dtype="float64",
            semantic_type=SemanticType.NUMERIC,
            null_count=0,
            null_fraction=0.0,
            unique_count=10,
            cardinality_fraction=1.0,
            min=100.0,
            max=1000.0,
            mean=550.0,
        )
        return DataProfile(
            source_path="test.csv",
            row_count=10,
            columns=[col],
            file_format="csv",
        )

    def test_column_names(self):
        profile = self._make_profile()
        assert profile.column_names() == ["revenue"]

    def test_get_column(self):
        profile = self._make_profile()
        col = profile.get_column("revenue")
        assert col is not None
        assert col.semantic_type == SemanticType.NUMERIC

    def test_get_column_missing(self):
        profile = self._make_profile()
        assert profile.get_column("nonexistent") is None

    def test_json_roundtrip(self):
        profile = self._make_profile()
        json_str = profile.to_json()
        restored = DataProfile.from_json(json_str)
        assert restored.source_path == profile.source_path
        assert restored.row_count == profile.row_count
        assert len(restored.columns) == 1
        assert restored.columns[0].name == "revenue"
        assert restored.columns[0].semantic_type == SemanticType.NUMERIC


class TestChartModels:
    def test_chart_spec_defaults(self):
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            data_mapping=DataMapping(x="category", y="value"),
            title="Test Chart",
        )
        assert spec.priority == ChartPriority.PRIMARY
        assert spec.figsize == (10, 6)

    def test_chart_plan_mutable(self):
        spec = ChartSpec(
            chart_type=ChartType.LINE,
            data_mapping=DataMapping(x="date", y="price"),
            title="Price Trend",
        )
        plan = ChartPlan(spec=spec, code="plt.plot()")
        plan.code = "plt.plot([1,2,3])"  # Should not raise
        assert plan.code == "plt.plot([1,2,3])"
        assert plan.renderer_type == RendererType.MATPLOTLIB
