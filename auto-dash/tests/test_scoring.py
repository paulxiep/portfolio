"""Tests for plotlint.scoring (score computation)."""

from plotlint.scoring import compute_score, SEVERITY_WEIGHTS, MAX_DEMERITS
from plotlint.models import Issue, DefectType, Severity


def test_compute_score_no_issues():
    assert compute_score([]) == 1.0


def test_compute_score_single_high():
    issues = [Issue(DefectType.LABEL_OVERLAP, Severity.HIGH, "test", "fix")]
    # 1.0 - 1.0/5.0 = 0.80
    assert compute_score(issues) == 0.80


def test_compute_score_single_medium():
    issues = [Issue(DefectType.LABEL_OVERLAP, Severity.MEDIUM, "test", "fix")]
    # 1.0 - 0.5/5.0 = 0.90
    assert compute_score(issues) == 0.90


def test_compute_score_single_low():
    issues = [Issue(DefectType.LABEL_OVERLAP, Severity.LOW, "test", "fix")]
    # 1.0 - 0.2/5.0 = 0.96
    assert compute_score(issues) == 0.96


def test_compute_score_multiple():
    issues = [
        Issue(DefectType.LABEL_OVERLAP, Severity.HIGH, "test1", "fix1"),
        Issue(DefectType.ELEMENT_CUTOFF, Severity.MEDIUM, "test2", "fix2"),
    ]
    # 1.0 - (1.0 + 0.5)/5.0 = 0.70
    assert compute_score(issues) == 0.70


def test_compute_score_clamps_at_zero():
    # 10 HIGH issues should give 0.0, not negative
    issues = [Issue(DefectType.LABEL_OVERLAP, Severity.HIGH, "test", "fix") for _ in range(10)]
    assert compute_score(issues) == 0.0


def test_compute_score_clamps_at_one():
    # Already tested with empty list
    assert compute_score([]) == 1.0


def test_severity_weights():
    assert SEVERITY_WEIGHTS[Severity.HIGH] == 1.0
    assert SEVERITY_WEIGHTS[Severity.MEDIUM] == 0.5
    assert SEVERITY_WEIGHTS[Severity.LOW] == 0.2


def test_max_demerits():
    assert MAX_DEMERITS == 5.0
