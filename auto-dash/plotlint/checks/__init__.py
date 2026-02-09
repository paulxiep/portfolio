# plotlint/checks/__init__.py

from __future__ import annotations

from typing import Protocol, runtime_checkable, Callable

from plotlint.elements import ElementMap
from plotlint.models import Issue


@runtime_checkable
class Check(Protocol):
    """Protocol for a single visual defect check.

    Each check receives the extracted elements and returns a list of issues.
    Checks are stateless, pure functions of the element map.
    """

    @property
    def name(self) -> str:
        """Unique name for this check (matches DefectType value)."""
        ...

    def __call__(self, elements: ElementMap) -> list[Issue]:
        """Run the check and return any detected issues."""
        ...


# --- Registry ---

_CHECKS: dict[str, Check] = {}


def register_check(check: Check) -> Check:
    """Register a check instance. Can also be used as a decorator on a class."""
    _CHECKS[check.name] = check
    return check


def get_registered_checks() -> dict[str, Check]:
    """Return all registered checks. Inspector calls this."""
    return dict(_CHECKS)


def check(name: str):
    """Decorator for registering a check class.

    Usage:
        @check("label_overlap")
        class LabelOverlapCheck:
            name = "label_overlap"
            def __call__(self, elements: ElementMap) -> list[Issue]:
                ...

    New checks in PL-1, PL-2 use this decorator.
    inspector.py never changes.
    """
    def decorator(cls):
        instance = cls()
        _CHECKS[name] = instance
        return cls
    return decorator


# --- Explicit imports to trigger registration ---
# Each check module uses @check() decorator which registers on import.
# New checks: add an import line here.
from plotlint.checks import overlap   # noqa: F401 — registers LabelOverlapCheck
from plotlint.checks import cutoff    # noqa: F401 — registers ElementCutoffCheck
