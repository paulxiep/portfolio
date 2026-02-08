"""Renderer bundle shell.

RendererBundle pairs a Renderer with its matching Extractor.
Fields typed as Any until MVP.6/7 define the Renderer and Extractor protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RendererBundle:
    """Pairs a Renderer with its matching Extractor.

    Prevents mismatches (e.g., matplotlib renderer with Plotly extractor).
    The convergence loop receives a bundle, not loose components.
    """

    renderer: Any  # Renderer protocol (MVP.6)
    extractor: Any  # Extractor protocol (MVP.7)
    renderer_type: str  # "matplotlib" or "plotly"


def matplotlib_bundle(**kwargs: Any) -> RendererBundle:
    """Create a matplotlib renderer+extractor bundle.

    Stub in MVP.1 — real implementation in MVP.6.
    """
    raise NotImplementedError(
        "matplotlib_bundle() requires MVP.6 (renderer) and MVP.7 (extractor)"
    )
