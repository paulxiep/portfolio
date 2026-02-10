"""Tests for plotlint.loop — convergence graph topology and stop conditions."""

import pytest

from plotlint.config import ConvergenceConfig
from plotlint.loop import (
    _make_should_continue,
    build_convergence_graph,
)
from plotlint.models import ConvergenceState
from plotlint.renderer import matplotlib_bundle


class TestGraphTopology:
    def test_graph_has_expected_nodes(self):
        graph = build_convergence_graph()
        node_names = set(graph.get_graph().nodes.keys())
        # LangGraph adds __start__ and __end__ nodes
        assert "render" in node_names
        assert "inspect" in node_names
        assert "patch" in node_names

    def test_graph_compiles(self):
        graph = build_convergence_graph()
        assert graph is not None

    def test_graph_with_explicit_bundle(self):
        bundle = matplotlib_bundle(dpi=150, timeout_seconds=10)
        graph = build_convergence_graph(bundle=bundle)
        assert graph is not None


class TestStubPassthrough:
    # Stubs replaced with real implementations in MVP.7
    # inspect_node is now a factory (_make_inspect_node)
    # patch_node remains a stub until MVP.8
    pass


class TestShouldContinue:
    def _make(self, **config_kwargs):
        config = ConvergenceConfig(**config_kwargs)
        return _make_should_continue(config)

    def test_perfect_score_stops(self):
        fn = self._make(target_score=1.0)
        state: ConvergenceState = {"score": 1.0, "iteration": 1}
        assert fn(state) == "stop"

    def test_max_iterations_stops(self):
        fn = self._make(max_iterations=3)
        state: ConvergenceState = {"score": 0.5, "iteration": 3}
        assert fn(state) == "stop"

    def test_render_error_stops(self):
        fn = self._make()
        state: ConvergenceState = {"score": 0.5, "iteration": 1, "render_error": "SyntaxError"}
        assert fn(state) == "stop"

    def test_stagnation_stops(self):
        fn = self._make(stagnation_window=2, score_improvement_threshold=0.01)
        state: ConvergenceState = {
            "score": 0.5,
            "iteration": 1,
            "score_history": [0.5, 0.5],
        }
        assert fn(state) == "stop"

    def test_improving_continues(self):
        fn = self._make(max_iterations=5, target_score=1.0)
        state: ConvergenceState = {
            "score": 0.5,
            "iteration": 1,
            "score_history": [0.3, 0.5],
        }
        assert fn(state) == "patch"

    def test_default_state_continues(self):
        """Empty state with no score defaults to continuing."""
        fn = self._make()
        state: ConvergenceState = {}
        assert fn(state) == "patch"

    def test_config_target_score_respected(self):
        fn = self._make(target_score=0.8)
        state: ConvergenceState = {"score": 0.8, "iteration": 1}
        assert fn(state) == "stop"

    def test_state_max_iterations_overrides_config(self):
        """max_iterations in state takes precedence over config."""
        fn = self._make(max_iterations=10)
        state: ConvergenceState = {"score": 0.5, "iteration": 2, "max_iterations": 2}
        assert fn(state) == "stop"
