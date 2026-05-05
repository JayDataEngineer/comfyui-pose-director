"""Unit tests for DAG — topological sort and adjacency."""

import pytest

from build_v2.dag import build_adjacency, topological_sort


class TestDAG:
    def test_simple_chain(self):
        """A -> B -> C should sort to [A, B, C]."""

        class A:
            name = "a"
            depends_on = []

        class B:
            name = "b"
            depends_on = ["a"]

        class C:
            name = "c"
            depends_on = ["b"]

        graph = build_adjacency([A(), B(), C()])
        order = topological_sort(graph)
        assert order.index("a") < order.index("b") < order.index("c")

    def test_diamond_dependency(self):
        """A -> B, A -> C, B -> D, C -> D."""

        class A:
            name = "a"
            depends_on = []

        class B:
            name = "b"
            depends_on = ["a"]

        class C:
            name = "c"
            depends_on = ["a"]

        class D:
            name = "d"
            depends_on = ["b", "c"]

        graph = build_adjacency([A(), B(), C(), D()])
        order = topological_sort(graph)
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_cycle_detection(self):
        class A:
            name = "a"
            depends_on = ["b"]

        class B:
            name = "b"
            depends_on = ["a"]

        graph = build_adjacency([A(), B()])
        with pytest.raises(ValueError, match="Circular"):
            topological_sort(graph)

    def test_no_deps(self):
        class A:
            name = "a"
            depends_on = []

        graph = build_adjacency([A()])
        order = topological_sort(graph)
        assert order == ["a"]


class TestDAGIntegration:
    """Verify DAG has no cycles with all real stages."""

    def test_stage_names_unique(self):
        from build_v2.orchestrator import ALL_STAGE_CLASSES
        names = [cls.name for cls in ALL_STAGE_CLASSES]
        assert len(names) == len(set(names)), f"Duplicate stage names: {names}"

    def test_dag_valid(self):
        from build_v2.orchestrator import ALL_STAGE_CLASSES
        stages = [cls(None) for cls in ALL_STAGE_CLASSES]
        graph = build_adjacency(stages)
        assert len(graph) == len(stages)
        order = topological_sort(graph)
        assert len(order) == len(stages)
