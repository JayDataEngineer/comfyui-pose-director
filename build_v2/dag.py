"""DAG utilities — topological sort and graph construction."""

from __future__ import annotations

from collections import deque


def build_adjacency(stages: list) -> dict[str, list[str]]:
    """Build adjacency list: stage_name -> [dependency_names]."""
    return {s.name: list(s.depends_on) for s in stages}


def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """Kahn's algorithm. Returns execution order. Raises on cycles."""
    in_degree = {k: len(v) for k, v in graph.items()}
    # Reverse edges: for each node, who depends on it?
    reverse: dict[str, list[str]] = {k: [] for k in graph}
    for node, deps in graph.items():
        for dep in deps:
            if dep in reverse:
                reverse[dep].append(node)

    queue = deque(k for k, d in in_degree.items() if d == 0)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for dependent in reverse.get(node, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(order) != len(graph):
        remaining = [k for k in graph if k not in order]
        raise ValueError(f"Circular dependency in stage graph, stuck at: {remaining}")

    return order
