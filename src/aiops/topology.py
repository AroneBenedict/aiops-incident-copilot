"""Service dependency topology.

An edge ``a -> b`` means "a depends on b". A fault in ``b`` therefore degrades ``a``:
impact flows against the edge direction, and root-cause search walks along it.
"""

from __future__ import annotations

import networkx as nx

DEPENDS_ON: dict[str, list[str]] = {
    "api-gateway": ["auth", "orders", "catalog", "search"],
    "auth": ["user-db", "cache"],
    "orders": ["orders-db", "payments", "message-queue"],
    "payments": ["payments-db", "message-queue"],
    "catalog": ["cache", "search-index"],
    "search": ["search-index"],
    "notifications": ["message-queue"],
    "user-db": ["core-router"],
    "orders-db": ["core-router"],
    "payments-db": ["core-router"],
    "cache": [],
    "search-index": [],
    "message-queue": [],
    "core-router": [],
}

KIND: dict[str, str] = {
    "api-gateway": "gateway",
    "auth": "app",
    "orders": "app",
    "payments": "app",
    "catalog": "app",
    "search": "app",
    "notifications": "app",
    "user-db": "database",
    "orders-db": "database",
    "payments-db": "database",
    "cache": "cache",
    "search-index": "index",
    "message-queue": "queue",
    "core-router": "network",
}

SERVICES: tuple[str, ...] = tuple(DEPENDS_ON)
KINDS: tuple[str, ...] = tuple(sorted(set(KIND.values())))


def build_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(SERVICES)
    for service, deps in DEPENDS_ON.items():
        graph.add_edges_from((service, dep) for dep in deps)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("service topology must be acyclic")
    return graph


def dependents(graph: nx.DiGraph, service: str) -> list[str]:
    """Services that call ``service`` directly."""
    return sorted(graph.predecessors(service))


def are_adjacent(graph: nx.DiGraph, a: str, b: str) -> bool:
    """True when the two services are the same or share a direct dependency edge."""
    return a == b or graph.has_edge(a, b) or graph.has_edge(b, a)
