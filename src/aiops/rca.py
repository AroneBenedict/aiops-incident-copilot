"""Root-cause ranking inside one correlated incident.

The main method is a random walk over the alarmed part of the dependency graph. The walk
moves from a service to the dependencies it calls, preferring the more anomalous ones, so
probability mass collects at the alarmed service that has no alarmed dependency of its own.
Two simple baselines are kept for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .correlate import Alarm, IncidentGroup


@dataclass(frozen=True)
class Candidate:
    service: str
    score: float
    first_alarm: int
    peak: float


def _summaries(group: IncidentGroup, alarms: list[Alarm]) -> dict[str, tuple[int, float]]:
    """service -> (earliest alarm start, highest peak) within the group."""
    out: dict[str, tuple[int, float]] = {}
    for alarm_id in group.alarm_ids:
        alarm = alarms[alarm_id]
        start, peak = out.get(alarm.service, (alarm.start, alarm.peak))
        out[alarm.service] = (min(start, alarm.start), max(peak, alarm.peak))
    return out


def _ranked(scores: dict[str, float], info: dict[str, tuple[int, float]]) -> list[Candidate]:
    total = sum(scores.values()) or 1.0
    candidates = [Candidate(s, scores[s] / total, info[s][0], info[s][1]) for s in scores]
    return sorted(candidates, key=lambda c: (-c.score, c.first_alarm, c.service))


def rank_earliest(group: IncidentGroup, alarms: list[Alarm]) -> list[Candidate]:
    """Baseline: whichever service alarmed first is the root cause."""
    info = _summaries(group, alarms)
    latest = max(start for start, _ in info.values())
    return _ranked({s: float(latest - start + 1) for s, (start, _) in info.items()}, info)


def rank_peak(group: IncidentGroup, alarms: list[Alarm]) -> list[Candidate]:
    """Baseline: the most anomalous service is the root cause."""
    info = _summaries(group, alarms)
    return _ranked({s: peak for s, (_, peak) in info.items()}, info)


def rank_random_walk(group: IncidentGroup, alarms: list[Alarm], graph: nx.DiGraph,
                     alpha: float = 0.85) -> list[Candidate]:
    """Personalised PageRank on the alarmed subgraph, walking towards dependencies."""
    info = _summaries(group, alarms)
    if len(info) == 1:
        return _ranked({s: 1.0 for s in info}, info)

    walk = nx.DiGraph()
    walk.add_nodes_from(info)
    for service in info:
        alarmed_deps = [dep for dep in graph.successors(service) if dep in info]
        for dep in alarmed_deps:
            walk.add_edge(service, dep, weight=info[dep][1])
        if not alarmed_deps:
            # A service with no alarmed dependency keeps the walker: it is a candidate root.
            walk.add_edge(service, service, weight=1.0)

    total_peak = sum(peak for _, peak in info.values())
    personalization = {s: peak / total_peak for s, (_, peak) in info.items()}
    scores = nx.pagerank(walk, alpha=alpha, personalization=personalization, weight="weight")
    return _ranked(scores, info)


METHODS = {
    "earliest_alarm": lambda group, alarms, graph: rank_earliest(group, alarms),
    "peak_score": lambda group, alarms, graph: rank_peak(group, alarms),
    "random_walk": rank_random_walk,
}
