"""Turn anomaly flags into alarms, then correlate alarms into incidents.

One fault usually raises alarms on several services. Correlation groups those alarms so an
operator sees one incident instead of a storm.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from .topology import SERVICES, are_adjacent


@dataclass(frozen=True)
class Alarm:
    id: int
    service: str
    start: int
    end: int
    peak: float  # highest anomaly score inside the alarm

    @property
    def duration(self) -> int:
        return self.end - self.start + 1


@dataclass
class IncidentGroup:
    id: int
    alarm_ids: list[int]
    services: list[str]
    start: int
    end: int


def build_alarms(scores: np.ndarray, threshold: float, min_duration: int = 3,
                 merge_gap: int = 2, offset: int = 0) -> list[Alarm]:
    """Contiguous anomalous runs per service become alarms; short blips are dropped."""
    alarms: list[Alarm] = []
    for i, service in enumerate(SERVICES):
        flagged = np.flatnonzero(scores[i] > threshold)
        if flagged.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(flagged) > merge_gap + 1)
        run_starts = np.concatenate([[0], breaks + 1])
        run_ends = np.concatenate([breaks, [flagged.size - 1]])
        for a, b in zip(run_starts, run_ends, strict=True):
            start, end = int(flagged[a]), int(flagged[b])
            if end - start + 1 < min_duration:
                continue
            peak = float(scores[i, start : end + 1].max())
            alarms.append(Alarm(-1, service, start + offset, end + offset, peak))
    alarms.sort(key=lambda alarm: (alarm.start, alarm.service))
    return [Alarm(n, a.service, a.start, a.end, a.peak) for n, a in enumerate(alarms)]


def label_alarms(alarms: list[Alarm], labels: np.ndarray) -> np.ndarray:
    """Ground-truth incident id per alarm (majority vote over its minutes), -1 if spurious."""
    index = {service: i for i, service in enumerate(SERVICES)}
    out = np.full(len(alarms), -1, dtype=np.int64)
    for n, alarm in enumerate(alarms):
        window = labels[index[alarm.service], alarm.start : alarm.end + 1]
        hits = window[window >= 0]
        if hits.size * 2 >= window.size and hits.size:
            values, counts = np.unique(hits, return_counts=True)
            out[n] = int(values[counts.argmax()])
    return out


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def correlate(alarms: list[Alarm], graph: nx.DiGraph | None, gap: int = 5) -> list[IncidentGroup]:
    """Group alarms that are close in time and, when a graph is given, adjacent in topology.

    Passing ``graph=None`` gives the time-window-only baseline.
    """
    sets = _UnionFind(len(alarms))
    ordered = sorted(range(len(alarms)), key=lambda n: alarms[n].start)
    active: list[int] = []
    for n in ordered:
        current = alarms[n]
        active = [m for m in active if alarms[m].end + gap >= current.start]
        for m in active:
            if graph is None or are_adjacent(graph, alarms[m].service, current.service):
                sets.union(m, n)
        active.append(n)

    members: dict[int, list[int]] = {}
    for n in range(len(alarms)):
        members.setdefault(sets.find(n), []).append(n)
    return _regroup(list(members.values()), alarms)


def _regroup(alarm_sets: list[list[int]], alarms: list[Alarm]) -> list[IncidentGroup]:
    groups = []
    for ids in sorted(alarm_sets, key=lambda ids: (min(alarms[n].start for n in ids), min(ids))):
        groups.append(IncidentGroup(
            id=len(groups),
            alarm_ids=sorted(ids),
            services=sorted({alarms[n].service for n in ids}),
            start=min(alarms[n].start for n in ids),
            end=max(alarms[n].end for n in ids),
        ))
    return groups


def split_by_root(groups: list[IncidentGroup], alarms: list[Alarm], graph: nx.DiGraph) -> list[IncidentGroup]:
    """Split a group that contains more than one independent root.

    Inside a group, a *root candidate* is an alarmed service that does not depend, directly or
    transitively, on any other alarmed service. One fault produces one root candidate. Two
    faults that overlap in time and touch neighbouring services produce two, and time-plus-
    adjacency correlation merges them. Here each alarm is handed to a root candidate it depends
    on (the one whose first alarm is closest in time), and the group is split accordingly.
    """
    depends_on = {service: nx.descendants(graph, service) for service in graph.nodes}
    out: list[list[int]] = []
    for group in groups:
        alarmed = set(group.services)
        roots = sorted(s for s in alarmed if not (depends_on[s] & alarmed))
        if len(roots) <= 1:
            out.append(list(group.alarm_ids))
            continue
        first = {root: min(alarms[n].start for n in group.alarm_ids if alarms[n].service == root)
                 for root in roots}
        parts: dict[str, list[int]] = {root: [] for root in roots}
        for n in group.alarm_ids:
            alarm = alarms[n]
            if alarm.service in parts:
                parts[alarm.service].append(n)
                continue
            reachable = [root for root in roots if root in depends_on[alarm.service]]
            owner = min(reachable, key=lambda root: (abs(first[root] - alarm.start), root))
            parts[owner].append(n)
        out.extend(part for part in parts.values() if part)
    return _regroup(out, alarms)


def group_assignment(groups: list[IncidentGroup], n_alarms: int) -> np.ndarray:
    out = np.empty(n_alarms, dtype=np.int64)
    for group in groups:
        out[group.alarm_ids] = group.id
    return out
