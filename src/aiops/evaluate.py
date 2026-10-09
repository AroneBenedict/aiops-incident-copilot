"""Evaluation metrics. Every function compares pipeline output with simulator ground truth."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score

from .correlate import Alarm, IncidentGroup, group_assignment
from .simulate import Incident
from .topology import SERVICES

_INDEX = {service: i for i, service in enumerate(SERVICES)}


def pointwise(flags: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    tp = int(np.count_nonzero(flags & truth))
    precision = tp / max(int(flags.sum()), 1)
    recall = tp / max(int(truth.sum()), 1)
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def event_level(flags: np.ndarray, incidents: list[Incident], offset: int) -> dict[str, float]:
    """Share of incidents caught on any affected service, and how fast."""
    delays = []
    for incident in incidents:
        first = None
        for service, (start, end) in incident.affected.items():
            hits = np.flatnonzero(flags[_INDEX[service], start - offset : end - offset + 1])
            if hits.size:
                minute = start + int(hits[0])
                first = minute if first is None else min(first, minute)
        if first is not None:
            delays.append(first - incident.start)
    detected = len(delays)
    return {
        "incidents": len(incidents),
        "detected": detected,
        "event_recall": round(detected / max(len(incidents), 1), 4),
        "median_delay_min": float(np.median(delays)) if delays else float("nan"),
    }


def correlation_quality(groups: list[IncidentGroup], alarm_labels: np.ndarray) -> dict[str, float]:
    assigned = group_assignment(groups, len(alarm_labels))
    real = alarm_labels >= 0
    return {
        "alarms": int(len(alarm_labels)),
        "groups": len(groups),
        "noise_reduction": round(1 - len(groups) / max(len(alarm_labels), 1), 4),
        "ari": round(float(adjusted_rand_score(alarm_labels[real], assigned[real])), 4) if real.any() else 0.0,
    }


def match_groups(groups: list[IncidentGroup], alarm_labels: np.ndarray) -> dict[int, IncidentGroup]:
    """Map each true incident to the group that holds most of its alarms."""
    best: dict[int, tuple[int, IncidentGroup]] = {}
    for group in groups:
        labels = alarm_labels[group.alarm_ids]
        for incident_id in np.unique(labels[labels >= 0]).tolist():
            count = int(np.count_nonzero(labels == incident_id))
            if incident_id not in best or count > best[incident_id][0]:
                best[incident_id] = (count, group)
    return {incident_id: group for incident_id, (_, group) in best.items()}


def rca_accuracy(rank, matched: dict[int, IncidentGroup], incidents: list[Incident],
                 alarms: list[Alarm], graph) -> dict[str, float]:
    """Top-1 and top-3 accuracy on detected incidents, and end to end over all incidents."""
    top1 = top3 = 0
    scored = [incident for incident in incidents if incident.id in matched]
    for incident in scored:
        ranked = [c.service for c in rank(matched[incident.id], alarms, graph)]
        top1 += ranked[0] == incident.root
        top3 += incident.root in ranked[:3]
    n = max(len(scored), 1)
    return {
        "scored": len(scored),
        "top1": round(top1 / n, 4),
        "top3": round(top3 / n, 4),
        "end_to_end_top1": round(top1 / max(len(incidents), 1), 4),
    }


def classification(truth: list[str], predicted: list[str]) -> dict[str, float]:
    if not truth:
        return {"n": 0, "accuracy": 0.0, "macro_f1": 0.0}
    accuracy = float(np.mean([t == p for t, p in zip(truth, predicted, strict=True)]))
    return {"n": len(truth), "accuracy": round(accuracy, 4),
            "macro_f1": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 4)}


def ranking(ranks: list[int | None]) -> dict[str, float]:
    """Hit@1, Hit@3 and MRR from 1-based ranks (None when the target was not returned)."""
    if not ranks:
        return {"n": 0, "hit1": 0.0, "hit3": 0.0, "mrr": 0.0}
    hit1 = np.mean([r == 1 for r in ranks])
    hit3 = np.mean([r is not None and r <= 3 for r in ranks])
    mrr = np.mean([1 / r if r else 0.0 for r in ranks])
    return {"n": len(ranks), "hit1": round(float(hit1), 4), "hit3": round(float(hit3), 4),
            "mrr": round(float(mrr), 4)}
