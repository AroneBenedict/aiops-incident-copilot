"""Fault-type classification from the metric signature at the suspected root service."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .config import METRICS
from .correlate import Alarm, IncidentGroup
from .simulate import LAT
from .topology import KIND, KINDS, SERVICES

_INDEX = {service: i for i, service in enumerate(SERVICES)}
FEATURE_NAMES = (
    [f"mean_z_{m}" for m in METRICS]
    + [f"extreme_z_{m}" for m in METRICS]
    + ["latency_trend", "onset_ratio", "log_duration", "group_size"]
    + [f"kind_{k}" for k in KINDS]
)


def service_window(group: IncidentGroup, alarms: list[Alarm], service: str) -> tuple[int, int]:
    """First and last alarmed minute of ``service`` inside the group."""
    mine = [alarms[n] for n in group.alarm_ids if alarms[n].service == service]
    if not mine:
        raise KeyError(f"{service} has no alarm in group {group.id}")
    return min(a.start for a in mine), max(a.end for a in mine)


def signature(z: np.ndarray, group: IncidentGroup, alarms: list[Alarm], service: str) -> np.ndarray:
    """Feature vector describing how ``service`` misbehaved during the incident."""
    start, end = service_window(group, alarms, service)
    window = z[_INDEX[service], start : end + 1]
    mean = window.mean(axis=0)
    extreme = window[np.abs(window).argmax(axis=0), np.arange(window.shape[1])]

    latency = window[:, LAT]
    if len(latency) >= 4 and latency.std() > 1e-9:
        trend = float(np.corrcoef(np.arange(len(latency)), latency)[0, 1])
    else:
        trend = 0.0
    third = max(1, len(latency) // 3)
    onset_ratio = float(np.clip(latency[:third].mean() / (abs(latency[-third:].mean()) + 1.0), -5, 5))

    kind = [1.0 if KIND[service] == k else 0.0 for k in KINDS]
    extras = [trend, onset_ratio, float(np.log(end - start + 1)), float(len(group.services))]
    return np.concatenate([mean, extreme, extras, kind])


def describe(z: np.ndarray, group: IncidentGroup, alarms: list[Alarm], service: str) -> dict[str, float]:
    """Mean deviation per metric, in robust standard deviations, for reports and prompts."""
    start, end = service_window(group, alarms, service)
    mean = z[_INDEX[service], start : end + 1].mean(axis=0)
    return {metric: round(float(value), 1) for metric, value in zip(METRICS, mean, strict=True)}


@dataclass
class FaultPrediction:
    fault_type: str
    confidence: float
    alternatives: list[tuple[str, float]]


class FaultClassifier:
    def __init__(self, seed: int = 0):
        self.model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                            class_weight="balanced", random_state=seed)

    def fit(self, features: np.ndarray, labels: list[str]) -> FaultClassifier:
        self.model.fit(features, labels)
        return self

    def predict(self, features: np.ndarray) -> list[FaultPrediction]:
        proba = self.model.predict_proba(np.atleast_2d(features))
        out = []
        for row in proba:
            order = np.argsort(row)[::-1]
            ranked = [(str(self.model.classes_[i]), round(float(row[i]), 3)) for i in order[:3]]
            out.append(FaultPrediction(ranked[0][0], ranked[0][1], ranked))
        return out


class KindLookup:
    """Baseline: predict the most common fault seen for that kind of service."""

    def fit(self, kinds: list[str], labels: list[str]) -> KindLookup:
        counts: dict[str, dict[str, int]] = {}
        overall: dict[str, int] = {}
        for kind, label in zip(kinds, labels, strict=True):
            counts.setdefault(kind, {})[label] = counts.get(kind, {}).get(label, 0) + 1
            overall[label] = overall.get(label, 0) + 1
        self.default = max(sorted(overall), key=overall.get)
        self.table = {k: max(sorted(v), key=v.get) for k, v in counts.items()}
        return self

    def predict(self, kinds: list[str]) -> list[str]:
        return [self.table.get(kind, self.default) for kind in kinds]
