"""Anomaly detectors with one shared interface, plus threshold tuning.

Each detector turns the metric cube (services, minutes, metrics) into an anomaly score
per service-minute. Higher means more anomalous.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.ensemble import IsolationForest

from .features import SeasonalBaseline, build_features, trailing_mean, transform


class Detector(Protocol):
    name: str

    def fit(self, clean: np.ndarray) -> Detector: ...

    def score(self, metrics: np.ndarray, offset: int = 0) -> np.ndarray: ...


class RollingZScore:
    """Baseline: deviation from each metric's own trailing hour. No seasonality, no training."""

    name = "rolling_zscore"

    def __init__(self, window: int = 60, warmup: int = 15):
        self.window, self.warmup = window, warmup

    def fit(self, clean: np.ndarray) -> RollingZScore:
        return self

    def score(self, metrics: np.ndarray, offset: int = 0) -> np.ndarray:
        x = transform(metrics)
        mean = trailing_mean(x, self.window)
        var = trailing_mean(x**2, self.window) - mean**2
        std = np.sqrt(np.clip(var, 1e-8, None))
        # Compare minute t with the window that ended at t-1 so a spike cannot mask itself.
        prev_mean = np.concatenate([mean[:, :1], mean[:, :-1]], axis=1)
        prev_std = np.concatenate([std[:, :1], std[:, :-1]], axis=1)
        z = np.abs(x - prev_mean) / prev_std
        z[:, : min(self.warmup, z.shape[1])] = 0.0  # too few samples for a meaningful spread
        return trailing_mean(z, 3).max(axis=2)


class SeasonalResidual:
    """Robust z-score against a per-service daily profile, smoothed over five minutes."""

    name = "seasonal_residual"

    def __init__(self) -> None:
        self.baseline: SeasonalBaseline | None = None

    def fit(self, clean: np.ndarray) -> SeasonalResidual:
        self.baseline = SeasonalBaseline.fit(clean)
        return self

    def score(self, metrics: np.ndarray, offset: int = 0) -> np.ndarray:
        if self.baseline is None:
            raise RuntimeError("fit() must be called before score()")
        z = self.baseline.residual_z(metrics, offset)
        return np.abs(trailing_mean(z, 5)).max(axis=2)


class IsolationForestDetector:
    """Multivariate detector on seasonal-residual features, pooled across services."""

    name = "isolation_forest"

    def __init__(self, seed: int = 0, max_fit_rows: int = 60_000):
        self.seed = seed
        self.max_fit_rows = max_fit_rows
        self.baseline: SeasonalBaseline | None = None
        self.model: IsolationForest | None = None

    def fit(self, clean: np.ndarray) -> IsolationForestDetector:
        self.baseline = SeasonalBaseline.fit(clean)
        features = build_features(self.baseline.residual_z(clean))
        rows = features.reshape(-1, features.shape[-1])
        rng = np.random.default_rng(self.seed)
        if len(rows) > self.max_fit_rows:
            rows = rows[rng.choice(len(rows), self.max_fit_rows, replace=False)]
        self.model = IsolationForest(n_estimators=150, max_samples=1024, random_state=self.seed)
        self.model.fit(rows)
        return self

    def score(self, metrics: np.ndarray, offset: int = 0) -> np.ndarray:
        if self.baseline is None or self.model is None:
            raise RuntimeError("fit() must be called before score()")
        features = build_features(self.baseline.residual_z(metrics, offset))
        flat = features.reshape(-1, features.shape[-1])
        return (-self.model.score_samples(flat)).reshape(features.shape[:2])


def best_f1_threshold(scores: np.ndarray, truth: np.ndarray) -> float:
    """Threshold with the best point-wise F1, found exactly. Call this on the history split only.

    Scores are sorted once; flagging the top k scores gives a precision/recall pair for every k.
    The returned threshold sits halfway between the last flagged score and the next one.
    """
    flat, target = scores.ravel(), truth.ravel().astype(bool)
    positives = int(target.sum())
    if positives == 0:
        return float(flat.max())
    order = np.argsort(-flat, kind="stable")
    ranked = flat[order]
    tp = np.cumsum(target[order])
    flagged = np.arange(1, flat.size + 1)
    f1 = 2 * tp / (flagged + positives)
    # A cut is only valid between two different score values.
    valid = np.append(ranked[:-1] > ranked[1:], True)
    best = int(np.argmax(np.where(valid, f1, -1.0)))
    lower = ranked[best + 1] if best + 1 < flat.size else ranked[best] - 1.0
    return float((ranked[best] + lower) / 2)
