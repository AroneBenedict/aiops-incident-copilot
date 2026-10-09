"""Feature engineering for per-minute service metrics.

Everything here is causal: a feature at minute t uses data up to and including t only,
so the same code is valid for streaming use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MINUTES_PER_DAY
from .simulate import CPU, ERR, LAT, THR


def transform(metrics: np.ndarray) -> np.ndarray:
    """Log-scale the multiplicative metrics so noise is roughly symmetric."""
    out = np.empty_like(metrics, dtype=np.float64)
    out[..., LAT] = np.log(metrics[..., LAT] + 1e-3)
    out[..., ERR] = np.log(metrics[..., ERR] + 1e-4)
    out[..., CPU] = metrics[..., CPU]
    out[..., THR] = np.log(metrics[..., THR] + 1.0)
    return out


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last ``window`` samples along the time axis (axis 1), causal."""
    csum = np.cumsum(values, axis=1)
    out = csum.copy()
    out[:, window:] = csum[:, window:] - csum[:, :-window]
    counts = np.minimum(np.arange(1, values.shape[1] + 1), window)
    shape = (1, -1) + (1,) * (values.ndim - 2)
    return out / counts.reshape(shape)


@dataclass
class SeasonalBaseline:
    """Per-service, per-minute-of-day expected value and robust spread, fitted on clean data."""

    profile: np.ndarray  # (services, 1440, metrics)
    scale: np.ndarray    # (services, metrics)

    @classmethod
    def fit(cls, clean: np.ndarray, smooth: int = 15) -> SeasonalBaseline:
        x = transform(clean)
        services, total, n_metrics = x.shape
        days = total // MINUTES_PER_DAY
        if days < 1:
            raise ValueError("need at least one full clean day to fit a seasonal baseline")
        by_day = x[:, : days * MINUTES_PER_DAY].reshape(services, days, MINUTES_PER_DAY, n_metrics)
        profile = np.median(by_day, axis=1)
        # Circular smoothing so the profile is not dominated by single-minute noise.
        padded = np.concatenate([profile[:, -smooth:], profile, profile[:, :smooth]], axis=1)
        kernel = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        profile = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), 1, padded)
        residual = by_day - profile[:, None]
        flat = residual.reshape(services, -1, n_metrics)
        mad = np.median(np.abs(flat - np.median(flat, axis=1, keepdims=True)), axis=1)
        return cls(profile, 1.4826 * mad + 1e-6)

    def residual_z(self, metrics: np.ndarray, offset: int = 0) -> np.ndarray:
        """Robust z-score of each metric against its seasonal expectation."""
        x = transform(metrics)
        minute_of_day = (np.arange(x.shape[1]) + offset) % MINUTES_PER_DAY
        return (x - self.profile[:, minute_of_day]) / self.scale[:, None]


def build_features(z: np.ndarray) -> np.ndarray:
    """Stack instantaneous, 5-minute and 15-minute views of the residual z-scores."""
    return np.concatenate([z, trailing_mean(z, 5), trailing_mean(z, 15)], axis=2)
