"""Central configuration. Every experiment is fully described by one Config."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

MINUTES_PER_DAY = 1440
METRICS = ("latency_ms", "error_rate", "cpu_pct", "throughput_rps")


@dataclass(frozen=True)
class Config:
    seed: int = 7
    # Timeline, in days. The three periods never overlap:
    #   clean    -> incident-free, used to fit seasonal baselines and unsupervised detectors
    #   history  -> labelled incidents, used to tune thresholds and train the fault classifier
    #   test     -> labelled incidents, used only for the reported numbers
    clean_days: int = 7
    history_days: int = 14
    test_days: int = 14
    incidents_per_day: float = 6.0
    concurrent_fraction: float = 0.3  # share of incidents followed by another within 20 minutes
    benign_bursts_per_day: float = 6.0
    silent_fraction: float = 0.25     # share of incidents whose root logs nothing useful

    # Alarm generation
    alarm_min_duration: int = 3   # minutes a service must stay anomalous to raise an alarm
    alarm_merge_gap: int = 2      # minutes; shorter gaps inside one alarm are bridged

    # Correlation
    correlation_gap: int = 5      # minutes between alarms that may still belong together

    # Agent
    agent_max_steps: int = 12
    rca_margin_for_log_check: float = 0.15

    @property
    def total_minutes(self) -> int:
        return (self.clean_days + self.history_days + self.test_days) * MINUTES_PER_DAY

    @property
    def history_start(self) -> int:
        return self.clean_days * MINUTES_PER_DAY

    @property
    def test_start(self) -> int:
        return (self.clean_days + self.history_days) * MINUTES_PER_DAY

    def fingerprint(self) -> str:
        """Short stable hash of the config, used to key experiment runs."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:10]


def small_config(seed: int = 7) -> Config:
    """A fast configuration for unit tests and local smoke runs (about one second)."""
    return Config(seed=seed, clean_days=2, history_days=3, test_days=3)
