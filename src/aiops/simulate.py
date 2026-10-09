"""Synthetic telemetry, alarms and logs with known ground truth.

Real operations data is confidential and rarely labelled with a root cause, so this module
generates a labelled stand-in: per-minute metrics for every service, with injected incidents
that start at one root service and propagate to the services that depend on it.

The simulator is deliberately not clean. It includes daily and weekly seasonality, noisy
metrics, unlabelled benign traffic bursts, mild incidents that barely move the metrics,
overlapping incidents and misleading log lines, so the downstream methods can fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from .config import METRICS, MINUTES_PER_DAY, Config
from .topology import KIND, SERVICES, build_graph

LAT, ERR, CPU, THR = range(len(METRICS))

BASE = {  # kind -> (latency_ms, error_rate, cpu_pct, throughput_rps)
    "gateway": (80.0, 0.004, 42.0, 900.0),
    "app": (60.0, 0.004, 45.0, 420.0),
    "database": (12.0, 0.001, 38.0, 800.0),
    "cache": (2.0, 0.0005, 30.0, 3000.0),
    "index": (25.0, 0.001, 40.0, 400.0),
    "queue": (8.0, 0.0005, 28.0, 1200.0),
    "network": (1.5, 0.0005, 22.0, 5000.0),
}


@dataclass(frozen=True)
class Fault:
    """How one fault type moves the root service's metrics, at full profile and severity 1."""

    kinds: tuple[str, ...]
    shape: str           # "step" (fast onset) or "ramp" (grows over the whole incident)
    onset: int           # minutes to reach full effect for "step" faults
    latency_x: float     # multiplicative latency increase
    error_add: float     # additive error-rate increase
    cpu_add: float       # additive CPU points
    throughput_x: float  # fractional throughput loss
    spread_latency: float  # latency impact passed on to dependents
    spread_error: float    # error impact passed on to dependents
    logs: tuple[str, ...] = field(default=())


FAULTS: dict[str, Fault] = {
    "cpu_saturation": Fault(
        ("app", "gateway", "index"), "step", 4, 1.8, 0.010, 45.0, 0.15, 0.9, 0.006,
        (
            "WARN worker pool saturated: {n} requests queued, load average {x}",
            "WARN request handler thread starvation detected, cpu throttled by cgroup",
            "ERROR health probe timed out after {ms}ms (process busy)",
            "WARN autoscaler: cpu utilisation {pct}% above target 70%",
        ),
    ),
    "memory_leak": Fault(
        ("app",), "ramp", 0, 1.2, 0.030, 15.0, 0.10, 0.7, 0.015,
        (
            "WARN heap usage at {pct}% of limit, full GC took {ms}ms",
            "WARN GC pause {ms}ms exceeded threshold; old generation not shrinking",
            "ERROR container OOMKilled, restarting (exit code 137)",
            "WARN resident memory grew {n}MB in the last 10 minutes",
        ),
    ),
    "db_connection_exhaustion": Fault(
        ("database",), "step", 2, 2.5, 0.120, -5.0, 0.40, 1.2, 0.060,
        (
            "ERROR could not acquire connection from pool within {ms}ms (active={n}, max={n})",
            "ERROR FATAL: remaining connection slots are reserved; too many clients already",
            "WARN connection pool exhausted, {n} callers waiting",
            "ERROR query aborted: connection checkout timeout",
        ),
    ),
    "disk_io_saturation": Fault(
        ("database", "index"), "step", 6, 3.0, 0.004, 12.0, 0.25, 1.4, 0.003,
        (
            "WARN fsync took {ms}ms; write-ahead log flush is slow",
            "WARN iowait at {pct}%, device queue depth {n}",
            "WARN checkpoint running long: {n} buffers pending, storage latency {ms}ms",
            "ERROR slow query: sequential scan blocked on disk read for {ms}ms",
        ),
    ),
    "packet_loss": Fault(
        ("network",), "step", 2, 2.0, 0.050, 0.0, 0.30, 1.1, 0.030,
        (
            "WARN interface ge-0/0/{n} input errors rising: CRC {n}, drops {n}",
            "ERROR BGP neighbor flap detected, session reset",
            "WARN packet loss {pct}% on uplink, retransmits increasing",
            "WARN TCP retransmission rate {pct}% above baseline",
        ),
    ),
    "bad_deploy": Fault(
        ("app", "gateway"), "step", 1, 0.15, 0.150, 0.0, 0.05, 0.1, 0.070,
        (
            "ERROR unhandled exception in request handler: NullPointerException at CheckoutController",
            "ERROR HTTP 500 returned for {n} requests since rollout of build {build}",
            "INFO deployment rollout completed: version {build} now serving 100% traffic",
            "ERROR config key 'feature.flags' missing after release {build}; fallback failed",
        ),
    ),
    "queue_backlog": Fault(
        ("queue",), "ramp", 0, 4.0, 0.003, 5.0, 0.35, 1.0, 0.004,
        (
            "WARN consumer lag {n} messages on partition {n}, lag growing",
            "WARN oldest unacknowledged message age {n}s exceeds SLO",
            "WARN consumer group rebalance in progress, {n} consumers active",
            "ERROR broker rejected publish: queue depth limit reached",
        ),
    ),
    "cache_eviction_storm": Fault(
        ("cache",), "step", 3, 2.2, 0.002, 25.0, 0.05, 1.0, 0.004,
        (
            "WARN eviction rate {n}/s, hit ratio dropped to {pct}%",
            "WARN maxmemory reached, evicting keys with allkeys-lru policy",
            "WARN cache miss stampede: {n} concurrent rebuilds for hot keys",
            "ERROR command latency {ms}ms, slowlog growing",
        ),
    ),
}
FAULT_TYPES: tuple[str, ...] = tuple(FAULTS)

DEPENDENT_LOGS = (
    "WARN upstream call to {dep} exceeded {ms}ms timeout",
    "ERROR received 503 from {dep}, retry {r}/3",
    "WARN circuit breaker for {dep} is half-open",
    "ERROR request failed: dependency {dep} unavailable",
)
NOISE_LOGS = (
    "INFO request completed in {ms}ms",
    "INFO scheduled job finished successfully",
    "INFO configuration reloaded",
    "WARN slow request {ms}ms on /internal/report",
    "ERROR client disconnected before response was sent",
    "WARN retrying idempotent call after transient timeout",
)


@dataclass
class Incident:
    id: int
    root: str
    fault_type: str
    start: int
    end: int
    severity: float
    affected: dict[str, tuple[int, int]]  # service -> (first, last) affected minute
    silent: bool = False                  # True when the root emitted no fault-specific logs


@dataclass
class Dataset:
    config: Config
    metrics: np.ndarray        # (services, minutes, metrics)
    labels: np.ndarray         # (services, minutes) incident id, or -1 when healthy
    incidents: list[Incident]
    logs: pd.DataFrame         # columns: ts, service, level, message

    @property
    def services(self) -> tuple[str, ...]:
        return SERVICES

    def incident(self, incident_id: int) -> Incident:
        return self.incidents[incident_id]

    def logs_between(self, service: str, start: int, end: int) -> pd.DataFrame:
        logs = self.logs
        mask = (logs["service"] == service) & (logs["ts"] >= start) & (logs["ts"] <= end)
        return logs.loc[mask]


def _build_id(rng: np.random.Generator) -> str:
    return f"2026.{int(rng.integers(1, 12))}.{int(rng.integers(0, 40))}"


def _fill(template: str, rng: np.random.Generator, dep: str = "", build: str = "") -> str:
    return template.format(
        n=int(rng.integers(3, 900)),
        r=int(rng.integers(1, 4)),
        x=round(float(rng.uniform(4, 32)), 1),
        ms=int(rng.integers(250, 9000)),
        pct=int(rng.integers(71, 99)),
        build=build or _build_id(rng),
        dep=dep,
    )


def _baseline(rng: np.random.Generator, total: int) -> np.ndarray:
    minutes = np.arange(total)
    day_phase = 2 * np.pi * (minutes / MINUTES_PER_DAY - 0.3)
    week_phase = 2 * np.pi * minutes / (7 * MINUTES_PER_DAY)
    load = 1.0 + 0.35 * np.sin(day_phase) + 0.08 * np.sin(week_phase)

    data = np.empty((len(SERVICES), total, len(METRICS)), dtype=np.float64)
    for i, service in enumerate(SERVICES):
        lat, err, cpu, thr = BASE[KIND[service]]
        scale = rng.uniform(0.85, 1.15, size=4)  # every service gets its own operating point
        noise = rng.uniform(0.6, 1.8)            # and its own noise level: some are hard to watch
        data[i, :, THR] = thr * scale[3] * load * rng.lognormal(0, 0.05 * noise, total)
        data[i, :, CPU] = cpu * scale[2] * (0.6 + 0.4 * load) + rng.normal(0, 2.5 * noise, total)
        data[i, :, LAT] = (lat * scale[0] * (1 + 0.25 * (load - 1))
                           * rng.lognormal(0, 0.06 * noise, total))
        data[i, :, ERR] = err * scale[1] * rng.lognormal(0, 0.35 * noise, total)
    return data


def _add_benign_bursts(rng: np.random.Generator, data: np.ndarray, cfg: Config) -> None:
    """Unlabelled traffic bursts: real load, not a fault. These are the false-positive bait."""
    days = cfg.total_minutes / MINUTES_PER_DAY
    for _ in range(rng.poisson(cfg.benign_bursts_per_day * days)):
        service = int(rng.integers(len(SERVICES)))
        start = int(rng.integers(0, cfg.total_minutes - 30))
        length = int(rng.integers(5, 21))
        window = slice(start, start + length)
        boost = rng.uniform(1.3, 1.8)
        data[service, window, THR] *= boost
        data[service, window, CPU] *= 1 + 0.45 * (boost - 1)
        data[service, window, LAT] *= 1 + 0.15 * (boost - 1)


def _profile(fault: Fault, length: int) -> np.ndarray:
    if fault.shape == "ramp":
        return np.linspace(0.05, 1.0, length)
    profile = np.ones(length)
    onset = min(fault.onset, length)
    profile[:onset] = np.linspace(1 / (onset + 1), 1.0, onset)
    tail = min(3, length)
    profile[-tail:] *= np.linspace(1.0, 0.4, tail)
    return profile


def _apply(data: np.ndarray, service: int, start: int, profile: np.ndarray,
           latency_x: float, error_add: float, cpu_add: float, throughput_x: float) -> None:
    window = slice(start, start + len(profile))
    data[service, window, LAT] *= 1 + latency_x * profile
    data[service, window, ERR] += error_add * profile
    data[service, window, CPU] += cpu_add * profile
    data[service, window, THR] *= 1 - throughput_x * profile


def simulate(cfg: Config) -> Dataset:
    rng = np.random.default_rng(cfg.seed)
    graph = build_graph()
    index = {service: i for i, service in enumerate(SERVICES)}
    total = cfg.total_minutes

    data = _baseline(rng, total)
    _add_benign_bursts(rng, data, cfg)
    labels = np.full((len(SERVICES), total), -1, dtype=np.int32)
    incidents: list[Incident] = []
    log_rows: list[tuple[int, str, str, str]] = []

    incident_days = cfg.history_days + cfg.test_days
    count = rng.poisson(cfg.incidents_per_day * incident_days)
    starts = rng.integers(cfg.history_start, total - 120, size=count)
    # Incidents cluster in practice (change windows, cascades), so some arrive close together.
    followers = starts[rng.random(count) < cfg.concurrent_fraction]
    starts = np.sort(np.concatenate([starts, followers + rng.integers(0, 21, followers.size)]))
    busy_until: dict[str, int] = {}

    for start in starts.tolist():
        fault_type = FAULT_TYPES[int(rng.integers(len(FAULT_TYPES)))]
        fault = FAULTS[fault_type]
        candidates = [s for s in SERVICES if KIND[s] in fault.kinds and busy_until.get(s, -1) < start]
        if not candidates:
            continue
        root = candidates[int(rng.integers(len(candidates)))]
        length = int(rng.integers(12, 51))
        severity = float(rng.uniform(0.35, 1.6))
        incident_id = len(incidents)
        busy_until[root] = start + length + 10
        build = _build_id(rng)  # one release per incident, so every log line names the same build
        silent = rng.random() < cfg.silent_fraction  # not every fault announces itself in logs

        profile = _profile(fault, length) * severity
        _apply(data, index[root], start, profile,
               fault.latency_x, fault.error_add, fault.cpu_add, fault.throughput_x)
        affected = {root: (start, start + length - 1)}

        # Propagate against the dependency edges, with delay, attenuation and drop-outs.
        frontier = [(root, 0, 0)]
        while frontier:
            service, hop, delay = frontier.pop(0)
            if hop >= 3:
                continue
            for dependent in sorted(graph.predecessors(service)):
                if dependent in affected or rng.random() > 0.85:
                    continue
                lag = delay + int(rng.integers(0, 4))  # synchronous calls can degrade at once
                if lag >= length - 3:
                    continue
                strength = (0.55 ** (hop + 1)) * rng.uniform(0.7, 1.1)
                _apply(data, index[dependent], start + lag, profile[: length - lag] * strength,
                       fault.spread_latency, fault.spread_error, 0.0, 0.08)
                affected[dependent] = (start + lag, start + length - 1)
                frontier.append((dependent, hop + 1, lag))

        for service, (first, last) in affected.items():
            row = labels[index[service], first : last + 1]
            row[row == -1] = incident_id
            is_root = service == root
            rate = (0.7 if is_root else 0.25) * min(severity, 1.2)
            if is_root and silent:
                rate = 0.0
            for minute in range(first, last + 1):
                if rng.random() < rate:
                    if is_root:
                        template = fault.logs[int(rng.integers(len(fault.logs)))]
                        message = _fill(template, rng, build=build)
                    else:
                        deps = [d for d in graph.successors(service) if d in affected]
                        dep = deps[int(rng.integers(len(deps)))] if deps else root
                        template = DEPENDENT_LOGS[int(rng.integers(len(DEPENDENT_LOGS)))]
                        message = _fill(template, rng, dep=dep)
                    level, _, text = message.partition(" ")
                    log_rows.append((minute, service, level, text))

        incidents.append(Incident(incident_id, root, fault_type, start, start + length - 1,
                                  round(severity, 3), affected, silent))

    # Background chatter, including benign WARN/ERROR lines that are not evidence of a fault.
    for service in SERVICES:
        for minute in np.flatnonzero(rng.random(total) < 1 / 40).tolist():
            if minute < cfg.history_start:
                continue
            template = NOISE_LOGS[int(rng.integers(len(NOISE_LOGS)))]
            level, _, text = _fill(template, rng).partition(" ")
            log_rows.append((minute, service, level, text))

    np.clip(data[:, :, CPU], 1.0, 100.0, out=data[:, :, CPU])
    np.clip(data[:, :, ERR], 0.0, 1.0, out=data[:, :, ERR])
    np.clip(data[:, :, THR], 0.0, None, out=data[:, :, THR])

    logs = pd.DataFrame(log_rows, columns=["ts", "service", "level", "message"])
    logs = logs.sort_values(["ts", "service"], kind="stable").reset_index(drop=True)
    return Dataset(cfg, data.astype(np.float32), labels, incidents, logs)


def service_graph() -> nx.DiGraph:
    return build_graph()
