import networkx as nx
import numpy as np

from aiops.config import METRICS, small_config
from aiops.simulate import FAULTS, simulate
from aiops.topology import KIND, SERVICES, build_graph


def test_shapes_and_ranges(dataset, cfg):
    assert dataset.metrics.shape == (len(SERVICES), cfg.total_minutes, len(METRICS))
    assert dataset.labels.shape == dataset.metrics.shape[:2]
    assert np.isfinite(dataset.metrics).all()
    cpu, err = dataset.metrics[..., 2], dataset.metrics[..., 1]
    assert cpu.min() >= 1 and cpu.max() <= 100
    assert err.min() >= 0 and err.max() <= 1


def test_same_seed_is_reproducible_and_seeds_differ(cfg, dataset):
    again = simulate(cfg)
    assert np.array_equal(again.metrics, dataset.metrics)
    assert again.logs.equals(dataset.logs)
    assert not np.array_equal(simulate(small_config(seed=8)).metrics, dataset.metrics)


def test_clean_period_has_no_incidents(dataset, cfg):
    assert (dataset.labels[:, : cfg.history_start] == -1).all()
    assert all(incident.start >= cfg.history_start for incident in dataset.incidents)
    assert dataset.logs["ts"].min() >= cfg.history_start


def test_incidents_follow_fault_and_topology_rules(dataset):
    graph = build_graph()
    assert len(dataset.incidents) > 10
    for incident in dataset.incidents:
        assert KIND[incident.root] in FAULTS[incident.fault_type].kinds
        assert incident.affected[incident.root] == (incident.start, incident.end)
        # Impact only reaches services that depend on the root, directly or transitively.
        callers = nx.ancestors(graph, incident.root)
        assert set(incident.affected) - {incident.root} <= callers
        for first, last in incident.affected.values():
            assert incident.start <= first <= last <= incident.end


def test_labels_cover_affected_windows(dataset):
    index = {service: i for i, service in enumerate(SERVICES)}
    for incident in dataset.incidents:
        for service, (first, last) in incident.affected.items():
            assert (dataset.labels[index[service], first : last + 1] >= 0).all()


def test_root_logs_respect_silent_flag(dataset):
    loud = [i for i in dataset.incidents if not i.silent and i.severity > 0.8 and i.end - i.start > 20]
    silent = [i for i in dataset.incidents if i.silent]
    assert loud and silent
    victim = ("upstream call", "received 503", "circuit breaker", "dependency ")
    noise = ("client disconnected", "retrying idempotent", "slow request", "request completed",
             "scheduled job", "configuration reloaded")

    def own_lines(incident):
        logs = dataset.logs_between(incident.root, incident.start, incident.end)["message"]
        return [m for m in logs if not any(p in m for p in victim + noise)]

    assert all(own_lines(i) for i in loud)
    # A silent root can still pick up lines from an overlapping incident, but most stay quiet.
    assert sum(not own_lines(i) for i in silent) >= len(silent) * 0.7


def test_topology_is_a_dag_with_known_kinds():
    graph = build_graph()
    assert nx.is_directed_acyclic_graph(graph)
    assert set(graph.nodes) == set(SERVICES) == set(KIND)
