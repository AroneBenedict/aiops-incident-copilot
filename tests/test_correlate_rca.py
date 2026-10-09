import numpy as np

from aiops.correlate import Alarm, build_alarms, correlate, group_assignment, label_alarms, split_by_root
from aiops.rca import rank_earliest, rank_peak, rank_random_walk
from aiops.topology import SERVICES, build_graph

GRAPH = build_graph()
IDX = {service: i for i, service in enumerate(SERVICES)}


def alarms_of(*specs):
    return [Alarm(n, service, start, end, peak) for n, (service, start, end, peak) in enumerate(specs)]


def test_build_alarms_applies_duration_gap_and_offset():
    scores = np.zeros((len(SERVICES), 60))
    scores[IDX["orders"], 10:14] = 5      # 4 minutes -> alarm
    scores[IDX["orders"], 16:20] = 7      # 2-minute gap -> merged into the same alarm
    scores[IDX["orders"], 40:42] = 9      # 2 minutes -> too short
    scores[IDX["cache"], 30:35] = 3
    alarms = build_alarms(scores, threshold=1.0, min_duration=3, merge_gap=2, offset=1000)
    assert [(a.service, a.start, a.end, a.peak) for a in alarms] == [
        ("orders", 1010, 1019, 7.0), ("cache", 1030, 1034, 3.0)]
    assert [a.id for a in alarms] == [0, 1]
    assert alarms[0].duration == 10


def test_label_alarms_uses_majority_and_marks_spurious():
    labels = np.full((len(SERVICES), 100), -1)
    labels[IDX["orders"], 10:20] = 4
    alarms = alarms_of(("orders", 10, 19, 1.0), ("orders", 18, 30, 1.0), ("cache", 50, 55, 1.0))
    assert label_alarms(alarms, labels).tolist() == [4, -1, -1]


def test_correlation_needs_time_and_topology():
    alarms = alarms_of(("orders-db", 100, 120, 9), ("orders", 102, 121, 5), ("api-gateway", 104, 119, 3),
                       ("search-index", 103, 118, 6),          # same time, unrelated subtree
                       ("orders", 300, 310, 4))                # same service, much later
    groups = correlate(alarms, GRAPH, gap=5)
    assert sorted(g.services for g in groups) == [["api-gateway", "orders", "orders-db"], ["orders"], ["search-index"]]
    time_only = correlate(alarms, None, gap=5)
    assert len(time_only) == 2 and len(time_only[0].alarm_ids) == 4
    assert group_assignment(groups, len(alarms)).tolist() == [0, 0, 0, 1, 2]


def test_split_by_root_separates_two_overlapping_faults():
    # Two faults at once: orders-db and search-index. api-gateway depends on both and glues them.
    alarms = alarms_of(("orders-db", 100, 130, 9), ("orders", 102, 130, 5), ("search-index", 110, 140, 8),
                       ("search", 111, 140, 5), ("api-gateway", 103, 140, 4))
    merged = correlate(alarms, GRAPH, gap=5)
    assert len(merged) == 1
    split = split_by_root(merged, alarms, GRAPH)
    assert sorted(g.services for g in split) == [["api-gateway", "orders", "orders-db"], ["search", "search-index"]]
    assert sorted(n for g in split for n in g.alarm_ids) == [0, 1, 2, 3, 4]


def test_split_by_root_keeps_a_single_fault_whole():
    # The intermediate service (orders) is not alarmed, but the gateway still depends on orders-db.
    alarms = alarms_of(("orders-db", 100, 130, 9), ("orders", 101, 130, 6), ("api-gateway", 103, 128, 4))
    groups = correlate(alarms, GRAPH, gap=5)
    assert split_by_root(groups, alarms, GRAPH) == groups


def test_random_walk_finds_the_dependency_even_when_a_caller_alarms_first_and_loudest():
    alarms = alarms_of(("api-gateway", 100, 130, 9.0), ("orders", 101, 130, 6.0), ("orders-db", 103, 130, 5.0))
    group = correlate(alarms, GRAPH, gap=5)[0]
    assert rank_earliest(group, alarms)[0].service == "api-gateway"
    assert rank_peak(group, alarms)[0].service == "api-gateway"
    ranked = rank_random_walk(group, alarms, GRAPH)
    assert ranked[0].service == "orders-db"
    assert abs(sum(c.score for c in ranked) - 1) < 1e-6
    assert [c.service for c in ranked] == ["orders-db", "orders", "api-gateway"]


def test_single_service_group_has_one_candidate():
    alarms = alarms_of(("cache", 10, 20, 3.0))
    group = correlate(alarms, GRAPH, gap=5)[0]
    for rank in (rank_earliest, rank_peak, lambda g, a: rank_random_walk(g, a, GRAPH)):
        ranked = rank(group, alarms)
        assert [(c.service, c.score) for c in ranked] == [("cache", 1.0)]
