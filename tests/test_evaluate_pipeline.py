import json
import math

import numpy as np

from aiops.correlate import Alarm, IncidentGroup
from aiops.detect import IsolationForestDetector, RollingZScore, SeasonalResidual
from aiops.evaluate import (
    classification,
    correlation_quality,
    event_level,
    match_groups,
    pointwise,
    ranking,
    rca_accuracy,
)
from aiops.pipeline import HEADLINE, aggregate, build, main, pooled_confidence
from aiops.rca import rank_peak
from aiops.simulate import Incident
from aiops.topology import SERVICES, build_graph
from aiops.tracking import log_run, read_runs


def test_pointwise_metrics():
    flags = np.array([[1, 1, 0, 0, 1]], dtype=bool)
    truth = np.array([[1, 0, 0, 1, 1]], dtype=bool)
    assert pointwise(flags, truth) == {"precision": 0.6667, "recall": 0.6667, "f1": 0.6667}
    assert pointwise(np.zeros_like(flags), truth)["f1"] == 0.0


def test_event_level_recall_and_delay():
    flags = np.zeros((len(SERVICES), 100), dtype=bool)
    flags[SERVICES.index("orders"), 23:30] = True
    caught = Incident(0, "orders-db", "disk_io_saturation", 1020, 1040, 1.0,
                      {"orders-db": (1020, 1040), "orders": (1021, 1040)})
    missed = Incident(1, "cache", "cache_eviction_storm", 1060, 1080, 1.0, {"cache": (1060, 1080)})
    result = event_level(flags, [caught, missed], offset=1000)
    assert result == {"incidents": 2, "detected": 1, "event_recall": 0.5, "median_delay_min": 3.0}
    assert math.isnan(event_level(flags, [missed], offset=1000)["median_delay_min"])


def _groups():
    alarms = [Alarm(0, "orders-db", 10, 20, 9.0), Alarm(1, "orders", 11, 20, 5.0),
              Alarm(2, "cache", 50, 60, 4.0), Alarm(3, "search", 90, 95, 2.0)]
    groups = [IncidentGroup(0, [0, 1], ["orders", "orders-db"], 10, 20),
              IncidentGroup(1, [2], ["cache"], 50, 60), IncidentGroup(2, [3], ["search"], 90, 95)]
    return alarms, groups, np.array([0, 0, 1, -1])


def test_correlation_quality_and_matching():
    alarms, groups, labels = _groups()
    quality = correlation_quality(groups, labels)
    assert quality == {"alarms": 4, "groups": 3, "noise_reduction": 0.25, "ari": 1.0}
    matched = match_groups(groups, labels)
    assert {incident: group.id for incident, group in matched.items()} == {0: 0, 1: 1}


def test_rca_accuracy_counts_undetected_incidents_as_misses():
    alarms, groups, labels = _groups()
    incidents = [Incident(0, "orders-db", "disk_io_saturation", 10, 20, 1.0, {}),
                 Incident(1, "auth", "bad_deploy", 50, 60, 1.0, {}),       # group blames cache: wrong
                 Incident(2, "search", "cpu_saturation", 200, 210, 1.0, {})]  # never alarmed
    result = rca_accuracy(lambda g, a, graph: rank_peak(g, a), match_groups(groups, labels), incidents,
                          alarms, build_graph())
    assert result == {"scored": 2, "top1": 0.5, "top3": 0.5, "end_to_end_top1": 0.3333}


def test_classification_and_ranking_metrics():
    assert classification(["a", "b", "a"], ["a", "b", "b"])["accuracy"] == 0.6667
    assert classification([], []) == {"n": 0, "accuracy": 0.0, "macro_f1": 0.0}
    assert ranking([1, 2, None, 4]) == {"n": 4, "hit1": 0.25, "hit3": 0.5, "mrr": 0.4375}
    assert ranking([])["n"] == 0


def test_pipeline_metrics_are_complete_and_sane(built, cfg):
    metrics = built.metrics
    assert {"config", "data", "detection", "correlation", "root_cause", "fault_classification",
            "retrieval", "agent"} <= set(metrics)
    assert metrics["detection"]["selected"] in metrics["detection"]["detectors"]
    assert metrics["correlation"]["selected"] in metrics["correlation"]["methods"]
    assert metrics["agent"]["completed"] == metrics["agent"]["investigated"] > 0
    for detector in metrics["detection"]["detectors"].values():
        assert 0 <= detector["precision"] <= 1 and 0 <= detector["recall"] <= 1
    assert metrics["retrieval"]["paraphrased_queries"]["tfidf"]["n"] == 32
    assert json.loads(json.dumps(metrics)) == metrics  # plain JSON types only
    assert len(built.alarm_labels) == len(built.workspace.alarms)
    assert all(alarm.start >= cfg.test_start for alarm in built.workspace.alarms)


def test_pipeline_is_reproducible(built, cfg):
    again = build(cfg, detectors=[RollingZScore(), SeasonalResidual()]).metrics
    for metrics in (again, built.metrics):
        for detector in metrics["detection"]["detectors"].values():
            detector.pop("seconds", None)
    assert again == built.metrics


def test_aggregate_skips_missing_optional_columns(built):
    summary = aggregate([built.metrics, built.metrics])
    assert summary["detection_f1"]["std"] == 0.0
    # The fixture ran with two detectors only, so the other columns are simply absent.
    assert "detection_f1_lstm_autoencoder" not in summary
    assert "detection_f1_isolation_forest" not in summary
    assert set(summary) <= set(HEADLINE)


def test_cli_writes_metrics_and_a_run_log(tmp_path, monkeypatch):
    def fast_detectors(seed):
        return [RollingZScore(), SeasonalResidual(), IsolationForestDetector(seed=seed, max_fit_rows=5000)]

    monkeypatch.setattr("aiops.pipeline._detectors", fast_detectors)
    result = main(["--small", "--seeds", "3", "--out", str(tmp_path)])
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["seeds"] == [3] and saved["summary"] == result["summary"]
    assert set(saved["confidence_calibration"]) == {"high", "medium", "low"}
    runs = read_runs(tmp_path / "runs.jsonl")
    assert len(runs) == 1 and runs[0]["small"] is True and "git_sha" in runs[0]
    report = (tmp_path / "REPORT.md").read_text()
    assert report.startswith("# Evaluation report") and "Seeds: 3." in report
    assert "| LSTM autoencoder (PyTorch) | n/a |" in report  # not part of this fast run
    assert "| Seasonal residual z-score | n/a |" not in report


def test_pooled_confidence_weights_by_count():
    def seed(high_n, high_acc):
        return {"agent": {"fully_correct_by_confidence": {
            "high": {"n": high_n, "accuracy": high_acc}, "medium": {"n": 0, "accuracy": None},
            "low": {"n": 2, "accuracy": 0.5}}}}

    pooled = pooled_confidence([seed(10, 1.0), seed(30, 0.8)])
    assert pooled == {"high": {"n": 40, "accuracy": 0.85}, "medium": {"n": 0, "accuracy": None},
                      "low": {"n": 4, "accuracy": 0.5}}


def test_tracking_appends_one_line_per_run(tmp_path):
    path = tmp_path / "nested" / "runs.jsonl"
    assert read_runs(path) == []
    log_run(path, {"note": "first"})
    record = log_run(path, {"note": "second"})
    assert [run["note"] for run in read_runs(path)] == ["first", "second"]
    assert {"timestamp", "git_sha", "python"} <= set(record)
