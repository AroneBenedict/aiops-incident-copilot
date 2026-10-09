import json
from pathlib import Path

import numpy as np
import pytest

from aiops.correlate import Alarm, IncidentGroup
from aiops.faults import FEATURE_NAMES, FaultClassifier, KindLookup, describe, service_window, signature
from aiops.retrieval import BM25Retriever, TfidfRetriever, build_query, chunk_runbooks, load_runbooks, tokenize
from aiops.simulate import FAULT_TYPES
from aiops.topology import SERVICES

ROOT = Path(__file__).resolve().parents[1]
RUNBOOKS = load_runbooks(ROOT / "runbooks")
CHUNKS = chunk_runbooks(RUNBOOKS)


def test_every_fault_type_has_exactly_one_runbook():
    faults = [book.fault_type for book in RUNBOOKS.values() if book.fault_type]
    assert sorted(faults) == sorted(FAULT_TYPES)
    assert len(RUNBOOKS) > len(faults)  # distractor runbooks exist


def test_runbook_parsing_extracts_sections_and_approval_flags():
    book = RUNBOOKS["bad-deploy"]
    assert book.title == "Error spike after a deployment"
    assert {"Symptoms", "Diagnosis", "Remediation", "Escalation"} <= set(book.sections)
    steps = book.remediation()
    assert len(steps) == 4
    assert steps[1].text == "Roll back to the previous version." and steps[1].requires_approval
    assert not steps[0].requires_approval
    assert all("(requires approval)" not in step.text for step in steps)


def test_every_runbook_gates_at_least_one_step():
    for book in RUNBOOKS.values():
        steps = book.remediation()
        assert steps and any(step.requires_approval for step in steps), book.name


def test_load_runbooks_fails_loudly_on_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_runbooks(tmp_path)


def test_tokenize_drops_stop_words_and_numbers():
    assert tokenize("The pool is exhausted, 42 callers waiting!") == ["pool", "exhausted", "callers", "waiting"]


@pytest.mark.parametrize("retriever", [TfidfRetriever(CHUNKS), BM25Retriever(CHUNKS)], ids=["tfidf", "bm25"])
@pytest.mark.parametrize("query,expected", [
    ("could not acquire connection from pool, too many clients already", "db-connection-exhaustion"),
    ("consumer lag growing on partition, broker rejected publish", "queue-backlog"),
    ("container OOMKilled exit code 137, heap usage near limit", "memory-leak"),
    ("interface input errors CRC drops, BGP neighbor flap", "network-packet-loss"),
])
def test_retrievers_find_the_matching_runbook(retriever, query, expected):
    hits = retriever.search(query, k=3)
    assert hits[0].runbook == expected
    assert len({hit.runbook for hit in hits}) == len(hits)  # one hit per runbook
    assert hits == sorted(hits, key=lambda hit: -hit.score)


def test_paraphrased_queries_file_covers_all_faults():
    queries = json.loads((ROOT / "data" / "paraphrased_queries.json").read_text())["queries"]
    assert {q["fault"] for q in queries} == set(FAULT_TYPES)
    assert len(queries) == 32


def test_build_query_describes_symptoms_and_dedupes_logs():
    deviations = {"latency_ms": 8.0, "error_rate": 0.4, "cpu_pct": 1.0, "throughput_rps": -5.0}
    logs = ["WARN fsync took 900ms", "WARN fsync took 1200ms", "WARN iowait at 80%"]
    query = build_query("database", deviations, logs)
    assert query.startswith("database. high latency, error rate normal, cpu normal, throughput drops.")
    assert query.count("fsync") == 1 and "iowait" in query


def _group_with_alarm(service="orders", start=50, end=79):
    alarms = [Alarm(0, service, start, end, 5.0)]
    return IncidentGroup(0, [0], [service], start, end), alarms


def test_signature_matches_feature_names_and_reads_the_window():
    z = np.zeros((len(SERVICES), 200, 4))
    index = SERVICES.index("orders")
    z[index, 50:80, 0] = np.linspace(1, 9, 30)   # latency ramps up
    z[index, 50:80, 3] = -4.0                    # throughput drops
    group, alarms = _group_with_alarm()
    features = signature(z, group, alarms, "orders")
    named = dict(zip(FEATURE_NAMES, features, strict=True))
    assert len(features) == len(FEATURE_NAMES)
    assert named["mean_z_latency_ms"] == pytest.approx(5.0)
    assert named["extreme_z_throughput_rps"] == pytest.approx(-4.0)
    assert named["latency_trend"] > 0.99
    assert named["kind_app"] == 1.0 and named["kind_database"] == 0.0
    assert describe(z, group, alarms, "orders") == {"latency_ms": 5.0, "error_rate": 0.0, "cpu_pct": 0.0,
                                                   "throughput_rps": -4.0}
    with pytest.raises(KeyError):
        service_window(group, alarms, "cache")


def test_classifier_learns_separable_faults_and_reports_alternatives():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.3, (40, 5)) + [6, 0, 0, 0, 0]
    b = rng.normal(0, 0.3, (40, 5)) + [0, 6, 0, 0, 0]
    model = FaultClassifier(seed=0).fit(np.vstack([a, b]), ["slow"] * 40 + ["failing"] * 40)
    prediction = model.predict(np.array([6, 0, 0, 0, 0]))[0]
    assert prediction.fault_type == "slow" and prediction.confidence > 0.9
    assert [name for name, _ in prediction.alternatives] == ["slow", "failing"]


def test_kind_lookup_baseline():
    lookup = KindLookup().fit(["database", "database", "app", "database"], ["io", "pool", "deploy", "pool"])
    assert lookup.predict(["database", "app", "queue"]) == ["pool", "deploy", "pool"]
