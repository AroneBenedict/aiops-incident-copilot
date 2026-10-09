"""End-to-end pipeline: simulate, detect, correlate, diagnose, retrieve, investigate, evaluate.

Model selection (detector threshold, detector, RCA method, retriever) uses the *history*
period only. Every number in the report comes from the later, untouched *test* period.

    python -m aiops.pipeline --seeds 7 11 13 17 19
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .agent import Agent, RulePlanner, Workspace, incident_log_lines, own_fault_lines
from .config import Config, small_config
from .correlate import Alarm, IncidentGroup, build_alarms, correlate, label_alarms, split_by_root
from .detect import IsolationForestDetector, RollingZScore, SeasonalResidual, best_f1_threshold
from .evaluate import classification, correlation_quality, event_level, match_groups, pointwise, ranking, rca_accuracy
from .faults import FaultClassifier, KindLookup, describe, signature
from .features import SeasonalBaseline
from .rca import METHODS
from .report import render
from .retrieval import BM25Retriever, TfidfRetriever, build_query, chunk_runbooks, load_runbooks
from .simulate import Dataset, simulate
from .topology import KIND, build_graph
from .tracking import log_run


def default_runbook_dir() -> Path:
    """AIOPS_RUNBOOKS if set, else ./runbooks, else the runbooks folder of a source checkout."""
    if os.environ.get("AIOPS_RUNBOOKS"):
        return Path(os.environ["AIOPS_RUNBOOKS"])
    local = Path.cwd() / "runbooks"
    return local if local.is_dir() else Path(__file__).resolve().parents[2] / "runbooks"


@dataclass
class Built:
    workspace: Workspace
    alarm_labels: np.ndarray     # ground-truth incident id per workspace alarm (evaluation only)
    metrics: dict


def _detectors(seed: int) -> list:
    detectors = [RollingZScore(), SeasonalResidual(), IsolationForestDetector(seed=seed)]
    try:  # the deep-learning detector is optional: it needs PyTorch
        from .detect_lstm import LSTMAutoencoderDetector
    except ImportError:
        return detectors
    return detectors + [LSTMAutoencoderDetector(seed=seed)]


def _rank_of(retriever, query: str, target: str) -> int | None:
    names = [hit.runbook for hit in retriever.search(query, k=5)]
    return names.index(target) + 1 if target in names else None


def _retrieval_ranks(retriever, runbook_for: dict[str, str], dataset: Dataset, z: np.ndarray,
                     alarms: list[Alarm], cases: list[tuple[IncidentGroup, str, str]]) -> list[int | None]:
    """Rank of the correct runbook for each (group, service, true fault) case."""
    ranks: list[int | None] = []
    for group, service, fault in cases:
        lines = own_fault_lines(incident_log_lines(dataset, group, service)[:12])
        query = build_query(KIND[service], describe(z, group, alarms, service), lines)
        ranks.append(_rank_of(retriever, query, runbook_for[fault]))
    return ranks


def build(cfg: Config, runbook_dir: Path | str | None = None, detectors: list | None = None,
          correlation: str | None = None) -> Built:
    """Run every stage. ``correlation`` forces one correlation method (used for ablations);
    by default the method with the best adjusted Rand index on the history period is used."""
    runbook_dir = Path(runbook_dir) if runbook_dir else default_runbook_dir()
    dataset = simulate(cfg)
    graph = build_graph()
    hs, ts = cfg.history_start, cfg.test_start
    truth = dataset.labels >= 0
    clean = dataset.metrics[:, :hs]
    history_incidents = [i for i in dataset.incidents if i.start < ts]
    test_incidents = [i for i in dataset.incidents if i.start >= ts]
    metrics: dict = {"config": {"seed": cfg.seed, "fingerprint": cfg.fingerprint()},
                     "data": {"services": len(dataset.services), "minutes": cfg.total_minutes,
                              "history_incidents": len(history_incidents),
                              "test_incidents": len(test_incidents), "log_lines": int(len(dataset.logs))}}

    # 1. Anomaly detection: tune each detector's threshold on history, compare on test.
    detection, fitted = {}, {}
    for detector in detectors or _detectors(cfg.seed):
        began = time.perf_counter()
        scores = detector.fit(clean).score(dataset.metrics)
        threshold = best_f1_threshold(scores[:, hs:ts], truth[:, hs:ts])
        test_flags = scores[:, ts:] > threshold
        detection[detector.name] = {
            "threshold": round(threshold, 4),
            "history_f1": pointwise(scores[:, hs:ts] > threshold, truth[:, hs:ts])["f1"],
            **pointwise(test_flags, truth[:, ts:]),
            **event_level(test_flags, test_incidents, ts),
            "seconds": round(time.perf_counter() - began, 2),
        }
        fitted[detector.name] = (scores, threshold)
    chosen = max(detection, key=lambda name: detection[name]["history_f1"])
    scores, threshold = fitted[chosen]
    metrics["detection"] = {"selected": chosen, "detectors": detection}  # alarm-level recall is added below

    # 2. Alarms and correlation.
    z = SeasonalBaseline.fit(clean).residual_z(dataset.metrics)

    def alarms_for(lo: int, hi: int) -> tuple[list[Alarm], np.ndarray]:
        found = build_alarms(scores[:, lo:hi], threshold, cfg.alarm_min_duration, cfg.alarm_merge_gap, lo)
        return found, label_alarms(found, dataset.labels)

    h_alarms, h_labels = alarms_for(hs, ts)
    t_alarms, t_labels = alarms_for(ts, cfg.total_minutes)
    def variants(found: list[Alarm]) -> dict[str, list[IncidentGroup]]:
        adjacent = correlate(found, graph, cfg.correlation_gap)
        return {"time_only": correlate(found, None, cfg.correlation_gap),
                "time_and_topology": adjacent,
                "time_topology_root_split": split_by_root(adjacent, found, graph)}

    h_variants, t_variants = variants(h_alarms), variants(t_alarms)
    history_ari = {name: correlation_quality(groups, h_labels)["ari"] for name, groups in h_variants.items()}
    correlation_name = correlation or max(history_ari, key=history_ari.get)
    h_groups, t_groups = h_variants[correlation_name], t_variants[correlation_name]
    metrics["correlation"] = {
        "selected": correlation_name,
        "false_alarms_per_day": round(float((t_labels < 0).sum()) / cfg.test_days, 2),
        "methods": {name: {"history_ari": history_ari[name], **correlation_quality(groups, t_labels)}
                    for name, groups in t_variants.items()},
    }

    # 3. Root-cause ranking: pick the method on history, report all on test.
    h_matched, t_matched = match_groups(h_groups, h_labels), match_groups(t_groups, t_labels)
    history_rca = {name: rca_accuracy(fn, h_matched, history_incidents, h_alarms, graph)
                   for name, fn in METHODS.items()}
    rca_name = max(history_rca, key=lambda name: history_rca[name]["top1"])
    rank = METHODS[rca_name]
    metrics["root_cause"] = {
        "selected": rca_name,
        "methods": {name: {"history_top1": history_rca[name]["top1"],
                           **rca_accuracy(fn, t_matched, test_incidents, t_alarms, graph)}
                    for name, fn in METHODS.items()},
    }

    # 4. Fault classification: train on history at the true root, test at the predicted root.
    train = [(i, h_matched[i.id]) for i in history_incidents
             if i.id in h_matched and i.root in h_matched[i.id].services]
    classifier = FaultClassifier(seed=cfg.seed).fit(
        np.array([signature(z, group, h_alarms, incident.root) for incident, group in train]),
        [incident.fault_type for incident, _ in train])
    lookup = KindLookup().fit([KIND[i.root] for i, _ in train], [i.fault_type for i, _ in train])

    scored = [i for i in test_incidents if i.id in t_matched]
    # Stricter than event_recall: the incident must own at least one alarm of three or more minutes.
    metrics["detection"]["alarm_level_incident_recall"] = round(len(scored) / max(len(test_incidents), 1), 4)
    predicted_root = {i.id: rank(t_matched[i.id], t_alarms, graph)[0].service for i in scored}
    true_faults = [i.fault_type for i in scored]
    if scored:
        features = np.array([signature(z, t_matched[i.id], t_alarms, predicted_root[i.id]) for i in scored])
        predicted_faults = [p.fault_type for p in classifier.predict(features)]
    else:
        predicted_faults = []
    majority = lookup.default
    metrics["fault_classification"] = {
        "train_incidents": len(train),
        "random_forest": classification(true_faults, predicted_faults),
        "kind_lookup_baseline": classification(
            true_faults, lookup.predict([KIND[predicted_root[i.id]] for i in scored])),
        "majority_baseline": classification(true_faults, [majority] * len(scored)),
    }

    # 5. Runbook retrieval: pick the retriever on history, report both on test.
    runbooks = load_runbooks(runbook_dir)
    runbook_for = {book.fault_type: name for name, book in runbooks.items() if book.fault_type}
    chunks = chunk_runbooks(runbooks)
    retrievers = {r.name: r for r in (TfidfRetriever(chunks), BM25Retriever(chunks))}
    h_cases = [(group, incident.root, incident.fault_type) for incident, group in train]
    t_cases = [(t_matched[i.id], predicted_root[i.id], i.fault_type) for i in scored]
    history_retrieval = {name: ranking(_retrieval_ranks(r, runbook_for, dataset, z, h_alarms, h_cases))
                         for name, r in retrievers.items()}
    retriever_name = max(history_retrieval, key=lambda name: history_retrieval[name]["mrr"])
    metrics["retrieval"] = {
        "selected": retriever_name,
        "runbooks": len(runbooks),
        "retrievers": {name: {"history_mrr": history_retrieval[name]["mrr"],
                              **ranking(_retrieval_ranks(r, runbook_for, dataset, z, t_alarms, t_cases))}
                       for name, r in retrievers.items()},
    }

    query_file = runbook_dir.parent / "data" / "paraphrased_queries.json"
    queries = json.loads(query_file.read_text(encoding="utf-8"))["queries"] if query_file.exists() else []
    metrics["retrieval"]["paraphrased_queries"] = {
        name: ranking([_rank_of(r, query["text"], runbook_for[query["fault"]]) for query in queries])
        for name, r in retrievers.items()}

    workspace = Workspace(cfg, dataset, z, t_alarms, t_groups, graph, classifier,
                          retrievers[retriever_name], runbooks, rank)

    # 6. Agent: full investigation of every detected test incident with the rule planner.
    agent = Agent(workspace, RulePlanner(cfg.rca_margin_for_log_check))
    root_ok = fault_ok = runbook_ok = all_ok = complete = reranked = calls = silent_total = silent_ok = 0
    by_confidence: dict[str, list[bool]] = {"high": [], "medium": [], "low": []}
    for incident in scored:
        report = agent.investigate(t_matched[incident.id].id)
        complete += report.status == "complete"
        calls += len(report.trace)
        reranked += "chosen over the top-ranked candidate" in report.summary
        silent_total += incident.silent
        r = report.root_cause == incident.root
        f = report.fault_type == incident.fault_type
        b = report.runbook == runbook_for[incident.fault_type]
        root_ok, fault_ok, runbook_ok, all_ok = root_ok + r, fault_ok + f, runbook_ok + b, all_ok + (r and f and b)
        silent_ok += incident.silent and r and f and b
        if report.status == "complete":
            by_confidence[report.confidence].append(bool(r and f and b))
    n = max(len(scored), 1)
    metrics["agent"] = {
        "planner": "rule",
        "investigated": len(scored),
        "completed": complete,
        "avg_tool_calls": round(calls / n, 2),
        "log_check_changed_root": reranked,
        "root_cause_accuracy": round(root_ok / n, 4),
        "fault_type_accuracy": round(fault_ok / n, 4),
        "runbook_accuracy": round(runbook_ok / n, 4),
        "fully_correct": round(all_ok / n, 4),
        "end_to_end_fully_correct": round(all_ok / max(len(test_incidents), 1), 4),
        "fully_correct_with_logs": round((all_ok - silent_ok) / max(len(scored) - silent_total, 1), 4),
        "fully_correct_without_logs": round(silent_ok / max(silent_total, 1), 4),
        "incidents_without_logs": int(silent_total),
        "fully_correct_by_confidence": {
            level: {"n": len(v), "accuracy": round(float(np.mean(v)), 4) if v else None}
            for level, v in by_confidence.items()},
    }
    return Built(workspace, t_labels, metrics)


# --------------------------------------------------------------------------- reporting

HEADLINE = {
    "detection_f1": lambda m: m["detection"]["detectors"][m["detection"]["selected"]]["f1"],
    "detection_f1_rolling_baseline": lambda m: m["detection"]["detectors"]["rolling_zscore"]["f1"],
    "detection_f1_isolation_forest": lambda m: m["detection"]["detectors"]["isolation_forest"]["f1"],
    "detection_f1_seasonal_residual": lambda m: m["detection"]["detectors"]["seasonal_residual"]["f1"],
    "event_recall": lambda m: m["detection"]["detectors"][m["detection"]["selected"]]["event_recall"],
    "median_delay_min": lambda m: m["detection"]["detectors"][m["detection"]["selected"]]["median_delay_min"],
    "alarm_level_incident_recall": lambda m: m["detection"]["alarm_level_incident_recall"],
    "noise_reduction": lambda m: m["correlation"]["methods"][m["correlation"]["selected"]]["noise_reduction"],
    "correlation_ari_time_only": lambda m: m["correlation"]["methods"]["time_only"]["ari"],
    "correlation_ari_time_and_topology": lambda m: m["correlation"]["methods"]["time_and_topology"]["ari"],
    "correlation_ari_root_split": lambda m: m["correlation"]["methods"]["time_topology_root_split"]["ari"],
    "rca_top1_random_walk": lambda m: m["root_cause"]["methods"]["random_walk"]["top1"],
    "rca_top1_earliest_alarm": lambda m: m["root_cause"]["methods"]["earliest_alarm"]["top1"],
    "rca_top1_peak_score": lambda m: m["root_cause"]["methods"]["peak_score"]["top1"],
    "rca_top3_random_walk": lambda m: m["root_cause"]["methods"]["random_walk"]["top3"],
    "fault_accuracy_random_forest": lambda m: m["fault_classification"]["random_forest"]["accuracy"],
    "fault_macro_f1_random_forest": lambda m: m["fault_classification"]["random_forest"]["macro_f1"],
    "fault_accuracy_kind_lookup": lambda m: m["fault_classification"]["kind_lookup_baseline"]["accuracy"],
    "retrieval_hit1_tfidf": lambda m: m["retrieval"]["retrievers"]["tfidf"]["hit1"],
    "retrieval_hit1_bm25": lambda m: m["retrieval"]["retrievers"]["bm25"]["hit1"],
    "retrieval_paraphrased_hit1_tfidf": lambda m: m["retrieval"]["paraphrased_queries"]["tfidf"]["hit1"],
    "retrieval_paraphrased_hit3_tfidf": lambda m: m["retrieval"]["paraphrased_queries"]["tfidf"]["hit3"],
    "retrieval_paraphrased_hit1_bm25": lambda m: m["retrieval"]["paraphrased_queries"]["bm25"]["hit1"],
    "agent_root_cause_accuracy": lambda m: m["agent"]["root_cause_accuracy"],
    "agent_fault_type_accuracy": lambda m: m["agent"]["fault_type_accuracy"],
    "agent_runbook_accuracy": lambda m: m["agent"]["runbook_accuracy"],
    "agent_fully_correct": lambda m: m["agent"]["fully_correct"],
    "agent_end_to_end_fully_correct": lambda m: m["agent"]["end_to_end_fully_correct"],
    "agent_avg_tool_calls": lambda m: m["agent"]["avg_tool_calls"],
    "agent_fully_correct_with_logs": lambda m: m["agent"]["fully_correct_with_logs"],
    "agent_fully_correct_without_logs": lambda m: m["agent"]["fully_correct_without_logs"],
    # Present only when PyTorch is installed.
    "detection_f1_lstm_autoencoder": lambda m: m["detection"]["detectors"].get("lstm_autoencoder", {}).get("f1"),
}


def _headline(getter, metrics: dict):
    try:
        return getter(metrics)
    except KeyError:  # that detector or method was not part of this run
        return None


def aggregate(per_seed: list[dict]) -> dict:
    summary = {}
    for key, getter in HEADLINE.items():
        values = [v for v in (_headline(getter, m) for m in per_seed) if v is not None]
        if values:
            summary[key] = {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4),
                            "min": round(float(np.min(values)), 4), "max": round(float(np.max(values)), 4)}
    return summary


def pooled_confidence(per_seed: list[dict]) -> dict:
    """Share of fully correct reports at each confidence level, pooled over all seeds."""
    out = {}
    for level in ("high", "medium", "low"):
        cells = [m["agent"]["fully_correct_by_confidence"][level] for m in per_seed]
        n = sum(cell["n"] for cell in cells)
        correct = sum(cell["n"] * cell["accuracy"] for cell in cells if cell["n"])
        out[level] = {"n": n, "accuracy": round(correct / n, 4) if n else None}
    return out


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13, 17, 19])
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--small", action="store_true", help="use the fast test-sized configuration")
    args = parser.parse_args(argv)

    per_seed = []
    for seed in args.seeds:
        cfg = small_config(seed) if args.small else Config(seed=seed)
        began = time.perf_counter()
        metrics = build(cfg).metrics
        metrics["seconds"] = round(time.perf_counter() - began, 1)
        per_seed.append(metrics)
        selected = metrics["detection"]["selected"]
        print(f"seed {seed}: detector={selected} f1={metrics['detection']['detectors'][selected]['f1']:.3f} "
              f"rca_top1={metrics['root_cause']['methods'][metrics['root_cause']['selected']]['top1']:.3f} "
              f"agent_fully_correct={metrics['agent']['fully_correct']:.3f} ({metrics['seconds']}s)")

    result = {"seeds": args.seeds, "summary": aggregate(per_seed),
              "confidence_calibration": pooled_confidence(per_seed), "per_seed": per_seed}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.out / "REPORT.md").write_text(render(result) + "\n", encoding="utf-8")
    log_run(args.out / "runs.jsonl", {"seeds": args.seeds, "small": args.small, "summary": result["summary"]})
    print(f"wrote {args.out / 'metrics.json'} and {args.out / 'REPORT.md'}")
    return result


if __name__ == "__main__":
    main()
