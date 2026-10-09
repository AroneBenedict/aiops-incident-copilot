"""Render the evaluation results as a Markdown report."""

from __future__ import annotations


def _cell(summary: dict, key: str, percent: bool = True) -> str:
    if key not in summary:
        return "n/a"
    mean, std = summary[key]["mean"], summary[key]["std"]
    return f"{mean:.1%} ± {std:.1%}" if percent else f"{mean:.2f} ± {std:.2f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows] + [""]


def render(result: dict) -> str:
    s, seeds = result["summary"], result["seeds"]
    data = [m["data"] for m in result["per_seed"]]
    tested = sum(d["test_incidents"] for d in data)
    selected = {
        stage: sorted({m[stage]["selected"] for m in result["per_seed"]})
        for stage in ("detection", "correlation", "root_cause", "retrieval")
    }
    out = [
        "# Evaluation report",
        "",
        f"Seeds: {', '.join(map(str, seeds))}. Values are mean ± standard deviation across seeds, "
        f"measured on the held-out test period ({tested} test incidents in total). "
        "Thresholds and model choices were made on the earlier history period only.",
        "",
        "## Anomaly detection (point-wise F1 over service-minutes)",
        "",
    ]
    out += _table(["Detector", "F1"], [
        ["Rolling z-score (baseline)", _cell(s, "detection_f1_rolling_baseline")],
        ["Isolation Forest", _cell(s, "detection_f1_isolation_forest")],
        ["LSTM autoencoder (PyTorch)", _cell(s, "detection_f1_lstm_autoencoder")],
        ["Seasonal residual z-score", _cell(s, "detection_f1_seasonal_residual")],
    ])
    out += [
        f"Selected on history: {', '.join(selected['detection'])}. "
        f"Incidents that raised at least one alarm of three or more minutes: "
        f"{_cell(s, 'alarm_level_incident_recall')}. "
        f"By the looser test of any single flagged minute on an affected service, recall is "
        f"{_cell(s, 'event_recall')} with a median delay of {_cell(s, 'median_delay_min', percent=False)} "
        "minutes; that test is easy to pass by chance and is reported only for completeness.",
        "",
        "## Alarm correlation (adjusted Rand index against true incidents)",
        "",
    ]
    out += _table(["Method", "ARI"], [
        ["Time window only (baseline)", _cell(s, "correlation_ari_time_only")],
        ["Time window + topology adjacency", _cell(s, "correlation_ari_time_and_topology")],
        ["Time + topology + root split", _cell(s, "correlation_ari_root_split")],
    ])
    out += [
        f"Selected on history: {', '.join(selected['correlation'])}. "
        f"Alarms collapsed into incidents: {_cell(s, 'noise_reduction')} fewer items to triage.",
        "",
        "## Root-cause ranking (top-1 accuracy on detected incidents)",
        "",
    ]
    out += _table(["Method", "Top-1"], [
        ["Earliest alarm (baseline)", _cell(s, "rca_top1_earliest_alarm")],
        ["Highest anomaly score (baseline)", _cell(s, "rca_top1_peak_score")],
        ["Random walk on dependency graph", _cell(s, "rca_top1_random_walk")],
    ])
    out += [
        f"Selected on history: {', '.join(selected['root_cause'])}. "
        f"Random-walk top-3 accuracy: {_cell(s, 'rca_top3_random_walk')}.",
        "",
        "## Fault-type classification (8 classes, at the predicted root)",
        "",
    ]
    out += _table(["Model", "Accuracy", "Macro F1"], [
        ["Service-kind lookup (baseline)", _cell(s, "fault_accuracy_kind_lookup"), "n/a"],
        ["Random Forest on metric signature", _cell(s, "fault_accuracy_random_forest"),
         _cell(s, "fault_macro_f1_random_forest")],
    ])
    out += ["## Runbook retrieval (12 runbooks)", ""]
    out += _table(["Query source", "Retriever", "Hit@1"], [
        ["Incident logs + metric symptoms", "TF-IDF", _cell(s, "retrieval_hit1_tfidf")],
        ["Incident logs + metric symptoms", "BM25", _cell(s, "retrieval_hit1_bm25")],
        ["32 paraphrased operator queries", "TF-IDF", _cell(s, "retrieval_paraphrased_hit1_tfidf")],
        ["32 paraphrased operator queries", "BM25", _cell(s, "retrieval_paraphrased_hit1_bm25")],
    ])
    out += [
        f"Paraphrased-query Hit@3 (TF-IDF): {_cell(s, 'retrieval_paraphrased_hit3_tfidf')}. "
        "The paraphrased set is fixed, so its spread across seeds is zero.",
        "",
        "## Agent, end to end (rule planner)",
        "",
    ]
    out += _table(["Measure", "Value"], [
        ["Root cause correct", _cell(s, "agent_root_cause_accuracy")],
        ["Fault type correct", _cell(s, "agent_fault_type_accuracy")],
        ["Runbook correct", _cell(s, "agent_runbook_accuracy")],
        ["All three correct (detected incidents)", _cell(s, "agent_fully_correct")],
        ["All three correct (all test incidents, including missed ones)", _cell(s, "agent_end_to_end_fully_correct")],
        ["All three correct when the root logged its fault", _cell(s, "agent_fully_correct_with_logs")],
        ["All three correct when the root logged nothing useful", _cell(s, "agent_fully_correct_without_logs")],
        ["Tool calls per investigation", _cell(s, "agent_avg_tool_calls", percent=False)],
    ])
    out += ["### Is the agent's confidence label informative?", ""]
    calibration = result["confidence_calibration"]
    out += _table(["Confidence", "Reports", "All three correct"], [
        [level, str(calibration[level]["n"]),
         "n/a" if calibration[level]["accuracy"] is None else f"{calibration[level]['accuracy']:.1%}"]
        for level in ("high", "medium", "low")
    ])
    return "\n".join(out)
