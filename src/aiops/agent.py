"""Incident-investigation agent.

The agent is a loop: a planner picks a tool, the tool runs against the workspace, and the
observation goes back to the planner, until the planner submits a report or runs out of
steps. Two planners implement the same interface:

* ``RulePlanner`` is deterministic and needs no network. Tests and the offline evaluation use it.
* ``ClaudePlanner`` (see ``llm.py``) lets an LLM choose the tool calls.

Guardrails that hold for every planner:

* Every tool is read-only. The agent can propose remediation but has no way to execute it.
* Whether a step needs human approval comes from the runbook, never from the planner.
* Tool calls are validated; an unknown tool or bad argument is returned as an error
  observation instead of crashing the loop.
* The loop is bounded by ``max_steps`` and every step is recorded in a trace.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import networkx as nx
import numpy as np

from .config import Config
from .correlate import Alarm, IncidentGroup
from .faults import FaultClassifier, describe, signature
from .rca import rank_random_walk
from .retrieval import Runbook, build_query
from .simulate import Dataset
from .topology import KIND

FINAL_TOOL = "submit_report"


# --------------------------------------------------------------------------- workspace


@dataclass
class Workspace:
    """Everything the tools are allowed to read."""

    cfg: Config
    dataset: Dataset
    z: np.ndarray                      # residual z-scores, (services, minutes, metrics)
    alarms: list[Alarm]
    groups: list[IncidentGroup]
    graph: nx.DiGraph
    classifier: FaultClassifier
    retriever: Any                     # anything with .search(query, k) -> list[Hit]
    runbooks: dict[str, Runbook]
    rank: Callable[..., list] = rank_random_walk

    def group(self, incident_id: int) -> IncidentGroup:
        if not 0 <= incident_id < len(self.groups):
            raise KeyError(f"unknown incident_id {incident_id}")
        return self.groups[incident_id]


# --------------------------------------------------------------------------- tools


def _require_service(ws: Workspace, incident_id: int, service: str) -> IncidentGroup:
    group = ws.group(incident_id)
    if service not in group.services:
        raise ValueError(f"{service!r} has no alarm in incident {incident_id}; choose from {group.services}")
    return group


def incident_log_lines(dataset: Dataset, group: IncidentGroup, service: str) -> list[str]:
    """WARN and ERROR lines of one service, from two minutes before the incident to its end."""
    logs = dataset.logs_between(service, group.start - 2, group.end)
    logs = logs[logs["level"].isin(["WARN", "ERROR"])]
    return [f"{level} {message}" for level, message in zip(logs["level"], logs["message"], strict=True)]


def tool_get_incident(ws: Workspace, incident_id: int) -> dict:
    group = ws.group(incident_id)
    return {
        "incident_id": group.id,
        "start_minute": group.start,
        "end_minute": group.end,
        "services": group.services,
        "alarms": [
            {"service": ws.alarms[n].service, "start": ws.alarms[n].start, "end": ws.alarms[n].end,
             "peak_score": round(ws.alarms[n].peak, 2)}
            for n in group.alarm_ids
        ],
    }


def tool_rank_root_cause(ws: Workspace, incident_id: int) -> dict:
    group = ws.group(incident_id)
    ranked = ws.rank(group, ws.alarms, ws.graph)
    return {"candidates": [
        {"service": c.service, "kind": KIND[c.service], "score": round(c.score, 3),
         "first_alarm_minute": c.first_alarm, "peak_score": round(c.peak, 2)}
        for c in ranked[:5]
    ]}


def tool_get_metric_deviation(ws: Workspace, incident_id: int, service: str) -> dict:
    group = _require_service(ws, incident_id, service)
    return {"service": service, "kind": KIND[service], "unit": "robust standard deviations from seasonal baseline",
            "deviation": describe(ws.z, group, ws.alarms, service)}


def tool_fetch_logs(ws: Workspace, incident_id: int, service: str, limit: int = 12) -> dict:
    group = _require_service(ws, incident_id, service)
    lines = incident_log_lines(ws.dataset, group, service)
    return {"service": service, "total_warn_error_lines": len(lines), "lines": lines[: max(1, int(limit))]}


def tool_classify_fault(ws: Workspace, incident_id: int, service: str) -> dict:
    group = _require_service(ws, incident_id, service)
    prediction = ws.classifier.predict(signature(ws.z, group, ws.alarms, service))[0]
    return {"fault_type": prediction.fault_type, "confidence": prediction.confidence,
            "alternatives": [{"fault_type": f, "probability": p} for f, p in prediction.alternatives]}


def tool_search_runbooks(ws: Workspace, query: str, k: int = 3) -> dict:
    hits = ws.retriever.search(query, k=max(1, min(int(k), 5)))
    return {"hits": [
        {"runbook": h.runbook, "title": h.title, "matched_section": h.section, "score": h.score,
         "fault_type": ws.runbooks[h.runbook].fault_type}
        for h in hits
    ]}


def tool_get_remediation(ws: Workspace, runbook: str) -> dict:
    if runbook not in ws.runbooks:
        raise KeyError(f"unknown runbook {runbook!r}")
    book = ws.runbooks[runbook]
    return {"runbook": runbook, "title": book.title,
            "steps": [asdict(step) for step in book.remediation()],
            "escalation": book.sections.get("Escalation", "")}


def _schema(properties: dict[str, dict], required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


_INCIDENT = {"incident_id": {"type": "integer", "description": "Id of the correlated incident."}}
_SERVICE = {"service": {"type": "string", "description": "A service that has an alarm in this incident."}}

TOOLS: dict[str, tuple[Callable[..., dict], str, dict]] = {
    "get_incident": (
        tool_get_incident,
        "Return the time window, the affected services and the alarms of one correlated incident. "
        "Call this first.",
        _schema(_INCIDENT, ["incident_id"]),
    ),
    "rank_root_cause": (
        tool_rank_root_cause,
        "Rank the alarmed services by how likely each is the root cause, using a random walk over "
        "the service dependency graph. Scores sum to 1; a small gap between the top two means the "
        "ranking is uncertain and logs should be checked.",
        _schema(_INCIDENT, ["incident_id"]),
    ),
    "get_metric_deviation": (
        tool_get_metric_deviation,
        "Return how far latency, error rate, CPU and throughput of one service moved from their "
        "seasonal baseline during the incident, in robust standard deviations.",
        _schema({**_INCIDENT, **_SERVICE}, ["incident_id", "service"]),
    ),
    "fetch_logs": (
        tool_fetch_logs,
        "Return WARN and ERROR log lines of one service during the incident. Lines that only "
        "blame a dependency (upstream timeout, 503 from another service) suggest the service is a "
        "victim, not the cause.",
        _schema({**_INCIDENT, **_SERVICE,
                 "limit": {"type": "integer", "description": "Maximum number of lines, default 12."}},
                ["incident_id", "service"]),
    ),
    "classify_fault": (
        tool_classify_fault,
        "Predict the fault type of a service from its metric signature with a trained classifier. "
        "Returns the top prediction, its probability and two alternatives.",
        _schema({**_INCIDENT, **_SERVICE}, ["incident_id", "service"]),
    ),
    "search_runbooks": (
        tool_search_runbooks,
        "Search the operations runbooks with a free-text query. Put observed symptoms and "
        "distinctive log phrases in the query. Returns the best-matching runbooks.",
        _schema({"query": {"type": "string", "description": "Symptoms and log phrases to search for."},
                 "k": {"type": "integer", "description": "Number of runbooks to return, default 3."}},
                ["query"]),
    ),
    "get_remediation": (
        tool_get_remediation,
        "Return the remediation steps of one runbook. Each step says whether it needs human approval.",
        _schema({"runbook": {"type": "string", "description": "Runbook name from search_runbooks."}},
                ["runbook"]),
    ),
    FINAL_TOOL: (
        lambda ws, **kwargs: kwargs,
        "Finish the investigation. Call this exactly once, after the evidence has been gathered.",
        _schema({**_INCIDENT,
                 "root_cause": {"type": "string", "description": "The service judged to be the root cause."},
                 "fault_type": {"type": "string", "description": "The diagnosed fault type."},
                 "runbook": {"type": "string", "description": "The runbook that applies."},
                 "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                 "summary": {"type": "string", "description": "Two or three sentences for the on-call engineer."}},
                ["incident_id", "root_cause", "fault_type", "runbook", "confidence", "summary"]),
    ),
}


def tool_specs() -> list[dict]:
    """Tool definitions in the JSON-schema form LLM tool-calling APIs expect."""
    return [{"name": name, "description": description, "input_schema": schema}
            for name, (_, description, schema) in TOOLS.items()]


# --------------------------------------------------------------------------- agent loop


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass(frozen=True)
class ToolResult:
    call: ToolCall
    ok: bool
    content: dict[str, Any]


class Planner(Protocol):
    name: str

    def start(self, incident_id: int) -> None: ...

    def decide(self) -> list[ToolCall]: ...

    def observe(self, results: list[ToolResult]) -> None: ...


@dataclass
class TraceStep:
    step: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    millis: float
    error: str = ""


@dataclass
class Report:
    incident_id: int
    status: str                       # "complete" or "incomplete"
    planner: str
    window: tuple[int, int]
    services: list[str]
    root_cause: str | None = None
    fault_type: str | None = None
    runbook: str | None = None
    confidence: str = "low"
    summary: str = ""
    remediation: list[dict] = field(default_factory=list)
    escalation: str = ""
    trace: list[TraceStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# Incident {self.incident_id}: {self.fault_type or 'undiagnosed'}",
            "",
            f"- **Status:** {self.status} (planner: {self.planner}, {len(self.trace)} tool calls)",
            f"- **Window:** minute {self.window[0]} to {self.window[1]}",
            f"- **Affected services:** {', '.join(self.services)}",
            f"- **Root cause:** {self.root_cause or 'not determined'}",
            f"- **Runbook:** {self.runbook or 'none'}",
            f"- **Confidence:** {self.confidence}",
            "",
            self.summary,
            "",
            "## Proposed remediation",
        ]
        for number, step in enumerate(self.remediation, 1):
            gate = " **[needs approval]**" if step["requires_approval"] else ""
            lines.append(f"{number}. {step['text']}{gate}")
        if self.escalation:
            lines += ["", "## Escalation", self.escalation]
        return "\n".join(lines)


def validate_call(call: ToolCall) -> str | None:
    """Return an error message when the call does not match the tool's schema."""
    if call.name not in TOOLS:
        return f"unknown tool {call.name!r}; available: {sorted(TOOLS)}"
    schema = TOOLS[call.name][2]
    properties = schema["properties"]
    missing = [key for key in schema["required"] if key not in call.arguments]
    if missing:
        return f"missing required argument(s): {missing}"
    unknown = [key for key in call.arguments if key not in properties]
    if unknown:
        return f"unexpected argument(s): {unknown}"
    python_types = {"integer": int, "string": str}
    for key, value in call.arguments.items():
        expected = python_types[properties[key]["type"]]
        if not isinstance(value, expected) or isinstance(value, bool):
            return f"argument {key!r} must be of type {properties[key]['type']}"
        if "enum" in properties[key] and value not in properties[key]["enum"]:
            return f"argument {key!r} must be one of {properties[key]['enum']}"
    return None


class Agent:
    def __init__(self, workspace: Workspace, planner: Planner, max_steps: int | None = None,
                 on_tool_call: Callable[[str, bool], None] | None = None):
        self.ws = workspace
        self.planner = planner
        self.max_steps = max_steps or workspace.cfg.agent_max_steps
        self.on_tool_call = on_tool_call

    def _execute(self, call: ToolCall, incident_id: int) -> ToolResult:
        error = validate_call(call)
        if error is None and call.arguments.get("incident_id", incident_id) != incident_id:
            error = f"this investigation is scoped to incident {incident_id}"
        if error is None and call.name == FINAL_TOOL:
            error = self._check_final(call.arguments)
        if error is not None:
            return ToolResult(call, False, {"error": error})
        try:
            return ToolResult(call, True, TOOLS[call.name][0](self.ws, **call.arguments))
        except (KeyError, ValueError) as exc:
            return ToolResult(call, False, {"error": str(exc).strip("'\"")})

    def _check_final(self, args: dict) -> str | None:
        group = self.ws.group(args["incident_id"])
        if args["root_cause"] not in group.services:
            return f"root_cause must be one of the alarmed services {group.services}"
        if args["runbook"] not in self.ws.runbooks:
            return f"unknown runbook {args['runbook']!r}; use a name returned by search_runbooks"
        return None

    def investigate(self, incident_id: int) -> Report:
        group = self.ws.group(incident_id)
        report = Report(incident_id, "incomplete", self.planner.name, (group.start, group.end), group.services)
        self.planner.start(incident_id)

        while len(report.trace) < self.max_steps:
            calls = self.planner.decide()
            if not calls:
                break
            results = []
            for call in calls[: self.max_steps - len(report.trace)]:
                began = time.perf_counter()
                result = self._execute(call, incident_id)
                millis = round((time.perf_counter() - began) * 1000, 2)
                report.trace.append(TraceStep(len(report.trace) + 1, call.name, call.arguments, result.ok,
                                              millis, result.content.get("error", "") if not result.ok else ""))
                if self.on_tool_call:
                    self.on_tool_call(call.name, result.ok)
                results.append(result)
                if call.name == FINAL_TOOL and result.ok:
                    return self._finish(report, call.arguments)
            self.planner.observe(results)
        report.summary = "Investigation stopped before a report was submitted; needs human review."
        return report

    def _finish(self, report: Report, args: dict) -> Report:
        # Remediation and approval flags are read from the runbook here, so a planner cannot
        # invent steps or mark a risky step as safe.
        remediation = tool_get_remediation(self.ws, args["runbook"])
        report.status = "complete"
        report.root_cause = args["root_cause"]
        report.fault_type = args["fault_type"]
        report.runbook = args["runbook"]
        report.confidence = args["confidence"]
        report.summary = args["summary"]
        report.remediation = remediation["steps"]
        report.escalation = remediation["escalation"]
        return report


# --------------------------------------------------------------------------- rule planner

_VICTIM_PHRASES = ("upstream call to", "received 503 from", "circuit breaker for", "dependency ")
_NOISE_PHRASES = ("client disconnected", "retrying idempotent call", "slow request")


def own_fault_lines(lines: list[str]) -> list[str]:
    """Log lines that describe a problem inside the service itself."""
    return [line for line in lines
            if not any(phrase in line for phrase in _VICTIM_PHRASES + _NOISE_PHRASES)]


class RulePlanner:
    """A fixed investigation policy that still branches on what the tools return.

    * Near-tie in the root-cause ranking: read the logs of both leading candidates.
    * Chosen root has no logs describing its own fault: read the next candidates' logs and
      switch if one of them does (verify the hypothesis, otherwise explore alternatives).
    * No usable logs at all: fall back to the metric classifier and search the runbooks for
      the fault it predicts.
    * Confidence counts independent signals that agree: log evidence, an unambiguous
      ranking, and a confident classifier that matches the final diagnosis.
    """

    name = "rule"

    def __init__(self, margin: float = 0.15, clear_margin: float = 0.5, max_candidates: int = 3):
        self.margin, self.clear_margin, self.max_candidates = margin, clear_margin, max_candidates

    def start(self, incident_id: int) -> None:
        self.incident_id = incident_id
        self.seen: dict[str, list[dict]] = {}
        self.logs: dict[str, list[str]] = {}
        self.root: str | None = None
        self.switched = False

    def _call(self, name: str, **arguments: Any) -> list[ToolCall]:
        return [ToolCall(name, arguments)]

    def observe(self, results: list[ToolResult]) -> None:
        for result in results:
            if not result.ok:
                self.seen.setdefault("errors", []).append(result.content)
                continue
            self.seen.setdefault(result.call.name, []).append(result.content)
            if result.call.name == "fetch_logs":
                self.logs[result.content["service"]] = result.content["lines"]

    def _evidence(self, service: str) -> int:
        return len(own_fault_lines(self.logs.get(service, [])))

    def _choose_root(self, order: list[str]) -> list[ToolCall] | None:
        """Fetch logs until a root can be chosen. Returns a tool call, or None once decided."""
        iid = self.incident_id
        scores = {c["service"]: c["score"] for c in self.seen["rank_root_cause"][0]["candidates"]}
        if order[0] not in self.logs:
            return self._call("fetch_logs", incident_id=iid, service=order[0])
        near_tie = len(order) > 1 and scores[order[0]] - scores[order[1]] < self.margin
        if near_tie and order[1] not in self.logs:
            return self._call("fetch_logs", incident_id=iid, service=order[1])
        best = max(self.logs, key=lambda service: (self._evidence(service), -order.index(service)))
        if self._evidence(best) == 0:
            unread = [service for service in order[: self.max_candidates] if service not in self.logs]
            if unread:
                return self._call("fetch_logs", incident_id=iid, service=unread[0])
            best = order[0]  # nobody logged their own fault: trust the ranking
        self.root, self.switched = best, best != order[0]
        return None

    def decide(self) -> list[ToolCall]:
        seen, iid = self.seen, self.incident_id
        if "errors" in seen:
            return []  # a deterministic policy has nothing new to try; hand over to a human
        if "get_incident" not in seen:
            return self._call("get_incident", incident_id=iid)
        if "rank_root_cause" not in seen:
            return self._call("rank_root_cause", incident_id=iid)
        if self.root is None:
            pending = self._choose_root([c["service"] for c in seen["rank_root_cause"][0]["candidates"]])
            if pending:
                return pending
        if "get_metric_deviation" not in seen:
            return self._call("get_metric_deviation", incident_id=iid, service=self.root)
        if "classify_fault" not in seen:
            return self._call("classify_fault", incident_id=iid, service=self.root)
        if "search_runbooks" not in seen:
            deviation = seen["get_metric_deviation"][0]
            lines = own_fault_lines(self.logs[self.root])
            return self._call("search_runbooks", query=build_query(deviation["kind"], deviation["deviation"], lines))
        return self._final()

    def _final(self) -> list[ToolCall]:
        seen = self.seen
        classified = seen["classify_fault"][0]
        searches = seen["search_runbooks"]
        hits = [hit for result in searches for hit in result["hits"]]
        by_fault = {hit["fault_type"]: hit["runbook"] for hit in reversed(hits) if hit["fault_type"]}
        has_log_evidence = self._evidence(self.root) > 0
        top = searches[0]["hits"][0]

        if has_log_evidence and top["fault_type"]:
            # Logs name the failure directly, so the log-matched runbook decides the diagnosis.
            fault_type, runbook = top["fault_type"], top["runbook"]
        elif classified["fault_type"] in by_fault:
            fault_type, runbook = classified["fault_type"], by_fault[classified["fault_type"]]
        elif len(searches) < 2:
            # No usable logs and the first search missed the classifier's diagnosis:
            # search again, this time for the diagnosed fault itself.
            kind = seen["get_metric_deviation"][0]["kind"]
            return self._call("search_runbooks", query=f"{classified['fault_type'].replace('_', ' ')} {kind}")
        else:
            fault_type, runbook = classified["fault_type"], top["runbook"]

        candidates = seen["rank_root_cause"][0]["candidates"]
        clear_ranking = candidates[0]["service"] == self.root and (
            len(candidates) == 1 or candidates[0]["score"] - candidates[1]["score"] >= self.clear_margin)
        classifier_backs_it = classified["fault_type"] == fault_type and classified["confidence"] >= 0.6
        signals = int(has_log_evidence) + int(clear_ranking) + int(classifier_backs_it)
        confidence = {3: "high", 2: "medium"}.get(signals, "low")

        deviation = seen["get_metric_deviation"][0]["deviation"]
        moved = ", ".join(f"{metric} {value:+.1f}σ" for metric, value in deviation.items() if abs(value) >= 3)
        others = [c["service"] for c in candidates if c["service"] != self.root][:2]
        basis = "log evidence" if has_log_evidence else "metrics only: no fault-specific logs found"
        summary = (
            f"{self.root} is the most likely root cause"
            f"{' (chosen over the top-ranked candidate after reading logs)' if self.switched else ''}. "
            f"Its metrics moved: {moved or 'no metric beyond 3σ'}. "
            f"The metric classifier says {classified['fault_type']} ({classified['confidence']:.0%}); "
            f"the best runbook match is '{top['title']}'. "
            f"Diagnosis rests on {basis}."
            f"{' Also check: ' + ', '.join(others) + '.' if others and not clear_ranking else ''}"
        )
        return self._call(FINAL_TOOL, incident_id=self.incident_id, root_cause=self.root,
                          fault_type=fault_type, runbook=runbook, confidence=confidence, summary=summary)
