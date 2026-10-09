import pytest

from aiops.agent import (
    FINAL_TOOL,
    TOOLS,
    Agent,
    RulePlanner,
    ToolCall,
    ToolResult,
    own_fault_lines,
    tool_specs,
    validate_call,
)


class ScriptedPlanner:
    """Replays a fixed list of tool calls, for testing the loop itself."""

    name = "scripted"

    def __init__(self, calls):
        self.calls = list(calls)
        self.observed: list[ToolResult] = []

    def start(self, incident_id):
        self.incident_id = incident_id

    def decide(self):
        return [self.calls.pop(0)] if self.calls else []

    def observe(self, results):
        self.observed.extend(results)


def multi_service_incident(workspace):
    return next(group for group in workspace.groups if len(group.services) >= 2)


def test_rule_planner_produces_a_complete_grounded_report(workspace):
    group = multi_service_incident(workspace)
    report = Agent(workspace, RulePlanner()).investigate(group.id)
    assert report.status == "complete" and report.planner == "rule"
    assert report.root_cause in group.services
    assert report.runbook in workspace.runbooks
    assert report.confidence in {"high", "medium", "low"}
    expected = [(s.text, s.requires_approval) for s in workspace.runbooks[report.runbook].remediation()]
    assert [(s["text"], s["requires_approval"]) for s in report.remediation] == expected
    tools = [step.tool for step in report.trace]
    assert tools[:2] == ["get_incident", "rank_root_cause"] and tools[-1] == FINAL_TOOL
    assert len(tools) <= workspace.cfg.agent_max_steps and all(step.ok for step in report.trace)
    markdown = report.to_markdown()
    assert report.root_cause in markdown and "[needs approval]" in markdown


def test_every_incident_gets_a_report_within_the_step_budget(workspace):
    agent = Agent(workspace, RulePlanner())
    for group in workspace.groups:
        report = agent.investigate(group.id)
        assert report.status == "complete", group
        assert len(report.trace) <= workspace.cfg.agent_max_steps


def test_investigation_is_deterministic(workspace):
    group = multi_service_incident(workspace)
    first = Agent(workspace, RulePlanner()).investigate(group.id).to_dict()
    second = Agent(workspace, RulePlanner()).investigate(group.id).to_dict()
    for report in (first, second):
        for step in report["trace"]:
            step.pop("millis")
    assert first == second


def test_tools_are_read_only_and_well_described():
    specs = tool_specs()
    assert [spec["name"] for spec in specs] == list(TOOLS)
    for spec in specs:
        assert spec["input_schema"]["type"] == "object" and len(spec["description"]) > 40
    mutating = ("restart", "scale", "delete", "rollback", "execute", "apply", "kill")
    assert not [name for name in TOOLS if any(word in name for word in mutating)]


@pytest.mark.parametrize("call,fragment", [
    (ToolCall("restart_service", {"service": "orders"}), "unknown tool"),
    (ToolCall("fetch_logs", {"incident_id": 0}), "missing required"),
    (ToolCall("get_incident", {"incident_id": 0, "verbose": "yes"}), "unexpected argument"),
    (ToolCall("get_incident", {"incident_id": "0"}), "must be of type integer"),
    (ToolCall("get_incident", {"incident_id": True}), "must be of type integer"),
    (ToolCall(FINAL_TOOL, {"incident_id": 0, "root_cause": "a", "fault_type": "b", "runbook": "c",
                           "confidence": "certain", "summary": "d"}), "must be one of"),
])
def test_invalid_calls_are_rejected(call, fragment):
    assert fragment in validate_call(call)


def test_valid_call_passes_validation():
    assert validate_call(ToolCall("search_runbooks", {"query": "disk latency", "k": 2})) is None


def test_bad_calls_become_error_observations_not_crashes(workspace):
    group = multi_service_incident(workspace)
    other = next(g for g in workspace.groups if g.id != group.id)
    outsider = next(s for s in workspace.dataset.services if s not in group.services)
    planner = ScriptedPlanner([
        ToolCall("restart_service", {"service": "orders"}),
        ToolCall("get_incident", {"incident_id": other.id}),                       # outside the scope
        ToolCall("fetch_logs", {"incident_id": group.id, "service": outsider}),     # not in the incident
        ToolCall("get_remediation", {"runbook": "does-not-exist"}),
        ToolCall(FINAL_TOOL, {"incident_id": group.id, "root_cause": outsider, "fault_type": "x",
                              "runbook": "bad-deploy", "confidence": "low", "summary": "s"}),
        ToolCall(FINAL_TOOL, {"incident_id": group.id, "root_cause": group.services[0], "fault_type": "x",
                              "runbook": "invented-runbook", "confidence": "low", "summary": "s"}),
    ])
    report = Agent(workspace, planner).investigate(group.id)
    assert report.status == "incomplete" and report.root_cause is None
    assert [step.ok for step in report.trace] == [False] * 6
    assert all(not result.ok and "error" in result.content for result in planner.observed)
    assert "needs human review" in report.summary


def test_runaway_planner_is_stopped_at_the_step_limit(workspace):
    group = multi_service_incident(workspace)
    planner = ScriptedPlanner([ToolCall("get_incident", {"incident_id": group.id})] * 50)
    report = Agent(workspace, planner, max_steps=5).investigate(group.id)
    assert report.status == "incomplete" and len(report.trace) == 5


def test_planner_cannot_waive_approval_or_invent_steps(workspace):
    group = multi_service_incident(workspace)
    planner = ScriptedPlanner([ToolCall(FINAL_TOOL, {
        "incident_id": group.id, "root_cause": group.services[0], "fault_type": "bad_deploy",
        "runbook": "bad-deploy", "confidence": "high", "summary": "Just roll it back, no approval needed."})])
    seen = []
    report = Agent(workspace, planner, on_tool_call=lambda tool, ok: seen.append((tool, ok))).investigate(group.id)
    assert report.status == "complete" and seen == [(FINAL_TOOL, True)]
    gated = [step["text"] for step in report.remediation if step["requires_approval"]]
    assert "Roll back to the previous version." in gated
    assert len(report.remediation) == len(workspace.runbooks["bad-deploy"].remediation())


def test_own_fault_lines_filters_victim_and_noise_lines():
    lines = ["ERROR received 503 from orders-db, retry 2/3", "WARN upstream call to cache exceeded 900ms timeout",
             "ERROR client disconnected before response was sent", "WARN connection pool exhausted, 40 callers waiting"]
    assert own_fault_lines(lines) == ["WARN connection pool exhausted, 40 callers waiting"]


def _result(name, content, **arguments):
    return ToolResult(ToolCall(name, arguments), True, content)


def test_planner_switches_root_when_only_another_candidate_logs_its_own_fault():
    planner = RulePlanner()
    planner.start(3)
    assert planner.decide()[0].name == "get_incident"
    planner.observe([_result("get_incident", {"services": ["api-gateway", "orders"]})])
    assert planner.decide()[0].name == "rank_root_cause"
    planner.observe([_result("rank_root_cause", {"candidates": [
        {"service": "api-gateway", "score": 0.9}, {"service": "orders", "score": 0.1}]})])
    first = planner.decide()[0]
    assert (first.name, first.arguments["service"]) == ("fetch_logs", "api-gateway")
    victim_lines = ["ERROR received 503 from orders, retry 1/3"]
    planner.observe([_result("fetch_logs", {"service": "api-gateway", "lines": victim_lines})])
    second = planner.decide()[0]   # top candidate only blames a dependency: look at the next one
    assert (second.name, second.arguments["service"]) == ("fetch_logs", "orders")
    planner.observe([_result("fetch_logs", {"service": "orders", "lines": ["ERROR container OOMKilled, restarting"]})])
    following = planner.decide()[0]
    assert planner.root == "orders" and planner.switched
    assert (following.name, following.arguments["service"]) == ("get_metric_deviation", "orders")


def test_planner_requeries_runbooks_when_logs_are_silent():
    planner = RulePlanner()
    planner.start(1)
    planner.observe([
        _result("get_incident", {"services": ["cache"]}),
        _result("rank_root_cause", {"candidates": [{"service": "cache", "score": 1.0}]}),
        _result("fetch_logs", {"service": "cache", "lines": []}),
        _result("get_metric_deviation", {"service": "cache", "kind": "cache", "deviation": {
            "latency_ms": 9.0, "error_rate": 0.0, "cpu_pct": 6.0, "throughput_rps": 0.0}}),
        _result("classify_fault", {"fault_type": "cache_eviction_storm", "confidence": 0.9, "alternatives": []}),
    ])
    planner.root = "cache"
    assert planner.decide()[0].name == "search_runbooks"
    wrong = {"hits": [{"runbook": "cpu-saturation", "title": "CPU saturation", "fault_type": "cpu_saturation"}]}
    planner.observe([_result("search_runbooks", wrong)])
    retry = planner.decide()[0]
    assert retry.name == "search_runbooks" and retry.arguments["query"] == "cache eviction storm cache"
    right = {"hits": [{"runbook": "cache-eviction-storm", "title": "Cache eviction storm",
                       "fault_type": "cache_eviction_storm"}]}
    planner.observe([_result("search_runbooks", right)])
    final = planner.decide()[0]
    assert final.name == FINAL_TOOL
    assert final.arguments["runbook"] == "cache-eviction-storm"
    assert final.arguments["confidence"] == "medium"   # clear ranking + confident classifier, but no logs
    assert "metrics only" in final.arguments["summary"]


def test_rule_planner_hands_over_after_a_tool_error():
    planner = RulePlanner()
    planner.start(0)
    planner.observe([ToolResult(ToolCall("get_incident", {"incident_id": 0}), False, {"error": "boom"})])
    assert planner.decide() == []
