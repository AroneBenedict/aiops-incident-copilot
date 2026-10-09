"""ClaudePlanner is tested against a fake Messages API client: no network, no API key."""

from types import SimpleNamespace

from aiops.agent import FINAL_TOOL, Agent, tool_specs
from aiops.llm import SYSTEM_PROMPT, ClaudePlanner


def tool_use(call_id, name, **arguments):
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


def text(value):
    return SimpleNamespace(type="text", text=value)


class FakeClient:
    """Stands in for anthropic.Anthropic(): returns queued responses and records requests."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        blocks = self.turns.pop(0)
        return SimpleNamespace(content=blocks, stop_reason="tool_use",
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20))


def group_of(workspace):
    return next(group for group in workspace.groups if len(group.services) >= 2)


def final_call(group, call_id="t9", **overrides):
    arguments = {"incident_id": group.id, "root_cause": group.services[0], "fault_type": "bad_deploy",
                 "runbook": "bad-deploy", "confidence": "medium", "summary": "Errors began with the rollout."}
    return tool_use(call_id, FINAL_TOOL, **{**arguments, **overrides})


def test_claude_planner_drives_the_agent_to_a_report(workspace):
    group = group_of(workspace)
    client = FakeClient([
        [text("I will start with the incident."), tool_use("t1", "get_incident", incident_id=group.id)],
        [tool_use("t2", "rank_root_cause", incident_id=group.id)],
        [final_call(group)],
    ])
    planner = ClaudePlanner(client=client, model="test-model")
    report = Agent(workspace, planner).investigate(group.id)

    assert report.status == "complete" and report.planner == "claude"
    assert [step.tool for step in report.trace] == ["get_incident", "rank_root_cause", FINAL_TOOL]
    assert planner.usage == {"input_tokens": 300, "output_tokens": 60, "requests": 3}

    first, second = client.requests[0], client.requests[1]
    assert first["model"] == "test-model" and first["system"] == SYSTEM_PROMPT
    assert first["tools"] == tool_specs()
    assert first["messages"] == [{"role": "user", "content": f"Investigate incident_id {group.id}."}]
    # The second request replays the assistant turn and answers its tool call by id.
    assistant, user = second["messages"][1], second["messages"][2]
    assert assistant["role"] == "assistant" and assistant["content"][1]["id"] == "t1"
    assert user["role"] == "user" and user["content"][0]["type"] == "tool_result"
    assert user["content"][0]["tool_use_id"] == "t1" and user["content"][0]["is_error"] is False


def test_errors_are_returned_to_the_model_so_it_can_recover(workspace):
    group = group_of(workspace)
    client = FakeClient([
        [tool_use("t1", "restart_service", service="orders")],            # not a real tool
        [final_call(group, "t2", runbook="made-up-runbook")],             # hallucinated runbook
        [final_call(group, "t3")],
    ])
    report = Agent(workspace, ClaudePlanner(client=client)).investigate(group.id)
    assert report.status == "complete" and report.runbook == "bad-deploy"
    assert [step.ok for step in report.trace] == [False, False, True]
    errors = [request["messages"][-1]["content"][0] for request in client.requests[1:]]
    assert all(block["is_error"] for block in errors)
    assert "unknown tool" in errors[0]["content"] and "unknown runbook" in errors[1]["content"]


def test_parallel_tool_calls_are_all_answered(workspace):
    group = group_of(workspace)
    client = FakeClient([
        [tool_use("a", "get_incident", incident_id=group.id), tool_use("b", "rank_root_cause", incident_id=group.id)],
        [final_call(group)],
    ])
    Agent(workspace, ClaudePlanner(client=client)).investigate(group.id)
    results = client.requests[1]["messages"][-1]["content"]
    assert [block["tool_use_id"] for block in results] == ["a", "b"]


def test_calls_cut_off_by_the_step_limit_still_get_a_result(workspace):
    group = group_of(workspace)
    client = FakeClient([
        [tool_use("a", "get_incident", incident_id=group.id), tool_use("b", "rank_root_cause", incident_id=group.id)],
        [tool_use("c", "rank_root_cause", incident_id=group.id), tool_use("d", "get_incident", incident_id=group.id)],
    ])
    planner = ClaudePlanner(client=client)
    report = Agent(workspace, planner, max_steps=3).investigate(group.id)
    assert report.status == "incomplete" and len(report.trace) == 3
    last = planner.messages[-1]["content"]
    assert [(block["tool_use_id"], block["is_error"]) for block in last] == [("c", False), ("d", True)]


def test_prose_only_reply_is_nudged_once_then_abandoned(workspace):
    group = group_of(workspace)
    nudged = FakeClient([[text("The root cause is probably orders.")], [final_call(group)]])
    assert Agent(workspace, ClaudePlanner(client=nudged)).investigate(group.id).status == "complete"
    assert FINAL_TOOL in nudged.requests[1]["messages"][-1]["content"]

    stubborn = FakeClient([[text("It is orders.")], [text("As I said, orders.")]])
    report = Agent(workspace, ClaudePlanner(client=stubborn)).investigate(group.id)
    assert report.status == "incomplete" and report.trace == []
