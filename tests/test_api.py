import pytest
from fastapi.testclient import TestClient

from aiops.agent import RulePlanner
from aiops.api import create_app


@pytest.fixture(scope="module")
def client(workspace):
    with TestClient(create_app(workspace, planner_factory=lambda ws: RulePlanner())) as test_client:
        yield test_client


def test_health_reports_loaded_data(client, workspace):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["incidents"] == len(workspace.groups) > 0
    assert body["runbooks"] == 12


def test_incident_listing_is_newest_first_and_filterable(client, workspace):
    listed = client.get("/incidents", params={"limit": 5}).json()
    assert len(listed) == 5
    assert [item["incident_id"] for item in listed] == sorted((i["incident_id"] for i in listed), reverse=True)
    multi = client.get("/incidents", params={"limit": 200, "min_services": 2}).json()
    assert multi and all(len(item["services"]) >= 2 for item in multi)


def test_incident_detail_and_missing_incident(client, workspace):
    group = workspace.groups[0]
    detail = client.get(f"/incidents/{group.id}").json()
    assert detail["services"] == group.services and len(detail["alarms"]) == len(group.alarm_ids)
    assert client.get("/incidents/999999").status_code == 404
    assert client.post("/incidents/999999/investigate").status_code == 404


def test_invalid_parameters_are_rejected(client):
    assert client.get("/incidents", params={"limit": 0}).status_code == 422
    assert client.get("/incidents/not-a-number").status_code == 422


def test_investigate_returns_report_and_updates_metrics(client, workspace):
    group = next(g for g in workspace.groups if len(g.services) >= 2)
    report = client.post(f"/incidents/{group.id}/investigate").json()
    assert report["status"] == "complete" and report["root_cause"] in group.services
    assert report["markdown"].startswith(f"# Incident {group.id}")
    assert any(step["requires_approval"] for step in report["remediation"])

    exposition = client.get("/metrics").text
    assert 'aiops_agent_investigations_total{status="complete"} 1' in exposition
    assert 'aiops_agent_tool_calls_total{tool="submit_report",outcome="ok"} 1' in exposition
    assert 'path="/incidents/{incident_id}/investigate",status="200"' in exposition
    assert 'path="/incidents/{incident_id}",status="404"' in exposition
