"""HTTP service that exposes correlated incidents and the investigation agent.

    uvicorn aiops.api:app --port 8000

Endpoints
    GET  /health                          liveness and basic counts
    GET  /incidents                       correlated incidents, newest first
    GET  /incidents/{id}                  one incident with its alarms
    POST /incidents/{id}/investigate      run the agent and return its report
    GET  /metrics                         Prometheus-format counters for monitoring

Set AIOPS_PROFILE=full to serve the full-size dataset (default: the small one, which starts
in about a second). Set AIOPS_PLANNER=claude and ANTHROPIC_API_KEY to let Claude plan.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from . import __version__
from .agent import Agent, RulePlanner, Workspace, tool_get_incident
from .config import Config, small_config


class IncidentSummary(BaseModel):
    incident_id: int
    start_minute: int
    end_minute: int
    services: list[str]
    alarm_count: int


class Telemetry:
    """In-process counters. Enough for a scrape endpoint without extra dependencies."""

    def __init__(self) -> None:
        self.requests: Counter[tuple[str, str, int]] = Counter()
        self.seconds: Counter[str] = Counter()
        self.tool_calls: Counter[tuple[str, str]] = Counter()
        self.investigations: Counter[str] = Counter()

    def render(self) -> str:
        lines = ["# TYPE aiops_http_requests_total counter"]
        for (method, path, status), count in sorted(self.requests.items()):
            lines.append(f'aiops_http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}')
        lines.append("# TYPE aiops_http_request_seconds_sum counter")
        for path, total in sorted(self.seconds.items()):
            lines.append(f'aiops_http_request_seconds_sum{{path="{path}"}} {total:.6f}')
        lines.append("# TYPE aiops_agent_investigations_total counter")
        for status, count in sorted(self.investigations.items()):
            lines.append(f'aiops_agent_investigations_total{{status="{status}"}} {count}')
        lines.append("# TYPE aiops_agent_tool_calls_total counter")
        for (tool, outcome), count in sorted(self.tool_calls.items()):
            lines.append(f'aiops_agent_tool_calls_total{{tool="{tool}",outcome="{outcome}"}} {count}')
        return "\n".join(lines) + "\n"


def _default_workspace() -> Workspace:
    from .pipeline import build  # imported here so importing the module stays cheap

    cfg = Config() if os.environ.get("AIOPS_PROFILE") == "full" else small_config()
    return build(cfg).workspace


def _make_planner(workspace: Workspace):
    if os.environ.get("AIOPS_PLANNER") == "claude":
        from .llm import ClaudePlanner

        return ClaudePlanner()
    return RulePlanner(workspace.cfg.rca_margin_for_log_check)


def create_app(workspace: Workspace | None = None, planner_factory=_make_planner) -> FastAPI:
    telemetry = Telemetry()
    state: dict[str, Workspace] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["ws"] = workspace if workspace is not None else _default_workspace()
        yield

    app = FastAPI(title="AIOps Incident Copilot", version=__version__, lifespan=lifespan)
    app.state.telemetry = telemetry

    @app.middleware("http")
    async def record(request: Request, call_next):
        began = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        path = route.path if route is not None else "unmatched"
        telemetry.requests[(request.method, path, response.status_code)] += 1
        telemetry.seconds[path] += time.perf_counter() - began
        return response

    def ws() -> Workspace:
        return state["ws"]

    def summary(group) -> IncidentSummary:
        return IncidentSummary(incident_id=group.id, start_minute=group.start, end_minute=group.end,
                               services=group.services, alarm_count=len(group.alarm_ids))

    def require(incident_id: int):
        try:
            return ws().group(incident_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"incident {incident_id} not found") from None

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "incidents": len(ws().groups),
                "alarms": len(ws().alarms), "runbooks": len(ws().runbooks)}

    @app.get("/incidents", response_model=list[IncidentSummary])
    def list_incidents(limit: int = Query(20, ge=1, le=200), min_services: int = Query(1, ge=1)):
        groups = [g for g in reversed(ws().groups) if len(g.services) >= min_services]
        return [summary(group) for group in groups[:limit]]

    @app.get("/incidents/{incident_id}")
    def get_incident(incident_id: int) -> dict:
        require(incident_id)
        return tool_get_incident(ws(), incident_id)

    @app.post("/incidents/{incident_id}/investigate")
    def investigate(incident_id: int) -> dict:
        require(incident_id)
        agent = Agent(ws(), planner_factory(ws()),
                      on_tool_call=lambda tool, ok: telemetry.tool_calls.update([(tool, "ok" if ok else "error")]))
        report = agent.investigate(incident_id)
        telemetry.investigations[report.status] += 1
        return {**report.to_dict(), "markdown": report.to_markdown()}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return telemetry.render()

    return app


app = create_app()
