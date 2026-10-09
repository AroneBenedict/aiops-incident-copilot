"""LLM-backed planner for the incident agent, using Claude tool calling.

The planner sends the conversation and the tool definitions to the Messages API. Claude
answers with ``tool_use`` blocks; the agent loop runs them and this planner sends the results
back as ``tool_result`` blocks. The model never touches the data directly: it can only pick
from the read-only tools in ``agent.TOOLS``.

The ``anthropic`` package is imported lazily so the rest of the project runs without it.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .agent import FINAL_TOOL, ToolCall, ToolResult, tool_specs

DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = f"""You are an operations assistant investigating one correlated incident \
in a microservice system. Work only through the tools you are given.

Method:
1. Call get_incident, then rank_root_cause.
2. If the top two candidates are close, read the logs of both before deciding. A service whose \
logs only blame a dependency is a victim, not the cause.
3. For your chosen root cause, look at its metric deviation, its logs and the fault classifier.
4. Search the runbooks using the symptoms and the distinctive log phrases you saw.
5. Call {FINAL_TOOL} exactly once. Use confidence "high" only when metrics, logs and the \
runbook agree.

Rules: base every claim on tool output. Never invent a service, runbook or log line. You \
cannot change the system; you only diagnose and point to the runbook."""


class PlannerError(RuntimeError):
    pass


def _block_to_dict(block: Any) -> dict:
    """Normalise an SDK content block (object or dict) to a plain dict we can resend."""
    if isinstance(block, dict):
        return block
    kind = getattr(block, "type", None)
    if kind == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input)}
    if kind == "text":
        return {"type": "text", "text": block.text}
    raise PlannerError(f"unsupported content block type: {kind!r}")


class ClaudePlanner:
    name = "claude"

    def __init__(self, client: Any = None, model: str | None = None, max_tokens: int = 1024):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise PlannerError("install the 'anthropic' package to use ClaudePlanner") from exc
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.client = client
        self.model = model or os.environ.get("AIOPS_LLM_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

    def start(self, incident_id: int) -> None:
        self.messages = [{"role": "user", "content": f"Investigate incident_id {incident_id}."}]

    def _request(self) -> list[dict]:
        response = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=SYSTEM_PROMPT,
            tools=tool_specs(), messages=self.messages,
        )
        usage = getattr(response, "usage", None)
        self.usage["requests"] += 1
        self.usage["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        self.usage["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        blocks = [_block_to_dict(block) for block in response.content]
        self.messages.append({"role": "assistant", "content": blocks})
        return blocks

    def decide(self) -> list[ToolCall]:
        blocks = self._request()
        calls = [b for b in blocks if b["type"] == "tool_use"]
        if not calls:
            # The model answered in prose. Ask once for a tool call, then give up.
            self.messages.append({"role": "user", "content":
                                  f"Continue with a tool call, and finish by calling {FINAL_TOOL}."})
            calls = [b for b in self._request() if b["type"] == "tool_use"]
        return [ToolCall(b["name"], dict(b["input"]), b["id"]) for b in calls]

    def observe(self, results: list[ToolResult]) -> None:
        answered = {result.call.call_id for result in results}
        content = [
            {"type": "tool_result", "tool_use_id": result.call.call_id,
             "content": json.dumps(result.content), "is_error": not result.ok}
            for result in results
        ]
        # Every tool_use block must get a tool_result, including calls cut off by the step limit.
        for block in self.messages[-1]["content"]:
            if block["type"] == "tool_use" and block["id"] not in answered:
                content.append({"type": "tool_result", "tool_use_id": block["id"],
                                "content": "not executed: step limit reached", "is_error": True})
        self.messages.append({"role": "user", "content": content})
