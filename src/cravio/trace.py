"""--trace: log every step Cravio takes, so you can watch a query travel through the agents and Swiggy MCP.

Each line is tagged with the agent doing the work, and subagents are indented under the main agent:
  STATE  the conversation loaded from and saved to SQLite, pauses and resumes
  MODEL  each model call: what goes in, the model's thinking, and the tool calls or reply that come out
  TOOL   each tool run and what it returned
  TASK   each hand-off to a subagent, and the one message it hands back
  HTTP   each request to Swiggy's MCP server: the JSON-RPC method and the MCP session it ran in
"""

import json
import logging
import sys
import time
from contextvars import ContextVar

import httpx
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from cravio.swiggy import client as swiggy

log = logging.getLogger("cravio.trace")

# Which agent is working and how many hand-offs deep. The task tool sets these for the subagent
# it starts, and everything that subagent runs (model calls, tools, HTTP) inherits them.
AGENT = ContextVar("agent", default="setup")
DEPTH = ContextVar("depth", default=0)

PREVIEW_CHARS = 300
# deepagents moves a tool result bigger than this to its virtual filesystem and shows the model a preview.
EVICT_TOKENS = 20_000
FS_TOOLS = {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute"}


class _AgentContext(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # The MCP session's own background task sends requests, so HTTP lines name it rather than an agent.
        record.agent = getattr(record, "agent", None) or AGENT.get()
        record.indent = "    " * DEPTH.get()
        return True


def enable() -> None:
    """Print Cravio's logs (every cravio.* logger) to the terminal."""
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_AgentContext())
    handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(indent)s[%(agent)s] %(message)s", "%H:%M:%S"))
    root = logging.getLogger("cravio")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False


def _short(value, limit: int = PREVIEW_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _result_text(result: ToolMessage | Command) -> str:
    if isinstance(result, Command) and isinstance(result.update, dict):
        result = next((m for m in result.update.get("messages", []) if isinstance(m, ToolMessage)), None)
    return result.text if isinstance(result, ToolMessage) else ""


def _tool_name(tool) -> str:
    return tool.get("name", "?") if isinstance(tool, dict) else tool.name


class Trace(AgentMiddleware):
    """Logs each model call and tool run of the agent it's attached to."""

    def __init__(self) -> None:
        self._listed: set[str] = set()

    async def awrap_model_call(self, request, handler):
        agent = AGENT.get()
        if agent not in self._listed:
            self._listed.add(agent)
            log.info("MODEL  tools this agent can call: %s", ", ".join(_tool_name(t) for t in request.tools))

        model = getattr(request.model, "model", None) or type(request.model).__name__
        history = len(request.state.get("messages", []))
        summarized = f" (of {history}; older ones replaced by a summary)" if len(request.messages) < history else ""
        log.info("MODEL  → %s · %d messages%s · %d tools", model, len(request.messages), summarized, len(request.tools))

        start = time.perf_counter()
        response = await handler(request)
        reply = next(m for m in reversed(response.result) if isinstance(m, AIMessage))
        usage = reply.usage_metadata or {}
        log.info(
            "MODEL  ← %.1fs · %s tokens in, %s out",
            time.perf_counter() - start, usage.get("input_tokens", "?"), usage.get("output_tokens", "?"),
        )
        if thinking := reply.additional_kwargs.get("reasoning_content"):
            log.info("         thinking: %s", _short(thinking))
        for call in reply.tool_calls:
            log.info("         calls %s %s", call["name"], _short(call["args"]))
        if not reply.tool_calls:
            log.info("         replies: %s", _short(reply.text))
        return response

    async def awrap_tool_call(self, request, handler):
        call = request.tool_call
        if call["name"] == "task":
            return await self._hand_off(call, request, handler)

        if call["name"] in FS_TOOLS:
            source = "deepagents virtual filesystem"
        elif request.tool is None:
            source = "not one of this agent's tools"
        elif (request.tool.metadata or {}).get("swiggy_name"):
            source = "Swiggy MCP"
        else:
            source = "Cravio tool"
        log.info("TOOL   %s · %s", call["name"], source)

        start = time.perf_counter()
        result = await handler(request)
        text = _result_text(result)
        failed = isinstance(result, ToolMessage) and result.status == "error"
        log.info(
            "TOOL   %s %s in %.1fs · %d chars (≈%d tokens): %s",
            call["name"], "FAILED" if failed else "done", time.perf_counter() - start,
            len(text), len(text) // 4, _short(text),
        )
        if len(text) // 4 > EVICT_TOKENS and call["name"] not in FS_TOOLS:
            log.info("         over %d tokens: deepagents keeps it under /large_tool_results/ and shows the model a preview", EVICT_TOKENS)
        return result

    async def _hand_off(self, call, request, handler):
        subagent = call["args"].get("subagent_type", "?")
        log.info("TASK   → %s starts a fresh chat holding only this: %s", subagent, _short(call["args"].get("description", ""), 600))
        agent, depth = AGENT.set(subagent), DEPTH.set(DEPTH.get() + 1)
        start = time.perf_counter()
        try:
            result = await handler(request)
        finally:
            AGENT.reset(agent)
            DEPTH.reset(depth)
        log.info(
            "TASK   ← %s finished in %.1fs; only its last message comes back: %s",
            subagent, time.perf_counter() - start, _short(_result_text(result), 600),
        )
        return result


MCP = {"agent": "mcp"}
_started: dict[int, tuple[float, str]] = {}


def _rpc(request: httpx.Request) -> str:
    """What an MCP request asks for: the JSON-RPC method, plus the tool name for tools/call."""
    if request.method == "DELETE":
        return "end MCP session"
    if request.method == "GET":
        return "listen for server messages"
    try:
        body = json.loads(request.content)
    except (httpx.RequestNotRead, ValueError):
        return ""  # not JSON-RPC: OAuth discovery, token refresh
    if not isinstance(body, dict):
        return ""
    method = body.get("method", "")
    if method == "tools/call":
        return f"tools/call {(body.get('params') or {}).get('name')}"
    return method


async def _on_request(request: httpx.Request) -> None:
    what = _rpc(request)
    _started[id(request)] = (time.perf_counter(), what)
    log.info("HTTP   → %s %s%s %s", request.method, request.url.host, request.url.path, what, extra=MCP)


async def _on_response(response: httpx.Response) -> None:
    start, what = _started.pop(id(response.request), (time.perf_counter(), ""))
    session = response.headers.get("mcp-session-id") or response.request.headers.get("mcp-session-id")
    log.info(
        "HTTP   ← %d %s in %d ms%s",
        response.status_code, what, (time.perf_counter() - start) * 1000,
        f" · MCP session {session[:8]}" if session else "",
        extra=MCP,
    )


def http_client(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:
    """Cravio's Swiggy HTTP client, logging each request and response. Never logs tokens or bodies."""
    client = swiggy.http_client(headers, timeout, auth)
    client.event_hooks = {"request": [_on_request], "response": [_on_response]}
    return client
