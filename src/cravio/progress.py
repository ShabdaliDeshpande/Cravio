"""What Cravio is doing right now: each step is running, then completed, failed, declined or waiting for you.

Progress is a middleware: attach it to the agent and it reports every tool run and hand-off to a subagent as
a Step, once when it starts and again when it ends. The CLI prints them; a web UI can show them the same way.
steps_in_last_turn() reads the same thing back from a saved conversation, which is how the CLI tells you
what was done, and what was left unfinished, when you come back after quitting halfway.
"""

import inspect
import itertools
import json
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

Status = Literal["running", "waiting", "completed", "error", "declined", "unfinished"]

# What each tool does, in the user's words.
LABELS = {
    "get_addresses": "Reading your saved addresses",
    "create_address": "Saving a new address",
    "search_restaurants": "Searching restaurants",
    "search_menu": "Searching dishes",
    "get_restaurant_menu": "Reading a menu",
    "get_food_cart": "Reading your Food cart",
    "update_food_cart": "Changing your Food cart",
    "flush_food_cart": "Clearing your Food cart",
    "fetch_food_coupons": "Looking for coupons",
    "apply_food_coupon": "Applying a coupon",
    "get_food_orders": "Reading your Food orders",
    "get_food_order_details": "Opening an order",
    "track_food_order": "Tracking your Food order",
    "search_products": "Searching Instamart",
    "your_go_to_items": "Reading your usual Instamart items",
    "get_cart": "Reading your Instamart cart",
    "update_cart": "Changing your Instamart cart",
    "get_orders": "Reading your Instamart orders",
    "get_delivery_status": "Checking an Instamart delivery",
    "create_escalation_ticket": "Opening a support ticket",
    "request_checkout": "Getting your cart ready for checkout",
    "write_todos": "Updating the plan",
}
SUBAGENTS = {
    "taste-profiler": "Learning what you usually order",
    "recommender": "Finding options for you",
    "support-escalation": "Looking into your order problem",
    "tracking": "Tracking your order",
    "instamart": "Shopping on Instamart",
}
HIDDEN = {"ls", "read_file", "write_file", "edit_file", "glob", "grep"}  # Deep Agents' scratch files

_AGENT = ContextVar("progress_agent", default="cravio")
_DEPTH = ContextVar("progress_depth", default=0)


@dataclass
class Step:
    id: int
    agent: str  # "cravio", or the subagent doing the work
    tool: str
    label: str
    depth: int  # 0 for Cravio's own steps, 1 inside a subagent
    status: Status = "running"
    detail: str = ""  # an error, or a subagent's answer, shortened
    plan: list[dict] = field(default_factory=list)  # write_todos only: [{"content", "status"}]
    started: float = field(default_factory=time.monotonic)
    seconds: float = 0.0


def plan_of(args: dict) -> list[dict]:
    """write_todos' list as [{"content", "status"}]. A local model sometimes sends plain strings, or the whole
    list as one JSON string; show those too instead of crashing the step display."""
    todos = args.get("todos") or []
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except ValueError:
            todos = [todos]
    if not isinstance(todos, list):
        return []
    return [t if isinstance(t, dict) else {"content": str(t), "status": "pending"} for t in todos]


def label_of(tool: str, args: dict) -> str:
    if tool == "task":
        subagent = args.get("subagent_type", "?")
        return SUBAGENTS.get(subagent, f"Asking {subagent}")
    return LABELS.get(tool, tool.replace("_", " ").capitalize())


def _status_of(result) -> tuple[Status, str]:
    if isinstance(result, Command) and isinstance(result.update, dict):
        result = next((m for m in result.update.get("messages", []) if isinstance(m, ToolMessage)), result)
    if not isinstance(result, ToolMessage):
        return "completed", ""
    text = " ".join(result.text.split())
    if result.status == "error":
        return ("declined", "you said no") if text.startswith("User rejected") else ("error", text[:200])
    return "completed", text[:200]


class Progress(AgentMiddleware):
    """Calls on_step(step) when each step starts and again when it ends. on_step may be async (a web UI)."""

    def __init__(self, on_step: Callable[[Step], None | Awaitable[None]]) -> None:
        self.on_step = on_step
        self._ids = itertools.count(1)

    async def _report(self, step: Step) -> None:
        result = self.on_step(step)
        if inspect.isawaitable(result):
            await result

    async def awrap_tool_call(self, request, handler):
        call = request.tool_call
        if call["name"] in HIDDEN:
            return await handler(request)
        step = Step(next(self._ids), _AGENT.get(), call["name"], label_of(call["name"], call["args"]), _DEPTH.get())
        if call["name"] == "write_todos":
            step.plan = plan_of(call["args"])
        await self._report(step)

        tokens = None
        if call["name"] == "task":  # everything the subagent does is one level down, under its name
            tokens = (_AGENT.set(call["args"].get("subagent_type", "?")), _DEPTH.set(_DEPTH.get() + 1))
        try:
            result = await handler(request)
        except GraphInterrupt:  # paused for the user's approval; the step runs again when they answer
            step.status, step.detail = "waiting", "waiting for your approval"
            raise
        except BaseException as exc:
            step.status, step.detail = "error", f"{type(exc).__name__}: {exc}"[:200]
            raise
        else:
            step.status, step.detail = _status_of(result)
            return result
        finally:
            if tokens:
                _AGENT.reset(tokens[0])
                _DEPTH.reset(tokens[1])
            step.seconds = time.monotonic() - step.started
            await self._report(step)


def steps_in_last_turn(messages: list) -> list[Step]:
    """Cravio's own steps since the user's last message, from a saved conversation.

    A tool call with no result yet is "unfinished": the run stopped (you quit, or it crashed) before it ended.
    Subagents' inner steps aren't saved in the conversation, so each subagent shows as one step.
    """
    start = max((i for i, m in enumerate(messages) if m.type == "human"), default=-1)
    results = {m.tool_call_id: m for m in messages[start + 1:] if isinstance(m, ToolMessage)}
    steps = []
    for m in messages[start + 1:]:
        for call in getattr(m, "tool_calls", None) or []:
            if call["name"] in HIDDEN:
                continue
            step = Step(len(steps) + 1, "cravio", call["name"], label_of(call["name"], call["args"]), 0)
            if call["name"] == "write_todos":
                step.plan = plan_of(call["args"])
            result = results.get(call["id"])
            step.status, step.detail = _status_of(result) if result else ("unfinished", "")
            steps.append(step)
    return steps
