"""A simple web chat for Cravio (Chainlit): the reply streams in, and each step shows while it runs.

  uv run chainlit run src/cravio/web.py      then open http://localhost:8000

Same agent, Swiggy connection and safety rules as the CLI. A cart change or checkout asks with Approve and
Decline buttons. Each browser tab is its own conversation, kept until the tab closes.
"""

import asyncio
import json
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # before cravio imports: they read settings at import

import chainlit as cl  # noqa: E402
from langchain.agents.middleware.internal_call_transformer import INTERNAL_CALL_METADATA_KEY  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from cravio.agent import MODEL, build_agent  # noqa: E402
from cravio.cli import decide, resume_with  # noqa: E402
from cravio.langsmith_tracing import langsmith_tracing  # noqa: E402
from cravio.progress import Progress, Step, label_of  # noqa: E402
from cravio.swiggy.client import build_client, swiggy_tools  # noqa: E402

ENDED = {"completed": "✔ done", "error": "✖ failed", "declined": "⊘ you said no", "waiting": "⏸ waiting for your approval"}
PLAN = {"completed": "✔", "in_progress": "▶", "pending": "○"}


class StepView:
    """Shows each Progress step as a Chainlit step: "running…", then how it ended. A subagent's own steps
    go inside its step."""

    def __init__(self) -> None:
        self.shown: dict[int, cl.Step] = {}
        self.subagent_at: dict[int, cl.Step] = {}  # depth -> the subagent step whose inner steps go under it

    async def __call__(self, step: Step) -> None:
        if step.id not in self.shown:
            parent = self.subagent_at.get(step.depth - 1) if step.depth else None
            view = cl.Step(name="Plan" if step.tool == "write_todos" else step.label, type="tool",
                           parent_id=parent.id if parent else None, auto_collapse=True)  # folds away when done
            self.shown[step.id] = view
            if step.tool == "task":
                self.subagent_at[step.depth] = view
            view.output = "\n".join(f"{PLAN.get(t.get('status'), '○')} {t.get('content')}" for t in step.plan) or "running…"
            await view.send()
            return
        view = self.shown[step.id]
        if step.tool != "write_todos":
            view.output = f"{ENDED.get(step.status, step.status)} · {step.seconds:.1f}s" + (f"\n{step.detail}" if step.status == "error" else "")
        await view.update()


async def _hold_swiggy(ready: asyncio.Future, close: asyncio.Event) -> None:
    """Keep this tab's Swiggy session open in one task: MCP sessions must close in the task that opened them."""
    try:
        async with swiggy_tools(build_client()) as tools:  # the servers in SWIGGY_SERVERS (.env)
            ready.set_result(tools)
            await close.wait()
    except Exception as exc:
        if not ready.done():
            ready.set_exception(exc)


@cl.on_chat_start
async def start() -> None:
    ready, close = asyncio.get_running_loop().create_future(), asyncio.Event()
    holder = asyncio.create_task(_hold_swiggy(ready, close))
    tools = await ready
    agent = build_agent(tools, middleware=[Progress(StepView())], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 80}
    cl.user_session.set("cravio", {"agent": agent, "config": config, "close": close, "holder": holder})
    await cl.Message(content=f"Hi, I'm Cravio ! . Ask for food, groceries, or help with an order.").send()


@cl.on_chat_end
async def end() -> None:
    if session := cl.user_session.get("cravio"):
        session["close"].set()
        await session["holder"]


async def _ask_approval(actions: list[dict]) -> dict:
    lines = [f"**{label_of(a['name'], a['args'])}**\n```json\n{json.dumps(a['args'], ensure_ascii=False, indent=1)[:1500]}\n```"
             for a in actions]
    answer = await cl.AskActionMessage(
        content="Cravio wants to:\n\n" + "\n\n".join(lines),
        actions=[cl.Action(name="approve", payload={"answer": "yes"}, label="Approve"),
                 cl.Action(name="decline", payload={"answer": "no"}, label="Decline")],
        timeout=600,
    ).send()
    return decide("yes" if (answer or {}).get("payload", {}).get("answer") == "yes" else "No.")


def _is_answer_text(chunk, meta: dict) -> bool:
    """Text Cravio writes to you. Not its middleware's own model calls, such as the summary it writes of a
    long chat, which would otherwise stream in as if it were the reply."""
    return (
        meta.get("lc_agent_name") == "cravio"
        and chunk.type == "AIMessageChunk"
        and isinstance(chunk.content, str)
        and INTERNAL_CALL_METADATA_KEY not in meta
        and meta.get("lc_source") != "summarization"
    )


@cl.on_message
async def on_message(message: cl.Message) -> None:
    session = cl.user_session.get("cravio")
    agent, config = session["agent"], session["config"]
    payload = {"messages": [{"role": "user", "content": message.content}]}
    with langsmith_tracing():  # traced and masked, as in the CLI, when CRAVIO_LANGSMITH=true
        while True:
            reply = cl.Message(content="")
            starts: dict[str, int] = {}  # model call -> where its text starts in the reply
            steps_only: set[str] = set()  # model calls that turned out to call tools: their text isn't the answer
            async for chunk, meta in agent.astream(payload, config, stream_mode="messages"):
                if not _is_answer_text(chunk, meta) or chunk.id in steps_only:
                    continue
                if chunk.tool_call_chunks:  # "I'll check your orders…" before a tool: take it back out
                    steps_only.add(chunk.id)
                    if chunk.id in starts:
                        reply.content = reply.content[: starts.pop(chunk.id)]
                        await reply.update()
                    continue
                if chunk.content:
                    starts.setdefault(chunk.id, len(reply.content))
                    await reply.stream_token(chunk.content)
            state = await agent.aget_state(config)
            if not state.interrupts:  # show exactly Cravio's final answer, whatever streamed on the way
                reply.content = state.values["messages"][-1].text
            if reply.content:
                await reply.send()
            if not state.interrupts:
                return
            actions = [a for intr in state.interrupts for a in intr.value["action_requests"]]
            decision = await _ask_approval(actions)
            payload = resume_with(state.interrupts, [decision] * len(actions))
