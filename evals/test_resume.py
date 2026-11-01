"""Quit halfway, come back: does Cravio continue where it stopped, without redoing finished steps?

  uv run python evals/test_resume.py

Real Swiggy, read-only ("Where is my order?"). The first run is stopped on purpose just as Cravio hands the
question to its tracking subagent, as if you closed the app. The second run starts from the saved
conversation, as `uv run cravio` does, and must finish the request without reading your addresses again.
"""

import asyncio
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # before importing cravio: it reads settings at import

from langchain.agents.middleware import wrap_tool_call  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from cravio.agent import build_agent  # noqa: E402
from cravio.progress import Progress, steps_in_last_turn  # noqa: E402
from cravio.swiggy.client import build_client, swiggy_tools  # noqa: E402


class AppClosed(Exception):
    """Stands in for you closing the app mid-run."""


@wrap_tool_call
async def close_at_handoff(request, handler):
    if request.tool_call["name"] == "task":
        raise AppClosed("closed while handing off to a subagent")
    return await handler(request)


async def main() -> int:
    saver = InMemorySaver()  # both runs share it, as the CLI's runs share .cravio/conversations.sqlite
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 80}
    question = {"messages": [{"role": "user", "content": "Use my Home address. Where is my order?"}]}

    async with swiggy_tools(build_client(["food", "im"])) as tools:
        first = build_agent(tools, middleware=[close_at_handoff], checkpointer=saver)
        try:
            await first.ainvoke(question, config)
            print("FAIL  the first run was supposed to stop halfway")
            return 1
        except AppClosed:
            pass

        state = await first.aget_state(config)
        saved = steps_in_last_turn(state.values["messages"])
        print("After quitting, the saved conversation shows:")
        for step in saved:
            print(f"   {step.status:10} {step.label}")
        checks = {
            "the run is marked unfinished": bool(state.next),
            "finished steps were saved": any(s.status == "completed" for s in saved),
            "the hand-off shows as unfinished": any(s.tool == "task" and s.status == "unfinished" for s in saved),
        }

        steps = []
        second = build_agent(tools, middleware=[Progress(steps.append)], checkpointer=saver)
        result = await asyncio.wait_for(second.ainvoke(None, config), 300)  # continue from the saved point
        cravio_steps = {s.id: s for s in steps if s.depth == 0}.values()  # each step's final state, Cravio's own
        reply = result["messages"][-1].text
        checks |= {
            "it finished with a reply": bool(reply.strip()),
            "the tracking hand-off ran": any(s.tool == "task" and s.status == "completed" for s in cravio_steps),
            "it didn't read your addresses again": not any(s.tool == "get_addresses" for s in cravio_steps),
        }
        print("\nAfter continuing, Cravio ran:", ", ".join(f"{s.label} ({s.status})" for s in cravio_steps))
        print("Reply:", " ".join(reply.split())[:200], "\n")

    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(asyncio.run(main()))
