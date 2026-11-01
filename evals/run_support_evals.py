"""Run Cravio's support examples (LangSmith dataset cravio-support-scenarios) on real Swiggy.

  uv run python evals/run_support_evals.py allergic    run the examples whose message has this word; upload nothing
  uv run python evals/run_support_evals.py --upload    run every example 3 times and score it in LangSmith
  uv run python evals/run_support_evals.py --upload --reps 1

Safe for your real account: every approval pause is declined (so the cart never changes),
order and payment tools are blocked anyway, and tickets go to a temporary file.
"""

import argparse
import asyncio
import json
import re
import sys
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # before importing cravio: it reads settings at import

import langsmith.env  # noqa: E402
from langchain.agents.middleware import wrap_tool_call  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langsmith import Client, aevaluate  # noqa: E402

import cravio.tools  # noqa: E402
from cravio.agent import MODEL, build_agent  # noqa: E402
from cravio.cli import decide, resume_with  # noqa: E402
from cravio.langsmith_tracing import masking_client  # noqa: E402
from cravio.pii import mask  # noqa: E402
from cravio.swiggy.client import build_client, swiggy_tools  # noqa: E402

DATASET = "cravio-support-scenarios"
TURN_TIMEOUT = 300  # seconds; a stuck model fails the example instead of hanging the run
# Their answers go in swiggy_said. get_orders and get_delivery_status are Instamart's.
ORDER_TOOLS = {"get_food_orders", "get_food_order_details", "track_food_order", "get_orders", "get_delivery_status"}
ORDER_ID = re.compile(r"Order (\d+)")  # Swiggy lists "1. Order 123456789012345 — Polar Bear | ...", newest first


async def run_example(tools, inputs: dict) -> dict:
    """Send one example's messages to Cravio and collect what it did."""
    events = []  # every tool Cravio and its subagents call, in order
    calls = []  # the same calls with their arguments, e.g. search_menu's vegFilter
    swiggy_said = []  # (tool name, Swiggy's answer) for the order tools

    @wrap_tool_call
    async def record(request, handler):
        name = request.tool_call["name"]
        events.append(name)
        calls.append({"name": name, "args": request.tool_call["args"]})
        result = await handler(request)
        if name in ORDER_TOOLS:
            swiggy_said.append((name, getattr(result, "text", "")))
        return result

    tickets_file = Path(tempfile.mkdtemp()) / "tickets.json"
    cravio.tools.TICKETS_FILE = tickets_file  # eval tickets stay out of .cravio/tickets.json
    agent = build_agent(tools, middleware=[record], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 80}

    async def send(text: str) -> str:
        result = await agent.ainvoke({"messages": [{"role": "user", "content": text}]}, config)
        for _ in range(5):  # a pause means a cart change or checkout: always say no
            pending = (await agent.aget_state(config)).interrupts
            if not pending:
                break
            actions = [a for intr in pending for a in intr.value["action_requests"]]
            result = await agent.ainvoke(resume_with(pending, [decide("No.")] * len(actions)), config)
        return result["messages"][-1].text

    def tickets() -> list[dict]:
        return json.loads(tickets_file.read_text(encoding="utf-8")) if tickets_file.exists() else []

    asked_before_ticket = False
    for text in inputs["turns"]:
        reply = await asyncio.wait_for(send(text), TURN_TIMEOUT)
    if inputs.get("follow_up") and not tickets() and "?" in reply:  # Cravio asked something first
        asked_before_ticket = True
        reply = await asyncio.wait_for(send(inputs["follow_up"]), TURN_TIMEOUT)
    order_lists = "\n".join(text for name, text in swiggy_said if name == "get_food_orders")
    return {
        "reply": reply,
        "events": events,
        "calls": calls,
        "tickets": tickets(),
        "asked_before_ticket": asked_before_ticket,
        "orders_seen": list(dict.fromkeys(ORDER_ID.findall(order_lists))),  # newest first, no repeats
        "swiggy_said": "\n\n".join(text for _, text in swiggy_said),
    }


async def upload(tools, reps: int) -> None:
    """Run every example and send the results to LangSmith, where your 5 evaluators score them."""
    langsmith.env.get_git_info = lambda *args, **kwargs: None  # LangSmith would attach your git name and email

    async def target(inputs: dict) -> dict:
        try:
            return await run_example(tools, inputs)
        except Exception as exc:  # a crash or timeout fails this example, not the whole run
            return {"error": f"{type(exc).__name__}: {exc}"}

    results = await aevaluate(
        target, data=DATASET, client=masking_client(),  # addresses and phones hidden before upload
        experiment_prefix="support-real", num_repetitions=reps,
        max_concurrency=1,  # one GPU: one example at a time
        metadata={"model": MODEL},
    )
    print(f"\nDone: {results.experiment_name}\nOpen LangSmith → Datasets & Experiments → {DATASET} → Experiments")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("word", nargs="?", default="", help="without --upload: only examples whose message has this word")
    parser.add_argument("--upload", action="store_true", help="run every example and score it in LangSmith")
    parser.add_argument("--reps", type=int, default=3, help="with --upload: runs per example")
    args = parser.parse_args()

    # Food and Instamart, as the app runs. orders_seen is Food's list, so "latest" means the newest Food order:
    # right while Instamart (which keeps 15 days) has nothing newer.
    async with swiggy_tools(build_client(["food", "im"])) as tools:  # one real Swiggy session for all examples
        if args.upload:
            await upload(tools, args.reps)
            return
        examples = [e for e in Client().list_examples(dataset_name=DATASET) if args.word.lower() in e.inputs["turns"][0].lower()]
        for example in examples:
            print("=" * 80 + "\nUSER: " + example.inputs["turns"][0])
            outputs = await run_example(tools, example.inputs)
            print(mask(json.dumps(outputs, indent=2, ensure_ascii=False)))  # hide addresses, phones in the printout


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
