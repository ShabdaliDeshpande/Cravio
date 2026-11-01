"""Does Cravio send each request to the right place? Food (restaurants), Instamart (groceries), support or tracking.

  uv run python evals/run_intent_evals.py            run here, print PASS/FAIL, upload nothing
  uv run python evals/run_intent_evals.py --upload   score it in LangSmith (dataset cravio-intent-scenarios)

Runs on real Swiggy (Food and Instamart) but stops at Cravio's first routing decision, before any subagent
works: nothing touches a cart, and each example takes seconds.
"""

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # before importing cravio: it reads settings at import

import langsmith.env  # noqa: E402
from langchain.agents.middleware import wrap_tool_call  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langsmith import aevaluate  # noqa: E402

from cravio.agent import MODEL, build_agent  # noqa: E402
from cravio.langsmith_tracing import masking_client  # noqa: E402
from cravio.swiggy.client import build_client, swiggy_tools  # noqa: E402

DATASET = "cravio-intent-scenarios"

# (what the user says, where it should go)
EXAMPLES = [
    ("I want to order chilli sauce for dinner.", "instamart"),
    ("Add 2 packets of Maggi and a litre of milk to my cart.", "instamart"),
    ("I need toothpaste and a bar of soap.", "instamart"),
    ("Get me tomatoes and onions, I'm cooking tonight.", "instamart"),
    ("I want chilli paneer for dinner.", "food"),
    ("Suggest a good biryani place nearby.", "food"),
    ("Order some momos for me.", "food"),
    ("Where is my order?", "tracking"),
    ("Where is my Instamart order?", "tracking"),
    ("My last order had a missing item.", "support"),
    ("My grocery order had a missing item.", "support"),
]

# Each subagent, and each tool the main agent uses itself, belongs to one place.
ROUTE_OF = {
    "instamart": "instamart",
    "recommender": "food", "taste-profiler": "food",
    "search_restaurants": "food", "search_menu": "food", "get_food_cart": "food", "update_food_cart": "food",
    "tracking": "tracking",
    "support-escalation": "support",
}


class Routed(Exception):
    """Raised at Cravio's first routing decision, to stop the run there."""

    def __init__(self, route: str) -> None:
        self.route = route


def _find_routed(exc: BaseException) -> Routed | None:
    if isinstance(exc, Routed):
        return exc
    for inner in getattr(exc, "exceptions", ()):  # the decision may come wrapped in an exception group
        if found := _find_routed(inner):
            return found
    return None


async def route_of(tools, text: str) -> dict:
    """Send one message and return where Cravio first sends it."""

    @wrap_tool_call
    async def stop_at_decision(request, handler):
        call = request.tool_call
        key = call["args"].get("subagent_type") if call["name"] == "task" else call["name"]
        if key in ROUTE_OF:
            raise Routed(ROUTE_OF[key])
        return await handler(request)  # e.g. get_addresses: let it run, it isn't a decision

    agent = build_agent(tools, middleware=[stop_at_decision], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 30}
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [{"role": "user", "content": f"Use my Home address. {text}"}]}, config), 300
        )
    except BaseException as exc:
        if routed := _find_routed(exc):
            return {"route": routed.route}
        return {"route": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {"route": "none", "reply": result["messages"][-1].text}  # answered without sending it anywhere


def right_place(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "right_place", "score": int(outputs.get("route") == reference_outputs["route"])}


def sync_dataset(client) -> None:
    """Create the LangSmith dataset from EXAMPLES, adding any example it doesn't have yet."""
    if client.has_dataset(dataset_name=DATASET):
        dataset = client.read_dataset(dataset_name=DATASET)
    else:
        dataset = client.create_dataset(DATASET, description="Where Cravio sends each request (evals/run_intent_evals.py)")
    have = {e.inputs["text"] for e in client.list_examples(dataset_id=dataset.id)}
    for text, route in EXAMPLES:
        if text not in have:
            client.create_example(inputs={"text": text}, outputs={"route": route}, dataset_id=dataset.id)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upload", action="store_true", help="score it in LangSmith")
    args = parser.parse_args()

    async with swiggy_tools(build_client(["food", "im"])) as tools:  # one real Swiggy session for all examples
        if args.upload:
            langsmith.env.get_git_info = lambda *a, **k: None  # LangSmith would attach your git name and email
            client = masking_client()
            sync_dataset(client)

            async def target(inputs: dict) -> dict:
                return await route_of(tools, inputs["text"])

            await aevaluate(
                target, data=DATASET, evaluators=[right_place],
                client=client, experiment_prefix="intent", max_concurrency=1, metadata={"model": MODEL},
            )
            return
        passed = 0
        for text, expected in EXAMPLES:
            got = await route_of(tools, text)
            ok = got["route"] == expected
            passed += ok
            print(f"{'PASS' if ok else 'FAIL'}  {expected:9} got {got['route']:9}  {text}  {got.get('error', '')}")
        print(f"\n{passed}/{len(EXAMPLES)} sent to the right place")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
