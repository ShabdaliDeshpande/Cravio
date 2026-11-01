"""For an order question, does Cravio look in the right store: Swiggy Food, Instamart, or both?

  uv run python evals/run_store_evals.py            run here, print PASS/FAIL, upload nothing
  uv run python evals/run_store_evals.py refund     only the examples whose message has this word
  uv run python evals/run_store_evals.py --upload   score it in LangSmith (dataset cravio-store-scenarios)

Runs the whole conversation on real Swiggy, like run_support_evals.py: approval pauses are declined and
tickets go to a temporary file. Both checks hold whatever is in your order history, so results stay stable:
  looked_in_right_store   it read the order list of the store the question is about (both when unclear)
  ticket_on_right_store   a grocery problem is never ticketed on a Food order, and the other way round
How Cravio handles a real Instamart order (which Swiggy keeps for 15 days) belongs to the watcher.
"""

import argparse
import asyncio
import re
import sys

from run_support_evals import ORDER_ID, run_example  # also loads .env before cravio is imported

import langsmith.env  # noqa: E402
from langsmith import aevaluate  # noqa: E402

from cravio.agent import MODEL  # noqa: E402
from cravio.langsmith_tracing import masking_client  # noqa: E402
from cravio.pii import mask  # noqa: E402
from cravio.swiggy.client import build_client, swiggy_tools  # noqa: E402

DATASET = "cravio-store-scenarios"

# (what the user says, answer if Cravio asks something first, which store it must look in)
EXAMPLES = [
    ("Where is my order?", None, "both"),
    ("Where is my Instamart order?", None, "instamart"),
    ("Where is my Swiggy Food order from the restaurant?", None, "food"),
    ("I want a refund for my latest order.", "My most recent order.", "both"),
    ("My Instamart grocery order had a missing item.", "My most recent Instamart order.", "instamart"),
]

FOOD_ORDER_TOOLS = {"get_food_orders", "get_food_order_details", "track_food_order"}
INSTAMART_ORDER_TOOLS = {"get_orders", "get_delivery_status"}


async def food_order_ids(tools) -> list[str]:
    """Every Food order ID on the account, read once, so a ticket can be checked against it."""
    by_name = {t.metadata["swiggy_name"]: t for t in tools if t.metadata.get("swiggy_server") == "food"}
    addresses = await by_name["get_addresses"].ainvoke({})
    address_id = re.search(r"\(ID: ([^)]+)\)", _text(addresses)).group(1)
    orders = await by_name["get_food_orders"].ainvoke({"addressId": address_id, "orderCount": 15})
    return ORDER_ID.findall(_text(orders))


def _text(result) -> str:
    return result if isinstance(result, str) else "\n".join(b.get("text", "") for b in result if isinstance(b, dict))


def looked_in_right_store(outputs: dict, reference_outputs: dict) -> dict:
    events = set(outputs.get("events", []))
    food, instamart = bool(events & FOOD_ORDER_TOOLS), bool(events & INSTAMART_ORDER_TOOLS)
    ok = {"both": food and instamart, "food": food, "instamart": instamart}[reference_outputs["store"]]
    return {"key": "looked_in_right_store", "score": int(ok)}


def ticket_on_right_store(outputs: dict, reference_outputs: dict) -> dict:
    food_ids, store = set(outputs.get("food_order_ids", [])), reference_outputs["store"]
    on_food = [t["order_id"] in food_ids for t in outputs.get("tickets", [])]
    ok = not any(on_food) if store == "instamart" else all(on_food) if store == "food" else True
    return {"key": "ticket_on_right_store", "score": int(ok)}


EVALUATORS = [looked_in_right_store, ticket_on_right_store]


async def run(tools, food_ids: list[str], inputs: dict) -> dict:
    try:
        outputs = await run_example(tools, inputs)
    except Exception as exc:  # a crash or timeout fails this example, not the whole run
        outputs = {"error": f"{type(exc).__name__}: {exc}"}
    return {**outputs, "food_order_ids": food_ids}


def _inputs(text: str, follow_up: str | None) -> dict:
    turns = {"turns": [f"Use my Home address. {text}"]}
    return {**turns, "follow_up": follow_up} if follow_up else turns


def sync_dataset(client) -> None:
    """Create the LangSmith dataset from EXAMPLES, adding any example it doesn't have yet."""
    if client.has_dataset(dataset_name=DATASET):
        dataset = client.read_dataset(dataset_name=DATASET)
    else:
        dataset = client.create_dataset(DATASET, description="Which store Cravio checks for an order question (evals/run_store_evals.py)")
    have = {e.inputs["turns"][0] for e in client.list_examples(dataset_id=dataset.id)}
    for text, follow_up, store in EXAMPLES:
        inputs = _inputs(text, follow_up)
        if inputs["turns"][0] not in have:
            client.create_example(inputs=inputs, outputs={"store": store}, dataset_id=dataset.id)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("word", nargs="?", default="", help="without --upload: only examples whose message has this word")
    parser.add_argument("--upload", action="store_true", help="score it in LangSmith")
    parser.add_argument("--reps", type=int, default=1, help="with --upload: runs per example")
    args = parser.parse_args()

    async with swiggy_tools(build_client(["food", "im"])) as tools:  # one real Swiggy session for all examples
        food_ids = await food_order_ids(tools)
        if args.upload:
            langsmith.env.get_git_info = lambda *a, **k: None  # LangSmith would attach your git name and email
            client = masking_client()
            sync_dataset(client)

            async def target(inputs: dict) -> dict:
                return await run(tools, food_ids, inputs)

            await aevaluate(
                target, data=DATASET, evaluators=EVALUATORS, client=client, experiment_prefix="store",
                num_repetitions=args.reps, max_concurrency=1, metadata={"model": MODEL},
            )
            return
        passed = 0
        examples = [e for e in EXAMPLES if args.word.lower() in e[0].lower()]
        for text, follow_up, store in examples:
            outputs = await run(tools, food_ids, _inputs(text, follow_up))
            scores = {e.__name__: e(outputs, {"store": store})["score"] for e in EVALUATORS}
            ok = all(scores.values())
            passed += ok
            looked = sorted(set(outputs.get("events", [])) & (FOOD_ORDER_TOOLS | INSTAMART_ORDER_TOOLS))
            print(f"{'PASS' if ok else 'FAIL'}  [{store}] {text}")
            print(f"      looked at: {looked}  tickets: {[t['order_id'] for t in outputs.get('tickets', [])]}  {outputs.get('error', '')}")
            if not ok:
                print(f"      failed: {[k for k, v in scores.items() if not v]}")
                print(f"      reply: {mask(' '.join((outputs.get('reply') or '').split()))[:300]}")
        print(f"\n{passed}/{len(examples)} looked in the right store")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
