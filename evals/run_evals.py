"""Run Cravio's cart scenarios against the fake Swiggy, and score them.

  uv run python evals/run_evals.py --local                 try every scenario once, print PASS/FAIL, upload nothing
  uv run python evals/run_evals.py                          a LangSmith experiment with the current prompt, 3 runs each
  uv run python evals/run_evals.py --prompt both --reps 5   current and pre-fix prompts, side by side in LangSmith
  uv run python evals/run_evals.py --local --only change-quantity

Each run gets a new fake Swiggy and a new conversation, so runs can't affect each other.
"""

import argparse
import asyncio
import re
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # before importing cravio: it reads settings at import

import langsmith.env  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langsmith import aevaluate  # noqa: E402

import cravio.agent  # noqa: E402
from cravio.cli import decide, resume_with  # noqa: E402
from cravio.langsmith_tracing import masking_client  # noqa: E402
from fake_swiggy import FakeSwiggy  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402

DATASET = "cravio-cart-scenarios"
CART_CHANGES = {"update_food_cart", "flush_food_cart"}
TURN_TIMEOUT = 300  # seconds; a stuck model fails the scenario instead of hanging the run

# The cart rules before the fixes of 1 Nov 2026, to show what the evals catch.
OLD_CART_RULES = (
    "- To add a dish, call search_menu with restaurantIdOfAddedItem to get its exact item, variant and addon IDs, "
    "then update_food_cart, then get_food_cart to show the result.\n"
    "- Swiggy can add an item and warn in the same reply (restaurant not taking orders, item out of stock). The item "
    "is then in the cart but can't be ordered: say exactly that, and offer to replace or remove it.\n"
    "- Changing the cart, clearing it and saving an address each pause for the user's approval. Say in one line what "
    "you are about to do before the call. If the user declines, ask what they'd like instead."
)


def _baseline(prompt: str) -> str:
    start, end = prompt.index("- Changing the cart, clearing it"), prompt.index("\n\nCheckout (demo mode)")
    return prompt[:start] + OLD_CART_RULES + prompt[end:]


PROMPTS = {"current": cravio.agent.MAIN_PROMPT, "baseline": _baseline(cravio.agent.MAIN_PROMPT)}


# --- Running a scenario -------------------------------------------------------------------------

async def _send(agent, config: dict, text: str, approve: bool) -> tuple[str, bool]:
    """One user message, answering any approval pause the way the scenario says. Returns (reply, paused)."""
    result = await agent.ainvoke({"messages": [{"role": "user", "content": text}]}, config)
    paused = False
    for _ in range(5):
        pending = (await agent.aget_state(config)).interrupts
        if not pending:
            break
        paused = True
        actions = [a for intr in pending for a in intr.value["action_requests"]]
        verdict = decide("yes" if approve else "No, leave my cart as it is.")
        result = await agent.ainvoke(resume_with(pending, [verdict] * len(actions)), config)
    return result["messages"][-1].text, paused


def _said_before_first_change(messages) -> str:
    """What Cravio wrote up to and including the message where it first tried to change the cart."""
    said = []
    for m in messages:
        if m.type == "ai":
            said.append(m.text)
            if any(c["name"] in CART_CHANGES for c in m.tool_calls):
                break
    return "\n".join(said)


async def run_scenario(inputs: dict) -> dict:
    fake = FakeSwiggy()
    fake.set_cart(*inputs["cart"])
    agent = cravio.agent.build_agent(fake.tools(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 80}
    app_changes = {turn: (rid, items) for turn, rid, items in inputs.get("app_changes_cart", [])}
    turns = []
    try:
        for i, text in enumerate(inputs["turns"]):
            if i in app_changes:
                fake.set_cart(*app_changes[i])
            first_call = len(fake.calls)
            reply, paused = await asyncio.wait_for(_send(agent, config, text, inputs.get("approve", True)), TURN_TIMEOUT)
            changed = any(name in CART_CHANGES for name, _ in fake.calls[first_call:])
            if inputs.get("follow_up") and not (paused or changed) and "?" in reply:
                reply, _ = await asyncio.wait_for(_send(agent, config, inputs["follow_up"], inputs.get("approve", True)), TURN_TIMEOUT)
            turns.append({"user": text, "calls": fake.calls[first_call:], "reply": reply})
    except Exception as exc:  # a crash or timeout fails this scenario rather than the whole run
        return {"error": f"{type(exc).__name__}: {exc}", "cart": [fake.restaurant_id, fake.cart], "turns": turns}
    messages = (await agent.aget_state(config)).values["messages"]
    return {"cart": [fake.restaurant_id, fake.cart], "turns": turns, "said_before_first_change": _said_before_first_change(messages)}


# --- Scoring --------------------------------------------------------------------------------------

_NEGATED = re.compile(r"(not|n't|never)\s+(been\s+)?(added|updated|changed)")
_CLAIMED = re.compile(r"\b(added|updated your cart)\b")


def _claims_success(reply: str) -> bool:
    """Rough: "I've added it" counts as a claim, "it was not added" doesn't."""
    return bool(_CLAIMED.search(_NEGATED.sub("", reply.lower())))


def check_scenario(outputs: dict, reference_outputs: dict) -> dict:
    """Score one run against its scenario's checks. LangSmith shows each check as a column, plus 'passed'."""
    checks = reference_outputs["checks"]
    if outputs.get("error"):
        return {"results": [{"key": "passed", "score": 0, "comment": outputs["error"]}]}
    turns = outputs["turns"]
    calls = [(name, args) for turn in turns for name, args in turn["calls"]]
    names = [name for name, _ in calls]
    reply = turns[-1]["reply"].lower()
    first_change = next((i for i, name in enumerate(names) if name in CART_CHANGES), None)

    results = {}
    if "cart" in checks:
        results["cart"] = outputs["cart"] == checks["cart"]
    if checks.get("reads_cart_before_change"):
        results["reads_cart_before_change"] = first_change is not None and "get_food_cart" in names[:first_change]
    if checks.get("reads_cart_in_last_turn"):
        results["reads_cart_in_last_turn"] = any(name == "get_food_cart" for name, _ in turns[-1]["calls"])
    if checks.get("no_restaurant_name"):
        results["no_restaurant_name"] = not any(args.get("restaurantName") for name, args in calls if name == "get_food_cart")
    if "reply_mentions" in checks:
        results["reply_mentions"] = all(word in reply for word in checks["reply_mentions"])
    if "reply_avoids" in checks:
        results["reply_avoids"] = not any(word in reply for word in checks["reply_avoids"])
    if "warns_before_change" in checks:
        said = outputs["said_before_first_change"].lower()
        results["warns_before_change"] = all(word in said for word in checks["warns_before_change"])
    if checks.get("no_success_claim"):
        results["no_success_claim"] = not _claims_success(reply)

    scores = [{"key": key, "score": int(ok)} for key, ok in results.items()]
    return {"results": scores + [{"key": "passed", "score": int(all(results.values()))}]}


def _example(scenario: dict) -> tuple[dict, dict]:
    inputs = {k: v for k, v in scenario.items() if k not in ("name", "about", "checks")}
    return inputs, {"checks": scenario["checks"]}


# --- Local runs ----------------------------------------------------------------------------------

async def run_local(prompt: str, reps: int, scenarios: list[dict]) -> None:
    passed: dict[str, int] = {}
    for scenario in scenarios:
        inputs, reference = _example(scenario)
        for rep in range(reps):
            outputs = await run_scenario(inputs)
            scores = {r["key"]: r["score"] for r in check_scenario(outputs, reference)["results"]}
            passed[scenario["name"]] = passed.get(scenario["name"], 0) + scores["passed"]
            failed = [key for key, score in scores.items() if not score and key != "passed"]
            detail = outputs.get("error") or (f"failed: {', '.join(failed)}" if failed else "")
            print(f"{'PASS' if scores['passed'] else 'FAIL'}  {scenario['name']} #{rep + 1}  {detail}")
            if not scores["passed"]:
                print(f"      cart: {outputs.get('cart')}\n      calls: {[n for t in outputs.get('turns', []) for n, _ in t['calls']]}"
                      f"\n      reply: {' '.join(outputs['turns'][-1]['reply'].split())[:300] if outputs.get('turns') else ''}")
    print(f"\n{prompt} prompt: " + ", ".join(f"{name} {n}/{reps}" for name, n in passed.items()))


# --- LangSmith experiments -----------------------------------------------------------------------

def sync_dataset(client) -> None:
    """Make the LangSmith dataset match SCENARIOS, keeping example IDs so experiments stay comparable."""
    if client.has_dataset(dataset_name=DATASET):
        dataset = client.read_dataset(dataset_name=DATASET)
    else:
        dataset = client.create_dataset(DATASET, description="Cravio cart behaviour against a fake Swiggy (evals/scenarios.py)")
    existing = {e.metadata.get("name"): e for e in client.list_examples(dataset_id=dataset.id)}
    for scenario in SCENARIOS:
        inputs, outputs = _example(scenario)
        metadata = {"name": scenario["name"], "about": scenario["about"]}
        example = existing.pop(scenario["name"], None)
        if example is None:
            client.create_example(inputs=inputs, outputs=outputs, metadata=metadata, dataset_id=dataset.id)
        elif example.inputs != inputs or example.outputs != outputs:
            client.update_example(example.id, inputs=inputs, outputs=outputs, metadata=metadata)
    for stale in existing.values():
        client.delete_example(stale.id)


async def run_experiment(client, prompt: str, reps: int) -> None:
    results = await aevaluate(
        run_scenario, data=DATASET, evaluators=[check_scenario], client=client,
        experiment_prefix=f"{prompt}-prompt", num_repetitions=reps, max_concurrency=1,  # one GPU: one run at a time
        metadata={"prompt": prompt, "model": cravio.agent.MODEL},
    )
    passed: dict[str, list[int]] = {}
    async for row in results:
        name = row["example"].metadata["name"]
        score = next(r.score for r in row["evaluation_results"]["results"] if r.key == "passed")
        passed.setdefault(name, []).append(int(score))
    print(f"\n{results.experiment_name}: " + ", ".join(f"{name} {sum(s)}/{len(s)}" for name, s in passed.items()))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true", help="run here and print results; upload nothing")
    parser.add_argument("--prompt", choices=["current", "baseline", "both"], default="current")
    parser.add_argument("--reps", type=int, default=None, help="runs per scenario (default: 1 local, 3 in LangSmith)")
    parser.add_argument("--only", help="run just this scenario (local only)")
    args = parser.parse_args()
    prompts = ["current", "baseline"] if args.prompt == "both" else [args.prompt]

    if args.local:
        scenarios = [s for s in SCENARIOS if not args.only or s["name"] == args.only]
        for prompt in prompts:
            cravio.agent.MAIN_PROMPT = PROMPTS[prompt]  # build_agent reads it on every call
            await run_local(prompt, args.reps or 1, scenarios)
        return

    # LangSmith would attach your git name and email to each experiment, unmasked.
    langsmith.env.get_git_info = lambda *args, **kwargs: None
    client = masking_client()
    sync_dataset(client)
    for prompt in prompts:
        cravio.agent.MAIN_PROMPT = PROMPTS[prompt]
        await run_experiment(client, prompt, args.reps or 3)
    print(f"\nCompare them in LangSmith: Datasets & Experiments → {DATASET}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
