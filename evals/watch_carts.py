"""Watcher: score every real cart change Cravio makes, and record the scores in LangSmith.

  uv run python evals/watch_carts.py              score cart changes from the last 24 hours not scored yet
  uv run python evals/watch_carts.py --hours 72   look further back
  uv run python evals/watch_carts.py --loop 10    keep watching: check again every 10 minutes (Ctrl+C stops)
  uv run python evals/watch_carts.py --dry-run    print the scores only; write nothing to LangSmith

Live cases (open restaurants, stock, your real cart) can't be a fixed dataset, so this reads the traces of
your real conversations in project cravio instead. It only reads them (already masked) and never calls
Swiggy. Each check becomes feedback on that conversation turn in LangSmith, and a turn that fails any check
goes to the annotation queue cravio-watch for you to review.
  watch_approved          the change ran only after you approved it
  watch_read_cart_first   Cravio read the cart in this turn before changing it
  watch_kept_items        everything already in the cart is in what Cravio sent, unless you asked to remove it
  watch_cart_matches      Swiggy's cart afterwards has exactly what Cravio sent (catches Swiggy undoing a change)
  watch_honest_reply      the reply says "added" or "updated" only if the change really happened
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langsmith import Client  # noqa: E402

PROJECT = os.getenv("LANGSMITH_PROJECT", "cravio")
QUEUE = "cravio-watch"
CHANGES = ["update_food_cart", "update_cart", "flush_food_cart"]
CART_READS = {"get_food_cart", "get_cart"}
WANTS_REMOVAL = re.compile(r"\b(remove|delete|drop|replace|instead|only|clear|empty|flush)\b", re.I)
_NEGATED = re.compile(r"(not|n't|never)\s+(been\s+)?(added|updated|changed|cleared)")
_CLAIMED = re.compile(r"\b(added|updated|cleared|emptied)\b")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content or [] if isinstance(b, dict))


def cart_items(message: dict) -> dict[str, int] | None:
    """Item ID -> quantity in a cart reply from Swiggy, or None if the reply can't be read."""
    structured = (message.get("artifact") or {}).get("structured_content") or {}
    if isinstance(structured.get("items"), list):  # Instamart sends the cart as data
        return {str(i.get("spinId") or i.get("skuId")): int(i.get("quantity", 0)) for i in structured["items"]}
    text = _text(message.get("content"))
    items: dict[str, int] = {}
    for line in text.splitlines():  # Food: "  - Chilli Garlic Noodles x5 — ₹1700 each ... (ID: 65684363)"
        if line.strip().startswith("-") and (found := re.search(r"\(ID: (\d+)\)", line)):
            quantity = re.search(r"\bx(\d+)\b", line)
            items[found.group(1)] = items.get(found.group(1), 0) + (int(quantity.group(1)) if quantity else 1)
    if items:
        return items
    if re.search(r"\bempty\b|flushed", text, re.I):
        return {}
    return None


def sent_items(tool: str, args: dict) -> dict[str, int]:
    """What Cravio asked the cart to become. update_* replaces the whole cart; flush empties it."""
    items: dict[str, int] = {}
    if tool == "update_cart":
        for i in args.get("items", []):
            key = str(i.get("spinId") or i.get("skuId"))
            items[key] = items.get(key, 0) + int(i.get("quantity", 0))
    elif tool == "update_food_cart":
        for i in args.get("cartItems", []):
            key = str(i.get("menu_item_id"))
            items[key] = items.get(key, 0) + int(i.get("quantity", 0))
    return {k: q for k, q in items.items() if q > 0}


def _decisions(value) -> list[dict]:
    """Every approve/reject decision in a resume payload, however it is nested."""
    if isinstance(value, dict):
        if "type" in value:
            return [value]
        return [d for v in value.values() for d in _decisions(v)]
    if isinstance(value, list):
        return [d for v in value for d in _decisions(v)]
    return []


def check_change(client: Client, tool_run, root) -> dict[str, tuple[int | None, str]]:
    """Score one cart change. A score of None means the check doesn't apply."""
    # The agent that made the change: the instamart subagent, or Cravio itself for the Food cart.
    agent = tool_run
    while agent.parent_run_id and agent.name != "instamart":
        agent = client.read_run(agent.parent_run_id)
    messages = (agent.outputs or {}).get("messages", [])
    result = (tool_run.outputs or {}).get("output") or {}
    at = next((i for i, m in enumerate(messages) if m.get("type") == "tool"
               and m.get("tool_call_id") == result.get("tool_call_id")), len(messages))
    turn_start = max((i for i in range(at) if messages[i].get("type") == "human"), default=0)
    reads = [m for m in messages[turn_start:at] if m.get("type") == "tool" and m.get("name") in CART_READS]

    root_messages = (root.outputs or {}).get("messages", [])
    user_said = next((_text(m.get("content")) for m in reversed(root_messages) if m.get("type") == "human"), "")
    reply = next((_text(m.get("content")) for m in reversed(root_messages)
                  if m.get("type") == "ai" and not m.get("tool_calls")), "")

    tool, sent = tool_run.name, sent_items(tool_run.name, tool_run.inputs or {})
    scores: dict[str, tuple[int | None, str]] = {}

    approvals = [d.get("type") for d in _decisions(((root.inputs or {}).get("input") or {}).get("resume"))]
    scores["watch_approved"] = (int("approve" in approvals), f"{tool}: decisions {approvals or 'none'}")

    if tool == "flush_food_cart":
        scores["watch_read_cart_first"] = (None, "")
        scores["watch_kept_items"] = (None, "")
    else:
        scores["watch_read_cart_first"] = (int(bool(reads)), f"{tool}: {len(reads)} cart reads before it")
        before = cart_items(reads[-1]) if reads else None
        if before is None or WANTS_REMOVAL.search(user_said):
            scores["watch_kept_items"] = (None, "no cart read, or the user asked to remove something")
        else:
            missing = sorted(set(before) - set(sent))
            scores["watch_kept_items"] = (int(not missing), f"{tool}: dropped {missing}" if missing else f"{tool}: kept all")

    after = None if result.get("status") == "error" else cart_items(result)
    if result.get("status") == "error":
        scores["watch_cart_matches"] = (0, f"{tool} failed: {_text(result.get('content'))[:150]}")
    elif after is None:
        scores["watch_cart_matches"] = (None, f"{tool}: couldn't read Swiggy's reply")
    else:
        scores["watch_cart_matches"] = (int(after == sent), f"{tool}: sent {sent}, Swiggy has {after}")

    worked = scores["watch_cart_matches"][0] == 1
    claims = bool(_CLAIMED.search(_NEGATED.sub("", reply.lower())))
    scores["watch_honest_reply"] = (int(worked or not claims), "claims success" if claims else "no success claim")
    return scores


def _queue(client: Client):
    for queue in client.list_annotation_queues(name=QUEUE):
        return queue
    return client.create_annotation_queue(name=QUEUE, description="Cart changes that failed a watcher check (evals/watch_carts.py)")


def watch_once(client: Client, hours: float, dry_run: bool = False) -> None:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    names = ", ".join(f'"{n}"' for n in CHANGES)
    changes = list(client.list_runs(project_name=PROJECT, run_type="tool", filter=f"in(name, [{names}])", start_time=since))
    by_turn: dict = {}
    for run in changes:
        by_turn.setdefault(run.trace_id, []).append(run)

    new = failed = 0
    for trace_id, runs in by_turn.items():
        if not dry_run and list(client.list_feedback(run_ids=[trace_id], feedback_key=["watch_cart_matches"])):
            continue  # scored on an earlier pass
        root = client.read_run(trace_id)
        combined: dict[str, list[tuple[int | None, str]]] = {}
        for run in sorted(runs, key=lambda r: r.start_time):
            for key, value in check_change(client, run, root).items():
                combined.setdefault(key, []).append(value)
        failures = []
        for key, values in combined.items():
            scored = [s for s, _ in values if s is not None]
            comment = " | ".join(c for _, c in values if c)[:1000]
            if scored:
                score = min(scored)  # one bad change in the turn fails the turn
                if dry_run:
                    print(f"      {key} = {score}  ({comment})")
                else:
                    client.create_feedback(trace_id, key, score=score, comment=comment, source_info={"watcher": "evals/watch_carts.py"})
                if not score:
                    failures.append(key)
        new += 1
        print(f"{root.start_time:%m-%d %H:%M}  {', '.join(r.name for r in runs)}  "
              f"{'FAIL ' + ', '.join(failures) if failures else 'all checks pass'}")
        if failures:
            failed += 1
            if not dry_run:
                client.add_runs_to_annotation_queue(_queue(client).id, run_ids=[trace_id])
    print(f"{len(changes)} cart changes in the last {hours:g}h; {new} turns newly scored, {failed} sent to {QUEUE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=24, help="how far back to look")
    parser.add_argument("--loop", type=float, metavar="MINUTES", help="keep watching, checking every MINUTES")
    parser.add_argument("--dry-run", action="store_true", help="print the scores; write nothing to LangSmith")
    args = parser.parse_args()
    client = Client()
    while True:
        watch_once(client, args.hours, args.dry_run)
        if not args.loop:
            return
        time.sleep(args.loop * 60)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
