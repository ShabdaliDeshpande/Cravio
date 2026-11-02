"""Watcher: score every real cart change Cravio makes, and record the scores in LangSmith.

  uv run python evals/watch_carts.py              score cart changes from the last 24 hours not scored yet
  uv run python evals/watch_carts.py --hours 72   look further back
  uv run python evals/watch_carts.py --loop 10    keep watching: check again every 10 minutes (Ctrl+C stops)
  uv run python evals/watch_carts.py --dry-run    print the scores only; write nothing to LangSmith
  uv run python evals/watch_carts.py --rescore    score turns again (after a watcher fix), replacing old scores

Live cases (open restaurants, stock, your real cart) can't be a fixed dataset, so this reads the traces of
your real conversations in project cravio instead. It only reads them (already masked) and never calls
Swiggy. Each check becomes feedback on that conversation turn in LangSmith, and a turn that fails any check
goes to the annotation queue cravio-watch for you to review.
  watch_approved          the change ran only after you approved it
  watch_read_cart_first   Cravio read the cart in this turn before changing it
  watch_kept_items        everything already in the cart is in what Cravio sent, unless you asked to remove it
                          or it was a switch to another restaurant (a Food cart holds one restaurant)
  watch_cart_matches      Swiggy's cart afterwards has exactly what Cravio sent (catches Swiggy undoing a change)
  watch_honest_reply      the reply says "added" or "updated" only if the change really happened
  watch_change_stuck      the cart still held what the change left, the last time it was read before the next
                          change: catches Swiggy bringing an old cart back minutes later. Scored once that's
                          known (a later change, or a read 5+ minutes after), so it can come on a later pass.
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
from langsmith.utils import LangSmithNotFoundError, LangSmithRateLimitError  # noqa: E402

from cravio.availability import STORE_CLOSED  # noqa: E402

PROJECT = os.getenv("LANGSMITH_PROJECT", "cravio")
QUEUE = "cravio-watch"
CHANGES = ["update_food_cart", "update_cart", "flush_food_cart"]
CHECKS = ["watch_approved", "watch_read_cart_first", "watch_kept_items", "watch_cart_matches", "watch_honest_reply"]
CART_READS = {"get_food_cart", "get_cart"}
STORE = {"update_food_cart": "food", "flush_food_cart": "food", "get_food_cart": "food", "update_cart": "im", "get_cart": "im"}
SETTLED = timedelta(minutes=5)  # Swiggy has brought old carts back 0.5 to 3 minutes after a change
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
    food = structured.get("data") if isinstance(structured.get("data"), dict) else {}
    if isinstance(food.get("items"), list):  # so does Food: data.items[].menu_item_id and quantity
        items: dict[str, int] = {}
        for i in food["items"]:
            items[str(i.get("menu_item_id"))] = items.get(str(i.get("menu_item_id")), 0) + int(i.get("quantity", 0))
        return items
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


def _patiently(call):
    """Run a LangSmith API call, waiting and trying again if LangSmith says we're asking too fast."""
    for wait in (10, 30, 60):
        try:
            return call()
        except LangSmithRateLimitError:
            print(f"   (LangSmith says too many requests; waiting {wait}s)")
            time.sleep(wait)
    return call()


def _agent_run(client: Client, tool_run, root):
    """The agent that made the change: the instamart subagent, or Cravio itself for the Food cart.
    One request: the subagent runs in this trace, and an enclosing one appears in the tool run's dotted order."""
    subagents = _patiently(lambda: list(client.list_runs(project_name=PROJECT, trace_id=tool_run.trace_id,
                                                         filter='eq(name, "instamart")')))
    return next((r for r in subagents if str(r.id) in (tool_run.dotted_order or "")), root)


def check_change(client: Client, tool_run, root, places: dict[str, str]) -> dict[str, tuple[int | None, str]]:
    """Score one cart change. A score of None means the check doesn't apply. places: menu item ID -> restaurant ID."""
    agent = _agent_run(client, tool_run, root)
    messages = (agent.outputs or {}).get("messages", [])
    result = (tool_run.outputs or {}).get("output") or {}
    at = next((i for i, m in enumerate(messages) if m.get("type") == "tool"
               and m.get("tool_call_id") == result.get("tool_call_id")), len(messages))
    def shows_cart(m: dict) -> bool:  # a cart read (even one Swiggy refused), or its reply to Cravio's own change
        if m.get("type") != "tool":
            return False
        return m.get("name") in CART_READS or (m.get("name") in CHANGES and m.get("status") != "error")

    turn_start = max((i for i in range(at) if messages[i].get("type") == "human"), default=0)
    asked_start = max((i for i in range(turn_start) if messages[i].get("type") == "human"), default=0)
    reads = [m for m in messages[turn_start:at] if shows_cart(m)]
    read_when = "in this turn"
    if not reads:  # you said "yes" to "Shall I add it?": what Cravio saw in the turn that asked counts
        reads = [m for m in messages[asked_start:turn_start] if shows_cart(m)]
        read_when = "in the turn that asked you to confirm"
    if not reads:  # Cravio's safety check (availability.py) reads the live cart in code, just before the change
        checked = _patiently(lambda: list(client.list_runs(project_name=PROJECT, trace_id=tool_run.trace_id, run_type="tool",
                                                           filter='in(name, ["get_food_cart", "get_cart"])')))
        reads = [(r.outputs or {}).get("output") or {} for r in sorted(checked, key=lambda r: r.start_time)
                 if r.start_time <= tool_run.start_time]
        read_when = "by Cravio's safety check just before the change"
    # Cravio told you the cart would be replaced (a dish from another restaurant), as its rules ask.
    announced = any(m.get("type") == "ai" and re.search(r"\breplac", _text(m.get("content")), re.I)
                    for m in messages[asked_start:at])

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
        scores["watch_read_cart_first"] = (int(bool(reads)), f"{tool}: read the cart {read_when}" if reads else f"{tool}: no cart read")
        before = cart_items(reads[-1]) if reads else None
        # Swiggy's cart data doesn't say which restaurant it's from, so a switch counts when Cravio announced it.
        if announced:  # "this will replace the noodles in your cart": the old items go on purpose
            scores["watch_kept_items"] = (None, f"{tool}: cart replaced, as Cravio told you before")
        elif before is None or WANTS_REMOVAL.search(user_said):
            scores["watch_kept_items"] = (None, "no cart read, or the user asked to remove something")
        else:
            missing = sorted(set(before) - set(sent))
            # A Food cart holds one restaurant, so a dish from another one replaces it all (Cravio's approval shows
            # what goes). Swiggy's cart doesn't name its restaurant, so go by the restaurant each item was added from.
            target = str((tool_run.inputs or {}).get("restaurantId", ""))
            old_places = {places[i] for i in before if i in places}
            if tool == "update_food_cart" and missing and not set(before) & set(sent) and old_places and target not in old_places:
                scores["watch_kept_items"] = (1, f"{tool}: switched restaurant, so the old items went: {missing}")
            else:
                unknown = " (Cravio didn't add it in the past week, so its restaurant is unknown)" if missing and not old_places else ""
                scores["watch_kept_items"] = (int(not missing), f"{tool}: dropped {missing}{unknown}" if missing else f"{tool}: kept all")

    after = None if result.get("status") == "error" else cart_items(result)
    if result.get("status") == "error" and STORE_CLOSED.search(_text(result.get("content"))):
        # Not a cart going wrong: the place wasn't taking orders (availability.py now stops this earlier).
        scores["watch_cart_matches"] = (None, f"{tool}: not taking orders: {_text(result.get('content'))[:120]}")
    elif result.get("status") == "error":
        scores["watch_cart_matches"] = (0, f"{tool} failed: {_text(result.get('content'))[:150]}")
    elif after is None:
        scores["watch_cart_matches"] = (None, f"{tool}: couldn't read Swiggy's reply")
    else:
        scores["watch_cart_matches"] = (int(after == sent), f"{tool}: sent {sent}, Swiggy has {after}")

    worked = scores["watch_cart_matches"][0] == 1
    claims = bool(_CLAIMED.search(_NEGATED.sub("", reply.lower())))
    scores["watch_honest_reply"] = (int(worked or not claims), "claims success" if claims else "no success claim")
    return scores


def check_stuck(change, later_reads: list, next_change_at) -> tuple[int | None, str]:
    """Did the change stick? Compare the cart it left with the last read before the next change."""
    result = (change.outputs or {}).get("output") or {}
    if result.get("status") == "error":
        return None, ""
    left = {} if change.name == "flush_food_cart" else cart_items(result)
    readable = [(r, items) for r in later_reads
                if ((r.outputs or {}).get("output") or {}).get("status") != "error"
                and (items := cart_items((r.outputs or {}).get("output") or {})) is not None]
    if left is None or not readable:
        return None, "no later cart read yet"
    read, seen = readable[-1]
    if next_change_at is None and read.start_time - change.start_time < SETTLED:
        return None, "too soon to tell"  # wait for a later read: old carts come back after a few minutes
    if seen == left:
        return 1, f"{change.name}: still {left or 'empty'} when read at {read.start_time:%H:%M}"
    return 0, (f"{change.name} left {left or 'an empty cart'}, but at {read.start_time:%H:%M} the cart was "
               f"{seen or 'empty'}: Swiggy brought back an older cart (or it was changed in the Swiggy app)")


def watch_stuck(client: Client, since, changes: list, dry_run: bool) -> tuple[int, int]:
    """Score watch_change_stuck for every change where it's known by now. Returns (scored, failed)."""
    reads = _patiently(lambda: list(client.list_runs(project_name=PROJECT, run_type="tool", start_time=since,
                                                     filter='in(name, ["get_food_cart", "get_cart"])')))
    ordered = sorted(changes, key=lambda r: r.start_time)
    by_turn: dict = {}
    for i, change in enumerate(ordered):
        store = STORE[change.name]
        next_at = next((c.start_time for c in ordered[i + 1:] if STORE[c.name] == store), None)
        later = sorted((r for r in reads if STORE[r.name] == store and r.start_time > change.start_time
                        and (next_at is None or r.start_time < next_at)), key=lambda r: r.start_time)
        score, comment = check_stuck(change, later, next_at)
        if score is not None:
            by_turn.setdefault(change.trace_id, []).append((score, comment, change))
    scored = failed = 0
    for trace_id, results in by_turn.items():
        if _patiently(lambda: list(client.list_feedback(run_ids=[trace_id], feedback_key=["watch_change_stuck"]))):
            continue  # already judged
        score, comment = min(s for s, _, _ in results), " | ".join(c for _, c, _ in results)[:1000]
        when = min(c.start_time for _, _, c in results)
        print(f"{when:%m-%d %H:%M}  {', '.join(c.name for _, _, c in results)}  watch_change_stuck = {score}  ({comment})")
        scored, failed = scored + 1, failed + (not score)
        if not dry_run:
            _patiently(lambda: client.create_feedback(trace_id, "watch_change_stuck", score=score, comment=comment,
                                                      source_info={"watcher": "evals/watch_carts.py"}))
            if not score:
                _patiently(lambda: client.add_runs_to_annotation_queue(_queue(client).id, run_ids=[trace_id]))
    return scored, failed


def _queue(client: Client):
    for queue in client.list_annotation_queues(name=QUEUE):
        return queue
    return client.create_annotation_queue(name=QUEUE, description="Cart changes that failed a watcher check (evals/watch_carts.py)")


def watch_once(client: Client, hours: float, dry_run: bool = False, rescore: bool = False) -> None:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    names = ", ".join(f'"{n}"' for n in CHANGES)
    changes = _patiently(lambda: list(client.list_runs(project_name=PROJECT, run_type="tool",
                                                       filter=f"in(name, [{names}])", start_time=since)))
    by_turn: dict = {}
    for run in changes:
        by_turn.setdefault(run.trace_id, []).append(run)
    # Which restaurant each dish was added from, by Cravio's own cart changes over the past week.
    food_changes = _patiently(lambda: list(client.list_runs(project_name=PROJECT, run_type="tool", filter='eq(name, "update_food_cart")',
                                                            start_time=since - timedelta(days=7))))
    places = {str(i.get("menu_item_id")): str((r.inputs or {}).get("restaurantId"))
              for r in sorted(food_changes, key=lambda r: r.start_time) for i in (r.inputs or {}).get("cartItems") or []}

    new = failed = 0
    for trace_id, runs in by_turn.items():
        earlier = {f.key: f for f in _patiently(lambda: list(client.list_feedback(
            run_ids=[trace_id], feedback_key=[*CHECKS, "watch_change_stuck"])))}
        if "watch_approved" in earlier and not rescore:
            continue  # scored on an earlier pass
        root = _patiently(lambda: client.read_run(trace_id))
        combined: dict[str, list[tuple[int | None, str]]] = {}
        for run in sorted(runs, key=lambda r: r.start_time):
            for key, value in check_change(client, run, root, places).items():
                combined.setdefault(key, []).append(value)
        failures = []
        for key, values in combined.items():
            scored = [s for s, _ in values if s is not None]
            comment = " | ".join(c for _, c in values if c)[:1000]
            if scored:
                score = min(scored)  # one bad change in the turn fails the turn
                if dry_run:
                    print(f"      {key} = {score}  ({comment})")
                elif key in earlier:  # --rescore: replace the old score rather than add a second one
                    _patiently(lambda: client.update_feedback(earlier[key].id, score=score, comment=comment))
                else:
                    _patiently(lambda: client.create_feedback(trace_id, key, score=score, comment=comment,
                                                              source_info={"watcher": "evals/watch_carts.py"}))
                if not score:
                    failures.append(key)
        new += 1
        print(f"{root.start_time:%m-%d %H:%M}  {', '.join(r.name for r in runs)}  "
              f"{'FAIL ' + ', '.join(failures) if failures else 'all checks pass'}")
        if failures:
            failed += 1
        if dry_run:
            continue
        was_in_queue = any(f.score == 0 for f in earlier.values())
        if failures and not was_in_queue:
            _patiently(lambda: client.add_runs_to_annotation_queue(_queue(client).id, run_ids=[trace_id]))
        elif was_in_queue and not failures and getattr(earlier.get("watch_change_stuck"), "score", None) != 0:
            try:  # --rescore found it passes now: take it off the review queue
                _patiently(lambda: client.delete_run_from_annotation_queue(_queue(client).id, run_id=trace_id))
            except LangSmithNotFoundError:
                pass  # already reviewed and removed
    print(f"{len(changes)} cart changes in the last {hours:g}h; {new} turns newly scored, {failed} sent to {QUEUE}")
    stuck, undone = watch_stuck(client, since, changes, dry_run)
    print(f"did the change stick: {stuck} changes judged now, {undone} undone afterwards")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=24, help="how far back to look")
    parser.add_argument("--loop", type=float, metavar="MINUTES", help="keep watching, checking every MINUTES")
    parser.add_argument("--dry-run", action="store_true", help="print the scores; write nothing to LangSmith")
    parser.add_argument("--rescore", action="store_true", help="score turns again, replacing the watcher's old scores")
    args = parser.parse_args()
    client = Client()
    while True:
        watch_once(client, args.hours, args.dry_run, args.rescore)
        if not args.loop:
            return
        time.sleep(args.loop * 60)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
