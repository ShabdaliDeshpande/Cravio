"""Only the exact change the user approved may run.

When a subagent pauses for approval, the run picks up again after the user's answer by running that subagent
from its start, and its model can then propose a slightly different change than the one the user saw. So the
CLI and web chat record what was approved (remember), and the safety middleware lets a change run only if it
matches an approval (take), once.
"""

import json
import time

APPROVAL_LIFETIME = 15 * 60  # seconds an approval stays usable
_approved: list[tuple[float, tuple]] = []


def _key(tool: str, args: dict) -> tuple:
    """What makes two changes the same for the user: items and quantities (and sizes/add-ons), not argument order."""
    def item(i: dict, id_key: str) -> tuple:
        return (str(i.get(id_key)), int(i.get("quantity", 0)),
                json.dumps(i.get("variants"), sort_keys=True), json.dumps(i.get("addons"), sort_keys=True))

    if tool == "update_cart":
        return tool, frozenset(item(i, "spinId") for i in args.get("items", []))
    if tool == "update_food_cart":
        return tool, str(args.get("restaurantId")), frozenset(item(i, "menu_item_id") for i in args.get("cartItems", []))
    if tool in ("flush_food_cart", "request_checkout"):
        return (tool,)
    return tool, json.dumps(args, sort_keys=True, default=str)


def remember(tool: str, args: dict) -> None:
    """The user approved this change."""
    now = time.monotonic()
    _approved[:] = [(t, k) for t, k in _approved if now - t < APPROVAL_LIFETIME]
    _approved.append((now, _key(tool, args)))


def take(tool: str, args: dict) -> bool:
    """True, once, if the user approved exactly this change. A different version of the same kind of change
    means the approval was for something that will never run, so it's dropped: the user is asked again."""
    key, now = _key(tool, args), time.monotonic()
    for entry in _approved:
        if entry[1] == key and now - entry[0] < APPROVAL_LIFETIME:
            _approved.remove(entry)
            return True
    forget(tool)
    return False


def forget(tool: str) -> None:
    """Drop approvals for this kind of change (the user declined, or a different version was proposed)."""
    _approved[:] = [(t, k) for t, k in _approved if k[0] != tool]
