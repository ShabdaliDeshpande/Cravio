"""Checks just before a cart change runs, in code, so the rules hold whatever the model does.

0. Only the exact change the user approved, once (approvals.py). After approval, a subagent is run again from
   its start, and its model may then propose something slightly different from what the user saw.
1. Only while the restaurant or store is taking orders. Swiggy can put items in a cart even when the
   restaurant is closed, and Instamart's store can stop taking orders (at night, say) while search still works.
2. Never delete what's in the cart by accident (Food and Instamart). update_food_cart and update_cart replace
   the whole cart, and the user may have changed it in the Swiggy app since Cravio last looked, so the live cart
   is read right before.
3. One cart change at a time. Two running together would each read the cart before the other changed it.
The prompts ask the agents to check first, so the user hears it early; this is the guarantee. Each check is
one quick Swiggy call and no model call.
"""

import asyncio
import re

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from cravio import approvals, catalog
from cravio.swiggy.client import needs_approval

STORE_CLOSED = re.compile(r"store is (?:unavailable|closed)|not (?:accepting|taking) orders", re.I)
WANTS_REMOVAL = re.compile(r"\b(remove|delete|drop|replace|instead|only|clear|empty)\b", re.I)
# search_menu lists "Gulab Jamun — ₹125 | Veg | 4.3★ | Haldiram's (restaurantId: 881451) (ID: ...)"
MENU_RESTAURANT = re.compile(r"\|\s*([^|\n]+?)\s*\(restaurantId:\s*(\d+)\)")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content or [] if isinstance(b, dict))


def _food_items(result: ToolMessage) -> dict[str, int] | None:
    """Menu item ID -> quantity in a Food cart reply, from Swiggy's data; None if it has none."""
    structured = (result.artifact or {}).get("structured_content") or {} if isinstance(result.artifact, dict) else {}
    cart = structured.get("data") if isinstance(structured.get("data"), dict) else None
    if cart is None:
        return None
    return {str(i.get("menu_item_id")): int(i.get("quantity", 0)) for i in cart.get("items") or []}


def _deletes(dropped: list[str], messages: list, read_tool: str) -> str | None:
    """Stop a change that would delete cart items, unless the user's request asked to remove something."""
    asked = next((m.text for m in reversed(messages) if getattr(m, "type", "") == "human"), "")
    if not dropped or WANTS_REMOVAL.search(asked):
        return None
    return (f"the cart right now also has {', '.join(dropped)}, and this change would delete it. Read the cart "
            f"with {read_tool} and send every item to keep, or ask the user.")


class TakingOrders(AgentMiddleware):
    """Checks with Swiggy just before update_food_cart (is the restaurant OPEN?) and Instamart's update_cart
    (is the store taking orders?), and stops the change if not."""

    def __init__(self, tools) -> None:
        self._swiggy = {((t.metadata or {}).get("swiggy_server"), (t.metadata or {}).get("swiggy_name")): t for t in tools}
        self.names: dict[str, str] = {}  # restaurant ID -> name, learned from Swiggy's search results
        self.item_place: dict[str, str] = {}  # menu item ID -> restaurant ID, from searches and cart changes
        self.confirmed: dict[str, int] | None = None  # the Food cart Swiggy confirmed after Cravio's last change
        self._cart_lock = asyncio.Lock()

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] in ("update_food_cart", "update_cart"):
            # Both replace the whole cart. Two at once (e.g. two subagent tasks) would each check the cart before
            # the other changed it, and the second would delete the first one's items. So one at a time.
            async with self._cart_lock:
                return await self._checked(request, handler)
        return await self._checked(request, handler)

    async def _checked(self, request, handler):
        call = request.tool_call
        metadata = (request.tool.metadata or {}) if request.tool else {}
        server = metadata.get("swiggy_server")
        problem = None
        if (needs_approval(metadata.get("swiggy_name", "")) or call["name"] == "request_checkout") \
                and not approvals.take(call["name"], call["args"]):
            problem = ("this isn't exactly the change the user approved (it may have been worked out again after "
                       "they answered). Make the change again, and the user will be asked to approve this one.")
        elif call["name"] == "update_food_cart":
            problem = await self._restaurant_problem(call["args"]) or await self._would_delete(
                call["args"], (request.state or {}).get("messages", []))
        elif call["name"] == "update_cart" and server == "im":
            problem = await self._store_problem(call["args"], (request.state or {}).get("messages", []))
        if problem:
            return ToolMessage(
                content=f"Not done: {problem} Nothing was changed. Tell the user, and offer another option.",
                tool_call_id=call["id"], name=call["name"], status="error",
            )
        result = await handler(request)
        self._learn(call, result)
        catalog.learn(call["name"], result)  # item names, so approvals can say "2 × Mixed Sprouts"
        catalog.shorten(call["name"], server, call["args"], result)  # Instamart search: ~10k tokens down to ~1k
        self._notice_restored_cart(call, result)
        return result

    def _notice_restored_cart(self, call: dict, result) -> None:
        """Swiggy sometimes brings back an older cart minutes after confirming a change (mostly after a
        restaurant switch or clearing the cart). When a later read shows that, say so in the result, so Cravio
        tells the user instead of trusting its own earlier "done"."""
        if not isinstance(result, ToolMessage) or result.status == "error":
            return
        if call["name"] == "flush_food_cart":
            self.confirmed = {}
        elif call["name"] == "update_food_cart":
            self.confirmed = _food_items(result)
        elif call["name"] == "get_food_cart" and self.confirmed is not None:
            now = _food_items(result)
            if now is not None and now != self.confirmed:
                note = ("\n\nNote from Cravio: this isn't the cart Swiggy confirmed after Cravio's last change. Swiggy "
                        "has brought back an older cart, which it sometimes does a few minutes after a change. Tell the "
                        "user plainly, and suggest fixing the cart in the Swiggy app: changing it again here may be "
                        "undone the same way.")
                result.content = result.content + note if isinstance(result.content, str) else [
                    *result.content, {"type": "text", "text": note}]
            self.confirmed = now

    async def _ask(self, server: str, tool: str, args: dict) -> ToolMessage | None:
        if (server, tool) not in self._swiggy:
            return None  # not connected (e.g. Instamart is off): can't check
        result = await self._swiggy[(server, tool)].ainvoke({"type": "tool_call", "id": f"check-{tool}", "name": tool, "args": args})
        return result if isinstance(result, ToolMessage) else None

    async def _restaurant_problem(self, args: dict) -> str | None:
        restaurant_id = str(args.get("restaurantId", ""))
        name = args.get("restaurantName") or self.names.get(restaurant_id)
        if not name:
            return None  # nothing to search by; Swiggy's own reply still warns if it's not taking orders
        found = await self._ask("food", "search_restaurants", {"query": name, "addressId": args.get("addressId")})
        restaurants = (((found.artifact or {}).get("structured_content") or {}).get("restaurants", []) if found else [])
        status = next((r.get("availabilityStatus") for r in restaurants if str(r.get("id")) == restaurant_id), None)
        if status and status != "OPEN":
            return f"{name} is {status.lower()} and not taking orders right now."
        return None

    async def _would_delete(self, args: dict, messages: list) -> str | None:
        """update_food_cart replaces the whole cart. Read the live cart now, because the user may have changed it
        in the Swiggy app since Cravio last looked, and stop the change if it would delete items they didn't ask
        to remove. Moving to another restaurant replaces the cart on purpose (Cravio warns first), so that's allowed."""
        live = await self._ask("food", "get_food_cart", {"addressId": args.get("addressId")})
        structured = (live.artifact or {}).get("structured_content") or {} if live and live.status != "error" else {}
        cart = structured.get("data") if isinstance(structured.get("data"), dict) else {}
        items = cart.get("items") or []
        # Swiggy's cart data doesn't say which restaurant it's from (its "name" echoes whatever Cravio passes),
        # so go by items: menu item IDs belong to one restaurant.
        in_cart = {str(i.get("menu_item_id")) for i in items}
        keeping = {str(i.get("menu_item_id")) for i in args.get("cartItems", [])}
        target = str(args.get("restaurantId", ""))
        same_place = bool(in_cart & keeping) or any(self.item_place.get(i) == target for i in in_cart)
        if not items or not same_place:
            return None  # empty cart, or another restaurant: replaced on purpose, after Cravio's warning
        dropped = [f"{i.get('name')} x{i.get('quantity')}" for i in items if str(i.get("menu_item_id")) not in keeping]
        return _deletes(dropped, messages, "get_food_cart")

    async def _store_problem(self, args: dict, messages: list) -> str | None:
        """Is Instamart taking orders? And, as with the Food cart, would this change delete items by accident?"""
        cart = await self._ask("im", "get_cart", {})
        if cart is None:
            return None
        if cart.status == "error":
            if STORE_CLOSED.search(_text(cart.content)):
                return "Instamart isn't taking orders at this address right now (Swiggy says the store is unavailable)."
            return None
        structured = (cart.artifact or {}).get("structured_content") or {} if isinstance(cart.artifact, dict) else {}
        keeping = {str(i.get("spinId")) for i in args.get("items", [])}
        dropped = [f"{i.get('itemName')} x{i.get('quantity')}" for i in structured.get("items") or []
                   if str(i.get("spinId")) not in keeping]
        return _deletes(dropped, messages, "get_cart")

    def _learn(self, call: dict, result) -> None:
        """Remember restaurant names, and which restaurant each menu item belongs to."""
        if not isinstance(result, ToolMessage) or result.status == "error":
            return
        structured = (result.artifact or {}).get("structured_content") or {} if isinstance(result.artifact, dict) else {}
        for r in structured.get("restaurants", []):
            if r.get("id") and r.get("name"):
                self.names[str(r["id"])] = r["name"]
        if call["name"] == "search_menu":
            searched_in = call["args"].get("restaurantIdOfAddedItem")  # a search inside one restaurant
            for line in _text(result.content).splitlines():
                item = re.search(r"\(ID: (\d+)\)", line)
                place = MENU_RESTAURANT.search(line)
                if place:
                    self.names[place.group(2)] = place.group(1)
                if item and (place or searched_in):
                    self.item_place[item.group(1)] = place.group(2) if place else str(searched_in)
        if call["name"] == "update_food_cart":  # everything in the cart now is from this restaurant
            cart = structured.get("data") if isinstance(structured.get("data"), dict) else {}
            for i in cart.get("items", []):
                self.item_place[str(i.get("menu_item_id"))] = str(call["args"].get("restaurantId", ""))
