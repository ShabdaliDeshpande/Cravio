"""Change a cart only while the restaurant or store is taking orders.

Swiggy can put items in a cart even when the restaurant is closed, and Instamart's store can stop taking
orders (at night, say) while its search still works. The prompts ask the agents to check first, so the user
hears it early. This middleware is the guarantee: just before a cart change runs, it asks Swiggy itself,
with no model call, and stops the change if the place isn't taking orders right now.
"""

import re

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

STORE_CLOSED = re.compile(r"store is (?:unavailable|closed)|not (?:accepting|taking) orders", re.I)
# search_menu lists "Gulab Jamun — ₹125 | Veg | 4.3★ | Haldiram's (restaurantId: 881451) (ID: ...)"
MENU_RESTAURANT = re.compile(r"\|\s*([^|\n]+?)\s*\(restaurantId:\s*(\d+)\)")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content or [] if isinstance(b, dict))


class TakingOrders(AgentMiddleware):
    """Checks with Swiggy just before update_food_cart (is the restaurant OPEN?) and Instamart's update_cart
    (is the store taking orders?), and stops the change if not."""

    def __init__(self, tools) -> None:
        self._swiggy = {((t.metadata or {}).get("swiggy_server"), (t.metadata or {}).get("swiggy_name")): t for t in tools}
        self.names: dict[str, str] = {}  # restaurant ID -> name, learned from Swiggy's search results

    async def awrap_tool_call(self, request, handler):
        call = request.tool_call
        server = ((request.tool.metadata or {}).get("swiggy_server") if request.tool else None)
        problem = None
        if call["name"] == "update_food_cart":
            problem = await self._restaurant_problem(call["args"])
        elif call["name"] == "update_cart" and server == "im":
            problem = await self._store_problem()
        if problem:
            return ToolMessage(
                content=f"Not done: {problem} Nothing was changed. Tell the user, and offer another option.",
                tool_call_id=call["id"], name=call["name"], status="error",
            )
        result = await handler(request)
        self._learn_names(result)
        return result

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

    async def _store_problem(self) -> str | None:
        cart = await self._ask("im", "get_cart", {})
        if cart and cart.status == "error" and STORE_CLOSED.search(_text(cart.content)):
            return "Instamart isn't taking orders at this address right now (Swiggy says the store is unavailable)."
        return None

    def _learn_names(self, result) -> None:
        if not isinstance(result, ToolMessage):
            return
        for name, restaurant_id in MENU_RESTAURANT.findall(_text(result.content)):
            self.names[restaurant_id] = name
        for r in ((result.artifact or {}).get("structured_content") or {}).get("restaurants", []) if isinstance(result.artifact, dict) else []:
            if r.get("id") and r.get("name"):
                self.names[str(r["id"])] = r["name"]
