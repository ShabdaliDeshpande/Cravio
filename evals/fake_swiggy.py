"""A fake Swiggy Food server for evals: Swiggy's real tool definitions, a cart kept in memory.

The model gets the same tools it gets from Swiggy (names, descriptions and arguments, from
docs/swiggy_tools/food.json) and the replies copy Swiggy's text format, quirks included.
Nothing touches a real account. A scenario sets the starting state and can change it between
turns, the way the Swiggy app or Swiggy itself does.
"""

import json
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool, ToolException

from cravio.swiggy.client import is_blocked

DEFINITIONS = Path(__file__).resolve().parent.parent / "docs" / "swiggy_tools" / "food.json"

# Made-up people and places, shaped like Swiggy's data.
HOME = "home00000000000000aa__FakeAddressHome00001"
WORK = "work00000000000000bb__FakeAddressWork00002"
ADDRESSES = [
    ("Home", "Asha Rao: 12, Lake View Apartments, Kondapur, Hyderabad, Telangana 500084, India", HOME),
    ("Work", "Asha Rao: 4th Floor, Cyber Towers, Madhapur, Hyderabad, Telangana 500081, India", WORK),
]
# Real Swiggy restaurant and menu IDs, from the bugs we chased. Menu: item ID -> (name, price, rating).
NIDHI, POLAR_BEAR = "351936", "614305"
RESTAURANTS = {
    NIDHI: {
        "name": "NIDHI Ki RASOI Se... North Indian (pure Veg)",
        "cuisines": ["North Indian", "Chinese", "Desserts"],
        "avgRating": 4.2, "areaName": "Kondapur", "distanceKm": 3.8, "deliveryTimeMinutes": 45,
        "menu": {
            "65684363": ("Chilli Garlic Noodles", 340, 4.1),
            "65684362": ("Veg Hakka Noodles (zero Spicy)", 340, 3.6),
            "68210756": ("Schezwan Noodle", 340, 3.3),
            "70503733": ("Gulab Jamun", 135, 4.5),
        },
    },
    POLAR_BEAR: {
        "name": "Polar Bear",
        "cuisines": ["Ice Cream", "Desserts"],
        "avgRating": 4.4, "areaName": "Serilingampally", "distanceKm": 2.5, "deliveryTimeMinutes": 29,
        "menu": {
            "101313396": ("Gudbud Sundae", 198.45, 4.6),
            "206776019": ("Bada Gudbud Sundae Tub (480 Ml)", 380.95, 4.3),
        },
    },
}

WIDGET_NOTE = (
    "⚠️ A rich UI widget may be shown to the user with this data. Avoid restating everything the widget "
    "already displays — instead, provide a brief recommendation or ask what they'd like to do next."
)


def _swiggy_error(message: str) -> ToolException:
    report = "ERR-FAKE0000-EVAL"
    return ToolException(
        f"{message} Report ID: {report} Something went wrong. If this persists, quote report ID: {report} "
        "when contacting support at mcp-support@swiggy.in."
    )


def _require_location(addressId, latitude) -> None:
    if not addressId and latitude is None:  # optional in the schemas, but Swiggy's server insists
        raise _swiggy_error("Either addressId from get_addresses or latitude and longitude are required for this tool.")


def _matches(query: str, *texts: str) -> int:
    """How many words of the query appear in the texts: a rough stand-in for Swiggy's search."""
    haystack = " ".join(texts).lower()
    return sum(word in haystack for word in query.lower().split())


class FakeSwiggy:
    def __init__(self) -> None:
        self.restaurant_id: str | None = None
        self.cart: dict[str, int] = {}  # menu item ID -> quantity, in the order they were added
        self.calls: list[tuple[str, dict]] = []  # every tool call, for the evaluators

    def set_cart(self, restaurant_id: str | None, items: dict[str, int]) -> None:
        """Change the cart from outside Cravio, as the Swiggy app (or Swiggy itself) does."""
        self.restaurant_id, self.cart = restaurant_id, dict(items)

    def tools(self) -> list[BaseTool]:
        """Every tool Cravio would get from real Swiggy, built from Swiggy's own definitions."""
        handlers = {
            "get_addresses": self.get_addresses,
            "search_restaurants": self.search_restaurants,
            "search_menu": self.search_menu,
            "get_food_cart": self.get_food_cart,
            "update_food_cart": self.update_food_cart,
            "flush_food_cart": self.flush_food_cart,
        }
        tools = []
        for spec in json.loads(DEFINITIONS.read_text(encoding="utf-8")):
            annotations = spec.get("annotations") or {}
            if is_blocked(spec["name"], bool(annotations.get("readOnlyHint"))):
                continue  # the same tools cravio.swiggy.client withholds
            tools.append(StructuredTool(
                name=spec["name"],
                description=spec["description"],
                args_schema=spec["inputSchema"],
                coroutine=self._tool(spec["name"], handlers.get(spec["name"])),
                response_format="content_and_artifact",
                handle_tool_error=True,  # a ToolException becomes an error reply, as with real Swiggy
                metadata={**annotations, "swiggy_name": spec["name"]},
            ))
        return tools

    def _tool(self, name: str, handler):
        async def call(**args):
            self.calls.append((name, args))
            if handler is None:
                raise ToolException(f"{name} isn't simulated in this eval.")
            return [{"type": "text", "text": handler(**args)}], None

        return call

    def get_addresses(self, **_) -> str:
        lines = [f"{i}. [{label}] {line} (ID: {id_})" for i, (label, line, id_) in enumerate(ADDRESSES, 1)]
        ids = ", ".join(id_ for *_, id_ in ADDRESSES)
        return (
            f"Found {len(ADDRESSES)} saved addresses (page 1 of 1, showing {len(ADDRESSES)}):\n" + "\n".join(lines)
            + f"\n\nThe saved addresses are ranked for display. Suggested/preselected address ID: {HOME}. Ask the "
            f"user to confirm the suggestion or choose from address IDs: {ids}. Do not call downstream tools until "
            f"they choose.\n\n{WIDGET_NOTE}"
        )

    def search_restaurants(self, query: str, addressId: str | None = None, latitude: float | None = None, **_) -> str:
        _require_location(addressId, latitude)
        # Swiggy always returns something; the closest matches come first.
        ranked = sorted(RESTAURANTS.items(), key=lambda r: -_matches(query, r[1]["name"], *r[1]["cuisines"]))
        restaurants = [{
            "id": rid, "name": r["name"], "cuisines": r["cuisines"], "avgRating": r["avgRating"],
            "totalRatings": "2K+", "costForTwo": "₹400 for two", "areaName": r["areaName"],
            "distanceKm": r["distanceKm"], "deliveryTimeMinutes": r["deliveryTimeMinutes"],
            "deliveryTimeRange": f"{r['deliveryTimeMinutes'] - 5}-{r['deliveryTimeMinutes'] + 5} MINS",
            "imageUrl": "https://media-assets.swiggy.com/fake.png", "availabilityStatus": "OPEN",
        } for rid, r in ranked]
        result = {"restaurants": restaurants, "dishes": [], "total": len(restaurants), "totalRestaurants": len(restaurants),
                  "totalDishes": 0, "query": query, "hasMore": False, "nextOffset": len(restaurants)}
        return json.dumps(result, ensure_ascii=False) + f"\n\n{WIDGET_NOTE}"

    def search_menu(self, query: str, addressId: str | None = None, latitude: float | None = None,
                    restaurantIdOfAddedItem: str | None = None, **_) -> str:
        _require_location(addressId, latitude)
        scope = {restaurantIdOfAddedItem: RESTAURANTS[restaurantIdOfAddedItem]} if restaurantIdOfAddedItem in RESTAURANTS else RESTAURANTS
        found = [(_matches(query, name), rid, item_id, name, price, rating)
                 for rid, r in scope.items() for item_id, (name, price, rating) in r["menu"].items()]
        found = sorted((f for f in found if f[0]), key=lambda f: -f[0])
        if not found:
            return f'No menu items found for "{query}".'  # wording guessed
        lines = []
        for i, (_, rid, item_id, name, price, rating) in enumerate(found, 1):
            if restaurantIdOfAddedItem:  # within one restaurant Swiggy shows the rating count, not the restaurant
                lines.append(f"{i}. {name} — ₹{price:g} | Veg | {rating}★ (120) (ID: {item_id})")
            else:
                lines.append(f"{i}. {name} — ₹{price:g} | Veg | {rating}★ | {RESTAURANTS[rid]['name']} (restaurantId: {rid}) (ID: {item_id})")
        return f'Found {len(found)} menu items for "{query}":\n' + "\n".join(lines) + f"\n\n{WIDGET_NOTE}"

    def get_food_cart(self, restaurantName: str | None = None, **_) -> str:
        if not self.cart:
            return "Your cart is empty."  # a guess: we haven't seen Swiggy's reply for an empty cart
        # Swiggy's quirk: it prints back any restaurantName it's given, whatever is in the cart.
        header = f"Restaurant: {restaurantName}\n" if restaurantName else ""
        return header + self._cart_text() + (
            "\n\nCart widget is displayed — do not repeat the cart items, prices, or totals in text.\n"
            "NOTE: The cart widget has quantity stepper controls. The user can adjust quantities and click "
            '"Confirm and Review Cart" — the widget will send you the changes to process via update_food_cart.'
            f"\n\n{WIDGET_NOTE}"
        )

    def update_food_cart(self, restaurantId: str, cartItems: list[dict], addressId: str | None = None,
                         latitude: float | None = None, restaurantName: str | None = None, **_) -> str:
        _require_location(addressId, latitude)
        if restaurantId not in RESTAURANTS:
            raise _swiggy_error(f"Restaurant {restaurantId} was not found.")  # wording guessed
        menu = RESTAURANTS[restaurantId]["menu"]
        cart: dict[str, int] = {}
        for item in cartItems:
            item_id, quantity = str(item["menu_item_id"]), int(item["quantity"])
            if item_id not in menu:
                raise _swiggy_error(f"Item {item_id} is not on this restaurant's menu.")  # wording guessed
            if quantity > 0:
                cart[item_id] = cart.get(item_id, 0) + quantity
        # Swiggy's quirks: the cart becomes exactly cartItems, so items left out are deleted, and a
        # different restaurant replaces the whole cart without a word.
        self.restaurant_id, self.cart = restaurantId, cart
        header = f"Restaurant: {restaurantName}\n" if restaurantName else ""
        return "Cart updated.\n" + header + self._cart_text() + (
            "\n\nNo widget for this response. Check the item(s) and variant(s) listed above actually match what was "
            "requested (the backend may have defaulted a variant) before confirming anything to the user, then call "
            "get_food_cart to show the updated cart."
        )

    def flush_food_cart(self, **_) -> str:
        self.restaurant_id, self.cart = None, {}
        return "Flushed Food cart successfully"

    def _cart_text(self) -> str:
        menu = RESTAURANTS[self.restaurant_id]["menu"]
        lines, total = [], 0.0
        for item_id, quantity in self.cart.items():
            name, price, _ = menu[item_id]
            total += price * quantity
            if quantity == 1:
                lines.append(f"  - {name} — ₹{price:g} (ID: {item_id})")
            else:  # Swiggy's own wording: the "each" price is really the subtotal
                lines.append(f"  - {name} x{quantity} — ₹{price * quantity:g} each (subtotal: ₹{price * quantity:g}) (ID: {item_id})")
        taxes = round(total * 0.1 + 27, 2)
        return (
            f"Items ({len(self.cart)}):\n" + "\n".join(lines)
            + f"\n\nItem total: ₹{total:g}\nDelivery: FREE\nTaxes & charges: ₹{taxes:g}\nTO PAY: ₹{round(total + taxes)}"
        )
