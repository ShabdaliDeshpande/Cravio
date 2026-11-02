"""Names for Swiggy's item IDs, so an approval says "2 × Mixed Sprouts 175 g" instead of a spinId.

Cart changes carry only IDs (Food's menu_item_id, Instamart's spinId). Every search and cart reply Cravio gets
names them, so learn() remembers each one, and describe() turns a pending cart change into plain words.
"""

import re

from langchain_core.messages import ToolMessage

from cravio.progress import label_of

# item ID -> "Mixed Sprouts (Molakalu) 175 g · ₹40"
NAMES: dict[str, str] = {}
# The cart as Swiggy last showed it, per store: item ID -> quantity. Lets an approval say what a change removes.
CART: dict[str, dict[str, int]] = {}
CART_REPLIES = {"get_cart": "im", "update_cart": "im", "get_food_cart": "food", "update_food_cart": "food"}

MENU_LINE = re.compile(r"^\s*\d+\.\s+(.+?)\s+—\s+₹([\d.]+).*\(ID: (\d+)\)", re.M)  # search_menu: "1. Name — ₹125 | ... (ID: 1)"


def _structured(result: ToolMessage) -> dict:
    return (result.artifact or {}).get("structured_content") or {} if isinstance(result.artifact, dict) else {}


def _text(content) -> str:
    return content if isinstance(content, str) else "\n".join(b.get("text", "") for b in content or [] if isinstance(b, dict))


def _price(value) -> str:
    return f" · ₹{value:g}" if isinstance(value, (int, float)) else ""


def learn(tool: str, result) -> None:
    """Remember the names in a Swiggy reply."""
    if not isinstance(result, ToolMessage) or result.status == "error":
        return
    data = _structured(result)
    for product in data.get("products", []) + data.get("similarProducts", []):  # Instamart search
        for v in product.get("variations", []):
            if v.get("spinId"):
                size = f" {v['quantityDescription']}" if v.get("quantityDescription") else ""
                NAMES[v["spinId"]] = f"{v.get('displayName') or product.get('displayName')}{size}{_price((v.get('price') or {}).get('offerPrice'))}"
    for item in data.get("items", []):  # Instamart cart
        if item.get("spinId"):
            size = f" {item['itemVariant']}" if item.get("itemVariant") else ""
            NAMES[item["spinId"]] = f"{item.get('itemName')}{size}{_price(item.get('discountedFinalPrice'))}"
    food = data.get("data") if isinstance(data.get("data"), dict) else {}
    for item in food.get("items", []):  # Food cart
        if item.get("menu_item_id"):
            NAMES[str(item["menu_item_id"])] = f"{item.get('name')}{_price(item.get('final_price'))}"
    if tool == "search_menu":  # Food search: names are only in the text
        for name, price, item_id in MENU_LINE.findall(_text(result.content)):
            NAMES.setdefault(item_id, f"{name} · ₹{float(price):g}")
    if tool in CART_REPLIES:
        items = data.get("items") if tool in ("get_cart", "update_cart") else food.get("items")
        if isinstance(items, list):
            key = "spinId" if CART_REPLIES[tool] == "im" else "menu_item_id"
            CART[CART_REPLIES[tool]] = {str(i.get(key)): int(i.get("quantity", 0)) for i in items}
    elif tool == "flush_food_cart":
        CART["food"] = {}


def _lines(items: list[dict], id_key: str) -> str:
    return "\n".join(f"- {i.get('quantity', '?')} × {NAMES.get(str(i.get(id_key)), f'item {i.get(id_key)}')}" for i in items)


def _removed(store: str, items: list[dict], id_key: str) -> str:
    """The items in the cart now that the change leaves out (update_* replaces the whole cart)."""
    keeping = {str(i.get(id_key)) for i in items}
    gone = [f"- {q} × {NAMES.get(i, f'item {i}')}" for i, q in CART.get(store, {}).items() if i not in keeping]
    return "\n\n⚠️ **Removes** from your cart:\n" + "\n".join(gone) if gone else ""


def cravio_note(messages: list) -> str:
    """What Cravio wrote with the change it's asking to make ("Adding the sundae replaces your noodles"), if any."""
    asking = next((m for m in reversed(messages) if getattr(m, "type", "") == "ai" and getattr(m, "tool_calls", None)), None)
    return " ".join(asking.text.split()) if asking is not None and asking.text else ""


def describe(tool: str, args: dict) -> str:
    """A pending change in plain words, for the approval question."""
    if tool == "update_cart":
        items = args.get("items", [])
        return "Your **Instamart** cart will be:\n" + _lines(items, "spinId") + _removed("im", items, "spinId")
    if tool == "update_food_cart":
        items, place = args.get("cartItems", []), f" from {args['restaurantName']}" if args.get("restaurantName") else ""
        return (f"Your **Food** cart will be{place}:\n" + _lines(items, "menu_item_id")
                + _removed("food", items, "menu_item_id"))
    if tool == "flush_food_cart":
        return "Empty your **Food** cart" + _removed("food", [], "menu_item_id")
    if tool == "request_checkout":
        return "Get your cart ready for checkout (Cravio never places the order)"
    if tool == "create_address":
        return f"Save a new {args.get('addressCategory', '').lower() or ''} address on your Swiggy account".replace("  ", " ")
    return label_of(tool, args)
