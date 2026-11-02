"""Cravio's own tools, for the parts of the journey Swiggy MCP doesn't cover."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

TICKETS_FILE = Path(os.getenv("CRAVIO_TICKETS_FILE", ".cravio/tickets.json"))


@tool
def request_checkout(address_id: str) -> str:
    """Final step: ask the user to approve their cart for checkout.

    Call this once the user says the cart is right. It pauses until the user approves.
    Cravio is in demo mode, so approval does NOT place or pay for an order.

    Args:
        address_id: The addressId the cart is being delivered to.
    """
    return (
        "The user approved the cart. Demo mode: Cravio does not place orders. "
        "Tell the user their cart is saved in the Swiggy app, ready for them to place there."
    )


@tool
def create_escalation_ticket(
    order_id: str,
    category: Literal["missing_item", "wrong_item", "quality", "late_delivery", "billing", "safety", "other"],
    severity: Literal["low", "medium", "high", "urgent"],
    summary: str,
) -> str:
    """Open a ticket for Cravio's support team when an order problem needs a human.

    Use for anything involving money, food safety or an allergic reaction, or when the user is
    upset or asks for a person. Keep the summary factual: what was ordered, what went wrong, and
    what the user wants.

    Before calling this, find the order with get_food_orders. Ask the user which order only if you can't
    tell from what they said.

    Args:
        order_id: The Swiggy order ID the problem is about.
        category: The kind of problem. Feeling sick, an allergic reaction or anything unsafe in the food is "safety".
        severity: "urgent" for safety or allergic reactions; "high" for refunds or missing meals.
        summary: Two or three factual sentences.
    """
    tickets = json.loads(TICKETS_FILE.read_text(encoding="utf-8")) if TICKETS_FILE.exists() else []
    ticket = {
        "id": f"CRV-{len(tickets) + 1:04d}",
        "order_id": order_id,
        "category": category,
        "severity": severity,
        "summary": summary,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tickets.append(ticket)
    TICKETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TICKETS_FILE.write_text(json.dumps(tickets, indent=2), encoding="utf-8")
    # The model writes its reply from this, so say plainly what happens next: nothing reaches the user.
    return json.dumps({
        **ticket,
        "what_happens_next": "Saved for Cravio's support team to review. Nobody will contact the user. "
        "For a quick answer, they can contact Swiggy support in the app.",
    })
