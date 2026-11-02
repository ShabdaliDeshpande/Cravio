"""Mask personal data in Cravio's traces before they leave this machine for LangSmith.

LangSmith calls mask() on every string in a trace (inputs, outputs, errors, metadata) inside
this process, before uploading. Anything no rule catches is uploaded as is, so look over your
first traces in the dashboard.
"""

import hashlib
import os
import re
from collections.abc import Sequence

from langsmith.anonymizer import DEFAULT_SECRET_RULES

# get_addresses lists "1. [Home] Full Name: flat, building, street, city, PIN, India (ID: ...)".
# Keep the [label] so a trace still shows which address was picked; hide the name and address. Only at the
# start of a numbered line, so menu items like "5. Gulab Jamun [2 Pieces] — ₹99 (ID: ...)" stay readable.
_SAVED_ADDRESS = re.compile(r"^(\s*\d+\.\s+\[[^\]\n]*\]) [^\n]*?(?= \(ID: )", re.M)
# Instamart's cart replies carry the delivery address as data (name, flat, area, city, phone, coordinates),
# and repeat it as JSON inside the reply text. Hide those fields whole.
# Food's cart data has one too: restaurant.deliverySubtitle = "10-15 mins to <name>: <full address>".
_ADDRESS_FIELDS = {"selectedAddressDetails", "selectedAddress", "deliveryAddress", "addressDetails", "deliverySubtitle"}
_ADDRESS_JSON = re.compile(
    r'"(selectedAddressDetails|selectedAddress|deliveryAddress|addressDetails|deliverySubtitle)"\s*:\s*(\{[^{}]*\}|"[^"]*")')
# What the model writes from it: "**Delivery Address:** 12, Lake View, Kondapur, Hyderabad ..."
_DELIVERY_LINE = re.compile(r"(delivery address\W*)[^\n]+", re.I)
# An address ID points at one address on your account.
_ADDRESS_ID = re.compile(r"\b[a-z0-9]{20}__[A-Za-z0-9_-]{20,}\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Indian mobile numbers, with or without +91 or a leading 0, written whole or as "98765 43210".
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
# A PIN code follows a place name ("Telangana 500046"); restaurant IDs are six digits too,
# but follow a quote or "ID: ", so they are kept.
_PIN_CODE = re.compile(r"(?:(?<=[A-Za-z] )|(?<=[A-Za-z], ))[1-9]\d{5}(?!\d)")
# Latitude and longitude have four or more decimals; prices and ratings have at most two.
_COORDINATE = re.compile(r"(?<![\d.])-?\d{1,3}\.\d{4,}(?![\d.])")
# Anything else you never want uploaded, such as your name or building: CRAVIO_PII_TERMS=Name,Building
_TERMS = [t.strip() for t in os.getenv("CRAVIO_PII_TERMS", "").split(",") if t.strip()]
_PRIVATE_TERMS = re.compile("|".join(map(re.escape, _TERMS)), re.IGNORECASE) if _TERMS else None


def _address_ref(match: re.Match) -> str:
    """The same address always gets the same short reference, so traces can still be compared."""
    return "addr_" + hashlib.sha256(match.group().encode()).hexdigest()[:8]


def mask(text: str, path: Sequence[str | int] = ()) -> str:
    """Hide personal data in one string. LangSmith passes the string's path (its field names) as well."""
    if any(part in _ADDRESS_FIELDS for part in path):
        return "<address>"
    for rule in DEFAULT_SECRET_RULES:  # API keys and tokens, from LangSmith's own list
        text = rule["pattern"].sub(rule["replace"], text)
    text = _ADDRESS_JSON.sub(r'"\1": "<address>"', text)
    text = _DELIVERY_LINE.sub(r"\1<address>", text)
    text = _SAVED_ADDRESS.sub(r"\1 <name and address>", text)
    text = _ADDRESS_ID.sub(_address_ref, text)
    text = _EMAIL.sub("<email>", text)
    text = _PHONE.sub("<phone>", text)
    text = _PIN_CODE.sub("<pin>", text)
    text = _COORDINATE.sub("<coordinate>", text)
    if _PRIVATE_TERMS:
        text = _PRIVATE_TERMS.sub("<private>", text)
    return text
