"""Cart scenarios for Cravio's evals, one per bug we found against real Swiggy.

Each scenario is plain data, so it can be uploaded as a LangSmith dataset:
  cart               the cart before the first message: [restaurant ID, {menu item ID: quantity}]
  turns              what the user says, one message per turn
  follow_up          sent once if Cravio stops to ask a question instead of changing the cart
  approve            the user's answer when Cravio pauses before changing the cart
  app_changes_cart   [before turn N, restaurant ID, items]: the cart changes behind Cravio's back
  checks             what must be true afterwards (see check_scenario in run_evals.py)
"""

from fake_swiggy import NIDHI, POLAR_BEAR

NOODLES, HAKKA, JAMUN, SUNDAE = "65684363", "65684362", "70503733", "101313396"

SCENARIOS = [
    {
        "name": "add-keeps-items",
        "about": "update_food_cart replaces the whole cart, so adding a dish must resend what's already there.",
        "cart": [NIDHI, {NOODLES: 5, JAMUN: 1}],
        "turns": ["Use my Home address. Add one Veg Hakka Noodles (zero spicy) from NIDHI Ki RASOI to my cart."],
        "follow_up": "Yes, go ahead.",
        "approve": True,
        "checks": {
            "cart": [NIDHI, {NOODLES: 5, JAMUN: 1, HAKKA: 1}],
            "reads_cart_before_change": True,
        },
    },
    {
        "name": "change-quantity",
        "about": "Changing one quantity must keep the other items.",
        "cart": [NIDHI, {NOODLES: 2, JAMUN: 1}],
        "turns": ["Use my Home address. Make the Chilli Garlic Noodles in my cart 4 instead of 2."],
        "follow_up": "Yes, go ahead.",
        "approve": True,
        "checks": {
            "cart": [NIDHI, {NOODLES: 4, JAMUN: 1}],
            "reads_cart_before_change": True,
        },
    },
    {
        "name": "rereads-after-app-change",
        "about": "The cart changed in the Swiggy app between turns; Cravio must read it again, not trust the chat.",
        "cart": [NIDHI, {NOODLES: 5, JAMUN: 1}],
        "turns": ["Use my Home address. What's in my cart?", "What's in my cart now?"],
        "app_changes_cart": [[1, POLAR_BEAR, {SUNDAE: 1}]],
        "checks": {
            "reads_cart_in_last_turn": True,
            "no_restaurant_name": True,
            "reply_mentions": ["sundae"],
            "reply_avoids": ["noodles", "jamun"],
        },
    },
    {
        "name": "switch-restaurant-warns",
        "about": "A dish from another restaurant wipes the cart without a warning from Swiggy, so Cravio must say so first.",
        "cart": [NIDHI, {NOODLES: 5}],
        "turns": ["Use my Home address. Add a Gudbud Sundae from Polar Bear to my cart."],
        "follow_up": "Yes, go ahead.",
        "approve": True,
        "checks": {
            "cart": [POLAR_BEAR, {SUNDAE: 1}],
            "reads_cart_before_change": True,
            "warns_before_change": ["noodle"],
        },
    },
    {
        "name": "declined-no-false-claim",
        "about": "When the user declines the cart change, Cravio must not say the item was added.",
        "cart": [NIDHI, {NOODLES: 1}],
        "turns": ["Use my Home address. Add a Gulab Jamun from NIDHI Ki RASOI to my cart."],
        "follow_up": "Yes, go ahead.",
        "approve": False,
        "checks": {
            "cart": [NIDHI, {NOODLES: 1}],
            "no_success_claim": True,
        },
    },
]
