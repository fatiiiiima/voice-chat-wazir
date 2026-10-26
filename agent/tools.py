"""Mock dealership data and the book_test_drive tool.

Nothing here touches a real database: bookings are appended to logs/bookings.jsonl.
"""

import json
import random
import re
import string
import time
from pathlib import Path

from loguru import logger

BRANCHES = [
    "AGMC Sheikh Zayed Road",
    "AGMC Motor City",
    "AGMC Sharjah",
]

# Small mock inventory of used cars. Prices in AED.
INVENTORY = [
    {"id": "C101", "car": "2021 BMW X5 xDrive40i", "km": 48000, "price": 189000, "branch": "AGMC Sheikh Zayed Road"},
    {"id": "C102", "car": "2020 BMW 330i", "km": 61000, "price": 98000, "branch": "AGMC Sheikh Zayed Road"},
    {"id": "C103", "car": "2022 MINI Cooper S", "km": 22000, "price": 89000, "branch": "AGMC Motor City"},
    {"id": "C104", "car": "2019 Toyota Land Cruiser GXR", "km": 95000, "price": 175000, "branch": "AGMC Sharjah"},
    {"id": "C105", "car": "2021 Nissan Patrol LE", "km": 70000, "price": 165000, "branch": "AGMC Sharjah"},
    {"id": "C106", "car": "2022 BMW X3 xDrive30i", "km": 35000, "price": 145000, "branch": "AGMC Motor City"},
    {"id": "C107", "car": "2020 Mercedes-Benz C200", "km": 58000, "price": 92000, "branch": "AGMC Motor City"},
    {"id": "C108", "car": "2023 BMW i4 eDrive40", "km": 15000, "price": 210000, "branch": "AGMC Sheikh Zayed Road"},
]

BOOKINGS_FILE = Path("logs/bookings.jsonl")


def inventory_for_prompt() -> str:
    lines = [
        f"- {c['car']}, {c['km']:,} km, AED {c['price']:,}, at {c['branch']}"
        for c in INVENTORY
    ]
    return "\n".join(lines)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def match_branch(branch: str) -> str | None:
    """Match spoken branch names like 'motor city' or 'SZR' to a branch."""
    b = _normalize(branch)
    aliases = {"szr": "AGMC Sheikh Zayed Road", "sheikh zayed": "AGMC Sheikh Zayed Road"}
    for alias, name in aliases.items():
        if alias in b:
            return name
    for name in BRANCHES:
        short = _normalize(name.replace("AGMC", ""))
        if short in b or b in _normalize(name):
            return name
    return None


def match_car(car: str) -> dict | None:
    """Return the inventory item whose name shares the most words with `car`."""
    words = set(_normalize(car).split())
    best, best_score = None, 0
    for item in INVENTORY:
        score = len(words & set(_normalize(item["car"]).split()))
        if score > best_score:
            best, best_score = item, score
    return best if best_score >= 1 else None


def validate_phone(phone: str) -> str | None:
    digits = re.sub(r"\D", "", phone)
    return digits if 9 <= len(digits) <= 15 else None


def book_test_drive_impl(name: str, phone: str, car: str, date_time: str, branch: str) -> dict:
    """Validate and record a booking. Returns a dict the LLM can read back to the caller."""
    errors = []
    if not name.strip():
        errors.append("customer name is missing")
    phone_digits = validate_phone(phone)
    if not phone_digits:
        errors.append("phone number looks invalid, ask the customer to repeat it")
    car_item = match_car(car)
    if not car_item:
        errors.append(f"'{car}' is not in our inventory")
    branch_name = match_branch(branch)
    if not branch_name:
        errors.append(f"unknown branch '{branch}', valid branches are: {', '.join(BRANCHES)}")
    if not date_time.strip():
        errors.append("date and time are missing")

    if errors:
        return {"status": "error", "errors": errors}

    ref = "TD-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    booking = {
        "ref": ref,
        "name": name.strip(),
        "phone": phone_digits,
        "car": car_item["car"],
        "car_id": car_item["id"],
        "date_time": date_time.strip(),
        "branch": branch_name,
        "created_at": time.time(),
    }
    BOOKINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with BOOKINGS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(booking) + "\n")
    logger.info(f"Test drive booked: {booking}")
    return {"status": "confirmed", "reference": ref, **{k: booking[k] for k in ("car", "date_time", "branch")}}
