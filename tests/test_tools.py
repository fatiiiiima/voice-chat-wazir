import json

import pytest

from agent import tools


@pytest.fixture(autouse=True)
def bookings_file(tmp_path, monkeypatch):
    path = tmp_path / "bookings.jsonl"
    monkeypatch.setattr(tools, "BOOKINGS_FILE", path)
    return path


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("motor city", "AGMC Motor City"),
        ("the Sharjah branch", "AGMC Sharjah"),
        ("Sheikh Zayed Road", "AGMC Sheikh Zayed Road"),
        ("SZR", "AGMC Sheikh Zayed Road"),
        ("Abu Dhabi", None),
    ],
)
def test_match_branch(spoken, expected):
    assert tools.match_branch(spoken) == expected


def test_match_car():
    assert tools.match_car("BMW X3")["id"] == "C106"
    assert tools.match_car("land cruiser")["id"] == "C104"
    assert tools.match_car("Ferrari Roma") is None


def test_booking_confirmed_and_recorded(bookings_file):
    result = tools.book_test_drive_impl(
        "Fatima Masood", "+971 50 123 4567", "BMW X3", "Tuesday 28 October 2026 at 17:00", "motor city"
    )
    assert result["status"] == "confirmed"
    assert result["reference"].startswith("TD-")
    assert result["branch"] == "AGMC Motor City"
    saved = json.loads(bookings_file.read_text().strip())
    assert saved["phone"] == "971501234567"
    assert saved["car_id"] == "C106"


def test_booking_reports_every_problem(bookings_file):
    result = tools.book_test_drive_impl("", "12", "Ferrari", "", "Abu Dhabi")
    assert result["status"] == "error"
    assert len(result["errors"]) == 5
    assert not bookings_file.exists()
