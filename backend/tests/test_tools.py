import json

import pytest

from app import config
from app.tools.claims import get_claim_status, submit_claim

VALID = {
    "policy_number": "POL-1092",
    "claim_type": "Water Damage",
    "amount": 800.0,
    "description": "A pipe burst under the kitchen sink.",
}


def stored_claims() -> list[dict]:
    return json.loads(config.CLAIMS_FILE.read_text())


def test_status_of_approved_claim():
    result = get_claim_status("CLM-8821")

    assert result["found"] is True
    assert result["status"] == "Approved"
    assert result["amount"] == 3500.0
    assert result["policy_number"] == "POL-1092"


def test_status_of_claim_under_review():
    result = get_claim_status("CLM-9014")

    assert result["found"] is True
    assert result["status"] == "Under Review"
    assert result["claim_type"] == "Personal Property"


def test_status_accepts_lower_case_and_spaces():
    assert get_claim_status(" clm-8821 ")["found"] is True


def test_unknown_claim_is_not_found():
    assert get_claim_status("CLM-0000") == {"found": False, "claim_id": "CLM-0000"}


@pytest.mark.parametrize("claim_id", ["8821", "CLM-12", "CLM-ABCD", "POL-1092", "", "CLM-8821; DROP TABLE"])
def test_malformed_claim_id_returns_an_error(claim_id):
    result = get_claim_status(claim_id)

    assert result["found"] is False
    assert "error" in result


def test_submit_valid_claim_appends_a_record():
    before = stored_claims()

    result = submit_claim(**VALID)

    assert result["confirmation_id"] == "CLM-9015"
    assert result["status"] == "Submitted"

    after = stored_claims()
    assert len(after) == 3
    assert after[:2] == before
    assert after[2]["claim_id"] == "CLM-9015"
    assert after[2]["status"] == "Submitted"
    assert after[2]["amount"] == 800.0
    assert after[2]["description"] == VALID["description"]
    assert after[2]["submitted_at"]
    assert get_claim_status("CLM-9015")["found"] is True


def test_submitted_claims_get_consecutive_ids():
    assert submit_claim(**VALID)["confirmation_id"] == "CLM-9015"
    assert submit_claim(**VALID)["confirmation_id"] == "CLM-9016"
    assert not list(config.CLAIMS_FILE.parent.glob("*.tmp"))


@pytest.mark.parametrize(
    "field, value",
    [
        ("policy_number", "1234"),
        ("policy_number", "POL-12345"),
        ("claim_type", "Earthquake"),
        ("amount", -50),
        ("amount", 0),
        ("amount", "a lot"),
        ("amount", "800"),
        ("amount", 2_000_000),
        ("description", "too short"),
    ],
)
def test_invalid_field_is_rejected_and_nothing_is_written(field, value):
    result = submit_claim(**{**VALID, field: value})

    assert "confirmation_id" not in result
    assert list(result["fields"]) == [field]
    assert len(stored_claims()) == 2


def test_all_invalid_fields_are_listed():
    result = submit_claim(policy_number="1234", claim_type="Fire", amount=-1, description="short")

    assert set(result["fields"]) == {"policy_number", "claim_type", "amount", "description"}


def test_claim_type_is_case_insensitive():
    result = submit_claim(**{**VALID, "claim_type": "water damage"})

    assert result["claim_type"] == "Water Damage"
    assert stored_claims()[2]["claim_type"] == "Water Damage"


def test_amount_is_rounded_to_two_decimals():
    assert submit_claim(**{**VALID, "amount": 99.999})["amount"] == 100.0


def test_amount_above_the_policy_limit_is_still_accepted():
    result = submit_claim(**{**VALID, "claim_type": "Personal Property", "amount": 12000})

    assert result["confirmation_id"] == "CLM-9015"
    assert result["amount"] == 12000.0
