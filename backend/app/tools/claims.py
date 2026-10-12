import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app import config

CLAIM_ID_PATTERN = re.compile(r"^CLM-\d{4,}$")
CLAIM_TYPES = ["Water Damage", "Personal Property"]
SECTION_LIMITS = {"Water Damage": 25_000, "Personal Property": 10_000}
FIELD_HINTS = {
    "policy_number": "must look like POL-1234 (POL- followed by 4 digits)",
    "claim_type": "must be Water Damage or Personal Property",
    "amount": "must be a number greater than 0 and at most 1,000,000",
    "description": "must be between 10 and 1000 characters",
}

_write_lock = threading.Lock()


class ClaimSubmission(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    policy_number: str = Field(pattern=r"^POL-\d{4}$")
    claim_type: Literal["Water Damage", "Personal Property"]
    amount: float = Field(strict=True, gt=0, le=1_000_000)
    description: str = Field(min_length=10, max_length=1000)

    @field_validator("claim_type", mode="before")
    @classmethod
    def normalise_claim_type(cls, value):
        if isinstance(value, str):
            for claim_type in CLAIM_TYPES:
                if value.strip().lower() == claim_type.lower():
                    return claim_type
        return value

    @field_validator("amount")
    @classmethod
    def round_amount(cls, value: float) -> float:
        value = round(value, 2)
        if value <= 0:
            raise ValueError("amount must be greater than 0")
        return value


def format_errors(error: ValidationError) -> dict:
    fields = {}
    for item in error.errors():
        name = str(item["loc"][0]) if item["loc"] else "claim"
        fields[name] = FIELD_HINTS.get(name, item["msg"])
    return {"error": "The claim was not submitted because some fields are invalid.", "fields": fields}


def _read_claims() -> list[dict]:
    with open(config.CLAIMS_FILE, encoding="utf-8") as file:
        return json.load(file)


def get_claim_status(claim_id: str) -> dict:
    claim_id = str(claim_id).strip().upper()
    if not CLAIM_ID_PATTERN.match(claim_id):
        return {"found": False, "error": "Invalid claim ID. Claim IDs look like CLM-1234."}

    for record in _read_claims():
        if record["claim_id"] == claim_id:
            return {"found": True, **record}
    return {"found": False, "claim_id": claim_id}


def submit_claim(policy_number: str, claim_type: str, amount: float, description: str) -> dict:
    try:
        claim = ClaimSubmission(
            policy_number=policy_number, claim_type=claim_type, amount=amount, description=description
        )
    except ValidationError as error:
        return format_errors(error)

    with _write_lock:
        claims = _read_claims()
        last_number = max((int(record["claim_id"].split("-")[1]) for record in claims), default=0)
        record = {
            "claim_id": f"CLM-{last_number + 1:04d}",
            "policy_number": claim.policy_number,
            "claim_type": claim.claim_type,
            "status": "Submitted",
            "amount": claim.amount,
            "description": claim.description,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        claims.append(record)

        # write a temp file next to the real one and swap it in, so a crash can't leave half a file
        fd, temp_path = tempfile.mkstemp(dir=config.CLAIMS_FILE.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(claims, file, indent=2)
        os.replace(temp_path, config.CLAIMS_FILE)

    result = {
        "confirmation_id": record["claim_id"],
        "status": "Submitted",
        "policy_number": claim.policy_number,
        "claim_type": claim.claim_type,
        "amount": claim.amount,
    }
    limit = SECTION_LIMITS[claim.claim_type]
    if claim.amount > limit:
        result["note"] = f"The amount is above the ${limit:,} coverage limit for {claim.claim_type}."
    return result
