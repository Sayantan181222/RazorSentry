# All PII is hashed before storage — raw identifiers never touch the audit log

import hashlib
import hmac
import os

PII_SALT = os.getenv("PII_SALT", "razorsentry-default-salt-change-in-prod")


# Returns a one-way HMAC-SHA256 hex digest of the input string using PII_SALT
def blind_identifier(raw_id: str) -> str:
    return hmac.new(
        PII_SALT.encode("utf-8"),
        raw_id.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()[:16]


# Blinds all PII fields in a transaction dict before any logging or storage
def blind_transaction(tx_dict: dict) -> dict:
    blinded = tx_dict.copy()
    for field in ("nameOrig", "nameDest", "transaction_id"):
        if field in blinded and blinded[field]:
            blinded[field] = blind_identifier(str(blinded[field]))
    return blinded


# Returns True if a string looks like an already-blinded identifier (16 hex chars)
def is_blinded(value: str) -> bool:
    return len(value) == 16 and all(c in "0123456789abcdef" for c in value)
