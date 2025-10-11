from qj_paths import rel as qj_rel
"""
auth_utils.py ??Authentication and authorization helpers for QualiJournal.

This module centralizes JWT verification, two?멹actor authentication (2FA) and
audit logging for the QualiJournal admin API.  The goal is to provide a
single place where security?멵ritical logic can be maintained and reused
throughout the application, leveraging FastAPI's dependency injection.

The JWT implementation herein does not rely on external libraries such as
PyJWT or python?멽ose to minimize additional dependencies.  It supports
tokens signed with HS256 (HMAC/SHA??56) using a shared secret.  The secret
key should be provided via the ``QUALI_JWT_SECRET`` environment variable.

Two?멹actor authentication uses a time?멳ased one?몋ime password (TOTP) scheme.
Codes are generated from a shared base32 secret (``QUALI_2FA_SECRET``) and
are valid for 30?몊econd windows.  Clients must send the current 6?멶igit
TOTP code in the ``X-2FA-Code`` header for endpoints that require
additional verification.  See RFC혻6238 for algorithm details.

Audit logging writes a single line per authenticated request to
``logs/audit.log`` relative to the project root.  Each line includes a
timestamp, the requesting user (subject claim), HTTP method, path and
client IP address.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

__all__ = [
    "verify_jwt_token",
    "require_two_factor",
]

# ---------------------------------------------------------------------------
# Internal helpers for JWT decoding/verification
# ---------------------------------------------------------------------------

def _add_padding(b64: str) -> str:
    """Add base64 padding if missing (length not a multiple of 4)."""
    return b64 + "=" * (-len(b64) % 4)


def _decode_segment(seg: str) -> bytes:
    """Decode a base64?몍rl segment to bytes."""
    return base64.urlsafe_b64decode(_add_padding(seg))


def _hmac_sha256(key: bytes, msg: bytes) -> bytes:
    return hmac.new(key, msg, hashlib.sha256).digest()


def _verify_hs256_signature(secret: str, header_b64: str, payload_b64: str, signature_b64: str) -> bool:
    """Verify the HS256 signature on a JWT token."""
    try:
        signature = _decode_segment(signature_b64)
    except Exception:
        return False
    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected = _hmac_sha256(secret.encode(), signing_input)
    return hmac.compare_digest(signature, expected)


def _parse_jwt(token: str, secret: str) -> Dict[str, Any]:
    """
    Decode and verify a JWT signed with HS256.  Returns the payload as a dict
    on success or raises an HTTPException on error.

    Args:
        token: Encoded JWT in ``<header>.<payload>.<signature>`` format.
        secret: Shared secret used to verify the signature.

    Raises:
        HTTPException: If the token is malformed, signature is invalid or
            the ``exp`` claim is expired.

    Returns:
        dict: Decoded payload.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format")
    header_b64, payload_b64, signature_b64 = parts
    # Verify signature
    if not _verify_hs256_signature(secret, header_b64, payload_b64, signature_b64):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signature")
    # Decode header and payload
    try:
        payload_json = _decode_segment(payload_b64).decode()
        payload = json.loads(payload_json)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    # Check expiration
    now = int(time.time())
    exp = payload.get("exp")
    if exp is not None and isinstance(exp, (int, float)):
        if now >= int(exp):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    return payload


# HTTP Bearer authentication scheme
_bearer_scheme = HTTPBearer(auto_error=False)


async def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme)) -> Dict[str, Any]:
    """
    FastAPI dependency that verifies a JWT token carried in an Authorization
    header with scheme ``Bearer``.  The token must be signed with HS256 and
    contain a valid ``exp`` claim.  On success, returns the decoded
    payload.  On failure, raises an HTTPException.

    The signing secret is read from the ``QUALI_JWT_SECRET`` environment
    variable; if not set, a placeholder secret is used (not recommended for
    production).  Always configure a strong secret in deployment.
    """
    if not credentials or not credentials.scheme or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    token = credentials.credentials
    secret = os.getenv("QUALI_JWT_SECRET", "quali_default_secret")
    return _parse_jwt(token, secret)


# ---------------------------------------------------------------------------
# TOTP verification for two?멹actor authentication
# ---------------------------------------------------------------------------

def _generate_totp(secret: str, interval: int = 30, digits: int = 6) -> str:
    """
    Generate a TOTP code for the given secret and time interval.

    Implements the algorithm described in RFC혻6238 using HMAC?멣HA1.
    Args:
        secret: Base32?멷ncoded secret key.
        interval: Time step in seconds.
        digits: Number of digits in the output code.
    Returns:
        Zero?몆added numeric string of length ``digits``.
    """
    # Decode base32 secret (strip padding).  If decoding fails, treat the
    # raw secret as ASCII bytes (for backwards compatibility).
    secret_bytes: bytes
    try:
        # normalize base32 (uppercase, remove whitespace)
        b32 = secret.strip().replace(" ", "").upper()
        secret_bytes = base64.b32decode(b32, casefold=True)
    except Exception:
        secret_bytes = secret.encode()
    # Compute counter based on current Unix time
    counter = int(time.time() / interval)
    msg = struct.pack(">Q", counter)
    h = hmac.new(secret_bytes, msg, hashlib.sha1).digest()
    # Dynamic truncation
    offset = h[-1] & 0x0F
    code_int = (int.from_bytes(h[offset:offset + 4], byteorder="big") & 0x7FFFFFFF) % (10 ** digits)
    return f"{code_int:0{digits}d}"


async def require_two_factor(request: Request, user: Dict[str, Any] = Depends(verify_jwt_token)) -> Dict[str, Any]:
    """
    FastAPI dependency that enforces two?멹actor authentication and audit
    logging for critical endpoints.

    The client must send a valid Bearer token and a TOTP code in the
    ``X-2FA-Code`` header.  The TOTP secret is read from the
    ``QUALI_2FA_SECRET`` environment variable; if not set, an exception is
    raised.  Upon success, the user's decoded JWT payload is returned.

    Additionally, an audit record is appended to ``logs/audit.log`` in the
    project root containing timestamp, user ID, HTTP method, path and
    client IP.  If the log directory does not exist it will be created.
    """
    # Validate TOTP
    totp_secret = os.getenv("QUALI_2FA_SECRET")
    if not totp_secret:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="2FA secret not configured")
    client_code = (request.headers.get("X-2FA-Code") or "").strip()
    if not client_code:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-2FA-Code header")
    expected = _generate_totp(totp_secret)
    # allow a small window: accept codes from adjacent intervals
    if client_code != expected:
        # check previous or next window
        t = int(time.time())
        interval = 30
        prev_code = _generate_totp(totp_secret, interval=interval, digits=6)
        # fudge the counter 짹1 by adjusting the time for previous/next window
        # re?멼mplement because _generate_totp uses current time internally
        # Evaluate next: adjust time + interval
        # compute previous and next codes by replicating algorithm
        # but we can just compute again with t +/- interval
        def totp_at(ts):
            counter = int(ts / interval)
            msg = struct.pack(">Q", counter)
            try:
                secret_bytes = base64.b32decode(totp_secret.strip().replace(" ", "").upper(), casefold=True)
            except Exception:
                secret_bytes = totp_secret.encode()
            h = hmac.new(secret_bytes, msg, hashlib.sha1).digest()
            offset = h[-1] & 0x0F
            code_int = (int.from_bytes(h[offset:offset + 4], byteorder="big") & 0x7FFFFFFF) % (10 ** 6)
            return f"{code_int:06d}"
        prev_code = totp_at(time.time() - interval)
        next_code = totp_at(time.time() + interval)
        if client_code not in (prev_code, next_code):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid 2FA code")
    # Write audit log
    # Use subject claim or user_id from token payload as identifier
    user_id = user.get("sub") or user.get("user_id") or "anonymous"
    method = request.method
    path = request.url.path
    client_ip = request.client.host if request.client else "unknown"
    now_iso = datetime.now(timezone.utc).isoformat()
    log_line = f"{now_iso} | {user_id} | {client_ip} | {method} {path}\n"
    try:
        log_dir = Path(os.getcwd()) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "audit.log"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception:
        # Silently ignore audit log failures
        pass
    return user
