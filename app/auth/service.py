"""
Authentication service.
Simple username/password with JWT tokens.
Pure Python implementation - no native dependencies.
"""
from __future__ import annotations

import base64
import hashlib
import hmac as hmac_lib
import json
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config.settings import get_settings

http_bearer = HTTPBearer(auto_error=False)


# ------------------------------------------------------------------ JWT (pure Python HMAC)

def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64_decode(s: str) -> bytes:
    padding = "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s + padding)


def create_access_token(username: str, expires_minutes: Optional[int] = None) -> str:
    """Create an HS256 JWT token (pure Python)."""
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.jwt_expire_minutes
    )
    header = _b64_encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64_encode(json.dumps(
        {"sub": username, "exp": int(expire.timestamp())},
        separators=(",", ":"),
    ).encode())
    signing_input = f"{header}.{payload}"
    sig = hmac_lib.new(
        settings.jwt_secret_key.encode(),
        signing_input.encode(),
        hashlib.sha256,
    ).digest()
    return f"{signing_input}.{_b64_encode(sig)}"


def decode_token(token: str) -> Optional[str]:
    """Decode and verify an HS256 JWT token. Returns username or None."""
    if not token:
        return None
    settings = get_settings()
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        signing_input = f"{parts[0]}.{parts[1]}"
        expected_sig = hmac_lib.new(
            settings.jwt_secret_key.encode(),
            signing_input.encode(),
            hashlib.sha256,
        ).digest()
        actual_sig = _b64_decode(parts[2])
        if not hmac_lib.compare_digest(expected_sig, actual_sig):
            return None
        payload = json.loads(_b64_decode(parts[1]))
        if payload.get("exp", 0) < datetime.now(timezone.utc).timestamp():
            return None
        return payload.get("sub")
    except Exception:
        return None


# ------------------------------------------------------------------ Password hashing

def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256 (no native dependencies)."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode(), 200_000)
    return f"pbkdf2sha256${salt}${dk.hex()}"


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password against its PBKDF2 hash."""
    try:
        if hashed.startswith("pbkdf2sha256$"):
            _, salt, stored_hex = hashed.split("$")
            dk = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt.encode(), 200_000)
            return hmac_lib.compare_digest(dk.hex(), stored_hex)
    except Exception:
        pass
    return False


# ------------------------------------------------------------------ FastAPI dependency

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
) -> str:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    username = decode_token(credentials.credentials)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return username


# ------------------------------------------------------------------ Auth service

class AuthService:
    """Handles user authentication against environment-configured credentials."""

    def __init__(self) -> None:
        settings = get_settings()
        self._username = settings.dashboard_username
        self._hashed_password = hash_password(settings.dashboard_password)

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Return JWT token if credentials are valid, else None."""
        if username != self._username:
            return None
        if not verify_password(password, self._hashed_password):
            return None
        return create_access_token(username)
