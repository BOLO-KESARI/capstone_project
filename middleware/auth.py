"""
RBAC Authentication Skeleton for FastAPI.

Implements:
  - JWT token creation & validation
  - Role-based access control (admin, analyst, viewer)
  - Password hashing
  - Dependency injection for route protection
  - In-memory user store (swap for DB in production)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional
import base64

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger("middleware.auth")

# ─── Configuration ──────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("JWT_SECRET", "airline-rm-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

# ─── Roles & Permissions ────────────────────────────────────────────────────
ROLES = {
    "admin": {
        "description": "Full system access",
        "permissions": [
            "read", "write", "delete", "ml:predict", "ml:train",
            "pricing:approve", "pricing:override", "reports:generate",
            "admin:users", "admin:config", "audit:read",
        ],
    },
    "analyst": {
        "description": "Analytics and ML access",
        "permissions": [
            "read", "ml:predict", "ml:train", "pricing:approve",
            "reports:generate", "audit:read",
        ],
    },
    "viewer": {
        "description": "Read-only dashboard access",
        "permissions": ["read", "reports:generate"],
    },
}


# ─── Password Hashing (no bcrypt dependency) ────────────────────────────────
def hash_password(password: str, salt: str = None) -> str:
    """PBKDF2-SHA256 password hash."""
    if salt is None:
        salt = os.urandom(16).hex()
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${key.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against stored hash."""
    try:
        salt, key_hex = hashed.split("$", 1)
        expected = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
        return hmac.compare_digest(expected.hex(), key_hex)
    except Exception:
        return False


# ─── JWT (minimal, no PyJWT dependency) ──────────────────────────────────────
def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * padding)


def create_token(payload: dict, expires_delta: timedelta = None) -> str:
    """Create a JWT-like token."""
    header = {"alg": ALGORITHM, "typ": "JWT"}
    now = time.time()
    payload = {**payload, "iat": now}

    if expires_delta:
        payload["exp"] = now + expires_delta.total_seconds()
    else:
        payload["exp"] = now + ACCESS_TOKEN_EXPIRE_MINUTES * 60

    header_b64 = _b64encode(json.dumps(header).encode())
    payload_b64 = _b64encode(json.dumps(payload).encode())
    message = f"{header_b64}.{payload_b64}"

    signature = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256).digest()
    sig_b64 = _b64encode(signature)

    return f"{message}.{sig_b64}"


def decode_token(token: str) -> dict:
    """Decode and verify a JWT-like token."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token format")

        header_b64, payload_b64, sig_b64 = parts
        message = f"{header_b64}.{payload_b64}"

        # Verify signature
        expected_sig = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256).digest()
        actual_sig = _b64decode(sig_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            raise ValueError("Invalid signature")

        payload = json.loads(_b64decode(payload_b64))

        # Check expiration
        if payload.get("exp", 0) < time.time():
            raise ValueError("Token expired")

        return payload
    except Exception as e:
        raise ValueError(f"Token validation failed: {e}")


# ─── In-Memory User Store ───────────────────────────────────────────────────
_users: dict[str, dict] = {
    "admin": {
        "username": "admin",
        "password_hash": hash_password("admin123"),
        "role": "admin",
        "full_name": "System Administrator",
        "created_at": datetime.now().isoformat(),
    },
    "analyst": {
        "username": "analyst",
        "password_hash": hash_password("analyst123"),
        "role": "analyst",
        "full_name": "Revenue Analyst",
        "created_at": datetime.now().isoformat(),
    },
    "viewer": {
        "username": "viewer",
        "password_hash": hash_password("viewer123"),
        "role": "viewer",
        "full_name": "Dashboard Viewer",
        "created_at": datetime.now().isoformat(),
    },
}


def authenticate_user(username: str, password: str) -> dict | None:
    """Authenticate user and return user dict (or None)."""
    user = _users.get(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


def create_user(username: str, password: str, role: str = "viewer", full_name: str = "") -> dict:
    """Create a new user."""
    if username in _users:
        raise ValueError(f"User '{username}' already exists")
    if role not in ROLES:
        raise ValueError(f"Invalid role: {role}")

    _users[username] = {
        "username": username,
        "password_hash": hash_password(password),
        "role": role,
        "full_name": full_name,
        "created_at": datetime.now().isoformat(),
    }
    return {"username": username, "role": role, "full_name": full_name}


# ─── FastAPI Dependencies ───────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """Extract current user from JWT token. Returns guest if no token."""
    if not credentials:
        # Allow unauthenticated access with limited permissions
        return {"username": "guest", "role": "viewer", "permissions": ROLES["viewer"]["permissions"]}

    try:
        payload = decode_token(credentials.credentials)
        username = payload.get("sub")
        user = _users.get(username)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return {
            "username": user["username"],
            "role": user["role"],
            "permissions": ROLES[user["role"]]["permissions"],
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


def require_permission(permission: str):
    """Dependency that checks if user has a specific permission."""

    async def checker(user: dict = Depends(get_current_user)):
        if permission not in user.get("permissions", []):
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required. Your role: {user['role']}",
            )
        return user

    return checker


def require_role(role: str):
    """Dependency that checks if user has a specific role."""

    async def checker(user: dict = Depends(get_current_user)):
        role_hierarchy = {"admin": 3, "analyst": 2, "viewer": 1}
        if role_hierarchy.get(user.get("role"), 0) < role_hierarchy.get(role, 0):
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' required. Your role: {user['role']}",
            )
        return user

    return checker


# ─── Token Generation Helpers ────────────────────────────────────────────────
def login(username: str, password: str) -> dict:
    """Authenticate and return tokens."""
    user = authenticate_user(username, password)
    if not user:
        return {"error": "Invalid credentials"}

    access_token = create_token(
        {"sub": username, "role": user["role"]},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh_token = create_token(
        {"sub": username, "type": "refresh"},
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "role": user["role"],
        "username": username,
    }


def list_users() -> list[dict]:
    """List all users (without password hashes)."""
    return [
        {"username": u["username"], "role": u["role"], "full_name": u["full_name"], "created_at": u["created_at"]}
        for u in _users.values()
    ]
