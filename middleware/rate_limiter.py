"""
Rate Limiting Middleware for FastAPI.

Implements:
  - Per-IP rate limiting (sliding window)
  - Per-endpoint rate limiting
  - Tiered limits (standard / premium API keys)
  - Custom rate limit headers (X-RateLimit-*)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("middleware.rate_limiter")


class SlidingWindowCounter:
    """In-memory sliding-window rate counter."""

    def __init__(self, window_seconds: int = 60, max_requests: int = 100):
        self.window = window_seconds
        self.max_requests = max_requests
        self._hits: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> tuple[bool, dict]:
        """Check if request is allowed. Returns (allowed, info)."""
        now = time.time()
        cutoff = now - self.window

        # Prune old entries
        self._hits[key] = [t for t in self._hits[key] if t > cutoff]

        current = len(self._hits[key])
        remaining = max(0, self.max_requests - current)
        reset_at = int(cutoff + self.window)

        if current >= self.max_requests:
            return False, {
                "limit": self.max_requests,
                "remaining": 0,
                "reset": reset_at,
                "retry_after": int(self._hits[key][0] + self.window - now) + 1,
            }

        self._hits[key].append(now)
        return True, {
            "limit": self.max_requests,
            "remaining": remaining - 1,
            "reset": reset_at,
        }

    def cleanup(self):
        """Remove expired entries to prevent memory leak."""
        now = time.time()
        expired = [k for k, v in self._hits.items() if not v or v[-1] < now - self.window * 2]
        for k in expired:
            del self._hits[k]


# Default rate limits
RATE_LIMITS = {
    "default": {"window": 60, "max_requests": 100},
    "ml_predict": {"window": 60, "max_requests": 30},
    "heavy_query": {"window": 60, "max_requests": 10},
    "auth": {"window": 300, "max_requests": 20},
}

# Endpoints that get stricter limits
HEAVY_ENDPOINTS = {
    "/api/predict", "/api/scenario", "/api/optimize",
    "/api/monte-carlo", "/api/time-series",
}
AUTH_ENDPOINTS = {"/api/auth/login", "/api/auth/register"}

# Premium API keys get higher limits
PREMIUM_KEYS = set()  # Populate from config/DB


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app, default_rpm: int = 100, ml_rpm: int = 30):
        super().__init__(app)
        self._default = SlidingWindowCounter(60, default_rpm)
        self._ml = SlidingWindowCounter(60, ml_rpm)
        self._heavy = SlidingWindowCounter(60, 10)
        self._auth = SlidingWindowCounter(300, 20)
        self._last_cleanup = time.time()

    def _get_client_key(self, request: Request) -> str:
        """Extract client identifier (IP or API key)."""
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            return f"key:{api_key}"

        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        return f"ip:{request.client.host}" if request.client else "ip:unknown"

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and static files
        path = request.url.path
        if path in ("/health", "/", "/docs", "/openapi.json") or path.startswith("/static"):
            return await call_next(request)

        client_key = self._get_client_key(request)

        # Check if premium
        is_premium = client_key.startswith("key:") and client_key[4:] in PREMIUM_KEYS

        # Select appropriate limiter
        if path in AUTH_ENDPOINTS:
            counter = self._auth
        elif any(path.startswith(ep) for ep in HEAVY_ENDPOINTS):
            counter = self._heavy if not is_premium else self._ml
        elif "/predict" in path or "/forecast" in path:
            counter = self._ml
        else:
            counter = self._default

        allowed, info = counter.is_allowed(client_key)

        if not allowed:
            logger.warning(f"[RateLimit] {client_key} exceeded limit on {path}")
            response = Response(
                content='{"error": "Rate limit exceeded", "retry_after": ' + str(info.get("retry_after", 60)) + '}',
                status_code=429,
                media_type="application/json",
            )
            response.headers["X-RateLimit-Limit"] = str(info["limit"])
            response.headers["X-RateLimit-Remaining"] = "0"
            response.headers["X-RateLimit-Reset"] = str(info["reset"])
            response.headers["Retry-After"] = str(info.get("retry_after", 60))
            return response

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(info["reset"])

        # Periodic cleanup
        if time.time() - self._last_cleanup > 300:
            self._default.cleanup()
            self._ml.cleanup()
            self._heavy.cleanup()
            self._auth.cleanup()
            self._last_cleanup = time.time()

        return response


def get_rate_limit_status() -> dict:
    """Return current rate limit configuration."""
    return {
        "tiers": RATE_LIMITS,
        "heavy_endpoints": list(HEAVY_ENDPOINTS),
        "premium_keys_count": len(PREMIUM_KEYS),
        "timestamp": datetime.now().isoformat(),
    }
