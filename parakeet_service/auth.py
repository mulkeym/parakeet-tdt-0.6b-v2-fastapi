"""
Optional API-key authentication.

Behaviour is controlled by the ``API_KEY`` environment variable:

* **unset / empty**  -> the service is open; every request is allowed.
* **set**            -> every REST request and WebSocket connection must present
                        the matching key, either as the ``x-api-key`` header
                        (configurable via ``API_KEY_HEADER``) or as an
                        ``Authorization: Bearer <key>`` header.

Comparison is constant-time to avoid leaking the key through timing.
"""
from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, WebSocket, status

from .config import ALLOWED_WS_ORIGINS, API_KEY, API_KEY_HEADER, logger


def _extract_key(headers) -> str | None:
    """Pull the presented key from the configured header or a Bearer token."""
    key = headers.get(API_KEY_HEADER)
    if key:
        return key
    auth = headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def _is_valid(presented: str | None) -> bool:
    if not presented:
        return False
    # constant-time comparison
    return hmac.compare_digest(presented, API_KEY)


async def require_api_key(request: Request) -> None:
    """FastAPI dependency enforcing the API key on REST endpoints."""
    if API_KEY is None:  # auth disabled
        return
    if not _is_valid(_extract_key(request.headers)):
        logger.warning("Rejected request to %s: missing/invalid API key", request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def websocket_authorized(ws: WebSocket) -> bool:
    """Validate a WebSocket handshake. Returns True if allowed.

    On failure the caller should close the socket with policy-violation (1008).
    Query-parameter fallback (``?api_key=``) is supported because browser
    WebSocket clients cannot set custom headers.
    """
    # Optional Origin allowlist (defense against cross-site WebSocket hijacking).
    # Only enforced when configured and when the client sends an Origin header
    # (browsers do; native/CLI clients typically do not).
    if ALLOWED_WS_ORIGINS is not None:
        origin = ws.headers.get("origin")
        if origin is not None and origin not in ALLOWED_WS_ORIGINS:
            logger.warning("Rejected WebSocket connection: disallowed Origin %r", origin)
            return False

    if API_KEY is None:  # auth disabled
        return True
    presented = _extract_key(ws.headers) or ws.query_params.get("api_key")
    if _is_valid(presented):
        return True
    logger.warning("Rejected WebSocket connection: missing/invalid API key")
    return False
