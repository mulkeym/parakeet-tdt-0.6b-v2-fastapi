import mimetypes
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Ensure fonts get a correct MIME type (Python's mimetypes lacks woff2), so they
# aren't blocked by X-Content-Type-Options: nosniff.
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("font/woff", ".woff")

from .model import lifespan
from .routes import router
from .config import SERVE_DASHBOARD, logger

from parakeet_service.stream_routes import router as stream_router

# Strict security headers applied to every HTTP response. The dashboard loads
# only same-origin CSS/JS/fonts (no inline scripts/styles, no external CDN), so
# a restrictive CSP holds without 'unsafe-inline'.
_CSP = (
    "default-src 'self'; "
    "img-src 'self' data:; "
    "style-src 'self'; "
    "script-src 'self'; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "base-uri 'none'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)
SECURITY_HEADERS = {
    "Content-Security-Policy": _CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}

_STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Parakeet-TDT 0.6B v2 STT service",
        version="0.0.1",
        description=(
            "High-accuracy English speech-to-text (FastConformer-TDT) "
            "with optional word/char/segment timestamps."
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def _security_headers(request, call_next):
        response = await call_next(request)
        for key, value in SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        return response

    app.include_router(router)

    # TODO: improve streaming and add support for other audio formats (maybe)
    app.include_router(stream_router)

    if SERVE_DASHBOARD:
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
        logger.info("Demo dashboard enabled (GET / and /static)")

    logger.info("FastAPI app initialised")
    return app


app = create_app()
