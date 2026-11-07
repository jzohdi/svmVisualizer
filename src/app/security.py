"""Security headers and the Content-Security-Policy, from one pure ASGI middleware.

Not @app.middleware("http"): Starlette's BaseHTTPMiddleware has a history of interfering with BackgroundTasks, and
training runs as one.
"""

import base64
import hashlib
import re

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from starlette.datastructures import MutableHeaders

FRAME_ANCESTORS = (
    "frame-ancestors 'self' https://jakezohdi.dev https://studio-zohdi-llc.com https://www.studio-zohdi-llc.com"
)

# "/", "/about" and everything else. Scripts are pinned to the exact CDN files the templates load. Both Tailwind
# runtimes and Plotly inject styles, and the markup has style attributes, hence 'unsafe-inline' for styles only.
PAGE_CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self' https://cdn.plot.ly/plotly-2.35.3.min.js"
    " https://unpkg.com/@tailwindcss/browser@4.3.5/dist/index.global.js https://cdn.tailwindcss.com/3.4.17",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    FRAME_ANCESTORS,
])

SECURITY_HEADERS = [
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "strict-origin-when-cross-origin"),
    ("Permissions-Policy", "camera=(), microphone=(), geolocation=()"),
]


def _inline_script_hashes(app: FastAPI) -> str:
    """CSP hashes of the inline scripts in FastAPI's /docs and /docs/oauth2-redirect pages.

    Rendered with the same arguments FastAPI's own docs routes pass (with no root_path), so the hashes follow the
    installed FastAPI version.
    """
    pages = [
        get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            init_oauth=app.swagger_ui_init_oauth,
            swagger_ui_parameters=app.swagger_ui_parameters,
        ),
        get_swagger_ui_oauth2_redirect_html(),
    ]
    scripts = [s for page in pages for s in re.findall(r"<script>(.*?)</script>", page.body.decode(), re.S)]
    return " ".join(f"'sha256-{base64.b64encode(hashlib.sha256(s.encode()).digest()).decode()}'" for s in scripts)


def docs_csp(app: FastAPI) -> str:
    """FastAPI's /docs (Swagger UI) and /redoc: jsDelivr and fastapi.tiangolo.com's favicon; Redoc also loads Google
    Fonts, the Redocly logo and a blob: worker, and injects styles."""
    return "; ".join([
        "default-src 'self'",
        f"script-src 'self' https://cdn.jsdelivr.net {_inline_script_hashes(app)}",
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com",
        "img-src 'self' data: https://fastapi.tiangolo.com https://cdn.redoc.ly",
        "worker-src 'self' blob:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        FRAME_ANCESTORS,
    ])


def docs_paths(app: FastAPI) -> set:
    return {p for p in (app.docs_url, app.redoc_url, app.swagger_ui_oauth2_redirect_url) if p}


class SecurityHeadersMiddleware:
    """Adds the header baseline to every HTTP response, and the docs or the page CSP by path."""

    def __init__(self, app, docs_paths: set, docs_csp: str):
        self.app = app
        self.docs_paths = docs_paths
        self.docs_csp = docs_csp

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        csp = self.docs_csp if scope["path"] in self.docs_paths else PAGE_CSP

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS:
                    headers[name] = value
                headers["Content-Security-Policy"] = csp
            await send(message)

        await self.app(scope, receive, send_with_headers)
