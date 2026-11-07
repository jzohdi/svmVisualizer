"""Smoke-test fixtures.

By default the tests start the app themselves: uvicorn runs from a fresh temp dir holding a copy of the committed
app.db, so ./app.db and ./error.log never touch the repo and every run starts from the same cached results.

Environment:
  SVM_BASE_URL           test a running site instead, read-only: tests marked `trains` or `posts` are skipped.
                         What remains is GETs, the home page's own automatic POST (a cached result) and POSTs
                         that fail validation (422).
  SVM_TEST_PORT          port for the local server (default: a free one).
  SVM_BROWSER_CHANNEL    Playwright browser channel, e.g. "chrome" for the system Chrome. Default: Playwright's
                         own Chromium (`python -m playwright install chromium`).
"""

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def pytest_configure(config):
    config.addinivalue_line("markers", "trains: trains a fresh model (writes a row); local server only")
    config.addinivalue_line("markers", "posts: sends a POST beyond the page's own automatic one; local server only")


def pytest_collection_modifyitems(config, items):
    if not os.environ.get("SVM_BASE_URL"):
        return
    skip = pytest.mark.skip(reason="trains or POSTs; skipped against SVM_BASE_URL (read-only)")
    for item in items:
        if "trains" in item.keywords or "posts" in item.keywords:
            item.add_marker(skip)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_healthy(url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(0.25)
    return False


@pytest.fixture(scope="session")
def base_url(tmp_path_factory):
    remote = os.environ.get("SVM_BASE_URL")
    if remote:
        url = remote.rstrip("/")
        # Fly machines stop when idle; give a cold start time to wake.
        assert _wait_healthy(url, 120), f"{url}/api/health never answered"
        yield url
        return

    workdir = tmp_path_factory.mktemp("svm-app")
    shutil.copyfile(REPO / "app.db", workdir / "app.db")
    port = int(os.environ.get("SVM_TEST_PORT") or _free_port())
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONUNBUFFERED": "1"}
    log = open(workdir / "server.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        if not _wait_healthy(url, 60):
            raise RuntimeError(f"server never answered; see {workdir / 'server.log'}")
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


@pytest.fixture(scope="session")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(channel=os.environ.get("SVM_BROWSER_CHANNEL") or None, headless=True)
        yield b
        b.close()


class PageProbe:
    """A page plus everything that went wrong on it: page errors, console errors and CSP violations."""

    def __init__(self, page):
        self.page = page
        self.errors = []
        page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        page.on("console", self._console)
        page.on("dialog", lambda d: d.accept())

    def _console(self, msg):
        # /about has no <link rel="icon">, so the browser's own /favicon.ico request 404s; that isn't the page's doing.
        if msg.type == "error" and not msg.location.get("url", "").endswith("/favicon.ico"):
            self.errors.append(f"console: {msg.text} ({msg.location.get('url')})")


@pytest.fixture
def probe(browser):
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    ctx.add_init_script(
        "document.addEventListener('securitypolicyviolation',"
        " e => console.error(`CSP violation: ${e.effectiveDirective} ${e.blockedURI}`));"
    )
    page = ctx.new_page()
    yield PageProbe(page)
    ctx.close()
