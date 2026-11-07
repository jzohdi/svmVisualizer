"""Smoke test of the behavior the site must keep: pages, the cached examples, the API contract and fresh training.

Model IDs are SHA-256 hashes of the request, so the committed app.db answers the sample datasets' requests at once.
"""

import hashlib
import json
import time
import urllib.error
import urllib.request

import pytest

EX2_LINEAR = "c9be2707b4c904e86f64122472bc7d633bd36e3637a876109fc09e303422973d"  # cached in app.db
EX1_LINEAR = "cfcbad7f9982c69b185bb53fe4a37bf221f6f0da73d64786ffe0cc95994532eb"  # cached in app.db
EX2_KNN = "3ef6a8c8a71e5df9306963807c5e6980e2c0fdb355c8053ebe007cc60b690902"  # not cached: trains fresh
METHODS = ["Linear", "Polynomial", "RBF", "Log Regression", "KNN", "Tree", "Forest"]
OPERATIONS = [
    "GET /", "GET /about", "GET /api/health", "GET /api/train_result/{id}", "POST /api/train_model",
]


def request(url, method="GET", body=None, headers=None):
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        with e:
            return e.code, dict(e.headers), e.read()


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def chart_points(page):
    """Wait for the Plotly chart, then return (trace type, {trace name: point count})."""
    page.wait_for_function("() => document.getElementById('myChart')?.data?.length > 0", timeout=60000)
    traces = page.evaluate("() => document.getElementById('myChart').data.map(t => [t.type, t.name, t.x.length])")
    return {t[0] for t in traces}, {t[1]: t[2] for t in traces}


# ---------- API ----------

def test_health(base_url):
    assert request(base_url + "/api/health")[::2] == (200, b'{"status":"healthy"}')


def test_static_sample_and_404(base_url):
    status, headers, body = request(base_url + "/static/example2.csv")
    assert status == 200 and headers["content-type"].startswith("text/csv")
    assert body.decode().count("\n") >= 13
    assert request(base_url + "/no-such-path")[::2] == (404, b'{"detail":"Not Found"}')


def test_cached_result(base_url):
    status, _, body = request(f"{base_url}/api/train_result/{EX1_LINEAR}")
    assert status == 200
    j = json.loads(body)
    assert j["status"] == "complete" and j["id"] == EX1_LINEAR
    r = j["result"]
    assert (r["method"], r["score"], r["params"]) == ("Linear", 1.0, '{"C": 0.1}')
    assert sha(r["result"]) == "39a1d7174b209b73de13bb07633cb251f802a808372e3ba4262d08f8f607baee"


def test_unknown_id_is_pending(base_url):
    status, _, body = request(base_url + "/api/train_result/smoke-test-unknown-id")
    assert status == 200 and json.loads(body) == {"id": "smoke-test-unknown-id", "status": "pending", "result": None}


@pytest.mark.parametrize("body,loc,type_", [
    ({"training_data": [[1, 2]], "labels": ["a"], "method": "Nope"}, ["body", "method"], "enum"),
    ({"training_data": [], "labels": ["a"], "method": "Linear"}, ["body", "training_data"], "too_short"),
    # 1-D rows pass validation (only the method fails here); they fail later, in training (pre-existing).
    ({"training_data": [[1.0]], "labels": ["a"], "method": "Nope"}, ["body", "method"], "enum"),
    # Size caps
    ({"training_data": [[1, 2]] * 10_001, "labels": ["a"], "method": "Linear"}, ["body", "training_data"], "too_long"),
    ({"training_data": [[0.0] * 101], "labels": ["a"], "method": "Linear"}, ["body", "training_data", 0], "too_long"),
    ({"training_data": [[1, 2]], "labels": ["a"] * 10_001, "method": "Linear"}, ["body", "labels"], "too_long"),
    ({"training_data": [[1, 2]], "labels": ["a" * 101], "method": "Linear"}, ["body", "labels", 0], "string_too_long"),
    ({"training_data": [[1, 2]], "labels": ["a"], "method": "Linear", "testing_data": [[1, 2]] * 20_001},
     ["body", "testing_data"], "too_long"),
    ({"training_data": [[1, 2]], "labels": ["a"], "method": "Linear", "testing_data": [[0.0] * 101]},
     ["body", "testing_data", 0], "too_long"),
], ids=["bad-method", "empty-training", "1d-rows-ok", "too-many-rows", "row-too-long", "too-many-labels",
        "label-too-long", "too-many-test-points", "test-row-too-long"])
def test_validation_errors(base_url, body, loc, type_):
    # A request that fails validation never reaches training, so this is safe against a live site.
    status, _, out = request(base_url + "/api/train_model", "POST", body)
    assert status == 422
    assert [(e["loc"], e["type"]) for e in json.loads(out)["detail"]] == [(loc, type_)]


def test_openapi(base_url):
    spec = json.loads(request(base_url + "/openapi.json")[2])
    ops = sorted(f"{m.upper()} {p}" for p, item in spec["paths"].items() for m in item)
    assert ops == OPERATIONS
    assert sorted(spec["components"]["schemas"]) == [
        "HTTPValidationError", "ModelMethod", "SvmResultResponse", "TrainingRequest", "TrainingResponse",
        "TrainingStatus", "ValidationError",
    ]


def test_cors(base_url):
    allowed = "https://svm-visualizer.fly.dev"
    _, h, _ = request(base_url + "/api/health", headers={"Origin": allowed})
    assert h.get("access-control-allow-origin") == allowed
    _, h, _ = request(base_url + "/api/health", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in {k.lower() for k in h}


# ---------- pages ----------

def test_home_trains_example2_on_load(base_url, probe):
    page = probe.page
    page.goto(base_url + "/")
    assert page.title() == "SVM Visualizer"
    assert [o for o in page.locator("#method option").all_inner_texts()] == METHODS
    assert page.locator("#sample-dataset option").all_inner_texts() == ["example2.csv", "example1.csv", "example0.csv"]
    types, points = chart_points(page)
    assert page.input_value("#polling-id") == EX2_LINEAR
    assert types == {"scatter3d"} and set(points) == {"no", "yes"} and sum(points.values()) == 25 ** 3
    assert page.locator("#dynamic-table tbody tr").count() == 14
    assert probe.errors == []


def test_home_example1_linear_2d(base_url, probe):
    page = probe.page
    page.goto(base_url + "/")
    chart_points(page)  # the automatic example2 run
    page.select_option("#sample-dataset", "example1.csv")
    page.click("#load-sample")
    page.wait_for_function("() => document.querySelectorAll('#dynamic-table tbody tr').length === 8")
    page.click("#submit")
    page.wait_for_function(f"() => document.getElementById('polling-id').value === '{EX1_LINEAR}'")
    page.wait_for_function("() => document.getElementById('myChart')?.data?.[0]?.type === 'scatter'", timeout=60000)
    types, points = chart_points(page)
    assert types == {"scatter"} and sum(points.values()) == 50 * 50
    assert probe.errors == []


def test_about(base_url, probe):
    page = probe.page
    assert page.goto(base_url + "/about").status == 200
    assert page.title() == "About - SVM Visualizer"
    page.wait_for_function("() => getComputedStyle(document.body).backgroundColor === 'rgb(243, 244, 246)'")
    assert probe.errors == []


def test_docs(base_url, probe):
    page = probe.page
    page.goto(base_url + "/docs")
    page.wait_for_selector(".opblock-summary", timeout=60000)
    ops = page.eval_on_selector_all(
        ".opblock-summary",
        "els => els.map(e => e.querySelector('.opblock-summary-method').textContent + ' ' +"
        " e.querySelector('.opblock-summary-path').getAttribute('data-path'))",
    )
    assert sorted(ops) == OPERATIONS
    assert probe.errors == []


# ---------- fresh training (local only) ----------

@pytest.mark.trains
def test_fresh_training_example2_knn(base_url, probe):
    page = probe.page
    page.goto(base_url + "/")
    chart_points(page)
    page.select_option("#method", "KNN")
    page.click("#submit")
    page.wait_for_function(f"() => document.getElementById('polling-id').value === '{EX2_KNN}'")
    deadline = time.monotonic() + 120
    while True:
        j = json.loads(request(f"{base_url}/api/train_result/{EX2_KNN}")[2])
        if j["status"] == "complete" or time.monotonic() > deadline:
            break
        time.sleep(1)
    assert j["status"] == "complete"
    r = j["result"]
    assert (r["method"], r["score"], r["params"]) == ("KNN", 0.8333333333333333, '{"leaf_size": 5, "n_neighbors": 5}')
    assert sha(r["result"]) == "ce872818039a341e7523bfbac65ae06ff553f0a3377d1dddfa17c57bc4839eed"
    confidence = json.loads(r["confidence"])
    assert len(confidence) == 25 ** 3 and sum(confidence) == pytest.approx(11326.4, abs=1e-6)
    # The page polls with backoff; wait until it has replaced the example2 + Linear chart with this result.
    page.wait_for_function(
        "() => document.getElementById('myChart')?.data?.find(t => t.name === 'no')?.x.length === 10081",
        timeout=60000,
    )
    _, points = chart_points(page)
    assert points == {"no": 10081, "yes": 5544}
    assert probe.errors == []


# ---------- headers and CSP ----------

def test_redoc(base_url, probe):
    page = probe.page
    page.goto(base_url + "/redoc")
    page.wait_for_selector('[data-section-id^="operation/"]', timeout=60000)
    assert page.locator('[data-section-id^="operation/"]').count() == len(OPERATIONS)
    assert probe.errors == []


@pytest.mark.parametrize("path", ["/", "/about", "/docs", "/redoc", "/api/health", "/static/index.js"])
def test_security_headers(base_url, path):
    _, h, _ = request(base_url + path)
    h = {k.lower(): v for k, v in h.items()}
    assert h["x-content-type-options"] == "nosniff"
    assert h["referrer-policy"] == "strict-origin-when-cross-origin"
    assert h["permissions-policy"] == "camera=(), microphone=(), geolocation=()"
    csp = h["content-security-policy"]
    assert "frame-ancestors 'self' https://jakezohdi.dev https://studio-zohdi-llc.com" in csp
    assert "object-src 'none'" in csp
    script_src = next(d for d in csp.split("; ") if d.startswith("script-src "))
    assert "'unsafe-inline'" not in script_src and "'unsafe-eval'" not in script_src
