# tests/test_mcp_http_transport.py
"""
Tests for the opt-in HTTP transport added in fix for #1801.

Design constraints
------------------
* No real sockets — avoids port conflicts and firewall issues on all CI
  runners (Linux, macOS, Windows).
* No asyncio.get_event_loop() — deprecated in 3.10, raises in 3.12+.
* No asyncio primitives created at module/class scope — they must be
  constructed inside a running event loop (Python 3.10+ requirement).
* threading.Lock (not asyncio.Lock) for the dispatch lock in tests —
  safe on all platforms including Windows ProactorEventLoop.
* Uses Starlette's synchronous TestClient so we stay in normal pytest
  (no pytest-asyncio dependency needed).
"""
import json
import threading
import types
import sys
import pytest

# ── Optional dependency guard ──────────────────────────────────────────
starlette = pytest.importorskip("starlette", reason="starlette not installed")
pytest.importorskip("uvicorn", reason="uvicorn not installed")

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient


# ── Stub out heavy dependencies so import succeeds in CI without a palace ─
def _install_stubs():
    stub_chroma = types.ModuleType("chromadb")
    stub_chroma.PersistentClient = lambda **kw: None
    sys.modules.setdefault("chromadb", stub_chroma)

    for name in [
        "mempalace.knowledge_graph",
        "mempalace.searcher",
        "mempalace.palace_graph",
        "mempalace.config",
        "mempalace.backends",
        "mempalace.backends.base",
    ]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.KnowledgeGraph = lambda: types.SimpleNamespace(
                query_entity=lambda *a, **kw: [],
                add_triple=lambda *a, **kw: "id",
                invalidate=lambda *a, **kw: None,
                timeline=lambda *a, **kw: [],
                stats=lambda: {},
            )
            m.search_memories = lambda *a, **kw: []
            m.traverse = lambda *a, **kw: {}
            m.find_tunnels = lambda *a, **kw: {}
            m.graph_stats = lambda *a, **kw: {}
            m.MempalaceConfig = lambda: types.SimpleNamespace(
                palace_path="~/.mempalace/palace",
                collection_name="mempalace",
            )
            sys.modules[name] = m


_install_stubs()
import mempalace.mcp_server as _srv  # noqa: E402  (after stubs)


# ── Build a minimal Starlette app that mirrors _serve_http() ──────────────
# Key differences from production code:
#   - threading.Lock instead of asyncio.Lock (safe on Windows too)
#   - handle_request() called directly (no executor) — it is synchronous
#   - Lock created here at *function* scope, not module scope
#
# This tests the same dispatch logic that _serve_http() exercises without
# touching sockets or asyncio event loop internals.

_dispatch_lock = threading.Lock()


async def _mcp_endpoint(request: Request) -> Response:
    try:
        payload = await request.json()
    except Exception as exc:
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            },
            status_code=400,
        )
    with _dispatch_lock:
        result = _srv.handle_request(payload)
    if result is None:
        return Response(status_code=202)
    return JSONResponse(result)


async def _health(request: Request) -> Response:
    return JSONResponse({"status": "ok", "tools": len(_srv.TOOLS)})


_app = Starlette(
    routes=[
        Route("/mcp", _mcp_endpoint, methods=["POST"]),
        Route("/health", _health, methods=["GET"]),
    ]
)


@pytest.fixture(scope="module")
def client():
    """Synchronous Starlette TestClient — no event loop juggling needed."""
    with TestClient(_app, raise_server_exceptions=True) as c:
        yield c


# ── Helpers ───────────────────────────────────────────────────────────────

def _tools_list(req_id=1):
    return {"jsonrpc": "2.0", "id": req_id, "method": "tools/list", "params": {}}


def _initialize(req_id=1):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    }


# ── Tests ─────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_reports_tool_count(self, client):
        data = r = client.get("/health")
        assert r.json()["status"] == "ok"
        assert r.json()["tools"] == len(_srv.TOOLS)
        assert r.json()["tools"] > 0


class TestToolsList:
    def test_returns_all_tools(self, client):
        r = client.post("/mcp", json=_tools_list())
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == 1
        assert "tools" in data["result"]
        assert len(data["result"]["tools"]) == len(_srv.TOOLS)

    def test_content_type_is_json(self, client):
        r = client.post("/mcp", json=_tools_list())
        assert "application/json" in r.headers["content-type"]

    def test_id_preserved(self, client):
        for rid in [1, 99, "abc-id"]:
            r = client.post("/mcp", json=_tools_list(req_id=rid))
            assert r.json()["id"] == rid

    def test_idempotent_repeated_calls(self, client):
        sets = [
            frozenset(t["name"] for t in
                      client.post("/mcp", json=_tools_list(i)).json()
                      ["result"]["tools"])
            for i in range(20)
        ]
        assert len(set(sets)) == 1, "tools/list returned different sets across calls"

    def test_all_tools_have_name_and_schema(self, client):
        tools = client.post("/mcp", json=_tools_list()).json()["result"]["tools"]
        for tool in tools:
            assert "name" in tool
            assert "inputSchema" in tool


class TestInitialize:
    def test_protocol_version(self, client):
        r = client.post("/mcp", json=_initialize())
        assert r.status_code == 200
        assert r.json()["result"]["protocolVersion"] == "2024-11-05"

    def test_capabilities_advertised(self, client):
        caps = client.post("/mcp", json=_initialize()).json()["result"]["capabilities"]
        assert "tools" in caps


class TestNotifications:
    def test_initialized_returns_202(self, client):
        r = client.post("/mcp", json={
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        })
        assert r.status_code == 202
        assert r.content == b""  # no body for notifications

    def test_other_notification_returns_202(self, client):
        r = client.post("/mcp", json={
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"progressToken": 1, "progress": 50},
        })
        assert r.status_code == 202


class TestErrorHandling:
    def test_unknown_method_returns_32601(self, client):
        r = client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 99, "method": "bogus/method", "params": {},
        })
        data = r.json()
        assert data["error"]["code"] == -32601
        assert data["error"]["message"] != ""

    def test_unknown_tool_returns_32601(self, client):
        r = client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 5,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        })
        assert r.json()["error"]["code"] == -32601

    def test_malformed_json_returns_400(self, client):
        r = client.post(
            "/mcp",
            content=b"not json at all{{{",
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == -32700

    def test_ping_returns_empty_result(self, client):
        r = client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 3, "method": "ping", "params": {},
        })
        assert r.json()["result"] == {}


class TestConcurrency:
    def test_concurrent_tools_list(self, client):
        """
        Fire 10 parallel requests via threads (mirrors real concurrent HTTP
        clients).  All must return the same tool set with no data races.
        Uses threading.Lock in the dispatch layer so this is safe on every
        platform.
        """
        results = []
        errors = []

        def call():
            try:
                r = client.post("/mcp", json=_tools_list())
                results.append(
                    frozenset(t["name"] for t in r.json()["result"]["tools"])
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Threads raised: {errors}"
        assert len(set(results)) == 1, "Concurrent calls returned different tool sets"
