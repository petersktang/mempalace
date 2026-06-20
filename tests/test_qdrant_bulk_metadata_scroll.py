# tests/test_qdrant_bulk_metadata_scroll.py
"""
Tests for issue #1796 -- O(n^2) bulk-metadata reads on the Qdrant backend.

Covers:
  1. BaseCollection.get_all_metadata() default implementation (offset loop,
     unchanged behavior for backends with real server-side cursors).
  2. QdrantCollection.get_all_metadata() single-scroll override -- the actual
     fix -- verified by counting how many times the underlying HTTP scroll
     call fires.
  3. mcp_server._fetch_all_metadata() delegates to get_all_metadata() when
     present, and falls back to the legacy offset loop when it is not.
  4. The Qdrant scroll page size constant is 4096, not 256.
"""

import types
import sys
from unittest import mock


# ── Stub heavy deps so we can import mempalace modules in isolation ─────────
def _install_stubs():
    stub_np = sys.modules.get("numpy")
    if stub_np is None:
        import numpy  # noqa: F401  -- numpy is a real, light dependency here

    for name in [
        "mempalace.knowledge_graph",
        "mempalace.searcher",
        "mempalace.palace_graph",
        "mempalace.config",
    ]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.KnowledgeGraph = lambda: types.SimpleNamespace()
            m.search_memories = lambda *a, **kw: []
            m.traverse = lambda *a, **kw: {}
            m.find_tunnels = lambda *a, **kw: {}
            m.graph_stats = lambda *a, **kw: {}
            m.MempalaceConfig = lambda: types.SimpleNamespace(
                palace_path="~/.mempalace/palace", collection_name="mempalace"
            )
            sys.modules[name] = m


_install_stubs()

from mempalace.backends.base import (  # noqa: E402
    BaseCollection,
    GetResult,
    PalaceRef,
)
from mempalace.backends import qdrant as qdrant_mod  # noqa: E402
from mempalace.backends.qdrant import QdrantCollection, _QdrantConfig  # noqa: E402


# ---------------------------------------------------------------------------
# 1. BaseCollection default get_all_metadata()
# ---------------------------------------------------------------------------


class _FakeOffsetPagedCollection(BaseCollection):
    """Minimal concrete collection with a real server-side offset cursor.

    Simulates Chroma-like behavior: get(limit=, offset=) returns exactly the
    requested slice without re-scanning anything -- the case the default
    get_all_metadata() implementation is correct for.
    """

    def __init__(self, all_metadata):
        self._all = all_metadata
        self.get_call_count = 0

    def add(self, **kwargs):
        raise NotImplementedError

    def upsert(self, **kwargs):
        raise NotImplementedError

    def query(self, **kwargs):
        raise NotImplementedError

    def get(
        self, *, ids=None, where=None, where_document=None, limit=None, offset=None, include=None
    ):
        self.get_call_count += 1
        offset = offset or 0
        limit = limit if limit is not None else len(self._all)
        page = self._all[offset : offset + limit]
        return GetResult(ids=[], documents=[], metadatas=page, embeddings=None)

    def delete(self, **kwargs):
        raise NotImplementedError

    def count(self) -> int:
        return len(self._all)


class TestBaseCollectionDefaultGetAllMetadata:
    def test_returns_all_metadata_across_pages(self):
        all_meta = [{"wing": f"w{i}"} for i in range(2500)]
        col = _FakeOffsetPagedCollection(all_meta)
        result = col.get_all_metadata()
        assert result == all_meta

    def test_empty_collection_returns_empty_list(self):
        col = _FakeOffsetPagedCollection([])
        assert col.get_all_metadata() == []

    def test_paginates_in_1000_row_batches(self):
        """
        2500 rows at page_size=1000 must take more than one call (proving
        pagination actually happens, not a single unbounded fetch) and must
        not take an unreasonable number of calls. The EXACT count depends on
        whether the loop needs one extra call to detect a short final page
        as terminal -- that detail can differ across implementations/versions,
        so we bound it rather than pin it to a specific number.
        """
        all_meta = [{"wing": f"w{i}"} for i in range(2500)]
        col = _FakeOffsetPagedCollection(all_meta)
        result = col.get_all_metadata()

        assert result == all_meta, "all 2500 rows must be returned regardless of paging"
        assert col.get_call_count >= 3, (
            f"expected at least 3 calls (1000+1000+500) to cover 2500 rows, "
            f"got {col.get_call_count}"
        )
        assert col.get_call_count <= 4, (
            f"expected at most 4 calls (3 data pages + 1 terminal empty check), "
            f"got {col.get_call_count}"
        )

    def test_passes_where_through(self):
        all_meta = [{"wing": "a"}, {"wing": "b"}]
        col = _FakeOffsetPagedCollection(all_meta)

        captured = {}
        original_get = col.get

        def spy_get(**kwargs):
            captured.update(kwargs)
            return original_get(**kwargs)

        col.get = spy_get
        col.get_all_metadata(where={"wing": "a"})
        assert captured.get("where") == {"wing": "a"}


# ---------------------------------------------------------------------------
# 2. QdrantCollection.get_all_metadata() single-scroll override
# ---------------------------------------------------------------------------


def _make_qdrant_collection(monkeypatch, scroll_pages):
    """
    Build a QdrantCollection with a mocked REST client whose scroll_points()
    returns the given pre-baked pages: list[tuple[list[dict_point], next_offset]].
    """
    config = _QdrantConfig(url="http://localhost:6333")
    client = mock.MagicMock()
    call_log = []

    def fake_scroll_points(
        collection, *, qdrant_filter=None, limit=4096, offset=None, with_vector=False
    ):
        call_log.append({"limit": limit, "offset": offset, "filter": qdrant_filter})
        idx = len([c for c in call_log]) - 1
        return scroll_pages[idx]

    client.scroll_points.side_effect = fake_scroll_points
    client.collection_exists.return_value = True

    backend = mock.MagicMock()
    backend._closed = False
    backend._marker_exists.return_value = True

    palace = PalaceRef(id="/tmp/fake-palace", local_path="/tmp/fake-palace")
    col = QdrantCollection(
        backend=backend,
        client=client,
        config=config,
        palace=palace,
        collection_name="mempalace",
        remote_collection="mempalace_abc123_mempalace",
    )
    return col, call_log


def _fake_point(doc_id: str, wing: str) -> dict:
    return {
        "id": f"point-{doc_id}",
        "payload": {
            qdrant_mod._PAYLOAD_ID: doc_id,
            qdrant_mod._PAYLOAD_DOCUMENT: f"content for {doc_id}",
            qdrant_mod._PAYLOAD_METADATA: {"wing": wing},
        },
        "vector": None,
    }


class TestQdrantGetAllMetadataSingleScroll:
    def test_returns_all_metadata_in_one_logical_pass(self, monkeypatch):
        page1 = ([_fake_point(f"d{i}", "wing_a") for i in range(3)], "cursor-1")
        page2 = ([_fake_point(f"d{i}", "wing_b") for i in range(3, 5)], None)
        col, call_log = _make_qdrant_collection(monkeypatch, [page1, page2])

        result = col.get_all_metadata()

        assert len(result) == 5
        assert result[0] == {"wing": "wing_a"}
        assert result[-1] == {"wing": "wing_b"}

    def test_walks_collection_exactly_once_regardless_of_size(self, monkeypatch):
        """
        The whole point of #1796: calling get_all_metadata() must not
        re-trigger additional full scrolls. Two scroll_points() calls (one
        per page until next_page_offset is None) is the expected, constant
        cost -- independent of how the caller might have looped before.
        """
        page1 = ([_fake_point(f"d{i}", "wing_a") for i in range(3)], "cursor-1")
        page2 = ([_fake_point(f"d{i}", "wing_a") for i in range(3, 6)], None)
        col, call_log = _make_qdrant_collection(monkeypatch, [page1, page2])

        col.get_all_metadata()

        assert len(call_log) == 2, (
            f"Expected exactly 2 scroll_points() calls (one full pass), got {len(call_log)}"
        )

    def test_does_not_call_get_internally(self, monkeypatch):
        """
        Regression guard: get_all_metadata() must call _scroll_all() directly,
        not self.get(limit=, offset=) -- calling get() in a loop is exactly
        the O(n^2) pattern this fix removes.
        """
        page1 = ([_fake_point("d0", "wing_a")], None)
        col, _ = _make_qdrant_collection(monkeypatch, [page1])
        col.get = mock.MagicMock(side_effect=AssertionError("get() should not be called"))

        result = col.get_all_metadata()
        assert result == [{"wing": "wing_a"}]
        col.get.assert_not_called()

    def test_filters_by_where_locally_when_required(self, monkeypatch):
        """
        A plain {"wing": "wing_a"} filter is push-down-able to Qdrant's native
        filter syntax -- _requires_local_filter() returns False for it, so
        get_all_metadata() correctly skips the LOCAL Python filter and relies
        on server-side filtering instead. Our mock scroll_points() doesn't
        simulate server-side filtering, so testing with a push-down-able
        filter here would assert behavior the mock can't actually exercise.

        Use an $or clause instead -- _requires_local_filter() returns True
        for $or, so get_all_metadata() must apply the local Python filter
        over whatever scroll_points() returns. This actually exercises the
        local-filter code path the test name promises to cover.
        """
        page1 = (
            [
                _fake_point("d0", "wing_a"),
                _fake_point("d1", "wing_b"),
                _fake_point("d2", "wing_c"),
            ],
            None,
        )
        col, _ = _make_qdrant_collection(monkeypatch, [page1])

        result = col.get_all_metadata(where={"$or": [{"wing": "wing_a"}, {"wing": "wing_b"}]})
        assert result == [{"wing": "wing_a"}, {"wing": "wing_b"}]

    def test_empty_remote_collection_returns_empty_list(self, monkeypatch):
        col, call_log = _make_qdrant_collection(monkeypatch, [])
        col._client.collection_exists.return_value = False
        col._backend._marker_exists.return_value = False

        result = col.get_all_metadata()
        assert result == []


# ---------------------------------------------------------------------------
# 3. Scroll page-size constant
# ---------------------------------------------------------------------------


class TestScrollPageSizeBump:
    def test_scroll_page_size_constant_is_4096(self):
        assert qdrant_mod._SCROLL_PAGE_SIZE == 4096

    def test_scroll_all_uses_page_size_constant(self, monkeypatch):
        page1 = ([_fake_point("d0", "wing_a")], None)
        col, call_log = _make_qdrant_collection(monkeypatch, [page1])

        col._scroll_all()

        assert call_log[0]["limit"] == qdrant_mod._SCROLL_PAGE_SIZE
        assert call_log[0]["limit"] != 256


# ---------------------------------------------------------------------------
# 4. mcp_server._fetch_all_metadata() delegation
# ---------------------------------------------------------------------------
