"""The archive must reject sessions missing cross-sectional features.

`rs_vwap_dev` is a deployed model input and can only come from the universe
feature path. A session written by a per-ticker path has it absent, and
training on a mix of present/absent is worse than either consistent state: the
model learns the feature is zero across most of history and real only recently.

This is not hypothetical — 2026-09-28 was archived by pre-v0.8.2 production and
had to be removed by hand before it reached a retrain.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from src.data import feature_archive as fa


@pytest.fixture()
def local_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(fa, "LOCAL_ROOT", tmp_path / "archive")
    monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
    return tmp_path / "archive"


def _rows(day: date, *, cross_sectional: bool):
    base = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=14)
    out = []
    for tk in ("AAA", "BBB"):
        for i in range(4):
            r = {"time": base + pd.Timedelta(minutes=i), "ticker": tk,
                 "ffsa_version": "v1", "rsi_14": 50.0 + i, "close": 100.0 + i}
            if cross_sectional:
                r["rs_vwap_dev"] = 0.001 * i
                r["rs_1m"] = 0.0002 * i
            out.append(r)
    return out


class _FakeSession:
    """archive_pending opens a SEPARATE session for the feature query and the
    price query, so the call counter has to live on the factory — not the
    session, or every session hands back the feature rows."""

    def __init__(self, state):
        self._state = state

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt):
        self._state["n"] += 1
        payload = (self._state["features"] if self._state["n"] == 1
                   else self._state["prices"])

        class _R:
            def all(_self):
                return payload
        return _R()


def _patch_db(monkeypatch, rows):
    """Feed archive_pending a fake DB returning `rows` for one session."""
    state = {
        "n": 0,
        "features": [
            (r["time"], r["ticker"],
             {k: v for k, v in r.items()
              if k not in ("time", "ticker", "ffsa_version")},
             r["ffsa_version"])
            for r in rows
        ],
        "prices": [(r["time"], r["ticker"], r["close"]) for r in rows],
    }

    import src.data.db as db
    monkeypatch.setattr(
        db, "get_session_factory", lambda: (lambda: _FakeSession(state)),
    )


@pytest.mark.asyncio
async def test_session_without_cross_sectional_is_refused(local_archive, monkeypatch):
    day = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    _patch_db(monkeypatch, _rows(day, cross_sectional=False))

    result = await fa.archive_pending(lookback_days=1)

    assert not fa.exists(day), "a session with no rs_* features was archived"
    assert not result.ok, "archive reported success while refusing a session"
    assert any("cross-sectional" in e for e in result.errors)


@pytest.mark.asyncio
async def test_session_with_cross_sectional_is_archived(local_archive, monkeypatch):
    day = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    _patch_db(monkeypatch, _rows(day, cross_sectional=True))

    result = await fa.archive_pending(lookback_days=1)

    assert fa.exists(day), "a valid session was not archived"
    assert result.ok and result.rows_written == 8


@pytest.mark.asyncio
async def test_a_refused_session_blocks_the_feature_matrix_prune(monkeypatch):
    """Refusing must not let the prune delete the rows anyway.

    db.prune_old_data skips feature_matrix when archiving reports failure —
    otherwise a refused session would be destroyed rather than retried.
    """
    import src.data.db as db

    executed: list[str] = []

    class _Conn:
        async def execute(self, stmt, params=None):
            executed.append(str(stmt))

            class _R:
                rowcount = 0
            return _R()

    class _Begin:
        async def __aenter__(self): return _Conn()
        async def __aexit__(self, *a): return False

    class _Engine:
        def begin(self): return _Begin()

    async def _failing(*a, **k):
        r = fa.ArchiveResult()
        r.errors.append("2026-09-28: no cross-sectional (rs_*) features")
        return r

    monkeypatch.setattr(db, "get_engine", lambda: _Engine())
    monkeypatch.setattr("src.data.feature_archive.archive_pending", _failing)

    results = await db.prune_old_data()
    assert results.get("feature_matrix") == 0
    assert not any("feature_matrix" in s for s in executed)
