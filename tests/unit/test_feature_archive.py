"""Feature archive — the durable training set that outlives DB retention.

The property under test is narrow but load-bearing: `feature_matrix` is pruned
at 3 days, so anything not copied out before the prune is gone permanently.
These tests pin the round-trip and, most importantly, that a failed archive
blocks the prune rather than silently letting it delete unarchived history.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.data import feature_archive as fa


@pytest.fixture()
def local_archive(tmp_path, monkeypatch):
    """Point the archive at a temp directory and force the local backend."""
    monkeypatch.setattr(fa, "LOCAL_ROOT", tmp_path / "archive")
    monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
    return tmp_path / "archive"


def _frame(day: date, tickers=("AAPL", "NVDA"), n=5) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=14)
    for tk in tickers:
        for i in range(n):
            rows.append({
                "time": base + pd.Timedelta(minutes=i),
                "ticker": tk,
                "ffsa_version": "v1",
                "rsi_14": 50.0 + i,
                "atr_pct": 0.01 * (i + 1),
                "close": 100.0 + i,
            })
    return pd.DataFrame(rows)


def test_round_trip_preserves_features_and_close(local_archive):
    day = date(2026, 6, 1)
    fa._write(day, _frame(day))

    out = fa.load_range(day, day)
    assert len(out) == 10
    assert {"time", "ticker", "rsi_14", "atr_pct", "close"} <= set(out.columns)
    # close must survive — without it no future retrain can build a label
    assert out["close"].notna().all()


def test_load_range_spans_days_and_skips_gaps(local_archive):
    d1, d3 = date(2026, 6, 1), date(2026, 6, 3)
    fa._write(d1, _frame(d1))
    fa._write(d3, _frame(d3))

    out = fa.load_range(d1, d3)
    assert len(out) == 20                      # d2 missing, not an error
    assert out["time"].is_monotonic_increasing or out.groupby("ticker")["time"].apply(
        lambda s: s.is_monotonic_increasing).all()


def test_ticker_filter(local_archive):
    day = date(2026, 6, 1)
    fa._write(day, _frame(day))
    out = fa.load_range(day, day, tickers=["AAPL"])
    assert set(out["ticker"]) == {"AAPL"}


def test_exists_and_available_days(local_archive):
    day = date(2026, 6, 1)
    assert not fa.exists(day)
    fa._write(day, _frame(day))
    assert fa.exists(day)
    assert fa.available_days() == ["2026-06-01"]


def test_load_range_empty_when_nothing_archived(local_archive):
    out = fa.load_range(date(2026, 6, 1), date(2026, 6, 5))
    assert out.empty


class _FailingArchive:
    """Stands in for archive_pending() raising."""

    async def __call__(self, *a, **k):
        raise RuntimeError("s3 unavailable")


@pytest.mark.asyncio
async def test_prune_skips_feature_matrix_when_archive_fails(monkeypatch):
    """THE critical invariant.

    feature_matrix is pruned at 3 days. If archiving fails and we prune anyway,
    the rows are gone for good — there is no second copy. Losing a day of disk
    is recoverable; losing a day of training data is not.
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

    monkeypatch.setattr(db, "get_engine", lambda: _Engine())
    monkeypatch.setattr(
        "src.data.feature_archive.archive_pending", _FailingArchive(),
    )

    results = await db.prune_old_data()

    assert results.get("feature_matrix") == 0, (
        "feature_matrix was pruned even though archiving failed — unarchived "
        "training data would have been destroyed"
    )
    assert not any("feature_matrix" in s for s in executed), (
        "a DELETE was issued against feature_matrix despite the archive failing"
    )


@pytest.mark.asyncio
async def test_prune_proceeds_for_other_tables_when_archive_fails(monkeypatch):
    """Only feature_matrix is protected — the rest still need pruning or the
    disk fills up, which is the incident that created the 3-day policy."""
    import src.data.db as db

    deleted: list[str] = []

    class _Conn:
        async def execute(self, stmt, params=None):
            deleted.append(str(stmt))

            class _R:
                rowcount = 1
            return _R()

    class _Begin:
        async def __aenter__(self): return _Conn()
        async def __aexit__(self, *a): return False

    class _Engine:
        def begin(self): return _Begin()

    monkeypatch.setattr(db, "get_engine", lambda: _Engine())
    monkeypatch.setattr(
        "src.data.feature_archive.archive_pending", _FailingArchive(),
    )
    monkeypatch.setattr(db, "create_async_engine", lambda *a, **k: None)

    await db.prune_old_data()
    assert any("ohlcv_1m" in s for s in deleted)
