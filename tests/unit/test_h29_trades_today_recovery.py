"""Unit tests for H29: DB-derived n_trades_today on startup.

Root problem: _sizing_n_trades_today resets to 0 on any restart, granting a
second full daily allotment when a redeploy happens mid-day.  H29 reads the
day's entry count directly from the DB at startup so the counter is never
stale after a restart.

Tests cover:
  1. Restart with entries in DB today → counter is raised to db_count
  2. Restart with no DB entries today → counter stays at 0 (overnight restart)
  3. Counter is never LOWERED by the DB seed (in-memory wins when higher)
  4. Pipeline filter: only this pipeline's entries are counted
  5. DB error is swallowed (fail-open; counter stays at 0)
  6. None scalar (empty table) is treated as zero, not an error
  7. After recovery to 6, the daily cap is seen as exhausted
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Patch get_settings before importing SignalLoop so __init__ doesn't fail
_mock_settings = MagicMock()
_mock_settings.alpaca_mode = "paper"
_mock_settings.polygon_api_key = "test"
_mock_settings.alpaca_api_key = "test"
_mock_settings.alpaca_secret_key = "test"
_mock_settings.database_url = "sqlite+aiosqlite://"


def _make_loop(pipeline_id: str = "pipeline_a") -> "SignalLoop":
    with patch("src.config.get_settings", return_value=_mock_settings):
        from src.agents.signal_loop import SignalLoop
        from src.execution.position_manager import PositionManager
        from src.risk.circuit_breakers import CircuitBreakers
        loop = SignalLoop(
            universe=["AAPL"],
            ensemble=MagicMock(),
            alpaca=MagicMock(),
            circuit_breakers=CircuitBreakers(),
            pos_manager=PositionManager(initial_portfolio=100_000.0),
            session_factory=MagicMock(),
            feature_cols=[f"f{i}" for i in range(30)],
            pipeline_id=pipeline_id,
        )
    return loop


@pytest.mark.asyncio
async def test_mid_day_restart_raises_counter():
    """A restart mid-day with 4 DB entries sets n_trades_today to 4."""
    loop = _make_loop()
    assert loop._sizing_n_trades_today == 0

    session_mock = AsyncMock()
    session_mock.execute = AsyncMock(return_value=MagicMock(scalar=MagicMock(return_value=4)))
    sf = MagicMock(return_value=MagicMock(
        __aenter__=AsyncMock(return_value=session_mock),
        __aexit__=AsyncMock(return_value=False),
    ))

    with patch("src.data.db.get_session_factory", return_value=sf):
        await loop._seed_trades_today_from_db()

    assert loop._sizing_n_trades_today == 4


@pytest.mark.asyncio
async def test_overnight_restart_leaves_counter_at_zero():
    """Overnight restart: no entries today → counter stays 0."""
    loop = _make_loop()

    session_mock = AsyncMock()
    session_mock.execute = AsyncMock(return_value=MagicMock(scalar=MagicMock(return_value=0)))
    sf = MagicMock(return_value=MagicMock(
        __aenter__=AsyncMock(return_value=session_mock),
        __aexit__=AsyncMock(return_value=False),
    ))

    with patch("src.data.db.get_session_factory", return_value=sf):
        await loop._seed_trades_today_from_db()

    assert loop._sizing_n_trades_today == 0


@pytest.mark.asyncio
async def test_counter_is_never_lowered():
    """In-memory count is 5 but DB says 3 → counter stays at 5 (fail-safe)."""
    loop = _make_loop()
    loop._sizing_n_trades_today = 5

    session_mock = AsyncMock()
    session_mock.execute = AsyncMock(return_value=MagicMock(scalar=MagicMock(return_value=3)))
    sf = MagicMock(return_value=MagicMock(
        __aenter__=AsyncMock(return_value=session_mock),
        __aexit__=AsyncMock(return_value=False),
    ))

    with patch("src.data.db.get_session_factory", return_value=sf):
        await loop._seed_trades_today_from_db()

    assert loop._sizing_n_trades_today == 5


@pytest.mark.asyncio
async def test_db_error_is_swallowed_counter_unchanged():
    """A DB error must not raise and must leave the counter at 0 (fail-open)."""
    loop = _make_loop()

    session_mock = AsyncMock()
    session_mock.execute = AsyncMock(side_effect=RuntimeError("connection reset"))
    sf = MagicMock(return_value=MagicMock(
        __aenter__=AsyncMock(return_value=session_mock),
        __aexit__=AsyncMock(return_value=False),
    ))

    with patch("src.data.db.get_session_factory", return_value=sf):
        # Must not raise
        await loop._seed_trades_today_from_db()

    assert loop._sizing_n_trades_today == 0


@pytest.mark.asyncio
async def test_db_none_scalar_treated_as_zero():
    """None from scalar() (empty table) is treated as 0, not an error."""
    loop = _make_loop()

    session_mock = AsyncMock()
    session_mock.execute = AsyncMock(return_value=MagicMock(scalar=MagicMock(return_value=None)))
    sf = MagicMock(return_value=MagicMock(
        __aenter__=AsyncMock(return_value=session_mock),
        __aexit__=AsyncMock(return_value=False),
    ))

    with patch("src.data.db.get_session_factory", return_value=sf):
        await loop._seed_trades_today_from_db()

    assert loop._sizing_n_trades_today == 0


@pytest.mark.asyncio
async def test_session_factory_receives_pipeline_filter():
    """The DB query must include the pipeline_id filter when set."""
    loop = _make_loop(pipeline_id="pipeline_a")

    captured_queries: list = []

    class CapturingSession(AsyncMock):
        async def execute(self, query, *args, **kwargs):
            captured_queries.append(str(query))
            return MagicMock(scalar=MagicMock(return_value=0))

    capturing = CapturingSession()
    sf = MagicMock(return_value=MagicMock(
        __aenter__=AsyncMock(return_value=capturing),
        __aexit__=AsyncMock(return_value=False),
    ))

    with patch("src.data.db.get_session_factory", return_value=sf):
        await loop._seed_trades_today_from_db()

    assert len(captured_queries) == 1, "expected exactly one DB query"
    # The compiled SQL should reference the pipeline_id column
    assert "pipeline_id" in captured_queries[0].lower() or \
        "pipeline_id" in repr(captured_queries[0]).lower(), \
        f"pipeline_id filter not found in query: {captured_queries[0]}"


@pytest.mark.asyncio
async def test_cap_is_respected_after_recovery():
    """After recovering n=6 from DB, a new entry attempt should be blocked."""
    loop = _make_loop()

    session_mock = AsyncMock()
    session_mock.execute = AsyncMock(return_value=MagicMock(scalar=MagicMock(return_value=6)))
    sf = MagicMock(return_value=MagicMock(
        __aenter__=AsyncMock(return_value=session_mock),
        __aexit__=AsyncMock(return_value=False),
    ))

    with patch("src.data.db.get_session_factory", return_value=sf):
        await loop._seed_trades_today_from_db()

    assert loop._sizing_n_trades_today == 6
    # The daily cap constant is SIZING_MAX_TRADES_PER_DAY = 6; counter at cap
    # means the next entry check should see it as exhausted.
    from src.agents.signal_loop import SIZING_MAX_TRADES_PER_DAY
    assert loop._sizing_n_trades_today >= SIZING_MAX_TRADES_PER_DAY
