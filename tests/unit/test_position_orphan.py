"""Held positions must stay syncable and exitable after a universe rotation.

Regression suite for the 2026-08-09 MSCI orphan. MSCI was bought 07-27 while in
the screener universe, the nightly rotation dropped it, and then:

  1. sync_from_broker skipped it (`if self._universe and ticker not in
     self._universe: continue`), so it never entered _positions;
  2. the exit path iterates _positions, so it was never exit-checked;
  3. even if it had been, the bar poller is universe-driven so there was no
     OHLCV row and `price <= 0` would have `continue`d anyway.

Net effect: open 13 days, $12.6k (12.9% of the account) stranded and invisible
to portfolio_heat — so risk limits were computed on understated exposure.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.execution.position_manager import PositionManager


def _broker(*tickers, price: float = 100.0, qty: float = 10.0):
    router = MagicMock()
    router.get_positions = AsyncMock(return_value=[
        {"ticker": t, "qty": qty, "avg_entry_price": price,
         "market_value": qty * price, "side": "long"}
        for t in tickers
    ])
    router.get_account = AsyncMock(return_value={"portfolio_value": 100_000.0})
    return router


# ─── sync must not drop what we hold ─────────────────────────────────────────

def test_held_ticker_outside_universe_is_still_synced():
    """THE bug: a rotated-out holding must survive sync."""
    pm = PositionManager(initial_portfolio=100_000.0, universe=["AAPL", "MSFT"])
    pm.set_owned_tickers({"MSCI"})            # open ledger row says we hold it

    asyncio.run(pm.sync_from_broker(_broker("AAPL", "MSCI")))

    assert "MSCI" in pm._positions, "held position dropped — it can never exit"
    assert "AAPL" in pm._positions


def test_unowned_ticker_outside_universe_is_still_skipped():
    """The A/B guard must survive: don't claim another pipeline's position."""
    pm = PositionManager(initial_portfolio=100_000.0, universe=["AAPL"])
    pm.set_owned_tickers(set())

    asyncio.run(pm.sync_from_broker(_broker("AAPL", "SOMEONE_ELSE")))

    assert "AAPL" in pm._positions
    assert "SOMEONE_ELSE" not in pm._positions


def test_managed_ticker_survives_rotation_even_without_ledger_seed():
    """A position opened this process is ours even if the ledger read failed."""
    pm = PositionManager(initial_portfolio=100_000.0, universe=["AAPL"])
    pm._managed_tickers.add("MSCI")

    asyncio.run(pm.sync_from_broker(_broker("AAPL", "MSCI")))
    assert "MSCI" in pm._positions


def test_empty_universe_syncs_everything():
    """No universe configured → no filtering (unchanged behaviour)."""
    pm = PositionManager(initial_portfolio=100_000.0, universe=None)
    asyncio.run(pm.sync_from_broker(_broker("AAPL", "MSCI")))
    assert {"AAPL", "MSCI"} <= set(pm._positions)


# ─── stranded capital shows up in heat ───────────────────────────────────────

def test_orphan_position_counts_toward_portfolio_heat():
    """Understated heat is the dangerous part: risk limits read exposure.

    With MSCI dropped, heat reported 0.0 while 12.9% of the account was
    actually deployed.
    """
    pm = PositionManager(initial_portfolio=100_000.0, universe=["AAPL"])

    pm.set_owned_tickers(set())               # pre-fix behaviour
    asyncio.run(pm.sync_from_broker(_broker("MSCI", price=564.0, qty=22.34)))
    heat_dropped = pm.portfolio_heat

    pm.set_owned_tickers({"MSCI"})            # post-fix
    asyncio.run(pm.sync_from_broker(_broker("MSCI", price=564.0, qty=22.34)))
    heat_seen = pm.portfolio_heat

    assert heat_dropped == pytest.approx(0.0)
    assert heat_seen > 0.10, heat_seen
    assert heat_seen > heat_dropped


def test_set_owned_tickers_normalises_case():
    pm = PositionManager(initial_portfolio=100_000.0, universe=["AAPL"])
    pm.set_owned_tickers(["msci", "Wdc"])
    assert pm._owned_tickers == {"MSCI", "WDC"}
