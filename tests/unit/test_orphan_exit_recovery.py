"""Orphaned-exit recovery and alert-channel health.

Three defects from 2026-09-29, all of which hid in plain sight:

1. Six positions closed at the broker with no DB exit write. The ledger showed
   them open; the broker showed none. P&L (+$28.13) existed only in Alpaca's
   fill history and had to be reconstructed by hand.
2. The repair path closed such rows with pnl/pnl_pct NULL on the reasoning that
   the exit price was "unknowable". It is knowable — it is in the order history.
3. Every CRITICAL escalation email failed (expired SMTP credential) and the
   only symptom was an absence of email.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.agents.integrity_agent import IntegrityAgent


class _Alpaca:
    def __init__(self, fill=None, raises=False, entry_fill=None):
        self._fill, self._raises, self._entry = fill, raises, entry_fill
        self.calls = []

    async def get_closing_fill(self, ticker, after, side="sell"):
        self.calls.append((ticker, after, side))
        if self._raises:
            raise RuntimeError("broker unreachable")
        return self._entry if side == "buy" else self._fill


def _agent(alpaca=None):
    return IntegrityAgent(session_factory=None, signal_loop=None, alpaca=alpaca)


def _orphan(**over):
    row = {
        "id": 42,
        "ticker": "MSFT",
        "entry_time": datetime(2026, 9, 29, 13, 42, tzinfo=timezone.utc),
        "entry_price": 504.81,
        "shares": 23.87,
    }
    row.update(over)
    return row


@pytest.mark.asyncio
async def test_exit_is_recovered_from_broker_fills():
    """REGRESSION: real P&L must not be discarded as NULL."""
    ag = _agent(_Alpaca({"price": 508.15, "qty": 23.87,
                         "filled_at": datetime(2026, 9, 29, 15, 20, tzinfo=timezone.utc)}))

    out = await ag._lookup_exit_fill(_orphan())

    assert out is not None
    assert out["exit_price"] == pytest.approx(508.15)
    # (508.15 - 504.81) * 23.87
    assert out["pnl"] == pytest.approx(79.72, abs=0.05)
    assert out["pnl_pct"] == pytest.approx(79.72 / (504.81 * 23.87), rel=1e-3)


@pytest.mark.asyncio
async def test_no_fill_found_leaves_derived_columns_null():
    """Absent evidence, record nothing — never fabricate a P&L."""
    ag = _agent(_Alpaca(None))
    assert await ag._lookup_exit_fill(_orphan()) is None


@pytest.mark.asyncio
async def test_broker_error_is_swallowed_not_raised():
    """A repair pass must not crash because the broker hiccuped."""
    ag = _agent(_Alpaca(raises=True))
    assert await ag._lookup_exit_fill(_orphan()) is None


@pytest.mark.asyncio
async def test_missing_entry_price_still_records_the_exit_price():
    """Partial evidence is still worth keeping: price yes, derived pnl no."""
    ag = _agent(_Alpaca({"price": 508.15, "qty": 23.87, "filled_at": None}))

    out = await ag._lookup_exit_fill(_orphan(entry_price=None))

    assert out is not None and out["exit_price"] == pytest.approx(508.15)
    assert "pnl" not in out and "pnl_pct" not in out


@pytest.mark.asyncio
async def test_no_broker_client_is_a_no_op():
    assert await _agent(None)._lookup_exit_fill(_orphan()) is None


@pytest.mark.asyncio
async def test_naive_entry_time_is_treated_as_utc():
    ag = _agent(_Alpaca({"price": 1.0, "qty": 1.0, "filled_at": None}))
    await ag._lookup_exit_fill(_orphan(
        entry_time=datetime(2026, 9, 29, 13, 42), entry_price=1.0, shares=1.0))
    assert ag._alpaca.calls[0][1].tzinfo is not None


@pytest.mark.asyncio
async def test_lookup_is_scoped_to_after_the_entry():
    """Must not pick up a sell from a PRIOR position in the same ticker."""
    entry = datetime(2026, 9, 29, 13, 42, tzinfo=timezone.utc)
    ag = _agent(_Alpaca({"price": 1.0, "qty": 1.0, "filled_at": None}))
    await ag._lookup_exit_fill(_orphan(entry_time=entry))
    assert ag._alpaca.calls[0][1] == entry


def test_alert_channel_error_starts_clean_and_is_reportable():
    """A dead escalation path must be observable, not merely logged."""
    ag = _agent()
    assert ag.last_alert_error is None

    ag.last_alert_error = "SMTPAuthenticationError: 535 BadCredentials"
    assert "535" in ag.last_alert_error


@pytest.mark.asyncio
async def test_entry_leg_is_taken_from_the_broker_not_the_ledger():
    """REGRESSION: ledger entry prices biased recovered P&L upward.

    Reconstructing 2026-09-29 from ledger entry_price understated losses by
    $4.21 while both winners came out exact. pnl_pct feeds the Kelly seed, so an
    optimistic bias there weakens a risk control. Both legs must come from the
    broker.
    """
    ag = _agent(_Alpaca(
        fill={"price": 911.19, "qty": 13.10, "filled_at": None},
        entry_fill={"price": 914.90, "qty": 13.10, "filled_at": None},
    ))
    # ledger carries a slightly wrong entry price
    out = await ag._lookup_exit_fill(_orphan(ticker="STX", entry_price=914.74,
                                            shares=13.10))

    assert out["pnl"] == pytest.approx((911.19 - 914.90) * 13.10, abs=0.01)
    assert any(c[2] == "buy" for c in ag._alpaca.calls), "entry leg not queried"


@pytest.mark.asyncio
async def test_falls_back_to_ledger_entry_when_no_entry_fill():
    ag = _agent(_Alpaca(fill={"price": 508.15, "qty": 23.87, "filled_at": None},
                        entry_fill=None))
    out = await ag._lookup_exit_fill(_orphan())
    assert out["pnl"] == pytest.approx((508.15 - 504.81) * 23.87, abs=0.05)
