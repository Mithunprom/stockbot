"""Unit tests for SmartPositionSizer — v0.3.3 bigger-position behavior."""

from __future__ import annotations

from src.execution.position_sizer import (
    SmartPositionSizer, _MAX_NOTIONAL_PCT,
)


def _size(ticker, dir_prob, pred_return, atr_pct, pv=98_000.0, heat=0.0,
          sector_notionals=None):
    return SmartPositionSizer(mode="paper").compute(
        ticker=ticker, dir_prob=dir_prob, pred_return=pred_return,
        atr_pct=atr_pct, price=150.0, portfolio_value=pv,
        portfolio_heat=heat, sector_notionals=sector_notionals or {},
        kelly_fraction=0.0,
    )


def test_strong_signal_hits_10pct_cap_large_account():
    """A strong signal on a big account targets ~10% per position."""
    r = _size("AAPL", dir_prob=0.72, pred_return=0.009, atr_pct=0.0006)
    assert r is not None
    assert abs(r.size_pct - _MAX_NOTIONAL_PCT) < 0.005   # ≈10%


def test_volatile_name_also_reaches_cap_but_with_wider_stops():
    """Volatile names still reach the cap; risk is controlled via wider exits,
    not (only) smaller size — the v0.3.3 design choice."""
    from src.agents.signal_loop import _atr_exits
    calm = _size("V", dir_prob=0.70, pred_return=0.008, atr_pct=0.0005)
    vol = _size("MU", dir_prob=0.70, pred_return=0.010, atr_pct=0.0021)
    assert calm is not None and vol is not None
    assert abs(vol.size_pct - _MAX_NOTIONAL_PCT) < 0.01
    # volatile name gets a wider stop + target than the calm name.
    # _atr_exits consumes a DAILY vol fraction; convert the 1m ATRs via the
    # sqrt(390) fallback exactly as _daily_vol_for does without daily bars.
    from src.agents.signal_loop import DAILY_VOL_SQRT_BARS
    sl_calm, _, tp_calm = _atr_exits(0.0005 * DAILY_VOL_SQRT_BARS)
    sl_vol, _, tp_vol = _atr_exits(0.0021 * DAILY_VOL_SQRT_BARS)
    assert sl_vol > sl_calm and tp_vol > tp_calm


def test_position_never_exceeds_breaker_cap():
    """Every sizing result stays under the 25% circuit-breaker position cap."""
    for dp, pr, atr in [(0.95, 0.02, 0.0004), (0.85, 0.015, 0.001),
                        (0.62, 0.006, 0.003)]:
        r = _size("AAPL", dir_prob=dp, pred_return=pr, atr_pct=atr)
        if r is not None:
            assert r.size_pct <= 0.25


def test_heat_ceiling_blocks_new_size():
    """At/above the heat ceiling the sizer returns None (no new entry)."""
    r = _size("AAPL", dir_prob=0.72, pred_return=0.009, atr_pct=0.0006, heat=0.65)
    assert r is None
