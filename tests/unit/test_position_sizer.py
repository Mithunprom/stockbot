"""Unit tests for SmartPositionSizer — v0.3.3 bigger-position behavior."""

from __future__ import annotations

from src.execution.position_sizer import (
    SmartPositionSizer, _MAX_NOTIONAL_PCT, _KELLY_INACTIVE_MAX_PCT,
    _KELLY_WINDOW_MIN_TRADES,
)


def _size(ticker, dir_prob, pred_return, atr_pct, pv=98_000.0, heat=0.0,
          sector_notionals=None, kelly_n_trades=_KELLY_WINDOW_MIN_TRADES):
    """Default kelly_n_trades to the activation threshold so existing tests
    exercise full-size behaviour; pass a smaller value to test the H36 guard."""
    return SmartPositionSizer(mode="paper").compute(
        ticker=ticker, dir_prob=dir_prob, pred_return=pred_return,
        atr_pct=atr_pct, price=150.0, portfolio_value=pv,
        portfolio_heat=heat, sector_notionals=sector_notionals or {},
        kelly_fraction=0.0, kelly_n_trades=kelly_n_trades,
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
    """At/above the 75% heat ceiling the sizer returns None (no new entry)."""
    r = _size("AAPL", dir_prob=0.72, pred_return=0.009, atr_pct=0.0006, heat=0.80)
    assert r is None
    # 60-75% band: half size, not blocked
    r_half = _size("AAPL", dir_prob=0.72, pred_return=0.009, atr_pct=0.0006, heat=0.65)
    assert r_half is not None


# ── H36: Kelly window inactive guard ──────────────────────────────────────────

def test_h36_inactive_window_caps_at_5pct():
    """When kelly_n_trades < KELLY_WINDOW_MIN_TRADES, size is capped at 5%.

    Root cause of Aug-31 (n=12 entries, −$938) and Oct-16 (n=6, −$1 412)
    runaway batches: Kelly was inactive or freshly seeded and positions were
    full-sized.  Stage 3b must cap them at _KELLY_INACTIVE_MAX_PCT = 5%.
    """
    # High-conviction signal that would ordinarily reach ~10% (the notional cap)
    r = _size("AAPL", dir_prob=0.72, pred_return=0.009, atr_pct=0.0006,
              kelly_n_trades=0)
    assert r is not None
    assert r.size_pct <= _KELLY_INACTIVE_MAX_PCT + 0.001, (
        f"Expected ≤{_KELLY_INACTIVE_MAX_PCT:.0%} but got {r.size_pct:.2%}"
    )
    assert r.to_dict()["stages"]["3b_inactive_guard"] is True


def test_h36_boundary_at_min_trades_is_uncapped():
    """Exactly at KELLY_WINDOW_MIN_TRADES the guard does not apply."""
    r_boundary = _size("AAPL", dir_prob=0.72, pred_return=0.009, atr_pct=0.0006,
                       kelly_n_trades=_KELLY_WINDOW_MIN_TRADES)
    assert r_boundary is not None
    assert r_boundary.size_pct > _KELLY_INACTIVE_MAX_PCT, (
        "At the activation threshold size should exceed the 5% inactive cap"
    )
    assert r_boundary.to_dict()["stages"]["3b_inactive_guard"] is False


def test_h36_guard_applies_across_all_conviction_levels():
    """Inactive guard applies regardless of dir_prob conviction bucket."""
    for dp, pr in [(0.60, 0.004), (0.72, 0.009), (0.85, 0.015)]:
        r = _size("NVDA", dir_prob=dp, pred_return=pr, atr_pct=0.0008,
                  kelly_n_trades=5)
        assert r is not None
        assert r.size_pct <= _KELLY_INACTIVE_MAX_PCT + 0.001, (
            f"dir_prob={dp}: expected ≤5% but got {r.size_pct:.2%}"
        )


def test_h36_audit_trail_surfaced_in_stages_dict():
    """kelly_n_trades is visible in the sizing result for diagnostic purposes."""
    r = _size("MSFT", dir_prob=0.70, pred_return=0.006, atr_pct=0.0007,
              kelly_n_trades=3)
    assert r is not None
    d = r.to_dict()
    assert d["inputs"]["kelly_n_trades"] == 3
    assert d["stages"]["3b_inactive_guard"] is True
