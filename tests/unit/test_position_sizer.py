"""Unit tests for SmartPositionSizer — v0.3.3 bigger-position behavior."""

from __future__ import annotations

from src.execution.position_sizer import (
    SmartPositionSizer, _MAX_NOTIONAL_PCT, _SECTOR_CAP_PCT, SECTOR_MAP,
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
    """At/above the 75% heat ceiling the sizer returns None (no new entry)."""
    r = _size("AAPL", dir_prob=0.72, pred_return=0.009, atr_pct=0.0006, heat=0.80)
    assert r is None
    # 60-75% band: half size, not blocked
    r_half = _size("AAPL", dir_prob=0.72, pred_return=0.009, atr_pct=0.0006, heat=0.65)
    assert r_half is not None


# ── H19 tests: SECTOR_MAP completeness (2026-08-20) ─────────────────────────
# Root event: INTC + KLAC entered 2026-08-19 as "other" (not in SECTOR_MAP)
# while MU + WDC already filled the "semis" cap. Result: 4 correlated
# semiconductor positions open simultaneously, net -$919.

def test_h19_semiconductor_missing_tickers_now_in_semis():
    """INTC, KLAC, AMAT, LRCX, TER, STX must be in 'semis' — not 'other'."""
    for ticker in ("INTC", "KLAC", "AMAT", "LRCX", "TER", "STX"):
        assert SECTOR_MAP.get(ticker) == "semis", (
            f"{ticker} maps to {SECTOR_MAP.get(ticker, 'other')!r}, expected 'semis'"
        )


def test_h19_intc_blocked_by_sector_cap_when_two_semis_deployed():
    """Sizer stage 4b: INTC must be blocked when 'semis' notional = 40% of portfolio.
    Before H19 fix, INTC was 'other' and stage 4b looked up sector_notionals['other']
    (empty), returning a full-sized position despite 2 semis already deployed."""
    pv = 98_000.0
    sector_notionals_full = {"semis": pv * _SECTOR_CAP_PCT}  # semis at cap
    result = SmartPositionSizer(mode="paper").compute(
        ticker="INTC", dir_prob=0.75, pred_return=0.009, atr_pct=0.0007,
        price=35.0, portfolio_value=pv, portfolio_heat=0.40,
        sector_notionals=sector_notionals_full, kelly_fraction=0.20,
    )
    assert result is None, "INTC must be blocked when semis notional is at sector cap"


def test_h19_seed_universe_equities_have_explicit_sector():
    """All equity tickers from the active seed universe must be in SECTOR_MAP.
    Any ticker not in the map falls into 'other', sharing a single cap slot with
    all other unmapped names — effectively no sector limit on the group."""
    equities = {
        # tech
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
        "ORCL", "CRM", "SNOW", "DDOG", "ARM", "AVGO",
        # growth
        "PLTR", "SMCI", "MSTR", "COIN", "HOOD", "UBER", "ABNB", "RIVN",
        # defense
        "LMT", "NOC", "RTX", "BA", "GD", "LHX", "KTOS", "HII", "LDOS",
        # finance
        "JPM", "GS", "V", "MA", "BAC", "BLK", "C", "WFC", "SCHW",
        # health
        "LLY", "NVO", "ABBV", "MRK", "PFE", "GILD", "AMGN", "REGN", "ISRG", "MRNA",
        # energy
        "XOM", "CVX", "OXY", "SLB", "COP", "PSX",
        # consumer
        "COST", "WMT", "HD", "NKE", "MCD", "SBUX", "TGT",
        # crypto-adjacent
        "MARA", "RIOT", "HUT", "CLSK",
    }
    missing = sorted(t for t in equities if t not in SECTOR_MAP)
    assert not missing, f"Tickers fall through to 'other' (no sector cap): {missing}"


def test_h19_defense_sector_cap_enforced():
    """LMT + RTX filling 'defense' at 40% should block a third defense entry (NOC)."""
    pv = 98_000.0
    sector_notionals_full = {"defense": pv * _SECTOR_CAP_PCT}
    result = SmartPositionSizer(mode="paper").compute(
        ticker="NOC", dir_prob=0.73, pred_return=0.008, atr_pct=0.0006,
        price=250.0, portfolio_value=pv, portfolio_heat=0.40,
        sector_notionals=sector_notionals_full, kelly_fraction=0.20,
    )
    assert result is None, "NOC must be blocked when defense sector cap is full"
