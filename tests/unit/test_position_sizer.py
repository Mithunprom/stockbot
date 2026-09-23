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
    """At/above the 75% heat ceiling the sizer returns None (no new entry)."""
    r = _size("AAPL", dir_prob=0.72, pred_return=0.009, atr_pct=0.0006, heat=0.80)
    assert r is None
    # 60-75% band: half size, not blocked
    r_half = _size("AAPL", dir_prob=0.72, pred_return=0.009, atr_pct=0.0006, heat=0.65)
    assert r_half is not None


# ── H27: SECTOR_MAP completeness (freeze-exempt correctness fix) ─────────────

def test_h27_live_universe_tickers_are_mapped():
    """Every ticker confirmed in the live trade ledger (Aug–Sep 2026) must be in
    SECTOR_MAP so it is subject to sector heat caps instead of landing in
    UNMAPPED_SECTOR.  Root: PCG, INTC, LITE, GS, COIN all appeared in
    live diagnostics/ledger without a sector assignment.
    Note: CIEN is in 'semis' per the fail-closed rewrite (photonics/optical
    shares the semiconductor demand cycle intraday).
    """
    from src.execution.position_sizer import SECTOR_MAP

    required = {
        # New additions in H27
        "INTC": "semis",
        "LITE": "semis",
        "KLAC": "semis",
        "LRCX": "semis",
        "AMAT": "semis",
        "GS": "financials",
        "COIN": "financials",
        "MSCI": "financials",
        "CIEN": "semis",  # optical networking — semis demand cycle per fail-closed rewrite
        "WDAY": "tech",
        "CRM": "tech",
        "MNST": "consumer",   # H22
        "PCG": "utilities",
        "RTX": "industrials",
        # Pre-existing entries (regression check)
        "AAPL": "tech",
        "NVDA": "semis",
        "MU": "semis",
        "JPM": "financials",
        "NFLX": "consumer",
        "XOM": "energy",
        "LLY": "healthcare",
    }
    for ticker, expected_sector in required.items():
        assert ticker in SECTOR_MAP, f"{ticker} missing from SECTOR_MAP — bypasses sector cap"
        assert SECTOR_MAP[ticker] == expected_sector, (
            f"{ticker}: expected sector '{expected_sector}', got '{SECTOR_MAP[ticker]}'"
        )


def test_h27_pcg_not_unmapped():
    """PCG (utilities) must resolve to 'utilities', not UNMAPPED_SECTOR.
    Sep-17 diagnostics confirmed PCG qualifies for entry and would have fired
    without sector protection if Kelly exits probation.
    """
    from src.execution.position_sizer import SECTOR_MAP, UNMAPPED_SECTOR
    assert SECTOR_MAP.get("PCG") == "utilities"
    assert SECTOR_MAP.get("PCG") != UNMAPPED_SECTOR


def test_h27_no_unmapped_recent_tickers_bypass_cap():
    """Verify that sizing PCG/INTC/LITE uses their correct sector, not UNMAPPED_SECTOR.
    A sector_notionals dict with 40% in each of their sectors should block entry
    (sector_cap = 40%), confirming they are subject to the cap.
    """
    from src.execution.position_sizer import SECTOR_MAP

    for ticker, sector in [("PCG", "utilities"), ("INTC", "semis"), ("LITE", "semis")]:
        assert SECTOR_MAP.get(ticker) == sector
        # At the 40% sector cap the sizer must return None for this ticker
        r = _size(
            ticker,
            dir_prob=0.78,
            pred_return=0.006,
            atr_pct=0.001,
            sector_notionals={sector: 39_200.0},  # 40% of 98k = exactly at cap
        )
        assert r is None, (
            f"{ticker} returned a sizing result even though '{sector}' sector is at cap"
        )
