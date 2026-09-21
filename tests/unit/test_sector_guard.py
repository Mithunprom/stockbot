"""Regression tests for the fail-closed correlation guard.

Every test here fails against the pre-2026-09-15 code, where
`SECTOR_MAP.get(ticker, "other")` dropped 35 of the 57 traded tickers into one
shared permissive bucket and silently defeated `MAX_POSITIONS_PER_SECTOR`.

The guard decides how much correlated capital may be at risk simultaneously,
so it is position-handling code and is covered accordingly.
"""

from __future__ import annotations

import pytest

from src.execution.position_sizer import SECTOR_MAP, SmartPositionSizer, _SECTOR_CAP_PCT

# Symbols added by the fail-closed fix are imported INSIDE the tests that need
# them. The behavioral regression tests below deliberately reference only
# pre-existing symbols, so that on the pre-fix code they still collect and then
# FAIL on behavior rather than erroring at import — which is what makes them
# regression tests rather than compile checks.
UNMAPPED_SECTOR = "unmapped"

# The four simultaneous semiconductor longs of 2026-08-19, which the old guard
# counted as "2 semis + 2 other" and waved through: -$1,237 in 31 minutes.
AUG_19_BASKET = ["KLAC", "INTC", "MU", "WDC"]
# The 2026-09-11 repeat: -$621.
SEP_11_BASKET = ["SNDK", "LITE", "LRCX", "AMD"]


# ─── Resolver ───────────────────────────────────────────────────────────────

def test_aug_19_basket_all_resolves_to_semis():
    """KLAC/INTC/MU/WDC are one correlation bucket, not two plus 'other'."""
    assert [SECTOR_MAP.get(t) for t in AUG_19_BASKET] == ["semis"] * 4


def test_sep_11_basket_all_resolves_to_semis():
    """Photonics and storage sit in the semis cycle, not a bucket of their own."""
    assert [SECTOR_MAP.get(t) for t in SEP_11_BASKET] == ["semis"] * 4


def test_every_ticker_traded_in_m2_is_mapped():
    """The names that produced the concentration episodes are now classified."""
    from src.execution.position_sizer import sector_of

    traded = AUG_19_BASKET + SEP_11_BASKET + [
        "AMAT", "QCOM", "MRVL", "COHR", "CIEN", "STX", "TER", "FLEX",
        "GS", "WFC", "MSCI", "HOOD", "ORCL", "NOW", "SNOW", "DDOG",
        "WDAY", "TTD", "ANET", "DELL", "JNJ", "PFE", "ABBV", "MRNA",
        "LMT", "LDOS", "LII", "ZBRA", "COP", "APTV", "GRMN",
    ]
    unmapped = [t for t in traded if sector_of(t) == UNMAPPED_SECTOR]
    assert unmapped == []


def test_unknown_ticker_never_lands_in_a_shared_permissive_bucket():
    """The old 'other' default is gone; unknowns are explicitly unmapped."""
    from src.execution.position_sizer import sector_of

    assert sector_of("ZZZZ") == UNMAPPED_SECTOR
    assert sector_of("ZZZZ") != "other"
    assert "ZZZZ" not in SECTOR_MAP


def test_sector_of_is_case_insensitive():
    from src.execution.position_sizer import sector_of

    assert sector_of("klac") == "semis"


# ─── Caps ───────────────────────────────────────────────────────────────────

def test_unmapped_bucket_is_never_treated_as_diversifying():
    """An unknown name gets no diversification credit — strictest caps apply.

    This is the core fail-closed property: whatever an unrecognized ticker is,
    the system must assume it is correlated with every other unrecognized
    ticker rather than assume it is not.
    """
    from src.execution.position_sizer import (
        MAX_POSITIONS_UNMAPPED, max_positions_for_sector, sector_cap_pct,
    )

    assert max_positions_for_sector(UNMAPPED_SECTOR) == MAX_POSITIONS_UNMAPPED
    assert (max_positions_for_sector(UNMAPPED_SECTOR)
            < max_positions_for_sector("semis"))
    assert sector_cap_pct(UNMAPPED_SECTOR) < sector_cap_pct("semis")


def test_known_sector_keeps_its_existing_caps():
    """The fix tightens the unknown case; it must not loosen the known one."""
    from src.execution.position_sizer import (
        MAX_POSITIONS_PER_SECTOR_DEFAULT, _UNMAPPED_SECTOR_CAP_PCT,
        max_positions_for_sector, sector_cap_pct,
    )

    assert max_positions_for_sector("semis") == MAX_POSITIONS_PER_SECTOR_DEFAULT
    assert sector_cap_pct("semis") == _SECTOR_CAP_PCT
    assert sector_cap_pct(UNMAPPED_SECTOR) == _UNMAPPED_SECTOR_CAP_PCT


# ─── Notional guard (_SECTOR_CAP_PCT path) ─────────────────────────────────

def _size(ticker: str, sector_notionals: dict[str, float] | None = None):
    return SmartPositionSizer(mode="paper").compute(
        ticker=ticker, dir_prob=0.70, pred_return=0.009, atr_pct=0.0006,
        price=100.0, portfolio_value=100_000.0, portfolio_heat=0.0,
        sector_notionals=sector_notionals or {}, kelly_fraction=0.0,
    )


def test_notional_guard_charges_unmapped_names_to_the_unmapped_bucket():
    """An unmapped ticker's size is drawn from the unmapped bucket's budget."""
    from src.execution.position_sizer import _UNMAPPED_SECTOR_CAP_PCT

    result = _size("ZZZZ")
    assert result is not None
    assert result.size_pct <= _UNMAPPED_SECTOR_CAP_PCT + 1e-9


def test_notional_guard_blocks_a_second_unmapped_name():
    """Once the unmapped bucket is full, another unknown ticker is refused.

    Under the old code both names hit the shared 'other' bucket at a 40% cap,
    so four unknown correlated longs could be opened before anything bound.
    """
    from src.execution.position_sizer import _UNMAPPED_SECTOR_CAP_PCT

    full = {UNMAPPED_SECTOR: _UNMAPPED_SECTOR_CAP_PCT * 100_000.0}
    assert _size("ZZZZ", full) is None


def test_notional_guard_does_not_leak_across_buckets():
    """A full unmapped bucket must not constrain a known sector."""
    from src.execution.position_sizer import _UNMAPPED_SECTOR_CAP_PCT

    full = {UNMAPPED_SECTOR: _UNMAPPED_SECTOR_CAP_PCT * 100_000.0}
    assert _size("NVDA", full) is not None


def test_semis_notional_bucket_now_includes_the_formerly_unmapped_names():
    """KLAC notional counts against the same budget as NVDA notional."""
    nearly_full = {"semis": _SECTOR_CAP_PCT * 100_000.0}
    assert _size("KLAC", nearly_full) is None
    assert _size("LRCX", nearly_full) is None


# ─── Position-count guard (signal loop entry gate) ─────────────────────────

def _loop_with_positions(tickers: list[str]):
    """Build a SignalLoop holding one open long in each given ticker."""
    from unittest.mock import MagicMock

    from src.agents.signal_loop import SignalLoop
    from src.execution.position_manager import PositionManager
    from src.risk.circuit_breakers import CircuitBreakers

    loop = SignalLoop(
        universe=list(tickers),
        ensemble=MagicMock(),
        alpaca=MagicMock(),
        circuit_breakers=CircuitBreakers(),
        pos_manager=PositionManager(initial_portfolio=100_000.0),
        session_factory=MagicMock(),
        feature_cols=[f"feat_{i}" for i in range(30)],
    )
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    for ticker in tickers:
        loop._pm.open_position(ticker, "long", 10, 100.0)
    return loop


def _signal(ticker: str):
    from src.models.ensemble import EnsembleSignal

    sig = EnsembleSignal(ticker=ticker, timestamp=None)
    sig.lgbm_pred_return = 0.009
    sig.lgbm_dir_prob = 0.65
    return sig


def test_aug_19_basket_cannot_all_be_opened():
    """REGRESSION: the fourth leg of the 2026-08-19 semi basket is refused.

    Fails on the pre-fix code, where KLAC/INTC counted as 'other' and the
    guard saw only two semis.
    """
    loop = _loop_with_positions(AUG_19_BASKET[:2])   # KLAC, INTC
    assert loop._sector_position_count("MU") == 2
    assert not loop._sizing_entry_gate_open(_signal("MU"))
    assert not loop._sizing_entry_gate_open(_signal("WDC"))


def test_sep_11_basket_cannot_all_be_opened():
    """REGRESSION: the 2026-09-11 repeat is refused for the same reason."""
    loop = _loop_with_positions(SEP_11_BASKET[:2])   # SNDK, LITE
    assert not loop._sizing_entry_gate_open(_signal("LRCX"))
    assert not loop._sizing_entry_gate_open(_signal("AMD"))


def test_second_unmapped_ticker_is_blocked_by_the_count_guard():
    """One unknown name at a time — no diversification credit for unknowns."""
    loop = _loop_with_positions(["ZZZZ"])
    assert loop._sector_position_count("YYYY") == 1
    assert not loop._sizing_entry_gate_open(_signal("YYYY"))


def test_unmapped_position_does_not_block_a_known_sector():
    """The unmapped bucket is strict, not global."""
    loop = _loop_with_positions(["ZZZZ"])
    assert loop._sizing_entry_gate_open(_signal("XOM"))


def test_sector_notionals_report_the_fail_closed_buckets():
    """Diagnostics must show the real buckets, not a catch-all."""
    loop = _loop_with_positions(["KLAC", "ZZZZ"])
    notionals = loop._compute_sector_notionals()
    assert notionals["semis"] == pytest.approx(1000.0)
    assert notionals[UNMAPPED_SECTOR] == pytest.approx(1000.0)
    assert "other" not in notionals
