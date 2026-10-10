"""Unit tests for compute_exit_stratification (H28).

All tests are pure-Python — no DB, no network, no signal_loop. Each test
encodes one invariant of the stratification logic so a future diff cannot
silently regress the calculation.

The final test (test_live_snapshot_oct9_2026) is a regression guard: it
encodes the Oct-9-2026 finding that the clean PF (ex integrity_broker_reconcile)
is < 1.0 while the overall PF is > 1.0, and that top-3 outliers represent
> 45 % of gross profit. Any code change that breaks this test is either a
genuine algorithmic regression or must update the numbers here.
"""

from __future__ import annotations

import pytest

from src.analysis.performance import compute_exit_stratification


# ── helpers ───────────────────────────────────────────────────────────────────

def _trade(
    pnl: float,
    exit_reason: str,
    ticker: str = "AAPL",
    entry_time: str = "2026-09-23T10:00:00+00:00",
    exit_time:  str = "2026-09-23T10:31:00+00:00",
) -> dict:
    return {
        "ticker": ticker,
        "pnl": pnl,
        "exit_reason": exit_reason,
        "entry_time": entry_time,
        "exit_time": exit_time,
    }


def _open(ticker: str = "SWKS") -> dict:
    return {
        "ticker": ticker,
        "pnl": None,
        "exit_reason": None,
        "entry_time": "2026-10-09T09:30:00+00:00",
        "exit_time": None,
    }


# ── test_empty_input ──────────────────────────────────────────────────────────

def test_empty_input_returns_zero_stats():
    result = compute_exit_stratification([])
    assert result["overall"]["n"] == 0
    assert result["overall"]["pf"] is None
    assert result["overall"]["win_rate"] is None
    assert result["n_open_skipped"] == 0
    assert result["n_exit_reasons"] == 0
    assert result["by_exit_reason"] == {}


# ── test_open_trades_skipped ──────────────────────────────────────────────────

def test_open_trades_are_excluded_from_stats():
    trades = [_trade(100.0, "stop_loss"), _open(), _open("NVDA")]
    result = compute_exit_stratification(trades)
    assert result["overall"]["n"] == 1
    assert result["n_open_skipped"] == 2


# ── test_profit_factor ────────────────────────────────────────────────────────

def test_profit_factor_gross_profit_over_gross_loss():
    trades = [
        _trade(300.0, "take_profit"),
        _trade(-100.0, "stop_loss"),
        _trade(-100.0, "stop_loss"),
    ]
    result = compute_exit_stratification(trades)
    overall = result["overall"]
    assert overall["gross_profit"] == 300.0
    assert overall["gross_loss"]   == 200.0
    assert overall["pf"] == pytest.approx(1.5, rel=1e-4)
    assert overall["net_pnl"] == pytest.approx(100.0, rel=1e-4)


def test_pf_is_none_when_no_losses():
    trades = [_trade(200.0, "take_profit"), _trade(50.0, "max_hold")]
    result = compute_exit_stratification(trades)
    assert result["overall"]["pf"] is None  # no denominator
    assert result["overall"]["gross_loss"] == 0.0


# ── test_win_rate ─────────────────────────────────────────────────────────────

def test_win_rate_is_fraction_of_winners():
    trades = [
        _trade(100.0, "max_hold"),
        _trade(-50.0, "stop_loss"),
        _trade(0.0,   "max_hold"),   # tie
    ]
    result = compute_exit_stratification(trades)
    overall = result["overall"]
    assert overall["wins"]   == 1
    assert overall["losses"] == 1
    assert overall["ties"]   == 1
    assert overall["win_rate"] == pytest.approx(1 / 3, rel=1e-4)


# ── test_by_exit_reason ───────────────────────────────────────────────────────

def test_by_exit_reason_separates_buckets():
    trades = [
        _trade(500.0, "integrity_broker_reconcile", ticker="COHR"),
        _trade(400.0, "integrity_broker_reconcile", ticker="MSTR"),
        _trade(50.0,  "take_profit",                ticker="LITE"),
        _trade(-80.0, "stop_loss",                  ticker="TER"),
    ]
    result = compute_exit_stratification(trades)
    by = result["by_exit_reason"]

    assert "integrity_broker_reconcile" in by
    assert by["integrity_broker_reconcile"]["n"] == 2
    assert by["integrity_broker_reconcile"]["gross_profit"] == 900.0
    assert by["integrity_broker_reconcile"]["losses"] == 0

    assert "stop_loss" in by
    assert by["stop_loss"]["wins"]   == 0
    assert by["stop_loss"]["losses"] == 1
    assert by["stop_loss"]["gross_loss"] == 80.0

    assert "take_profit" in by
    assert by["take_profit"]["n"] == 1


# ── test_clean_excludes_reconcile ─────────────────────────────────────────────

def test_clean_pf_excludes_integrity_broker_reconcile():
    trades = [
        # Three halt-recovery outliers — should NOT appear in clean bucket
        _trade(1451.0, "integrity_broker_reconcile", ticker="COHR"),
        _trade(1008.0, "integrity_broker_reconcile", ticker="MSTR"),
        _trade(968.0,  "integrity_broker_reconcile", ticker="LITE"),
        # Regular strategy exits
        _trade(-200.0, "max_hold",  ticker="NVDA"),
        _trade(-150.0, "max_hold",  ticker="AMD"),
        _trade(80.0,   "max_hold",  ticker="AAPL"),
    ]
    result = compute_exit_stratification(trades)
    clean = result["clean"]

    assert clean["n"] == 3
    assert clean["gross_profit"] == pytest.approx(80.0)
    assert clean["gross_loss"]   == pytest.approx(350.0)
    # pf is rounded to 3 decimal places: round(80/350, 3) = 0.229
    assert clean["pf"] == pytest.approx(0.229, abs=0.001)
    assert clean["net_pnl"] == pytest.approx(80.0 - 350.0, rel=1e-3)

    # Reconcile exits inflate overall PF above clean PF
    assert result["overall"]["pf"] > clean["pf"]


# ── test_strategy_only ────────────────────────────────────────────────────────

def test_strategy_only_includes_designed_exit_reasons():
    trades = [
        _trade(100.0, "take_profit"),
        _trade(-50.0, "stop_loss"),
        _trade(30.0,  "trailing_stop"),
        _trade(-20.0, "max_hold"),
        _trade(5.0,   "signal_reversal"),
        _trade(800.0, "integrity_broker_reconcile"),  # excluded
        _trade(10.0,  "unknown_exit"),                # excluded
    ]
    result = compute_exit_stratification(trades)
    strat = result["strategy_only"]
    assert strat["n"] == 5
    assert strat["gross_profit"] == pytest.approx(135.0)
    assert strat["gross_loss"]   == pytest.approx(70.0)
    assert strat["pf"] == pytest.approx(135.0 / 70.0, rel=1e-3)


# ── test_concentration ────────────────────────────────────────────────────────

def test_concentration_sorts_by_pnl_descending():
    trades = [
        _trade(1000.0, "integrity_broker_reconcile", ticker="COHR"),
        _trade(500.0,  "integrity_broker_reconcile", ticker="MSTR"),
        _trade(200.0,  "take_profit",                ticker="LITE"),
        _trade(50.0,   "take_profit",                ticker="AAPL"),
        _trade(-100.0, "stop_loss",                  ticker="TER"),
    ]
    result = compute_exit_stratification(trades, top_n_concentration=3)
    conc = result["concentration"]

    assert conc["top_n"] == 3
    assert conc["total_gross_profit"] == pytest.approx(1750.0)
    assert conc["top_n_gross_profit"] == pytest.approx(1700.0)
    expected_pct = 1700.0 / 1750.0 * 100
    assert conc["top_n_pct_of_gross"] == pytest.approx(expected_pct, rel=1e-2)
    assert len(conc["tickers"]) == 3
    # Largest winner first
    assert conc["tickers"][0]["ticker"] == "COHR"
    assert conc["tickers"][1]["ticker"] == "MSTR"
    assert conc["tickers"][2]["ticker"] == "LITE"


def test_concentration_when_fewer_wins_than_top_n():
    trades = [_trade(100.0, "max_hold"), _trade(-50.0, "stop_loss")]
    result = compute_exit_stratification(trades, top_n_concentration=5)
    conc = result["concentration"]
    # Only 1 winner — no error, tickers list is length 1
    assert len(conc["tickers"]) == 1
    assert conc["top_n_gross_profit"] == pytest.approx(100.0)


# ── test_hold_time ────────────────────────────────────────────────────────────

def test_avg_hold_minutes_computed_correctly():
    trades = [
        _trade(100.0, "max_hold",
               entry_time="2026-10-09T10:00:00+00:00",
               exit_time="2026-10-09T10:31:00+00:00"),   # 31 min
        _trade(-50.0, "max_hold",
               entry_time="2026-10-09T14:00:00+00:00",
               exit_time="2026-10-09T14:30:00+00:00"),   # 30 min
    ]
    result = compute_exit_stratification(trades)
    assert result["overall"]["avg_hold_minutes"] == pytest.approx(30.5, rel=1e-2)


def test_missing_entry_time_skipped_in_avg_hold():
    trades = [
        {"pnl": 50.0, "exit_reason": "max_hold",
         "entry_time": None,
         "exit_time": "2026-10-09T10:31:00+00:00"},
        _trade(50.0, "max_hold",
               entry_time="2026-10-09T14:00:00+00:00",
               exit_time="2026-10-09T14:30:00+00:00"),
    ]
    result = compute_exit_stratification(trades)
    # Only the second trade has valid hold → avg = 30 min
    assert result["overall"]["avg_hold_minutes"] == pytest.approx(30.0, rel=1e-2)


# ── test_exit_reasons_seen ────────────────────────────────────────────────────

def test_exit_reasons_seen_sorted():
    trades = [
        _trade(10.0,  "trailing_stop"),
        _trade(-5.0,  "stop_loss"),
        _trade(20.0,  "max_hold"),
    ]
    result = compute_exit_stratification(trades)
    assert result["exit_reasons_seen"] == sorted(["trailing_stop", "stop_loss", "max_hold"])
    assert result["n_exit_reasons"] == 3


# ── regression: Oct-9-2026 live data ─────────────────────────────────────────

def test_live_snapshot_oct9_2026_clean_pf_below_one():
    """Regression: Oct-9 2026 live data (n=99).

    Three integrity_broker_reconcile exits (COHR +$1,451, MSTR +$1,008,
    LITE +$968) accounted for ~48 % of gross profit. Excluding those, the
    clean PnL is negative. This test encodes that finding as a permanent
    guard so it remains visible even as more trades accumulate.

    Win/loss counts are approximated from the published summary:
      WR = 41.4 %, avg win ≈ $173, avg loss ≈ $108 on 96 regular exits.
    """
    reconcile_wins = [
        _trade(1451.0, "integrity_broker_reconcile", ticker="COHR"),
        _trade(1008.0, "integrity_broker_reconcile", ticker="MSTR"),
        _trade(968.0,  "integrity_broker_reconcile", ticker="LITE"),
    ]
    # 96 regular trades: 38W @ $97 avg, 57L @ $108 avg, 1 breakeven.
    # Regular avg-win is ~$97 (not $173 — the $173 figure is inflated by
    # the 3 reconcile outliers). Gross profit: 38×97=$3,686; gross loss:
    # 57×108=$6,156 → clean net ≈ -$2,470 (negative edge confirmed).
    regular = (
        [_trade(97.0,    "max_hold")] * 38
        + [_trade(-108.0, "max_hold")] * 57
        + [_trade(0.0,    "max_hold")] * 1
    )
    result = compute_exit_stratification(reconcile_wins + regular)

    # Overall: inflated by reconcile wins (3 + 38 + 57 + 1 = 99)
    assert result["overall"]["n"] == 99
    assert result["overall"]["pf"] is not None
    assert result["overall"]["pf"] > 1.0

    # Clean: reconcile wins removed → 96 trades, net negative
    clean = result["clean"]
    assert clean["n"] == 96  # 38 + 57 + 1
    assert clean["gross_profit"] < clean["gross_loss"]
    assert clean["pf"] < 1.0
    assert clean["net_pnl"] < 0

    # Concentration: top-3 reconcile wins > 45 % of overall gross profit
    conc = result["concentration"]
    assert conc["top_n_pct_of_gross"] is not None
    assert conc["top_n_pct_of_gross"] > 45.0
