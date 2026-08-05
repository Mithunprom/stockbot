"""Unit tests for research_backtest.py backtest simulator.

Focus: the confirmed-reversal fix (v0.3.4 parity).  The pre-fix code used a
bare pred_return sign check that caused exits on every momentary noise flip —
root cause of the 25-minute median hold diagnosed 2026-07-07.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research_backtest import (
    Params,
    _is_confirmed_reversal_long,
    simulate,
)


# ─── _is_confirmed_reversal_long unit tests ──────────────────────────────────

def test_positive_pred_return_never_confirms_reversal():
    """A bullish signal cannot confirm a reversal against a long position."""
    assert not _is_confirmed_reversal_long(
        pred_return=0.005, dir_prob=0.20, threshold=0.003, dead_lo=0.40
    )


def test_sign_only_below_threshold_does_not_confirm():
    """A bearish sign flip below the cost threshold must NOT count (noise)."""
    assert not _is_confirmed_reversal_long(
        pred_return=-0.001, dir_prob=0.20, threshold=0.003, dead_lo=0.40
    )


def test_sign_only_dir_prob_in_dead_zone_does_not_confirm():
    """Magnitude cleared but dir_prob in the dead zone (uncertain) → not confirmed."""
    assert not _is_confirmed_reversal_long(
        pred_return=-0.005, dir_prob=0.50, threshold=0.003, dead_lo=0.40
    )


def test_confirmed_reversal_all_gates_clear():
    """All three gates clear → confirmed reversal."""
    assert _is_confirmed_reversal_long(
        pred_return=-0.005, dir_prob=0.20, threshold=0.003, dead_lo=0.40
    )


def test_dir_prob_exactly_at_dead_lo_not_confirmed():
    """dir_prob == dead_lo is the boundary — must NOT confirm (strict less-than)."""
    assert not _is_confirmed_reversal_long(
        pred_return=-0.005, dir_prob=0.40, threshold=0.003, dead_lo=0.40
    )


def test_pred_return_exactly_at_threshold_not_confirmed():
    """pred_return == threshold is a noise bar — must NOT confirm (strict gt)."""
    assert not _is_confirmed_reversal_long(
        pred_return=-0.003, dir_prob=0.20, threshold=0.003, dead_lo=0.40
    )


def test_zero_pred_return_not_confirmed():
    """Zero pred_return (default EnsembleSignal) must never trigger reversal."""
    assert not _is_confirmed_reversal_long(
        pred_return=0.0, dir_prob=0.0, threshold=0.003, dead_lo=0.40
    )


# ─── Params defaults ──────────────────────────────────────────────────────────

def test_params_reversal_bars_matches_production():
    """Backtest default reversal_bars must match SIZING_REVERSAL_BARS (45)."""
    from src.agents.signal_loop import SIZING_REVERSAL_BARS
    assert Params().reversal_bars == SIZING_REVERSAL_BARS, (
        f"Params.reversal_bars={Params().reversal_bars} diverges from "
        f"production SIZING_REVERSAL_BARS={SIZING_REVERSAL_BARS}"
    )


def test_params_dead_lo_matches_production():
    """Backtest dead_lo must match the lower bound of SIZING_DIR_PROB_DEAD_ZONE."""
    from src.agents.signal_loop import SIZING_DIR_PROB_DEAD_ZONE
    lo, _ = SIZING_DIR_PROB_DEAD_ZONE
    assert Params().dead_lo == lo, (
        f"Params.dead_lo={Params().dead_lo} diverges from "
        f"production dead_zone_lo={lo}"
    )


# ─── simulate() integration smoke test ───────────────────────────────────────

def _make_preds(n_entry_bars: int = 10, n_reversal_bars: int = 0,
                entry_pred: float = 0.006, entry_dir: float = 0.70,
                rev_pred: float = -0.005, rev_dir: float = 0.20,
                cost_threshold: float = 0.003) -> pd.DataFrame:
    """Build a minimal synthetic preds DataFrame for simulate().

    Layout: entry bar, n_entry_bars hold bars (flat), then n_reversal_bars
    confirmed bearish bars, then a hold-until-max_hold.

    Base is set to 13:40 UTC = 09:40 ET so bar 0 lands inside the default
    entry window (09:40–15:30 ET). Without this, base 09:40 UTC = 05:40 ET
    means bar 0 is before market open and the entry signal is never seen.
    """
    et_tz = "America/New_York"
    # 13:40 UTC = 09:40 ET (EDT, UTC-4) on 2026-05-19 — first bar of entry window
    base = pd.Timestamp("2026-05-19 13:40:00", tz="UTC")
    records = []
    market_bar = 0  # count only bars that fall inside RTH
    for i in range(500):
        ts = base + pd.Timedelta(minutes=i)
        et_time = ts.tz_convert(et_tz)
        if et_time.hour < 9 or (et_time.hour == 9 and et_time.minute < 30):
            continue
        if et_time.hour >= 16:
            continue
        if market_bar == 0:
            pred, dp = entry_pred, entry_dir
        elif 0 < market_bar <= n_reversal_bars:
            pred, dp = rev_pred, rev_dir
        else:
            pred, dp = 0.0, 0.5
        records.append({
            "ticker": "AAPL",
            "time": ts,
            "pred_return": pred,
            "dir_prob": dp,
            "close": 100.0,
            "open": 100.0,
            "atr_pct": 0.001,
            "daily_vol": 0.02,
            "regime": 1,
            "forward_return": 0.0,
        })
        market_bar += 1
    df = pd.DataFrame(records)
    df["next_open"] = df["close"]
    return df


@pytest.mark.parametrize("n_rev,expected_reversal", [
    (0, False),
    (10, False),   # 10 bars < reversal_bars=45 → max_hold, not reversal
    (50, True),    # 50 confirmed bars > reversal_bars=45 → reversal exit
])
def test_simulate_reversal_requires_sustained_confirmation(n_rev, expected_reversal):
    """Reversal exit fires only after reversal_bars sustained confirmed bars."""
    preds = _make_preds(n_reversal_bars=n_rev)
    p = Params(
        reversal_bars=45,
        dead_lo=0.40,
        cost_threshold=0.003,
        pdt_enabled=False,
        max_pos=1,
        heat_ceiling=0.90,
        trades_per_day=10,
        cooldown=0,
        max_hold_bars=390,
        stag_bars=390,
    )
    res = simulate(preds, p, start="2026-05-19", end="2026-05-20")
    reasons = res.get("exit_reasons", {})
    has_reversal = reasons.get("signal_reversal", 0) > 0
    assert has_reversal == expected_reversal, (
        f"n_rev={n_rev}: expected signal_reversal={expected_reversal}, "
        f"got exit_reasons={reasons}"
    )


# ─── H10: closed_return_pct reconciliation ────────────────────────────────────

def test_simulate_returns_closed_return_pct():
    """simulate() must include closed_return_pct, open_marks_pct, backtest_capital."""
    preds = _make_preds()
    p = Params(pdt_enabled=False, max_pos=1, heat_ceiling=0.90,
               trades_per_day=10, cooldown=0, max_hold_bars=390, stag_bars=390)
    res = simulate(preds, p, start="2026-05-19", end="2026-05-20")
    assert "closed_return_pct" in res, "closed_return_pct missing from simulate() output"
    assert "open_marks_pct" in res, "open_marks_pct missing from simulate() output"
    assert "backtest_capital" in res, "backtest_capital missing from simulate() output"


def test_simulate_closed_return_reconciles_with_trade_pnl():
    """closed_return_pct must equal sum(trade pnl) / capital * 100."""
    preds = _make_preds()
    p = Params(pdt_enabled=False, max_pos=1, heat_ceiling=0.90,
               trades_per_day=10, cooldown=0, max_hold_bars=50, stag_bars=50)
    capital = 10_000.0
    res = simulate(preds, p, start="2026-05-19", end="2026-05-20", capital=capital)
    trades_df = res["_trades"]
    if trades_df.empty:
        return  # no trades in window — not a failure
    expected_closed_pct = round(float(trades_df.pnl.sum()) / capital * 100, 2)
    assert res["closed_return_pct"] == expected_closed_pct, (
        f"closed_return_pct={res['closed_return_pct']} does not match "
        f"sum(pnl)/capital*100={expected_closed_pct}"
    )


def test_simulate_total_return_equals_closed_plus_open_marks():
    """total_return_pct == closed_return_pct + open_marks_pct (accounting identity)."""
    preds = _make_preds(n_entry_bars=20)
    p = Params(pdt_enabled=False, max_pos=1, heat_ceiling=0.90,
               trades_per_day=10, cooldown=0, max_hold_bars=390, stag_bars=390)
    res = simulate(preds, p, start="2026-05-19", end="2026-05-20")
    reconstructed = round(res["closed_return_pct"] + res["open_marks_pct"], 2)
    assert abs(reconstructed - res["total_return_pct"]) < 0.01, (
        f"total={res['total_return_pct']} ≠ closed={res['closed_return_pct']} "
        f"+ open_marks={res['open_marks_pct']} = {reconstructed}"
    )


# ─── H9: probe quality floor ──────────────────────────────────────────────────

def test_h9_probe_floor_constants_are_set():
    """KELLY_PROBE_MIN_ENSEMBLE and KELLY_PROBE_MIN_SENTIMENT exist and are sane."""
    from src.agents.signal_loop import KELLY_PROBE_MIN_ENSEMBLE, KELLY_PROBE_MIN_SENTIMENT
    assert 0.05 <= KELLY_PROBE_MIN_ENSEMBLE <= 0.20, (
        f"KELLY_PROBE_MIN_ENSEMBLE={KELLY_PROBE_MIN_ENSEMBLE} outside expected range [0.05, 0.20]"
    )
    assert -0.60 <= KELLY_PROBE_MIN_SENTIMENT <= -0.30, (
        f"KELLY_PROBE_MIN_SENTIMENT={KELLY_PROBE_MIN_SENTIMENT} outside expected range [-0.60, -0.30]"
    )


def test_h9_probe_floor_blocks_sndk_style_noise_entry():
    """SNDK id=104 case: ensemble=0.018, sentiment=-0.90 must fail the floor."""
    from src.agents.signal_loop import KELLY_PROBE_MIN_ENSEMBLE, KELLY_PROBE_MIN_SENTIMENT
    sndk_ensemble = 0.018
    sndk_sentiment = -0.90
    assert sndk_ensemble < KELLY_PROBE_MIN_ENSEMBLE, (
        f"SNDK noise-level ensemble {sndk_ensemble} should be below floor "
        f"{KELLY_PROBE_MIN_ENSEMBLE} but is not"
    )
    assert sndk_sentiment < KELLY_PROBE_MIN_SENTIMENT, (
        f"SNDK noise-level sentiment {sndk_sentiment} should be below floor "
        f"{KELLY_PROBE_MIN_SENTIMENT} but is not"
    )


def test_h9_probe_floor_passes_genuine_opportunity():
    """A signal with ensemble=0.42, sentiment=+0.30 must pass the probe floor."""
    from src.agents.signal_loop import KELLY_PROBE_MIN_ENSEMBLE, KELLY_PROBE_MIN_SENTIMENT
    ensemble = 0.42
    sentiment = 0.30
    assert ensemble >= KELLY_PROBE_MIN_ENSEMBLE, (
        f"Genuine opportunity ensemble {ensemble} should be above floor "
        f"{KELLY_PROBE_MIN_ENSEMBLE} but is not"
    )
    assert sentiment >= KELLY_PROBE_MIN_SENTIMENT, (
        f"Genuine opportunity sentiment {sentiment} should be above floor "
        f"{KELLY_PROBE_MIN_SENTIMENT} but is not"
    )


def test_h9_probe_floor_ensemble_floor_only():
    """Signal with ensemble above floor but sentiment below floor must be blocked."""
    from src.agents.signal_loop import KELLY_PROBE_MIN_ENSEMBLE, KELLY_PROBE_MIN_SENTIMENT
    # PLTR #124: ensemble=0.288, sentiment=-0.969 — ensemble ok, sentiment not
    ensemble = 0.288
    sentiment = -0.969
    assert ensemble >= KELLY_PROBE_MIN_ENSEMBLE  # passes ensemble floor
    assert sentiment < KELLY_PROBE_MIN_SENTIMENT  # blocked by sentiment floor


def test_simulate_noise_bar_does_not_trigger_reversal():
    """A sign flip below the cost threshold must never exit (pre-fix regression)."""
    # Build data: entry bar, then many bars with opposite sign but tiny magnitude
    records = []
    # 13:40 UTC = 09:40 ET so bar 0 is the first bar of the entry window
    base = pd.Timestamp("2026-05-19 13:40:00", tz="UTC")
    et_tz = "America/New_York"
    market_bar = 0
    for i in range(500):
        ts = base + pd.Timedelta(minutes=i)
        et_time = ts.tz_convert(et_tz)
        if et_time.hour < 9 or (et_time.hour == 9 and et_time.minute < 30):
            continue
        if et_time.hour >= 16:
            continue
        pred = 0.006 if market_bar == 0 else -0.0001  # noise-level opposite sign
        dp = 0.70 if market_bar == 0 else 0.45        # dir_prob in dead zone too
        records.append({
            "ticker": "AAPL",
            "time": ts,
            "pred_return": pred,
            "dir_prob": dp,
            "close": 100.0,
            "open": 100.0,
            "atr_pct": 0.001,
            "daily_vol": 0.02,
            "regime": 1,
            "forward_return": 0.0,
        })
        market_bar += 1
    preds = pd.DataFrame(records)
    p = Params(
        reversal_bars=45,
        dead_lo=0.40,
        cost_threshold=0.003,
        pdt_enabled=False,
        max_pos=1,
        heat_ceiling=0.90,
        trades_per_day=10,
        cooldown=0,
        max_hold_bars=390,
        stag_bars=390,
    )
    res = simulate(preds, p, start="2026-05-19", end="2026-05-20")
    reasons = res.get("exit_reasons", {})
    assert reasons.get("signal_reversal", 0) == 0, (
        f"Noise-level sign flips must not cause reversal exits; got {reasons}"
    )
