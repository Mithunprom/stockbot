"""Unit tests for H14 — exit barriers scaled to the holding window.

The defect these guard against: `_atr_exits` derived barriers from DAILY sigma
while `SIZING_MAX_HOLD_BARS` capped the trade at 30 bars. Volatility scales
with sqrt(time), so every barrier was inflated by 1/sqrt(30/390) = 3.6x and
none could be reached — 111 of 111 v0.6.0 trades exited on `max_hold` and zero
on any barrier.

These functions decide when a position is closed, so they are money-handling
code and are covered as such.
"""

from __future__ import annotations

import math

import pytest

from src.agents.signal_loop import (
    CATASTROPHIC_STOP_MULT,
    DAILY_VOL_CEIL,
    DAILY_VOL_FLOOR,
    SIZING_MAX_HOLD_BARS,
    SIZING_STOP_LOSS_CAP,
    SIZING_STOP_LOSS_FLOOR,
    SIZING_STOP_LOSS_HVOL_MULT,
    SIZING_TAKE_PROFIT_CAP,
    SIZING_TAKE_PROFIT_FLOOR,
    SIZING_TAKE_PROFIT_HVOL_MULT,
    SIZING_TRAILING_HVOL_MULT,
    _BARS_PER_SESSION,
    _atr_exits,
    _hold_window_vol,
)

HOLD_SCALE = math.sqrt(SIZING_MAX_HOLD_BARS / _BARS_PER_SESSION)


# ─── The conversion itself ──────────────────────────────────────────────────

def test_hold_window_vol_applies_sqrt_of_time():
    """A 30-bar window sees sqrt(30/390) = 0.277 of a session's sigma."""
    assert _hold_window_vol(0.04, 30) == pytest.approx(0.04 * math.sqrt(30 / 390))
    assert _hold_window_vol(0.04, 390) == pytest.approx(0.04)


def test_hold_window_vol_is_monotonic_in_time():
    """Longer holds can make larger moves."""
    vols = [_hold_window_vol(0.03, b) for b in (5, 30, 120, 390)]
    assert vols == sorted(vols)


def test_hold_window_vol_clamps_the_daily_input():
    """A degenerate vol read cannot produce a degenerate barrier."""
    assert _hold_window_vol(0.0, 30) == pytest.approx(DAILY_VOL_FLOOR * HOLD_SCALE)
    assert _hold_window_vol(9.9, 30) == pytest.approx(DAILY_VOL_CEIL * HOLD_SCALE)


def test_hold_window_vol_handles_a_zero_hold():
    """Guards against a divide-by-zero / negative sqrt if the cap is ever 0."""
    assert _hold_window_vol(0.03, 0) == 0.0
    assert _hold_window_vol(0.03, -5) == 0.0


# ─── Barriers are reachable, which is the whole point ──────────────────────

def test_stop_is_reachable_within_the_hold_window():
    """REGRESSION: the pre-fix stop was 3.97 sigma of a 30-bar hold — dead code.

    Fails on the old `_atr_exits`, which returned max(0.02*1.1, 0.010) = 2.2%
    for this input: 3.97 sigma of the window, a barrier no ordinary trade
    reaches.
    """
    daily_vol = 0.02
    stop, _trail, _tp = _atr_exits(daily_vol)
    sigma = _hold_window_vol(daily_vol, SIZING_MAX_HOLD_BARS)
    assert stop / sigma == pytest.approx(SIZING_STOP_LOSS_HVOL_MULT, rel=1e-6)
    assert stop / sigma < 3.0, "stop is still too wide to ever fire"


def test_every_barrier_is_within_reach_across_the_vol_range():
    """No barrier may exceed 4 sigma of the hold window at any vol."""
    for daily_vol in (0.005, 0.01, 0.02, 0.04, 0.08, 0.15):
        stop, trail, take_profit = _atr_exits(daily_vol)
        sigma = _hold_window_vol(daily_vol, SIZING_MAX_HOLD_BARS)
        for name, barrier in (("stop", stop), ("trail", trail), ("tp", take_profit)):
            assert barrier / sigma <= 4.0, (
                f"{name} is {barrier / sigma:.2f} sigma at daily_vol={daily_vol} "
                f"— unreachable inside a {SIZING_MAX_HOLD_BARS}-bar hold"
            )


def test_floors_do_not_resurrect_the_daily_regime():
    """REGRESSION: the floors were the worse half of the bug.

    At the 0.5% daily-vol floor the hold-window sigma is ~0.139%. The old 1.0%
    stop floor was therefore a ~7 sigma stop for exactly the calm names most
    likely to clamp. DAILY_VOL_FLOOR must now bind first, leaving the barrier
    floors as an inactive safety net.
    """
    stop, _trail, take_profit = _atr_exits(DAILY_VOL_FLOOR)
    sigma = _hold_window_vol(DAILY_VOL_FLOOR, SIZING_MAX_HOLD_BARS)
    assert stop / sigma == pytest.approx(SIZING_STOP_LOSS_HVOL_MULT, rel=1e-6), (
        "the stop floor is clamping at the daily-vol floor — it is active again"
    )
    assert take_profit / sigma == pytest.approx(
        SIZING_TAKE_PROFIT_HVOL_MULT, rel=1e-6
    )
    assert SIZING_STOP_LOSS_FLOOR < 0.01, "stop floor still on the daily scale"
    assert SIZING_TAKE_PROFIT_FLOOR < 0.015, "TP floor still on the daily scale"


def test_barriers_are_strictly_tighter_than_the_pre_fix_geometry():
    """The fix may only make the stop MORE binding, never less.

    Reproduces the old formula inline and asserts the new stop is never wider.
    """
    for daily_vol in (0.005, 0.01, 0.02, 0.05, 0.10, 0.15):
        clamped = min(max(daily_vol, DAILY_VOL_FLOOR), DAILY_VOL_CEIL)
        old_stop = min(max(clamped * 1.1, 0.010), SIZING_STOP_LOSS_CAP)
        new_stop, _trail, _tp = _atr_exits(daily_vol)
        assert new_stop <= old_stop + 1e-12, (
            f"stop LOOSENED at daily_vol={daily_vol}: {old_stop} -> {new_stop}"
        )


# ─── Geometry invariants ────────────────────────────────────────────────────

def test_reward_exceeds_risk_at_every_volatility():
    """A take-profit inside the stop loses money at any win rate below 50%."""
    for daily_vol in (0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30):
        stop, _trail, take_profit = _atr_exits(daily_vol)
        assert take_profit > stop, f"R:R inverted at daily_vol={daily_vol}"


def test_trailing_is_never_tighter_than_the_stop():
    """On a monotone adverse path the disaster stop must fire first.

    The ledger has no intra-hold peak, so the trailing stop's fire rate is
    unmeasurable. This ordering is what keeps it from quietly pre-empting the
    stop that WAS calibrated on evidence.
    """
    assert SIZING_TRAILING_HVOL_MULT >= SIZING_STOP_LOSS_HVOL_MULT
    for daily_vol in (0.005, 0.02, 0.08, 0.15):
        stop, trail, _tp = _atr_exits(daily_vol)
        assert trail >= stop - 1e-12


def test_barriers_are_monotonic_in_volatility():
    """A more volatile name gets proportionally more room, never less."""
    prev = (0.0, 0.0, 0.0)
    for daily_vol in (0.005, 0.01, 0.02, 0.04, 0.08, 0.15):
        current = _atr_exits(daily_vol)
        assert all(c >= p - 1e-12 for c, p in zip(current, prev))
        prev = current


def test_take_profit_cap_stays_above_the_stop_cap():
    """At the high-vol end the clamps must not invert the geometry."""
    assert SIZING_TAKE_PROFIT_CAP > SIZING_STOP_LOSS_CAP


def test_atr_exits_honours_an_explicit_hold_window():
    """A longer hold widens the barriers; the default tracks the live cap."""
    short = _atr_exits(0.04, hold_bars=30)
    long = _atr_exits(0.04, hold_bars=390)
    assert all(l > s for l, s in zip(long, short))
    assert _atr_exits(0.04) == _atr_exits(0.04, hold_bars=SIZING_MAX_HOLD_BARS)


# ─── Catastrophic stop becomes live ────────────────────────────────────────

def test_catastrophic_threshold_is_reachable_but_wider_than_the_stop():
    """H14 turns this path from dead code into live behavior — deliberately.

    It must stay strictly wider than the normal stop (or it stops meaning
    "materially worse"), and stay inside 4 sigma (or it is decorative again).
    """
    assert CATASTROPHIC_STOP_MULT > 1.0
    effective = SIZING_STOP_LOSS_HVOL_MULT * CATASTROPHIC_STOP_MULT
    assert effective > SIZING_STOP_LOSS_HVOL_MULT
    assert effective <= 4.0, "catastrophic override is unreachable again"


def test_catastrophic_threshold_only_ever_tightens_the_pdt_path():
    """It may bring an exit forward, never defer one.

    The override relabels a deferred exit as `stop_loss` so it may spend a PDT
    day-trade. A smaller multiple can only make that happen sooner.
    """
    assert CATASTROPHIC_STOP_MULT <= 2.0, (
        "raising this above the previous 2.0 would DELAY an emergency exit"
    )


# ─── End-to-end through the exit checker ───────────────────────────────────

def _loop_with_open_long(daily_vol: float, entry_price: float = 100.0):
    """A SignalLoop holding one long that has not yet reached max_hold."""
    from unittest.mock import MagicMock

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
        feature_cols=[f"feat_{i}" for i in range(30)],
    )
    loop._pm.portfolio_value = 100_000.0
    loop._entry_prices["AAPL"] = entry_price
    loop._entry_directions["AAPL"] = 1
    loop._peak_prices["AAPL"] = entry_price
    loop._bars_held["AAPL"] = 1          # well short of max_hold
    loop._ticker_daily_vol["AAPL"] = daily_vol
    return loop


def _flat_signal():
    from src.models.ensemble import EnsembleSignal

    sig = EnsembleSignal(ticker="AAPL", timestamp=None)
    sig.lgbm_pred_return = 0.0
    sig.lgbm_dir_prob = 0.5
    return sig


def test_stop_loss_now_actually_fires():
    """REGRESSION: with the pre-fix barriers this returned None.

    Old stop at daily_vol=2% was 2.2%; a -1.5% move sat inside it and the
    position simply ran to the timer. New stop is 1.11%, so the move trips it.
    """
    loop = _loop_with_open_long(0.02)
    assert loop._check_sizing_exit("AAPL", 98.5, _flat_signal()) == "stop_loss"


def test_normal_wander_still_does_not_stop_the_trade():
    """The stop must not truncate the 15-30 minute edge v0.6.0 protects.

    -0.58% is the median absolute realized move of the M2 book; a disaster stop
    that fires there would be a trading stop.
    """
    loop = _loop_with_open_long(0.02)
    assert loop._check_sizing_exit("AAPL", 99.42, _flat_signal()) is None


def test_take_profit_now_actually_fires():
    """The TP is attainable inside the window instead of a 5.41 sigma fantasy."""
    loop = _loop_with_open_long(0.02)
    assert loop._check_sizing_exit("AAPL", 102.0, _flat_signal()) == "take_profit"


def test_max_hold_still_governs_an_untriggered_trade():
    """The timer remains the primary exit — the barriers are insurance."""
    loop = _loop_with_open_long(0.02)
    loop._bars_held["AAPL"] = SIZING_MAX_HOLD_BARS
    assert loop._check_sizing_exit("AAPL", 100.0, _flat_signal()) == "max_hold"
