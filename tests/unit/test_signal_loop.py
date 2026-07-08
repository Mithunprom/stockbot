"""Unit tests for SignalLoop — Phase 5 paper trading pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.signal_loop import SignalLoop
from src.execution.position_manager import PositionManager
from src.models.ensemble import EnsembleSignal
from src.risk.circuit_breakers import CircuitBreakers


def _make_loop(broadcast=None) -> SignalLoop:
    """Create a SignalLoop with mocked dependencies."""
    ensemble = MagicMock()
    alpaca = MagicMock()
    cb = CircuitBreakers()
    pm = PositionManager(initial_portfolio=100_000.0)
    sf = MagicMock()
    feature_cols = [f"feat_{i}" for i in range(30)]
    return SignalLoop(
        universe=["AAPL", "MSFT"],
        ensemble=ensemble,
        alpaca=alpaca,
        circuit_breakers=cb,
        pos_manager=pm,
        session_factory=sf,
        feature_cols=feature_cols,
        broadcast_fn=broadcast,
    )


def test_signal_loop_instantiation():
    loop = _make_loop()
    assert loop._n_features == 30
    assert loop._universe == ["AAPL", "MSFT"]
    assert not loop._stopped


def test_to_tensors_shapes():
    loop = _make_loop()
    arr = np.random.randn(60, 30).astype(np.float32)
    feat_1m, feat_5m = loop._to_tensors(arr)
    assert feat_1m.shape == (60, 30)
    # Every 5th of 60 rows starting at index 4: indices 4,9,14,...,59 → 12 bars
    assert feat_5m.shape == (12, 30)


def test_to_tensors_chronological_order():
    """Verify 5m tensor is every 5th bar from the 1m tensor (not reversed)."""
    loop = _make_loop()
    arr = np.arange(60 * 30, dtype=np.float32).reshape(60, 30)
    feat_1m, feat_5m = loop._to_tensors(arr)
    # feat_5m[0] should be 1m row at index 4 (arr[4])
    assert torch.allclose(feat_5m[0], feat_1m[4])
    # feat_5m[1] should be 1m row at index 9 (arr[9])
    assert torch.allclose(feat_5m[1], feat_1m[9])


def test_get_latest_signals_empty():
    loop = _make_loop()
    assert loop.get_latest_signals() == []


def test_market_hours_weekend():
    """Saturday → not market hours."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    loop = _make_loop()
    # Patch datetime inside the method
    with patch("src.agents.signal_loop.datetime") as mock_dt:
        # Saturday
        fake_now = datetime(2026, 3, 7, 10, 30, tzinfo=ZoneInfo("America/New_York"))
        mock_dt.now.return_value = fake_now
        assert not loop._is_market_hours()


def test_market_hours_weekday_open():
    """Wednesday 10:30 ET → market hours."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    loop = _make_loop()
    with patch("src.agents.signal_loop.datetime") as mock_dt:
        fake_now = datetime(2026, 3, 4, 10, 30, tzinfo=ZoneInfo("America/New_York"))
        mock_dt.now.return_value = fake_now
        assert loop._is_market_hours()


def test_circuit_breaker_blocks_order():
    """When trading is halted, _act_on_signal should return immediately."""
    loop = _make_loop()
    loop._cb._halted = True

    # _act_on_signal with halted CB should return without calling alpaca
    import asyncio

    async def run():
        sig = MagicMock(spec=EnsembleSignal)
        sig.ensemble_signal = 0.8
        sig.ticker = "AAPL"
        await loop._act_on_signal(sig, 200.0)

    asyncio.run(run())
    loop._alpaca.get_latest_quote.assert_not_called()
    loop._alpaca.submit_order.assert_not_called()


def test_position_size_respects_cap():
    """Vol-scaled position size never exceeds max_position_pct."""
    pm = PositionManager(initial_portfolio=100_000.0, max_position_pct=0.25)
    # With very low realized vol, vol_scalar is capped at 2x
    # base 5% × 2 = 10% < 25% cap
    size = pm.compute_position_size("AAPL", recent_returns=[0.0001] * 30)
    assert size <= pm.portfolio_value * pm.max_position_pct
    assert size > 0


# ─── Swing-mode exit geometry ──────────────────────────────────────────────────

def test_atr_exits_geometry():
    """Stops/targets scale with daily vol, respect floors/caps, and TP ≈ 2× stop."""
    from src.agents.signal_loop import (
        _atr_exits,
        SIZING_STOP_LOSS_FLOOR, SIZING_STOP_LOSS_CAP,
        SIZING_TAKE_PROFIT_FLOOR, SIZING_TAKE_PROFIT_CAP,
    )
    # _atr_exits now takes a TRUE daily-vol fraction (not 1m ATR).
    # Low-vol name (JPM-like ~0.8% daily) → floors apply
    sl_lo, ts_lo, tp_lo = _atr_exits(0.008)
    assert sl_lo >= SIZING_STOP_LOSS_FLOOR
    assert tp_lo >= SIZING_TAKE_PROFIT_FLOOR
    # High-vol name (SNDK-like ~12% daily) → caps apply
    sl_hi, ts_hi, tp_hi = _atr_exits(0.12)
    assert sl_hi <= SIZING_STOP_LOSS_CAP
    assert tp_hi <= SIZING_TAKE_PROFIT_CAP
    # Monotonic in volatility, and reward ≥ risk
    assert sl_hi >= sl_lo and tp_hi >= tp_lo
    assert tp_lo >= 1.8 * sl_lo


# ─── Kelly governor (window + probation, no deadlock) ─────────────────────────

def _stamp(days_ago: float):
    from datetime import datetime, timedelta, timezone
    return datetime.now(timezone.utc) - timedelta(days=days_ago)


def test_kelly_window_prunes_stale_trades():
    """Outcomes older than the lookback never freeze the governor."""
    from src.agents.signal_loop import KELLY_LOOKBACK_DAYS
    loop = _make_loop()
    # 30 old losing trades (the May disaster) — all outside the window
    loop._sizing_recent_outcomes = [
        (_stamp(KELLY_LOOKBACK_DAYS + 5), -0.01) for _ in range(30)
    ]
    assert loop._kelly_mode() == "inactive"   # pruned → not enough recent data
    assert loop._sizing_recent_outcomes == []


def test_kelly_probation_allows_single_probe():
    """Negative recent expectancy degrades to probation, not a permanent block."""
    from src.agents.signal_loop import KELLY_MIN_TRADES, TICKER_IC_MIN_N
    loop = _make_loop()
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    loop._sizing_recent_outcomes = [
        (_stamp(1), -0.01) for _ in range(KELLY_MIN_TRADES + 2)
    ]
    loop._update_kelly()
    assert loop._kelly_mode() == "probation"

    sig = EnsembleSignal(ticker="AAPL", timestamp=_stamp(0))
    sig.lgbm_pred_return = 0.009
    sig.lgbm_dir_prob = 0.62

    # No IC history → no probe (probes require demonstrated positive IC)
    assert not loop._sizing_entry_gate_open(sig)

    # Positive-IC ticker → one probe allowed. The probe reads the 30d probe
    # cache (_ticker_ic_probe), NOT the 7d block cache — the 7d cache's ~250
    # sample ceiling made n>=300 unreachable and deadlocked the governor.
    loop._ticker_ic_probe = {"AAPL": (0.15, TICKER_IC_MIN_N + 50)}
    assert loop._sizing_entry_gate_open(sig)

    # Probe budget spent → blocked until tomorrow
    loop._probation_entries_today = 1
    assert not loop._sizing_entry_gate_open(sig)


def test_kelly_probation_probe_ignores_7d_block_cache():
    """Regression: a full 7d block cache must NOT satisfy the probe.

    The 2026-06 deadlock was the probe reading the 7d cache whose n topped out
    at ~250 (< TICKER_IC_MIN_N=300), so probation could never release. The probe
    now reads only the 30d cache; a populated 7d cache alone leaves it blocked.
    """
    from src.agents.signal_loop import KELLY_MIN_TRADES, TICKER_IC_MIN_N
    loop = _make_loop()
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    loop._sizing_recent_outcomes = [
        (_stamp(1), -0.01) for _ in range(KELLY_MIN_TRADES + 2)
    ]
    loop._update_kelly()
    assert loop._kelly_mode() == "probation"

    sig = EnsembleSignal(ticker="AAPL", timestamp=_stamp(0))
    sig.lgbm_pred_return = 0.009
    sig.lgbm_dir_prob = 0.62

    # 7d block cache full & positive, but probe cache empty → still blocked
    loop._ticker_ic = {"AAPL": (0.15, TICKER_IC_MIN_N + 50)}
    loop._ticker_ic_probe = {}
    assert not loop._sizing_entry_gate_open(sig)

    # Once the 30d probe cache clears the bar, the probe fires
    loop._ticker_ic_probe = {"AAPL": (0.15, TICKER_IC_MIN_N + 50)}
    assert loop._sizing_entry_gate_open(sig)


def test_ticker_ic_gate_blocks_proven_negative():
    """Tickers the model is provably wrong on (e.g. JPM −0.32) are skipped."""
    from src.agents.signal_loop import TICKER_IC_MIN_N
    loop = _make_loop()
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    loop._ticker_ic = {"JPM": (-0.32, TICKER_IC_MIN_N + 100)}

    sig = EnsembleSignal(ticker="JPM", timestamp=_stamp(0))
    sig.lgbm_pred_return = 0.009
    sig.lgbm_dir_prob = 0.61
    assert not loop._sizing_entry_gate_open(sig)

    # Insufficient sample fails open (other gates still apply)
    loop._ticker_ic = {"JPM": (-0.32, 10)}
    assert loop._sizing_entry_gate_open(sig)


# ─── PDT-aware exits ───────────────────────────────────────────────────────────

def _exit_fixture(loop, ticker="AAPL", entry_price=100.0, days_ago_entered=0):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    et_today = datetime.now(ZoneInfo("America/New_York")).date()
    loop._entry_prices[ticker] = entry_price
    loop._entry_directions[ticker] = 1
    loop._entry_dates[ticker] = et_today - timedelta(days=days_ago_entered)
    loop._peak_prices[ticker] = entry_price
    loop._ticker_atr[ticker] = 0.0005  # → stop 1.09%, tp 2.2%


def test_pdt_defers_same_day_max_hold():
    """Same-day non-stop exits wait for the next session (not a day trade then)."""
    from src.agents.signal_loop import SIZING_MAX_HOLD_BARS
    loop = _make_loop()
    pm = loop._pm
    pm.portfolio_value = 10_000.0   # under the $25k PDT threshold
    _exit_fixture(loop, days_ago_entered=0)
    loop._bars_held["AAPL"] = SIZING_MAX_HOLD_BARS
    loop._daytrade_count = 0

    sig = EnsembleSignal(ticker="AAPL", timestamp=_stamp(0))
    # flat price → only max_hold can fire, but it's a same-day round trip
    assert loop._check_sizing_exit("AAPL", 100.0, sig) is None


def test_pdt_allows_same_day_stop_loss_with_budget():
    loop = _make_loop()
    loop._pm.portfolio_value = 10_000.0
    _exit_fixture(loop, days_ago_entered=0)
    loop._bars_held["AAPL"] = 5
    loop._daytrade_count = 0
    sig = EnsembleSignal(ticker="AAPL", timestamp=_stamp(0))
    # -2% on a ~1.1% stop → stop_loss, budget available → allowed
    assert loop._check_sizing_exit("AAPL", 98.0, sig) == "stop_loss"
    # Budget exhausted → even the stop defers (Alpaca would reject it anyway)
    loop._daytrade_count = 3
    assert loop._check_sizing_exit("AAPL", 98.0, sig) is None


def test_next_day_exits_unrestricted():
    from src.agents.signal_loop import SIZING_MAX_HOLD_BARS
    loop = _make_loop()
    loop._pm.portfolio_value = 10_000.0
    _exit_fixture(loop, days_ago_entered=1)
    loop._bars_held["AAPL"] = SIZING_MAX_HOLD_BARS
    loop._daytrade_count = 3   # no budget — irrelevant for overnight positions
    sig = EnsembleSignal(ticker="AAPL", timestamp=_stamp(0))
    assert loop._check_sizing_exit("AAPL", 100.0, sig) == "max_hold"


def test_stagnation_exit_frees_dead_capital():
    """Stagnation logic fires when its threshold sits below max hold."""
    import src.agents.signal_loop as sl
    loop = _make_loop()
    loop._pm.portfolio_value = 10_000.0
    _exit_fixture(loop, days_ago_entered=2)
    with patch.object(sl, "SIZING_STAGNATION_BARS", 100), \
         patch.object(sl, "SIZING_MAX_HOLD_BARS", 200):
        loop._bars_held["AAPL"] = 150
        sig = EnsembleSignal(ticker="AAPL", timestamp=_stamp(0))
        # +0.1% after the stagnation window → dead trade
        assert loop._check_sizing_exit("AAPL", 100.1, sig) == "stagnation"


# ─── Entry discipline ──────────────────────────────────────────────────────────

def test_execute_entries_caps_per_tick():
    """Best signals first, hard cap per tick — no more 6-position bursts."""
    import asyncio
    from src.agents.signal_loop import MAX_ENTRIES_PER_TICK
    loop = _make_loop()
    fills: list[str] = []

    async def fake_act(sig, price, feat, regime=0):
        fills.append(sig.ticker)
        return True

    loop._act_on_signal = fake_act
    sigs = []
    for i, t in enumerate(["NVDA", "AMD", "SMCI", "GOOGL", "AMZN"]):
        s = EnsembleSignal(ticker=t, timestamp=_stamp(0))
        s.lgbm_pred_return = 0.001 * (i + 1)   # AMZN strongest
        sigs.append(s)
    prices = {t: 100.0 for t in ["NVDA", "AMD", "SMCI", "GOOGL", "AMZN"]}

    n = asyncio.run(loop._execute_entries(sigs, prices, {}))
    assert n == MAX_ENTRIES_PER_TICK
    assert len(fills) == MAX_ENTRIES_PER_TICK
    assert fills[0] == "AMZN"   # ranked by |pred_return|


def test_sector_position_cap_blocks_third_semi():
    from src.agents.signal_loop import MAX_POSITIONS_PER_SECTOR
    loop = _make_loop()
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    loop._pm.open_position("NVDA", "long", 10, 100.0)
    loop._pm.open_position("AMD", "long", 10, 100.0)

    sig = EnsembleSignal(ticker="SMCI", timestamp=_stamp(0))
    sig.lgbm_pred_return = 0.009
    sig.lgbm_dir_prob = 0.65
    assert loop._sector_position_count("SMCI") == MAX_POSITIONS_PER_SECTOR
    assert not loop._sizing_entry_gate_open(sig)

    # Different sector still allowed
    sig2 = EnsembleSignal(ticker="XOM", timestamp=_stamp(0))
    sig2.lgbm_pred_return = 0.009
    sig2.lgbm_dir_prob = 0.65
    assert loop._sizing_entry_gate_open(sig2)


def test_heat_ceiling_blocks_entries():
    from src.agents.signal_loop import PORTFOLIO_HEAT_CEILING
    loop = _make_loop()
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    pm = loop._pm
    pm.portfolio_value = 10_000.0
    pm.open_position("AAPL", "long", 35, 100.0)
    pm.open_position("XOM", "long", 30, 100.0)
    assert pm.managed_heat >= PORTFOLIO_HEAT_CEILING

    sig = EnsembleSignal(ticker="CVX", timestamp=_stamp(0))
    sig.lgbm_pred_return = 0.009
    sig.lgbm_dir_prob = 0.65
    assert not loop._sizing_entry_gate_open(sig)


# ─── Dynamic entry threshold ───────────────────────────────────────────────────

def test_dynamic_threshold_self_calibrates():
    """Threshold tracks the trailing percentile of |pred|, with floor + fallback."""
    from src.agents.signal_loop import (
        DYN_THRESH_FALLBACK, DYN_THRESH_FLOOR, DYN_THRESH_MIN_SAMPLES,
    )
    loop = _make_loop()
    # Too few samples → fallback
    assert loop._dynamic_cost_threshold() == DYN_THRESH_FALLBACK

    # Small-magnitude model (like the 2026-06-12 retrain): P92 of |pred| ≈ 0.0018
    loop._pred_magnitudes.extend([0.001] * DYN_THRESH_MIN_SAMPLES + [0.002] * 100)
    thr = loop._dynamic_cost_threshold()
    assert DYN_THRESH_FLOOR <= thr < 0.0025   # calibrated near the floor

    # Large-magnitude model → threshold scales up automatically
    loop._pred_magnitudes.clear()
    loop._pred_magnitudes.extend([0.004] * DYN_THRESH_MIN_SAMPLES + [0.02] * 200)
    assert loop._dynamic_cost_threshold() > 0.004


def test_gate_uses_dynamic_threshold():
    loop = _make_loop()
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    sig = EnsembleSignal(ticker="XOM", timestamp=_stamp(0))
    sig.lgbm_pred_return = 0.004   # above floor, below a high calibrated bar
    sig.lgbm_dir_prob = 0.70
    # Fallback threshold (0.003) → passes
    assert loop._sizing_entry_gate_open(sig)
    # Calibrate to a strong-magnitude model → 0.004 no longer top-8%
    from src.agents.signal_loop import DYN_THRESH_MIN_SAMPLES
    loop._pred_magnitudes.extend([0.008] * (DYN_THRESH_MIN_SAMPLES + 500))
    assert not loop._sizing_entry_gate_open(sig)


# ─── Confirmed-reversal exits + positive-IC gate (v0.3.4, 2026-07-07 diagnosis) ─

def test_reversal_ignores_bare_sign_flips():
    """A pred_return sign flip below the cost threshold must NOT count toward
    reversal — sign-only flicker truncated 1-day holds to a 25-min median."""
    loop = _make_loop()
    loop._entry_directions["AAPL"] = 1
    sig = EnsembleSignal(ticker="AAPL", timestamp=_stamp(0))
    sig.lgbm_pred_return = -0.0001   # opposite sign, but tiny (noise)
    sig.lgbm_dir_prob = 0.55         # inside dead zone too
    assert not loop._confirmed_opposite_signal(sig, entry_dir=1)


def test_reversal_requires_tradeable_opposite_signal():
    """Only an opposite signal that would itself qualify for entry counts."""
    loop = _make_loop()
    sig = EnsembleSignal(ticker="AAPL", timestamp=_stamp(0))
    sig.lgbm_pred_return = -0.009    # clears fallback cost threshold (0.003)
    sig.lgbm_dir_prob = 0.20         # strong P(down), outside dead zone
    assert loop._confirmed_opposite_signal(sig, entry_dir=1)

    # Same magnitude but dir_prob in the dead zone → not confirmed
    sig.lgbm_dir_prob = 0.50
    assert not loop._confirmed_opposite_signal(sig, entry_dir=1)

    # Same-direction signal never counts as a reversal
    sig.lgbm_pred_return = 0.009
    sig.lgbm_dir_prob = 0.80
    assert not loop._confirmed_opposite_signal(sig, entry_dir=1)


def test_reversal_needs_sustained_confirmation():
    """One confirmed opposite bar must not exit — SIZING_REVERSAL_BARS required."""
    from src.agents.signal_loop import SIZING_REVERSAL_BARS
    assert SIZING_REVERSAL_BARS >= 30, (
        "reversal confirmation must span ~30+ bars; 4-bar flicker exits "
        "destroyed the swing design (2026-07-07 diagnosis)"
    )
    loop = _make_loop()
    _exit_fixture(loop, "AAPL", entry_price=100.0, days_ago_entered=1)
    loop._bars_held["AAPL"] = 10

    sig = EnsembleSignal(ticker="AAPL", timestamp=_stamp(0))
    sig.lgbm_pred_return = -0.009
    sig.lgbm_dir_prob = 0.20

    # Confirmed opposite bars accumulate but don't exit until the bar count
    for _ in range(SIZING_REVERSAL_BARS - 1):
        assert loop._check_sizing_exit("AAPL", 100.0, sig) is None
    assert loop._check_sizing_exit("AAPL", 100.0, sig) == "signal_reversal"


def test_reversal_counter_resets_on_neutral_bar():
    """A non-confirmed bar resets the reversal streak."""
    loop = _make_loop()
    _exit_fixture(loop, "AAPL", entry_price=100.0, days_ago_entered=1)
    loop._bars_held["AAPL"] = 10

    opposite = EnsembleSignal(ticker="AAPL", timestamp=_stamp(0))
    opposite.lgbm_pred_return = -0.009
    opposite.lgbm_dir_prob = 0.20
    neutral = EnsembleSignal(ticker="AAPL", timestamp=_stamp(0))
    neutral.lgbm_pred_return = -0.0001
    neutral.lgbm_dir_prob = 0.50

    for _ in range(10):
        loop._check_sizing_exit("AAPL", 100.0, opposite)
    assert loop._reversal_counts["AAPL"] == 10
    loop._check_sizing_exit("AAPL", 100.0, neutral)
    assert loop._reversal_counts["AAPL"] == 0


def test_ticker_ic_gate_requires_positive_ic():
    """At full sample, non-positive live IC blocks entry (was: only < −0.05)."""
    from src.agents.signal_loop import TICKER_IC_MIN_N
    loop = _make_loop()
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    sig = EnsembleSignal(ticker="WDC", timestamp=_stamp(0))
    sig.lgbm_pred_return = 0.009
    sig.lgbm_dir_prob = 0.70

    # Mildly negative IC at full sample → blocked (previously slipped through)
    loop._ticker_ic = {"WDC": (-0.01, TICKER_IC_MIN_N + 50)}
    assert not loop._sizing_entry_gate_open(sig)

    # Exactly zero → blocked (no proven edge, why pay costs)
    loop._ticker_ic = {"WDC": (0.0, TICKER_IC_MIN_N + 50)}
    assert not loop._sizing_entry_gate_open(sig)

    # Positive IC at full sample → open
    loop._ticker_ic = {"WDC": (0.04, TICKER_IC_MIN_N + 50)}
    assert loop._sizing_entry_gate_open(sig)

    # Small sample fails open regardless of IC sign
    loop._ticker_ic = {"WDC": (-0.04, 50)}
    assert loop._sizing_entry_gate_open(sig)
