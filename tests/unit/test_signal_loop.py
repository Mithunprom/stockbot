"""Unit tests for SignalLoop — Phase 5 paper trading pipeline."""

from __future__ import annotations

import numpy as np
import pytest
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
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


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_to_tensors_shapes():
    loop = _make_loop()
    arr = np.random.randn(60, 30).astype(np.float32)
    feat_1m, feat_5m = loop._to_tensors(arr)
    assert feat_1m.shape == (60, 30)
    # Every 5th of 60 rows starting at index 4: indices 4,9,14,...,59 → 12 bars
    assert feat_5m.shape == (12, 30)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
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
    pm.open_position("AAPL", "long", 40, 100.0)
    pm.open_position("XOM", "long", 40, 100.0)
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


# ─── H5: Signal-reconfirmed hold extension ────────────────────────────────────

def _h5_fixture(loop, ticker="AAPL", entry_price=100.0, days_ago_entered=1,
                pred_ret=0.009, dir_prob=0.70, bars=None):
    """Set up a position that has reached max_hold with a good signal."""
    from src.agents.signal_loop import SIZING_MAX_HOLD_BARS
    from datetime import timedelta
    from zoneinfo import ZoneInfo
    from datetime import datetime
    et_today = datetime.now(ZoneInfo("America/New_York")).date()
    loop._entry_prices[ticker] = entry_price
    loop._entry_directions[ticker] = 1
    loop._entry_dates[ticker] = et_today - timedelta(days=days_ago_entered)
    loop._peak_prices[ticker] = entry_price
    loop._ticker_atr[ticker] = 0.0005
    loop._bars_held[ticker] = bars if bars is not None else SIZING_MAX_HOLD_BARS
    # seed a realistic daily vol so daily_vol_for works
    loop._ticker_daily_vol[ticker] = 0.02  # 2% daily vol

    sig = EnsembleSignal(ticker=ticker, timestamp=_stamp(0))
    sig.lgbm_pred_return = pred_ret
    sig.lgbm_dir_prob = dir_prob
    return sig


def test_h5_extension_granted_on_fresh_signal():
    """Extension fires when signal re-qualifies and position is not a loser."""
    from src.agents.signal_loop import SIZING_MAX_HOLD_BARS, MAX_HOLD_EXTENSIONS
    loop = _make_loop()
    loop._pm.portfolio_value = 100_000.0
    sig = _h5_fixture(loop)

    # _check_sizing_exit should return None (extension granted, not max_hold)
    result = loop._check_sizing_exit("AAPL", 100.0, sig)
    assert result is None, f"Expected None (extension), got {result!r}"
    # Extension counter incremented
    assert loop._hold_extension_count.get("AAPL", 0) == 1
    # bars_held reset to 0 so the new window starts fresh
    assert loop._bars_held.get("AAPL", -1) == 0


def test_h5_max_hold_fires_after_two_extensions():
    """After MAX_HOLD_EXTENSIONS, max_hold exits regardless of signal."""
    from src.agents.signal_loop import SIZING_MAX_HOLD_BARS, MAX_HOLD_EXTENSIONS
    loop = _make_loop()
    loop._pm.portfolio_value = 100_000.0
    sig = _h5_fixture(loop)

    # Exhaust both extensions
    loop._hold_extension_count["AAPL"] = MAX_HOLD_EXTENSIONS

    result = loop._check_sizing_exit("AAPL", 100.0, sig)
    # No more extensions available → must exit
    assert result == "max_hold"


def test_h5_extension_denied_when_losing_more_than_daily_vol():
    """Extension denied if unrealized < -1× daily vol (don't compound losers).

    Setup: daily_vol=0.5% so stop_loss floor (1.0%) doesn't fire before the
    extension check. A -0.7% loss exceeds -1× daily_vol but sits inside the stop,
    so the only exit reason that can fire is 'max_hold' (no extension granted).
    """
    loop = _make_loop()
    loop._pm.portfolio_value = 100_000.0
    sig = _h5_fixture(loop, pred_ret=0.009, dir_prob=0.75)
    # Override fixture's 2% daily_vol with 0.5%; stop = max(0.5*1.1, floor=1%) = 1%
    loop._ticker_daily_vol["AAPL"] = 0.005
    # -0.7% < -0.5% (1× daily_vol) → extension denied; -0.7% > -1% (stop) → no stop
    result = loop._check_sizing_exit("AAPL", 99.3, sig)
    assert result == "max_hold"
    assert loop._hold_extension_count.get("AAPL", 0) == 0


def test_h5_extension_denied_when_signal_weak():
    """Extension denied when |pred_return| is below the dynamic cost threshold."""
    from src.agents.signal_loop import DYN_THRESH_FALLBACK
    loop = _make_loop()
    loop._pm.portfolio_value = 100_000.0
    # pred_ret below fallback threshold (0.003) → signal not fresh enough
    sig = _h5_fixture(loop, pred_ret=0.001, dir_prob=0.75)

    result = loop._check_sizing_exit("AAPL", 100.0, sig)
    assert result == "max_hold"
    assert loop._hold_extension_count.get("AAPL", 0) == 0


def test_h5_extension_denied_when_dir_prob_in_dead_zone():
    """Extension denied when dir_prob < 0.60 (dead zone — no conviction)."""
    loop = _make_loop()
    loop._pm.portfolio_value = 100_000.0
    sig = _h5_fixture(loop, pred_ret=0.009, dir_prob=0.55)

    result = loop._check_sizing_exit("AAPL", 100.0, sig)
    assert result == "max_hold"
    assert loop._hold_extension_count.get("AAPL", 0) == 0


def test_h5_second_extension_resets_bars_again():
    """Second extension resets bars_held a second time."""
    from src.agents.signal_loop import SIZING_MAX_HOLD_BARS
    loop = _make_loop()
    loop._pm.portfolio_value = 100_000.0
    sig = _h5_fixture(loop)

    # First extension
    r1 = loop._check_sizing_exit("AAPL", 100.0, sig)
    assert r1 is None
    assert loop._hold_extension_count["AAPL"] == 1
    assert loop._bars_held["AAPL"] == 0

    # Simulate another day of holding — bring bars back to max_hold
    loop._bars_held["AAPL"] = SIZING_MAX_HOLD_BARS

    # Second extension
    r2 = loop._check_sizing_exit("AAPL", 100.0, sig)
    assert r2 is None
    assert loop._hold_extension_count["AAPL"] == 2
    assert loop._bars_held["AAPL"] == 0


def test_h5_clear_sizing_state_removes_extension_count():
    """_clear_sizing_state must remove the extension counter to prevent stale state."""
    loop = _make_loop()
    loop._hold_extension_count["AAPL"] = 2
    loop._bars_held["AAPL"] = 0
    loop._entry_prices["AAPL"] = 100.0
    loop._entry_directions["AAPL"] = 1

    loop._clear_sizing_state("AAPL")
    assert "AAPL" not in loop._hold_extension_count


def test_h5_existing_pdt_test_unchanged():
    """Regression: PDT guard still defers same-day max_hold when signal is flat."""
    from src.agents.signal_loop import SIZING_MAX_HOLD_BARS
    loop = _make_loop()
    loop._pm.portfolio_value = 10_000.0   # under $25k PDT threshold
    _exit_fixture(loop, days_ago_entered=0)
    loop._bars_held["AAPL"] = SIZING_MAX_HOLD_BARS
    loop._daytrade_count = 0

    sig = EnsembleSignal(ticker="AAPL", timestamp=_stamp(0))
    # Default signal: lgbm_pred_return=0.0, dir_prob=0.5 → extension denied →
    # max_hold reason → PDT defer → None
    assert loop._check_sizing_exit("AAPL", 100.0, sig) is None


# ─── H3: Earnings-calendar blackout ───────────────────────────────────────────

def _et_today():
    """Return today's date in US/Eastern (matches _in_earnings_blackout's today)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York")).date()


def test_h3_in_blackout_same_day():
    """Earnings day itself is in the blackout window."""
    loop = _make_loop()
    loop._ticker_earnings["AAPL"] = _et_today()
    assert loop._in_earnings_blackout("AAPL")


def test_h3_in_blackout_one_day_before():
    """One calendar day before earnings → in blackout."""
    from datetime import timedelta
    loop = _make_loop()
    loop._ticker_earnings["AAPL"] = _et_today() + timedelta(days=1)
    assert loop._in_earnings_blackout("AAPL")


def test_h3_in_blackout_two_days_before():
    """Two calendar days before earnings → still in blackout."""
    from datetime import timedelta
    loop = _make_loop()
    loop._ticker_earnings["AAPL"] = _et_today() + timedelta(days=2)
    assert loop._in_earnings_blackout("AAPL")


def test_h3_outside_blackout_three_days_before():
    """Three days out is clear."""
    from datetime import timedelta
    loop = _make_loop()
    loop._ticker_earnings["AAPL"] = _et_today() + timedelta(days=3)
    assert not loop._in_earnings_blackout("AAPL")


def test_h3_in_blackout_one_day_after():
    """One day after earnings is still inside the symmetric window."""
    from datetime import timedelta
    loop = _make_loop()
    loop._ticker_earnings["AAPL"] = _et_today() - timedelta(days=1)
    assert loop._in_earnings_blackout("AAPL")


def test_h3_outside_blackout_three_days_after():
    """Three days after earnings → clear."""
    from datetime import timedelta
    loop = _make_loop()
    loop._ticker_earnings["AAPL"] = _et_today() - timedelta(days=3)
    assert not loop._in_earnings_blackout("AAPL")


def test_h3_unknown_earnings_fails_open():
    """None earnings date must never block entry."""
    loop = _make_loop()
    loop._ticker_earnings["AAPL"] = None
    assert not loop._in_earnings_blackout("AAPL")


def test_h3_ticker_not_in_cache_fails_open():
    """Missing earnings cache entry must never block entry."""
    loop = _make_loop()
    assert not loop._in_earnings_blackout("AAPL")


def test_h3_entry_gate_blocked_in_blackout():
    """Gate 0c: earnings blackout blocks entry even on a great signal."""
    from datetime import timedelta
    loop = _make_loop()
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    loop._ticker_earnings["AAPL"] = _et_today() + timedelta(days=1)

    sig = EnsembleSignal(ticker="AAPL", timestamp=_stamp(0))
    sig.lgbm_pred_return = 0.015
    sig.lgbm_dir_prob = 0.85
    assert not loop._sizing_entry_gate_open(sig)


def test_h3_entry_gate_open_when_not_in_blackout():
    """Gate 0c: no blackout → gate stays open (signal quality may still block)."""
    from datetime import timedelta
    loop = _make_loop()
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    # Far-future earnings → not in blackout
    loop._ticker_earnings["AAPL"] = _et_today() + timedelta(days=30)

    sig = EnsembleSignal(ticker="AAPL", timestamp=_stamp(0))
    sig.lgbm_pred_return = 0.015
    sig.lgbm_dir_prob = 0.85
    # Gate 0c passes; gate 6 (signal quality) also passes (pred > fallback)
    assert loop._sizing_entry_gate_open(sig)


def test_h3_compute_earnings_blackout_fails_open():
    """_compute_earnings_blackout returns None per ticker on yfinance error."""
    from src.agents.signal_loop import _compute_earnings_blackout
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.calendar = None
        result = _compute_earnings_blackout(["AAPL", "MSFT"])
    assert result == {"AAPL": None, "MSFT": None}


def test_h3_compute_earnings_blackout_parses_timestamp():
    """_compute_earnings_blackout extracts the date from a yfinance Timestamp."""
    from src.agents.signal_loop import _compute_earnings_blackout
    from datetime import date, timezone
    from unittest.mock import patch
    import pandas as pd

    target_date = date(2026, 7, 25)
    ts = pd.Timestamp("2026-07-25", tz="UTC")

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.calendar = {"Earnings Date": [ts]}
        result = _compute_earnings_blackout(["AAPL"])
    assert result == {"AAPL": target_date}


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


# ─── v0.3.5 regression: the Jul 8-9 exit-path outage ──────────────────────────

def test_exit_branch_executes_without_nameerror():
    """REGRESSION (2026-07-08/09 outage): the sizing_exit branch of
    _act_on_signal crashed with NameError (unbound atr_pct in the log call)
    the moment any exit condition fired, killing exits/entries/CB checks for
    two sessions. This test drives the REAL exit branch end-to-end."""
    import asyncio
    from src.agents.signal_loop import SIZING_MAX_HOLD_BARS

    loop = _make_loop()
    pm = loop._pm
    pm.portfolio_value = 97_000.0
    pm.open_position("AAPL", "long", 10, 100.0)
    _exit_fixture(loop, "AAPL", entry_price=100.0, days_ago_entered=1)
    loop._bars_held["AAPL"] = SIZING_MAX_HOLD_BARS  # max_hold must fire

    submitted = {}

    async def fake_submit(req):
        submitted["side"] = getattr(req, "side", None)
        from datetime import datetime, timezone

        class R:  # minimal fill result
            status = "filled"
            filled_avg_price = 100.0
            filled_qty = 10.0
            filled_at = datetime.now(timezone.utc)
            id = "test-order"
        return R()

    loop._alpaca.submit_order = fake_submit
    loop._alpaca.get_latest_quote = MagicMock()

    sig = EnsembleSignal(ticker="AAPL", timestamp=_stamp(0))
    sig.lgbm_pred_return = 0.001
    sig.lgbm_dir_prob = 0.55

    # Must not raise — the outage bug threw NameError before order submission
    asyncio.run(loop._act_on_signal(sig, 100.0, None, regime=1))
    assert str(submitted.get("side", "")).lower().endswith("sell"), (
        f"exit branch must reach order submission; got {submitted or 'no order'}"
    )


def test_exit_loop_isolates_per_ticker_failures():
    """A poisoned ticker in the exit loop must not abort the tick for others
    (per-ticker try/except added after the Jul 9 outage)."""
    import asyncio
    src = open("src/agents/signal_loop.py").read()
    # The exit loop must wrap _act_on_signal in a per-ticker try/except
    assert "sizing_exit_tick_error" in src


def test_daily_reset_fires_after_930_any_minute():
    """REGRESSION: reset used to require a tick landing exactly at 9:30-9:31;
    now any post-9:30 tick on a new date resets (Jul 9: counter stuck at cap)."""
    from datetime import datetime as dt
    from zoneinfo import ZoneInfo

    loop = _make_loop()
    loop._sizing_n_trades_today = 4
    loop._last_reset_date = None

    with patch("src.agents.signal_loop.datetime") as mock_dt:
        # 14:45 ET — far outside the old 9:30-9:31 window
        fake_now = dt(2026, 7, 10, 14, 45, tzinfo=ZoneInfo("America/New_York"))
        mock_dt.now.return_value = fake_now
        loop._maybe_reset_daily_value()

    assert loop._sizing_n_trades_today == 0
    assert loop._last_reset_date == fake_now.date()


def test_exit_of_short_position_buys_to_cover():
    """REGRESSION (2026-07-10 MSTR spiral): exiting a SHORT must BUY.
    Hardcoded side="sell" compounded an accidental short ~2x/minute."""
    import asyncio
    from src.agents.signal_loop import SIZING_MAX_HOLD_BARS

    loop = _make_loop()
    pm = loop._pm
    pm.portfolio_value = 97_000.0
    pm.open_position("AAPL", "short", 100, 100.0)
    loop._entry_prices["AAPL"] = 100.0
    loop._entry_directions["AAPL"] = -1
    loop._peak_prices["AAPL"] = 100.0
    loop._ticker_atr["AAPL"] = 0.0005
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    loop._entry_dates["AAPL"] = (
        datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
    )
    loop._bars_held["AAPL"] = SIZING_MAX_HOLD_BARS  # force max_hold exit

    submitted = {}

    async def fake_submit(req):
        submitted["side"] = req.side
        submitted["qty"] = req.qty
        from datetime import timezone as tz
        class R:
            status = "filled"
            filled_avg_price = 100.0
            filled_qty = 100.0
            filled_at = datetime.now(tz.utc)
            id = "t"
        return R()

    loop._alpaca.submit_order = fake_submit
    sig = EnsembleSignal(ticker="AAPL", timestamp=_stamp(0))
    sig.lgbm_pred_return = 0.001
    sig.lgbm_dir_prob = 0.55
    asyncio.run(loop._act_on_signal(sig, 100.0, None, regime=1))
    assert submitted.get("side") == "buy", (
        f"short exit must BUY to cover, got {submitted}"
    )
