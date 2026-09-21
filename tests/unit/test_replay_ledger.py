"""Unit tests for the ledger replay harness.

The harness produces every P&L claim in the correlation/exit-calibration work,
so its arithmetic is money-handling code and is tested as such.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.replay_ledger import (
    DailyVolLookup,
    Fill,
    block_count_placebo,
    effective_sigma_multiple,
    filter_window,
    load_ledger,
    placebo_z,
    replay_concurrency_guard,
    replay_entry_burst_guard,
    replay_stop_barrier,
    summarize,
)

BASE = datetime(2026, 8, 10, 13, 40, tzinfo=timezone.utc)


def _fill(trade_id: int, ticker: str, pnl_pct: float, *, entry_min: int = 0,
          hold_min: int = 30, shares: float = 100.0, price: float = 100.0,
          side: str = "buy") -> Fill:
    """Build a synthetic fill whose pnl is exactly consistent with pnl_pct."""
    entry = BASE + timedelta(minutes=entry_min)
    direction = -1 if side in ("sell", "short") else 1
    return Fill(
        trade_id=trade_id,
        ticker=ticker,
        side=side,
        entry_time=entry,
        exit_time=entry + timedelta(minutes=hold_min),
        entry_price=price,
        exit_price=price * (1 + pnl_pct * direction),
        shares=shares,
        pnl=pnl_pct * shares * price,
        pnl_pct=pnl_pct,
        exit_reason="max_hold",
        ensemble_signal=0.5,
    )


# ─── summarize ──────────────────────────────────────────────────────────────

def test_summarize_reports_risk_alongside_pnl():
    """Win rate, profit factor, drawdown and Sharpe travel with net P&L."""
    fills = [_fill(1, "AAA", 0.02), _fill(2, "BBB", -0.01), _fill(3, "CCC", 0.01)]
    stats = summarize(fills)
    assert stats.n == 3
    assert stats.wins == 2 and stats.losses == 1
    assert stats.win_rate == pytest.approx(2 / 3)
    # gross profit 200 + 100 = 300; gross loss 100
    assert stats.gross_profit == pytest.approx(300.0)
    assert stats.gross_loss == pytest.approx(100.0)
    assert stats.profit_factor == pytest.approx(3.0)
    assert stats.net_pnl == pytest.approx(200.0)
    assert stats.expectancy == pytest.approx(200.0 / 3)


def test_summarize_drawdown_follows_exit_order():
    """Drawdown is peak-to-trough on the closed-trade curve, ordered by exit."""
    fills = [
        _fill(1, "AAA", 0.03, entry_min=0),    # +300, exits first
        _fill(2, "BBB", -0.02, entry_min=1),   # -200
        _fill(3, "CCC", 0.01, entry_min=2),    # +100
    ]
    stats = summarize(fills)
    # curve: 300 -> 100 -> 200; peak 300, trough 100
    assert stats.max_drawdown == pytest.approx(200.0)


def test_summarize_empty_is_all_zero():
    stats = summarize([])
    assert stats.n == 0 and stats.net_pnl == 0.0 and stats.profit_factor == 0.0


def test_summarize_profit_factor_infinite_without_losses():
    stats = summarize([_fill(1, "AAA", 0.01)])
    assert stats.profit_factor == float("inf")


# ─── load_ledger ────────────────────────────────────────────────────────────

def test_load_ledger_drops_integrity_voided_rows(tmp_path):
    """Rows voided by integrity repair (pnl=null) must never enter a statistic."""
    payload = {"trades": [
        {"id": 1, "ticker": "AAA", "side": "buy",
         "entry_time": "2026-08-10T13:40:00+00:00",
         "exit_time": "2026-08-10T14:10:00+00:00",
         "entry_price": 100.0, "exit_price": 101.0, "shares": 10.0,
         "pnl": 10.0, "pnl_pct": 0.01, "exit_reason": "max_hold",
         "ensemble_signal": 0.4},
        {"id": 2, "ticker": "BBB", "side": "buy",
         "entry_time": "2026-08-10T13:40:00+00:00",
         "exit_time": "2026-08-10T14:10:00+00:00",
         "entry_price": 100.0, "exit_price": 101.0, "shares": 10.0,
         "pnl": None, "pnl_pct": None, "exit_reason": "max_hold",
         "ensemble_signal": 0.4},
    ]}
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(payload))
    fills = load_ledger(path)
    assert [f.trade_id for f in fills] == [1]


def test_filter_window_selects_on_exit_date():
    from datetime import date
    early = _fill(1, "AAA", 0.01, entry_min=0)
    late = _fill(2, "BBB", 0.01, entry_min=60 * 24 * 40)
    selected = filter_window([early, late], since=date(2026, 9, 1))
    assert [f.trade_id for f in selected] == [2]


# ─── concurrency guard ──────────────────────────────────────────────────────

def test_concurrency_guard_blocks_third_overlapping_position():
    """With cap=2, a third simultaneous same-bucket entry is blocked."""
    fills = [
        _fill(1, "AAA", 0.01, entry_min=0, hold_min=30),
        _fill(2, "BBB", 0.01, entry_min=1, hold_min=30),
        _fill(3, "CCC", -0.05, entry_min=2, hold_min=30),
    ]
    result = replay_concurrency_guard(fills, lambda t: "semis", max_per_bucket=2)
    assert [f.trade_id for f in result.admitted] == [1, 2]
    assert [f.trade_id for f in result.blocked] == [3]
    assert result.blocked_pnl == pytest.approx(-500.0)


def test_concurrency_guard_allows_reentry_after_exit():
    """A closed position must free its slot."""
    fills = [
        _fill(1, "AAA", 0.01, entry_min=0, hold_min=10),
        _fill(2, "BBB", 0.01, entry_min=0, hold_min=10),
        _fill(3, "CCC", 0.01, entry_min=20, hold_min=10),
    ]
    result = replay_concurrency_guard(fills, lambda t: "semis", max_per_bucket=2)
    assert len(result.blocked) == 0


def test_concurrency_guard_separates_buckets():
    """Positions in different buckets never contend for the same cap."""
    fills = [
        _fill(1, "AAA", 0.01, entry_min=0),
        _fill(2, "BBB", 0.01, entry_min=1),
        _fill(3, "CCC", 0.01, entry_min=2),
    ]
    result = replay_concurrency_guard(fills, lambda t: t, max_per_bucket=2)
    assert len(result.blocked) == 0


# ─── burst guard ────────────────────────────────────────────────────────────

def test_burst_guard_caps_entries_per_rolling_window():
    fills = [_fill(i, f"T{i}", 0.01, entry_min=i) for i in range(5)]
    result = replay_entry_burst_guard(fills, window_minutes=10.0, max_entries=2)
    assert [f.trade_id for f in result.admitted] == [0, 1]
    assert [f.trade_id for f in result.blocked] == [2, 3, 4]


def test_burst_guard_window_rolls_forward():
    """Entries spaced beyond the window are all admitted."""
    fills = [_fill(i, f"T{i}", 0.01, entry_min=i * 20) for i in range(4)]
    result = replay_entry_burst_guard(fills, window_minutes=10.0, max_entries=1)
    assert len(result.blocked) == 0


# ─── stop barrier ───────────────────────────────────────────────────────────

class _FixedVol(DailyVolLookup):
    """DailyVolLookup stub with a constant daily vol (bypasses the cache)."""

    def __init__(self, daily: float) -> None:  # noqa: D107 - test stub
        self._fixed = daily
        self._floor, self._ceil = 0.0, 1.0
        self.misses = 0

    def daily_vol(self, ticker, as_of):  # type: ignore[override]
        return self._fixed


def test_stop_barrier_only_fires_on_trades_past_it_at_exit():
    """A trade that exits inside the barrier is never counted as stopped."""
    # daily 3.9% -> 30-bar sigma = 0.039 * sqrt(30/390) = 1.0817%
    vol = _FixedVol(0.039)
    inside = _fill(1, "AAA", -0.005)
    outside = _fill(2, "BBB", -0.05)
    scan = replay_stop_barrier([inside, outside], vol, sigma_mult=1.0, hold_bars=30)
    assert scan.fires == 1
    assert scan.fire_rate == pytest.approx(0.5)


def test_stop_barrier_reprices_loss_to_the_barrier():
    """A stopped trade's P&L becomes exactly the barrier distance x notional."""
    vol = _FixedVol(0.039)
    stop_pct = 0.039 * (30 / 390) ** 0.5
    fill = _fill(1, "AAA", -0.05, shares=100.0, price=100.0)   # -$500 on $10k
    scan = replay_stop_barrier([fill], vol, sigma_mult=1.0, hold_bars=30)
    assert scan.net_after == pytest.approx(-stop_pct * 10_000.0)
    assert scan.pnl_delta == pytest.approx(-stop_pct * 10_000.0 - (-500.0))
    assert scan.pnl_delta > 0     # a stop can only help a trade it catches


def test_stop_barrier_uses_position_return_not_price_return():
    """`pnl_pct` is already signed by direction; the barrier must not re-sign it.

    A losing SHORT carries a negative pnl_pct just like a losing long, so it
    must trip the stop. Multiplying by `direction` would flip it into a winner.
    """
    vol = _FixedVol(0.039)
    short_loser = _fill(1, "AAA", -0.05, side="sell")
    assert short_loser.trade_return < 0
    scan = replay_stop_barrier([short_loser], vol, sigma_mult=1.0, hold_bars=30)
    assert scan.fires == 1


def test_stop_barrier_wide_enough_never_fires():
    vol = _FixedVol(0.039)
    fills = [_fill(i, f"T{i}", -0.02) for i in range(5)]
    scan = replay_stop_barrier(fills, vol, sigma_mult=10.0, hold_bars=30)
    assert scan.fires == 0
    assert scan.pnl_delta == pytest.approx(0.0)


def test_effective_sigma_multiple_documents_the_h14_unit_bug():
    """1.1 daily sigmas is ~3.97 sigmas of a 30-bar hold."""
    assert effective_sigma_multiple(1.1, 30) == pytest.approx(3.966, abs=0.01)
    assert effective_sigma_multiple(1.0, 390) == pytest.approx(1.0)


# ─── volatility lookup ──────────────────────────────────────────────────────

def test_daily_vol_lookup_has_no_lookahead(tmp_path):
    """Only bars strictly BEFORE the entry date may inform a counterfactual."""
    from datetime import date
    cache = tmp_path / "vol.json"
    cache.write_text(json.dumps({
        "AAA": {"2026-08-10": 0.02, "2026-08-11": 0.09},
    }))
    lookup = DailyVolLookup(cache, fallback=0.5)
    assert lookup.daily_vol("AAA", date(2026, 8, 11)) == pytest.approx(0.02)
    assert lookup.daily_vol("AAA", date(2026, 8, 12)) == pytest.approx(0.09)


def test_daily_vol_lookup_clamps_like_production(tmp_path):
    from datetime import date
    cache = tmp_path / "vol.json"
    cache.write_text(json.dumps({"AAA": {"2026-08-10": 0.90}}))
    lookup = DailyVolLookup(cache, floor=0.005, ceil=0.15)
    assert lookup.daily_vol("AAA", date(2026, 8, 11)) == pytest.approx(0.15)


def test_hold_window_vol_scales_by_sqrt_time(tmp_path):
    from datetime import date
    cache = tmp_path / "vol.json"
    cache.write_text(json.dumps({"AAA": {"2026-08-10": 0.04}}))
    lookup = DailyVolLookup(cache)
    full = lookup.hold_window_vol("AAA", date(2026, 8, 11), 390)
    half = lookup.hold_window_vol("AAA", date(2026, 8, 11), 30)
    assert full == pytest.approx(0.04)
    assert half == pytest.approx(0.04 * (30 / 390) ** 0.5)


# ─── placebo control ────────────────────────────────────────────────────────

def test_placebo_null_recovers_expected_removal():
    """Removing k of n fills at random recovers the proportional mean."""
    fills = [_fill(i, f"T{i}", -0.01) for i in range(10)]   # each -$100
    mean, std = block_count_placebo(fills, n_blocked=5, trials=500)
    assert mean == pytest.approx(-500.0)
    assert std == pytest.approx(0.0)


def test_placebo_z_flags_a_rule_that_is_no_better_than_chance():
    """A guard that blocks an unremarkable subset scores near zero."""
    fills = [_fill(i, f"T{i}", -0.01 if i % 2 else 0.01) for i in range(20)]
    result = replay_concurrency_guard(fills, lambda t: "one", max_per_bucket=2)
    assert abs(placebo_z(fills, result)) < 3.0
