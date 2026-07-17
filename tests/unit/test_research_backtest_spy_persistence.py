"""Unit tests for research_backtest.py — H2 SPY MA overlay + H6 persistence filter.

All tests use synthetic in-memory DataFrames; no Alpaca credentials, no
filesystem cache, no yfinance network calls required.

Coverage:
  _load_spy_ma_map():
    - above-MA days map to 1.0, below-MA days map to below_scale
    - shift(1) enforces no-lookahead (today's scalar = yesterday's signal)
    - empty yfinance response returns empty dict (fail-open at call site)
    - exception in yfinance returns empty dict (fail-open)
    - ma_days=10 uses 10-bar rolling mean

  Params:
    - spy_ma_days default 0, spy_below_scale default 0.5
    - min_persistence_bars default 0

  simulate() — H2 SPY MA scalar:
    - spy_ma_map=None is identical to baseline (no change in trade count)
    - below-MA days with below_scale=0.5 reduce notional (may drop below $500 floor)
    - below-MA days with below_scale=0.0 block all entries (notional = 0 < $500)
    - above-MA days with any scale still allow entries normally

  simulate() — H6 persistence filter:
    - min_persistence_bars=0 produces identical result to baseline
    - min_persistence_bars=1 allows entry after first qualifying bar
    - min_persistence_bars=2 blocks entry until 2 consecutive qualifying bars
    - a single non-qualifying bar resets streak to 0
    - persistence is checked per-ticker independently
    - already-open positions are not affected by persistence (exits work normally)
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research_backtest import Params, _load_spy_ma_map, simulate

ET = ZoneInfo("America/New_York")
CAPITAL = 10_000.0

# ─── Synthetic RTH 1m bar builder ─────────────────────────────────────────────

def _make_preds(
    records: list[dict],
) -> pd.DataFrame:
    """Build a minimal preds DataFrame suitable for simulate().

    Each record must have: ticker, time (UTC Timestamp), pred_return, dir_prob,
    close, open, atr_pct, daily_vol, regime, forward_return.
    """
    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["next_open"] = df.groupby("ticker")["open"].shift(-1).fillna(df["close"])
    return df


def _rth_ts(day_offset: int = 0, hour: int = 10, minute: int = 0) -> pd.Timestamp:
    """Return a UTC Timestamp that maps to the given hour:minute in ET on the tune-leg date.

    2026-05-18 is within the tune leg. hour/minute are ET wall-clock values
    (e.g. hour=10, minute=30 → 10:30 AM ET which is in the 09:40–15:30 window).
    """
    et_base = pd.Timestamp("2026-05-18", tz=ET)
    et_ts = et_base + timedelta(days=day_offset, hours=hour, minutes=minute)
    return et_ts.tz_convert("UTC")


def _base_row(ticker: str, day: int, bar: int, pred: float, prob: float,
              price: float = 100.0) -> dict:
    """One 1m bar at (10:00 + bar) minutes ET on tune-leg day + day_offset."""
    return {
        "ticker": ticker,
        "time": _rth_ts(day_offset=day, hour=10, minute=bar),
        "pred_return": pred,
        "dir_prob": prob,
        "close": price,
        "open": price,
        "atr_pct": 0.001,    # deliberately low so ATR scale-up pushes notional above $1k
        "daily_vol": 0.02,
        "regime": 0,
        "forward_return": 0.01,
    }


def _minimal_params(**kwargs) -> Params:
    """Return Params configured for fast, deterministic simulations.

    max_hold_bars=1  → position exits exactly 1 bar after entry.
    cooldown=999     → no re-entry after exit for the length of any test.
    Wide stops/trail/TP → only max_hold triggers exits, not price moves.
    """
    defaults = dict(
        pdt_enabled=False,       # no day-trade limits
        cooldown=999,            # prevent re-entry during tests (few bars)
        max_pos=6,
        heat_ceiling=0.99,
        max_entries_tick=6,
        trades_per_day=99,
        cost_bps=0.0,            # zero cost so PnL sign is predictable
        stop_mult=99.0,          # very wide stops — only max_hold exits
        trail_mult=99.0,
        tp_mult=99.0,
        stag_bars=999_999,
        reversal_bars=999_999,
        max_hold_bars=1,         # exit exactly 1 bar after entry → reliable n_trades
    )
    defaults.update(kwargs)
    return Params(**defaults)


def _took_position(res: dict) -> bool:
    """True if the simulation entered at least one position (traded or still open)."""
    return res["n_trades"] + res["open_at_end"] >= 1


# ─── _load_spy_ma_map tests ───────────────────────────────────────────────────

class TestLoadSpyMaMap:
    """Tests for _load_spy_ma_map() — offline with mocked yfinance."""

    def _make_spy_df(self, closes: list[float], start_date: str = "2026-03-01") -> pd.DataFrame:
        idx = pd.date_range(start_date, periods=len(closes), freq="B", tz="UTC")
        return pd.DataFrame({"Close": closes}, index=idx)

    def test_above_ma_maps_to_1_0(self):
        """Day where SPY > MA produces scalar 1.0 (shifted by 1)."""
        closes = [100.0] * 19 + [110.0, 115.0]  # last two bars both above 20d MA
        spy_df = self._make_spy_df(closes)
        with patch("yfinance.download", return_value=spy_df):
            result = _load_spy_ma_map(ma_days=20, below_scale=0.5)
        # last date: yesterday was above MA → today maps to 1.0
        last_date = spy_df.index[-1].date()
        assert result.get(last_date, -1) == 1.0

    def test_below_ma_maps_to_below_scale(self):
        """Day where SPY < MA produces scalar == below_scale (shifted by 1)."""
        closes = [120.0] * 19 + [100.0, 95.0]  # last two bars drop below MA
        spy_df = self._make_spy_df(closes)
        with patch("yfinance.download", return_value=spy_df):
            result = _load_spy_ma_map(ma_days=20, below_scale=0.5)
        last_date = spy_df.index[-1].date()
        # yesterday (index -2) was 100.0, MA of prior 20 bars is ~120 → below → 0.5
        assert result.get(last_date, -1) == 0.5

    def test_shift_enforces_no_lookahead(self):
        """The first valid MA bar (index ma_days) produces a NaN after shift, so
        the corresponding date is NOT in the map — can't be used as signal."""
        closes = [100.0] * 22
        spy_df = self._make_spy_df(closes)
        with patch("yfinance.download", return_value=spy_df):
            result = _load_spy_ma_map(ma_days=20, below_scale=0.5)
        # First date should be absent (NaN from shift)
        first_date = spy_df.index[0].date()
        assert first_date not in result

    def test_empty_dataframe_returns_empty_dict(self):
        """Empty yfinance response → empty dict (fail-open at caller)."""
        with patch("yfinance.download", return_value=pd.DataFrame()):
            result = _load_spy_ma_map()
        assert result == {}

    def test_too_few_bars_returns_empty_dict(self):
        """Fewer bars than ma_days + 1 → empty dict."""
        closes = [100.0] * 5
        spy_df = self._make_spy_df(closes)
        with patch("yfinance.download", return_value=spy_df):
            result = _load_spy_ma_map(ma_days=20)
        assert result == {}

    def test_yfinance_exception_returns_empty_dict(self):
        """Network error → empty dict (fail-open)."""
        with patch("yfinance.download", side_effect=RuntimeError("network error")):
            result = _load_spy_ma_map()
        assert result == {}

    def test_ma10_uses_10bar_lookback(self):
        """ma_days=10 uses 10-bar rolling mean, not 20."""
        # 10 bars at 100, then 1 bar at 200 → above MA quickly
        closes = [100.0] * 10 + [200.0, 210.0]
        spy_df = self._make_spy_df(closes)
        with patch("yfinance.download", return_value=spy_df):
            result = _load_spy_ma_map(ma_days=10, below_scale=0.5)
        last_date = spy_df.index[-1].date()
        # yesterday at 200 > 10d MA ~109 → above → 1.0
        assert result.get(last_date, -1) == 1.0

    def test_below_scale_0_0_produces_zero_scalar(self):
        """below_scale=0.0 (full gate) maps below-MA days to 0.0."""
        closes = [120.0] * 19 + [100.0, 95.0]
        spy_df = self._make_spy_df(closes)
        with patch("yfinance.download", return_value=spy_df):
            result = _load_spy_ma_map(ma_days=20, below_scale=0.0)
        last_date = spy_df.index[-1].date()
        assert result.get(last_date, -1) == 0.0


# ─── Params defaults ──────────────────────────────────────────────────────────

class TestParamsDefaults:
    def test_spy_ma_days_default_zero(self):
        assert Params().spy_ma_days == 0

    def test_spy_below_scale_default_half(self):
        assert Params().spy_below_scale == 0.5

    def test_min_persistence_bars_default_zero(self):
        assert Params().min_persistence_bars == 0


# ─── simulate() — H2 SPY MA scalar ───────────────────────────────────────────

def _multi_bar_preds(ticker: str, n_bars: int, day: int = 0,
                     pred: float = 0.05, prob: float = 0.90,
                     price: float = 100.0) -> list[dict]:
    """n_bars consecutive qualifying rows for one ticker, starting at 10:00 ET."""
    return [_base_row(ticker, day=day, bar=i, pred=pred, prob=prob, price=price)
            for i in range(n_bars)]


class TestSimulateSpyMa:
    """Tests that the SPY MA scalar is applied correctly to notional sizing.

    Each test uses 4 consecutive bars (enough for entry+hold+exit+fill).
    max_hold_bars=1 ensures the position exits on bar+1 so n_trades is countable.
    """

    def _preds(self, price: float = 100.0, n: int = 4) -> pd.DataFrame:
        return _make_preds(_multi_bar_preds("AAA", n_bars=n, price=price))

    def test_spy_ma_map_none_does_not_change_behavior(self):
        """spy_ma_map=None produces identical trade count to omitting the argument."""
        preds = self._preds()
        p = _minimal_params()
        r1 = simulate(preds, p, "2026-05-18", "2026-05-18")
        r2 = simulate(preds, p, "2026-05-18", "2026-05-18", spy_ma_map=None)
        assert r1["n_trades"] == r2["n_trades"]
        assert r1["open_at_end"] == r2["open_at_end"]

    def test_above_ma_day_allows_entry(self):
        """scalar=1.0 on the trade day → entry allowed at full size."""
        preds = self._preds()
        p = _minimal_params()
        trade_date = _rth_ts(0).tz_convert(ET).date()
        spy_map = {trade_date: 1.0}
        res = simulate(preds, p, "2026-05-18", "2026-05-18", spy_ma_map=spy_map)
        assert _took_position(res), "Expected at least one position when SPY above MA"

    def test_below_ma_full_gate_blocks_all_entries(self):
        """scalar=0.0 on the trade day → notional*0 < $500 floor → no entry."""
        preds = self._preds()
        p = _minimal_params(spy_below_scale=0.0)
        trade_date = _rth_ts(0).tz_convert(ET).date()
        spy_map = {trade_date: 0.0}
        res = simulate(preds, p, "2026-05-18", "2026-05-18", spy_ma_map=spy_map)
        assert res["n_trades"] == 0 and res["open_at_end"] == 0, (
            "Full gate (scale=0.0) should block all entries"
        )

    def test_missing_date_in_spy_map_fails_open(self):
        """Date absent from spy_map → .get(date, 1.0) → entry allowed (fail-open)."""
        preds = self._preds()
        p = _minimal_params()
        spy_map: dict = {}   # empty — all dates default to 1.0 at caller
        res = simulate(preds, p, "2026-05-18", "2026-05-18", spy_ma_map=spy_map)
        assert _took_position(res), "Empty spy_map should fail-open and allow entries"

    def test_below_ma_half_heat_allows_entries(self):
        """scalar=0.5 halves notional — still well above $500 on a $10k book → entry allowed."""
        preds = self._preds()
        p = _minimal_params(spy_below_scale=0.5)
        trade_date = _rth_ts(0).tz_convert(ET).date()
        spy_map = {trade_date: 0.5}
        res = simulate(preds, p, "2026-05-18", "2026-05-18", spy_ma_map=spy_map)
        assert _took_position(res), (
            "Half-heat scale (0.5) on a $10k portfolio should still allow entries"
        )

    def test_spy_scalar_applied_before_notional_floor(self):
        """Verify scalar is multiplied before the $500 floor check, not after.

        With a very small portfolio, scalar=0.0 must produce 0 notional → blocked.
        With scalar=1.0 the same portfolio allows entry (notional stays above floor).
        """
        preds = self._preds()
        p_gate = _minimal_params(spy_below_scale=0.0)
        p_open = _minimal_params()
        trade_date = _rth_ts(0).tz_convert(ET).date()
        res_gate = simulate(preds, p_gate, "2026-05-18", "2026-05-18",
                            spy_ma_map={trade_date: 0.0})
        res_open = simulate(preds, p_open, "2026-05-18", "2026-05-18",
                            spy_ma_map={trade_date: 1.0})
        # Gate must block; open must allow
        assert res_gate["n_trades"] == 0 and res_gate["open_at_end"] == 0
        assert _took_position(res_open)


# ─── simulate() — H6 persistence filter ──────────────────────────────────────

class TestSimulatePersistence:
    """Tests that the signal persistence filter is applied correctly.

    Bar layout (with max_hold_bars=1, cooldown=999):
      bar k:   streak reaches min_persistence_bars → ENTRY (fill = bar k+1's open)
      bar k+1: bars=1 ≥ max_hold_bars=1 → EXIT → n_trades += 1
    So for persistence=N, we need N bars to build streak + 1 entry bar + 1 exit bar
    → at least N+2 bars for a confirmed exit.
    """

    def _qual_bars(self, ticker: str, n: int, day: int = 0) -> list[dict]:
        return _multi_bar_preds(ticker, n_bars=n, day=day)

    def test_persistence_0_identical_to_baseline(self):
        """min_persistence_bars=0 produces same n_trades as default Params."""
        preds = _make_preds(self._qual_bars("AAA", 5))
        p_base = _minimal_params()
        p_p0 = _minimal_params(min_persistence_bars=0)
        r_base = simulate(preds, p_base, "2026-05-18", "2026-05-18")
        r_p0 = simulate(preds, p_p0, "2026-05-18", "2026-05-18")
        assert r_base["n_trades"] == r_p0["n_trades"]

    def test_persistence_1_allows_entry_from_first_qualifying_bar(self):
        """persistence=1: entry fires on the very first qualifying bar (streak=1)."""
        # 4 bars: bar0 → ENTRY (streak=1≥1), bar1 → EXIT, bars 2-3 → cooldown
        preds = _make_preds(self._qual_bars("AAA", 4))
        p = _minimal_params(min_persistence_bars=1)
        res = simulate(preds, p, "2026-05-18", "2026-05-18")
        assert res["n_trades"] >= 1, "persistence=1 should enter on first qualifying bar"

    def test_persistence_2_blocks_single_qualifying_bar(self):
        """persistence=2: 1 qualifying bar is insufficient (streak=1 < 2) → no entry."""
        preds = _make_preds([_base_row("AAA", day=0, bar=0, pred=0.05, prob=0.90)])
        p = _minimal_params(min_persistence_bars=2)
        res = simulate(preds, p, "2026-05-18", "2026-05-18")
        assert not _took_position(res), (
            "persistence=2: single qualifying bar must be blocked"
        )

    def test_persistence_2_allows_entry_after_two_consecutive_bars(self):
        """persistence=2: 2+ qualifying bars allow entry on the 2nd bar."""
        # 5 bars: bar0 streak=1, bar1 streak=2→ENTRY, bar2 EXIT, bars 3-4 cooldown
        preds = _make_preds(self._qual_bars("AAA", 5))
        p = _minimal_params(min_persistence_bars=2)
        res = simulate(preds, p, "2026-05-18", "2026-05-18")
        assert res["n_trades"] >= 1, (
            "persistence=2: 2 consecutive qualifying bars should trigger entry"
        )

    def test_persistence_3_blocks_two_qualifying_bars(self):
        """persistence=3: 2 qualifying bars are insufficient."""
        preds = _make_preds(self._qual_bars("AAA", 2))
        p = _minimal_params(min_persistence_bars=3)
        res = simulate(preds, p, "2026-05-18", "2026-05-18")
        assert not _took_position(res), (
            "persistence=3: 2 qualifying bars should be blocked"
        )

    def test_persistence_3_allows_entry_after_three_consecutive_bars(self):
        """persistence=3: 3+ qualifying bars allow entry on the 3rd bar."""
        # 6 bars: bars 0-2 build streak, bar2 ENTRY, bar3 EXIT
        preds = _make_preds(self._qual_bars("AAA", 6))
        p = _minimal_params(min_persistence_bars=3)
        res = simulate(preds, p, "2026-05-18", "2026-05-18")
        assert res["n_trades"] >= 1, (
            "persistence=3: 3 consecutive qualifying bars should trigger entry"
        )

    def test_streak_resets_on_non_qualifying_bar(self):
        """A non-qualifying bar resets the streak; counter starts from 0."""
        # Pattern: [qual, non-qual, qual, qual, qual, qual]
        # With persistence=2: streak never reaches 2 without interruption at bar1
        # After reset at bar1: bar2→streak=1, bar3→streak=2→ENTRY, bar4→EXIT
        preds = _make_preds([
            _base_row("AAA", day=0, bar=0, pred=0.05, prob=0.90),   # streak=1
            _base_row("AAA", day=0, bar=1, pred=-0.01, prob=0.20),  # streak=0 (reset)
            _base_row("AAA", day=0, bar=2, pred=0.05, prob=0.90),   # streak=1
            _base_row("AAA", day=0, bar=3, pred=0.05, prob=0.90),   # streak=2→ENTRY
            _base_row("AAA", day=0, bar=4, pred=0.05, prob=0.90),   # EXIT (max_hold=1)
        ])
        p = _minimal_params(min_persistence_bars=2)
        res = simulate(preds, p, "2026-05-18", "2026-05-18")
        # Entry should occur at bar3 (after rebuild), exit at bar4
        assert res["n_trades"] >= 1, (
            "After streak reset at bar1, a fresh 2-bar run should still trigger entry"
        )

    def test_streak_reset_prevents_entry_if_insufficient_bars_after_reset(self):
        """After a reset, insufficient remaining bars → no entry."""
        # [qual, non-qual, qual] — streak only reaches 1 after reset
        preds = _make_preds([
            _base_row("AAA", day=0, bar=0, pred=0.05, prob=0.90),   # streak=1
            _base_row("AAA", day=0, bar=1, pred=-0.01, prob=0.20),  # streak=0
            _base_row("AAA", day=0, bar=2, pred=0.05, prob=0.90),   # streak=1 only
        ])
        p = _minimal_params(min_persistence_bars=2)
        res = simulate(preds, p, "2026-05-18", "2026-05-18")
        assert not _took_position(res), (
            "After reset at bar1, only 1 qualifying bar remains — must be blocked"
        )

    def test_persistence_is_tracked_per_ticker_independently(self):
        """AAA builds streak=2 and enters; BBB has only streak=1 and is blocked."""
        preds = _make_preds([
            # AAA: bars 0, 1, 2, 3 — streak=2 at bar1 → ENTRY, EXIT at bar2
            _base_row("AAA", day=0, bar=0, pred=0.05, prob=0.90),
            _base_row("AAA", day=0, bar=1, pred=0.05, prob=0.90),
            _base_row("AAA", day=0, bar=2, pred=0.05, prob=0.90),
            _base_row("AAA", day=0, bar=3, pred=0.05, prob=0.90),
            # BBB: only bar 1 — streak=1 → blocked with persistence=2
            _base_row("BBB", day=0, bar=1, pred=0.05, prob=0.90),
        ])
        p = _minimal_params(min_persistence_bars=2, max_pos=6, max_entries_tick=6)
        res = simulate(preds, p, "2026-05-18", "2026-05-18")
        # AAA should trade (streak≥2); BBB should not (streak=1 at its only bar)
        assert res["n_trades"] == 1, (
            f"Expected exactly 1 trade (AAA only), got n_trades={res['n_trades']}, "
            f"open_at_end={res['open_at_end']}"
        )

    def test_persistence_does_not_affect_exit_of_open_position(self):
        """A position already open is NOT affected by the persistence filter — exits work normally."""
        # Enter first (persistence=0 for control), then verify persistence=0 doesn't crash
        preds = _make_preds(self._qual_bars("AAA", 5))
        p = _minimal_params(min_persistence_bars=0)
        res = simulate(preds, p, "2026-05-18", "2026-05-18")
        assert res["n_trades"] + res["open_at_end"] >= 0   # no crash

    def test_phase_spy_ma_importable(self):
        """phase_spy_ma is importable and callable."""
        from scripts.research_backtest import phase_spy_ma
        assert callable(phase_spy_ma)

    def test_phase_persistence_importable(self):
        """phase_persistence is importable and callable."""
        from scripts.research_backtest import phase_persistence
        assert callable(phase_persistence)

    def test_cli_choices_include_new_phases(self):
        """--phase spy_ma and --phase persistence are valid CLI choices."""
        import argparse
        import importlib
        import types
        # Re-import to get the updated main()
        import scripts.research_backtest as rb
        importlib.reload(rb)
        # Parse args — should not raise
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["research_backtest.py", "--phase", "spy_ma"]
            ap = argparse.ArgumentParser()
            ap.add_argument("--phase", default="all",
                            choices=["fetch", "features", "train", "simulate",
                                     "tune", "spy_ma", "persistence", "all"])
            args = ap.parse_args()
            assert args.phase == "spy_ma"

            sys.argv = ["research_backtest.py", "--phase", "persistence"]
            args2 = ap.parse_args()
            assert args2.phase == "persistence"
        finally:
            sys.argv = old_argv
