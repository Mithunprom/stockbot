"""Unit tests for the Profitability Diagnosis Agent metric + detector logic."""

from __future__ import annotations

import math

import pytest

from src.agents.profitability_agent import (
    StrategyDesign,
    attribution_by,
    core_metrics,
    daily_sharpe,
    detect_horizon_mismatch,
    detect_negative_expectancy,
    detect_stop_hemorrhage,
    detect_ticker_bleeders,
    detect_undesigned_exit_dominance,
    diagnose,
    hold_minutes,
)


def _trade(
    ticker: str = "AAPL",
    pnl: float = 10.0,
    entry: str = "2026-07-01T14:00:00+00:00",
    exit_: str = "2026-07-01T14:30:00+00:00",
    exit_reason: str = "take_profit",
    pnl_pct: float = 0.5,
) -> dict:
    return {
        "ticker": ticker,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "entry_time": entry,
        "exit_time": exit_,
        "exit_reason": exit_reason,
    }


class TestMetrics:
    def test_hold_minutes(self):
        assert hold_minutes(_trade()) == 30.0

    def test_hold_minutes_open_trade(self):
        t = _trade()
        t["exit_time"] = None
        assert hold_minutes(t) is None

    def test_core_metrics_math(self):
        trades = [_trade(pnl=30.0), _trade(pnl=-10.0), _trade(pnl=-10.0)]
        m = core_metrics(trades)
        assert m["n_trades"] == 3
        assert m["total_pnl"] == 10.0
        assert m["win_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert m["profit_factor"] == pytest.approx(30.0 / 20.0)
        assert m["avg_win"] == 30.0
        assert m["avg_loss"] == -10.0

    def test_profit_factor_no_losses_is_inf(self):
        m = core_metrics([_trade(pnl=5.0)])
        assert m["profit_factor"] == math.inf

    def test_daily_sharpe_positive_days(self):
        trades = [
            _trade(pnl=100.0, exit_="2026-07-01T15:00:00+00:00"),
            _trade(pnl=110.0, exit_="2026-07-02T15:00:00+00:00"),
            _trade(pnl=90.0, exit_="2026-07-03T15:00:00+00:00"),
        ]
        risk = daily_sharpe(trades, portfolio_value=100_000)
        assert risk["n_days"] == 3
        assert risk["sharpe"] > 0
        assert risk["max_drawdown_pct"] == 0.0

    def test_daily_sharpe_insufficient_days(self):
        risk = daily_sharpe([_trade()], portfolio_value=100_000)
        assert risk["sharpe"] is None

    def test_attribution_groups_and_sorts_worst_first(self):
        trades = [
            _trade(ticker="AMD", pnl=-50.0),
            _trade(ticker="MSTR", pnl=100.0),
            _trade(ticker="AMD", pnl=-25.0),
        ]
        att = attribution_by(trades, "ticker")
        assert list(att.keys())[0] == "AMD"
        assert att["AMD"]["total_pnl"] == -75.0
        assert att["AMD"]["n"] == 2
        assert att["MSTR"]["win_rate"] == 1.0


class TestDetectors:
    def test_horizon_mismatch_fires_on_short_holds(self):
        # 25-minute holds against a 390-minute design
        trades = [_trade() for _ in range(10)]
        f = detect_horizon_mismatch(trades, StrategyDesign())
        assert f is not None
        assert f.severity == "CRITICAL"
        assert f.metrics["hold_ratio"] < 0.5

    def test_horizon_mismatch_silent_when_holds_match_design(self):
        trades = [
            _trade(entry="2026-07-01T14:00:00+00:00", exit_="2026-07-02T14:00:00+00:00")
            for _ in range(10)
        ]
        assert detect_horizon_mismatch(trades, StrategyDesign()) is None

    def test_undesigned_exit_dominance(self):
        trades = [_trade(exit_reason="signal_reversal", pnl=-5.0) for _ in range(6)]
        trades += [_trade(exit_reason="stop_loss", pnl=-20.0) for _ in range(4)]
        f = detect_undesigned_exit_dominance(trades, StrategyDesign())
        assert f is not None
        assert f.metrics["reason"] == "signal_reversal"
        assert f.metrics["share"] == 0.6

    def test_undesigned_exit_silent_when_designed_exits_dominate(self):
        trades = [_trade(exit_reason="take_profit") for _ in range(9)]
        trades += [_trade(exit_reason="signal_reversal")]
        assert detect_undesigned_exit_dominance(trades, StrategyDesign()) is None

    def test_stop_hemorrhage(self):
        trades = [
            _trade(exit_reason="stop_loss", pnl=-300.0),
            _trade(exit_reason="trailing_stop", pnl=-200.0),
            _trade(exit_reason="take_profit", pnl=400.0),
        ]
        f = detect_stop_hemorrhage(trades)
        assert f is not None
        assert f.metrics["stop_damage"] == -500.0

    def test_stop_hemorrhage_silent_when_tp_pays(self):
        trades = [
            _trade(exit_reason="stop_loss", pnl=-100.0),
            _trade(exit_reason="take_profit", pnl=400.0),
        ]
        assert detect_stop_hemorrhage(trades) is None

    def test_ticker_bleeders_requires_min_trades(self):
        trades = [_trade(ticker="AMD", pnl=-10.0) for _ in range(3)]
        trades += [_trade(ticker="NVDA", pnl=-99.0)]  # only 1 trade — excluded
        f = detect_ticker_bleeders(trades)
        assert f is not None
        assert "AMD" in f.metrics["bleeders"]
        assert "NVDA" not in f.metrics["bleeders"]

    def test_negative_expectancy(self):
        trades = [_trade(pnl=-40.0), _trade(pnl=-40.0), _trade(pnl=30.0)]
        f = detect_negative_expectancy(trades)
        assert f is not None
        assert f.metrics["profit_factor"] < 1.0

    def test_negative_expectancy_silent_when_profitable(self):
        trades = [_trade(pnl=50.0), _trade(pnl=-20.0)]
        assert detect_negative_expectancy(trades) is None


class TestDiagnose:
    def test_findings_ranked_by_severity(self):
        # Short holds (CRITICAL) + losing book (HIGH) + tiny sample (INFO)
        trades = [_trade(pnl=-10.0, exit_reason="signal_reversal") for _ in range(5)]
        findings = diagnose(trades, window_days=30)
        assert findings, "expected at least one finding"
        severities = [f.severity for f in findings]
        rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
        assert severities == sorted(severities, key=lambda s: rank[s])
        assert findings[0].code in ("HORIZON_MISMATCH", "UNDESIGNED_EXIT_DOMINANT")

    def test_diagnose_empty_trades_no_crash(self):
        findings = diagnose([], window_days=30)
        codes = [f.code for f in findings]
        assert "SMALL_SAMPLE" in codes
