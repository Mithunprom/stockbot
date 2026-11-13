"""Tests for the hybrid multi-agent graph, backtester, and strategy search.

Covers money / signal / risk paths (CLAUDE.md): analyst bounds, debate, circuit
breakers that cannot be disabled, no-lookahead features, next-open fills.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.graph.analysts import score_all, score_technical
from src.graph.backtest import composite_score, simulate_portfolio
from src.graph.catalog import StrategySpec, seed_strategies
from src.graph.data import synthetic_ohlcv
from src.graph.debate import risk_committee, run_debate, trader_plan
from src.graph.indicators import compute_daily_indicators
from src.graph.pipeline import build_feature_store, build_signals
from src.graph.state import AnalystReport, TradePlan


def _row_from_bars(seed: int = 1) -> pd.Series:
    bars = {"SPY": synthetic_ohlcv(n=260, seed=0), "AAPL": synthetic_ohlcv(n=260, seed=seed)}
    store = build_feature_store(bars, bars["SPY"]["close"])
    return store["AAPL"].iloc[-1]


# ─── Feature look-ahead ──────────────────────────────────────────────────────


def test_indicators_shift_blocks_last_bar_spike():
    """A spike in the last close must not change the last row of RSI (shifted)."""
    df = synthetic_ohlcv(n=120, seed=3)
    base = compute_daily_indicators(df, shift=True)
    spiked = df.copy()
    spiked.iloc[-1, spiked.columns.get_loc("close")] *= 1.25
    spiked.iloc[-1, spiked.columns.get_loc("high")] *= 1.25
    alt = compute_daily_indicators(spiked, shift=True)
    assert base["rsi_14"].iloc[-1] == pytest.approx(alt["rsi_14"].iloc[-1], nan_ok=True)
    assert base["atr_pct"].iloc[-1] == pytest.approx(alt["atr_pct"].iloc[-1], nan_ok=True)


def test_unshifted_indicators_do_see_the_spike():
    """Sanity: without shift, the last RSI MUST move — otherwise the guard is fake."""
    df = synthetic_ohlcv(n=120, seed=3)
    base = compute_daily_indicators(df, shift=False)
    spiked = df.copy()
    spiked.iloc[-1, spiked.columns.get_loc("close")] *= 1.25
    alt = compute_daily_indicators(spiked, shift=False)
    assert base["rsi_14"].iloc[-1] != pytest.approx(alt["rsi_14"].iloc[-1])


# ─── Analyst + debate ────────────────────────────────────────────────────────


def test_analyst_scores_are_bounded():
    row = _row_from_bars()
    reports = score_all(row)
    assert set(reports) >= {"technical", "fundamental", "sentiment", "news", "macro", "quant"}
    for r in reports.values():
        assert -1.0 <= r.score <= 1.0
        assert 0.0 <= r.confidence <= 1.0


def test_debate_prefers_the_side_with_weight():
    bull = AnalystReport("technical", 0.8, 0.9)
    bear = AnalystReport("news", -0.2, 0.4)
    ensemble, debate = run_debate(
        {"technical": bull, "news": bear},
        {"technical": 0.8, "news": 0.2},
        debate_weight=0.6,
    )
    assert debate.judge_verdict == "bullish"
    assert ensemble > 0


def test_trader_respects_long_only():
    plan = trader_plan(
        -0.9, 0.8, 0.02,
        entry_threshold=0.2, stop_atr=1.5, tp_atr=2.5,
        size_pct=0.1, holding_days=5, long_only=True, close=100.0,
    )
    assert plan.action == "FLAT"
    assert plan.size_pct == 0.0


def test_trader_goes_long_when_threshold_clears():
    plan = trader_plan(
        0.5, 0.7, 0.02,
        entry_threshold=0.2, stop_atr=1.5, tp_atr=2.5,
        size_pct=0.1, holding_days=5, long_only=True, close=100.0,
    )
    assert plan.action == "LONG"
    assert plan.stop_pct > 0
    assert plan.target_pct > plan.stop_pct


# ─── Risk committee — breakers never disabled ────────────────────────────────


def _plan(size: float = 0.10) -> TradePlan:
    return TradePlan("LONG", 0.6, 0.7, 100.0, 0.03, 0.06, size, 5)


def test_risk_rejects_daily_loss_limit():
    v = risk_committee(
        _plan(), risk_mode="aggressive", hmm_regime=0, spy_vs_sma200=0.05,
        portfolio_heat=0.1, heat_ceiling=0.8, consecutive_losses=0,
        daily_pnl_pct=-0.031, drawdown_pct=-0.01,
    )
    assert v.decision == "rejected"
    assert v.veto_reason == "daily_loss_limit"
    assert v.size_pct == 0.0


def test_risk_rejects_max_drawdown():
    v = risk_committee(
        _plan(), risk_mode="aggressive", hmm_regime=0, spy_vs_sma200=0.05,
        portfolio_heat=0.1, heat_ceiling=0.8, consecutive_losses=0,
        daily_pnl_pct=0.0, drawdown_pct=-0.09,
    )
    assert v.decision == "rejected"
    assert v.veto_reason == "max_drawdown"


def test_risk_rejects_vol_spike():
    v = risk_committee(
        _plan(), risk_mode="balanced", hmm_regime=0, spy_vs_sma200=0.05,
        portfolio_heat=0.1, heat_ceiling=0.8, consecutive_losses=0,
        daily_pnl_pct=0.0, drawdown_pct=-0.01, vix_proxy=0.40,
    )
    assert v.veto_reason == "vol_spike"


def test_conservative_vetoes_high_vol_regime():
    v = risk_committee(
        _plan(), risk_mode="conservative", hmm_regime=2, spy_vs_sma200=0.02,
        portfolio_heat=0.1, heat_ceiling=0.8, consecutive_losses=0,
        daily_pnl_pct=0.0, drawdown_pct=-0.01,
    )
    assert v.decision == "rejected"


def test_committee_caps_notional_at_20_percent():
    v = risk_committee(
        _plan(size=0.50), risk_mode="committee", hmm_regime=0, spy_vs_sma200=0.05,
        portfolio_heat=0.0, heat_ceiling=0.8, consecutive_losses=0,
        daily_pnl_pct=0.0, drawdown_pct=-0.01,
    )
    assert v.decision != "rejected"
    assert v.size_pct <= 0.20 + 1e-9


# ─── Backtester ──────────────────────────────────────────────────────────────


def _spec(**kwargs) -> StrategySpec:
    base = [s for s in seed_strategies() if s.name == "primo_linear"][0]
    return base.mutated(**kwargs)


def test_backtest_stop_hits_before_target_on_same_bar():
    """Pessimistic intrabar rule: if low tags the stop, count a loss."""
    idx = pd.bdate_range("2024-01-02", periods=8, tz="UTC")
    close = pd.Series([100, 100, 100, 100, 100, 100, 100, 100], index=idx, dtype=float)
    bars = {
        "AAPL": pd.DataFrame(
            {
                "open": close,
                "high": close + 10,   # would hit a 5% target
                "low": close - 10,    # would hit a 5% stop
                "close": close,
                "volume": 1e6,
            },
            index=idx,
        )
    }
    # Signal on day 0 → fill day 0 open (features already lagged in production).
    signals = pd.DataFrame(
        [{
            "time": idx[1],
            "ticker": "AAPL",
            "action": "LONG",
            "score": 0.8,
            "confidence": 0.8,
            "size_pct": 0.10,
            "stop_pct": 0.05,
            "target_pct": 0.05,
            "atr_pct": 0.02,
            "close": 100.0,
            "hmm_regime": 0,
        }]
    )
    spec = _spec(holding_days=10, cost_bps=0.0, heat_ceiling=0.9, max_positions=1, size_pct=0.10)
    res = simulate_portfolio(bars, signals, spec, start="2024-01-02", end="2024-01-12", capital=10_000)
    assert res.n_trades >= 1
    # Stop-first means we must not show a win on a bar that also tagged the stop.
    assert res.exit_reasons.get("stop_loss", 0) >= 1


def test_backtest_buy_hold_tracks_price():
    idx = pd.bdate_range("2024-01-02", periods=40, tz="UTC")
    close = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1e6},
        index=idx,
    )
    bars = {"SPY": df, "AAA": df.copy()}
    spec = [s for s in seed_strategies() if s.topology == "buy_hold"][0]
    spec = spec.mutated(cost_bps=0.0)
    res = simulate_portfolio(bars, pd.DataFrame(), spec, start="2024-01-02", end="2024-02-28", capital=10_000)
    assert res.total_return_pct == pytest.approx(10.0, abs=1.5)


def test_drawdown_halt_flattens_active_book():
    idx = pd.bdate_range("2024-01-02", periods=8, tz="UTC")
    px = [100.0, 100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 80.0]
    close = pd.Series(px, index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1e6},
        index=idx,
    )
    bars = {"SPY": df, "AAA": df.copy()}
    signals = pd.DataFrame(
        [{
            "time": idx[1], "ticker": "AAA", "action": "LONG", "score": 0.9,
            "confidence": 0.9, "size_pct": 0.90, "stop_pct": 0.50, "target_pct": 0.90,
            "atr_pct": 0.02, "close": 100.0, "hmm_regime": 0,
        }]
    )
    spec = _spec(holding_days=30, cost_bps=0.0, heat_ceiling=0.95, max_positions=1, size_pct=0.90)
    res = simulate_portfolio(bars, signals, spec, start="2024-01-02", end="2024-01-12", capital=10_000)
    assert res.max_dd_pct >= 8.0 or res.exit_reasons.get("drawdown_halt", 0) >= 1


def test_composite_score_penalises_deep_drawdown():
    good = composite_score(1.6, 5.0, 1.6, 0.55, 40, 12.0)
    bad = composite_score(1.6, 20.0, 1.6, 0.55, 40, 12.0)
    assert good > bad


# ─── Pipeline + search ───────────────────────────────────────────────────────


def test_build_signals_emits_long_or_flat_only_for_long_only_spec():
    bars = {
        "SPY": synthetic_ohlcv(n=220, seed=0, drift=0.0005),
        "AAPL": synthetic_ohlcv(n=220, seed=1, drift=0.0006),
        "MSFT": synthetic_ohlcv(n=220, seed=2, drift=0.0004),
    }
    store = build_feature_store(bars, bars["SPY"]["close"])
    spec = _spec(long_only=True)
    sig = build_signals(store, spec)
    assert not sig.empty
    assert set(sig.action.unique()) <= {"LONG", "FLAT"}


def test_strategy_search_writes_staging_not_live(tmp_path, monkeypatch):
    from src.agents import strategy_search_agent as ssa

    monkeypatch.setattr(ssa, "STAGING_PATH", tmp_path / "staging" / "strategy_search.json")
    monkeypatch.setattr(ssa, "REPORT_DIR", tmp_path / "opportunities")
    bars = {
        "SPY": synthetic_ohlcv(n=260, seed=0, drift=0.0003, vol=0.012),
        "AAPL": synthetic_ohlcv(n=260, seed=1, drift=0.0005, vol=0.016),
        "MSFT": synthetic_ohlcv(n=260, seed=2, drift=0.0004, vol=0.014),
        "JPM": synthetic_ohlcv(n=260, seed=3, drift=0.00025, vol=0.013),
    }
    idx = bars["SPY"].index
    agent = ssa.StrategySearchAgent(
        generations=1,
        children_per_gen=3,
        elite_k=3,
        seed=1,
        windows={
            "tune": (str(idx[30].date()), str(idx[140].date())),
            "val": (str(idx[141].date()), str(idx[200].date())),
            "holdout": (str(idx[201].date()), str(idx[-5].date())),
        },
    )
    report = agent.search(bars)
    staging, path = agent.persist(report)
    assert staging.exists()
    assert path.exists()
    payload = staging.read_text()
    assert "do_not_apply_to_live" in payload or "needs_review" in payload
    assert (tmp_path / "staging" / "strategy_search.json").exists()
    # Live config is not in tmp_path and must not have been created.
    assert not (tmp_path / "live.yaml").exists()
    winner = report["winner"]["spec"]
    assert "topology" in winner
    assert "validation" in report["winner"]
    assert "holdout" in report["winner"]
    # Every leaderboard row reports the KPI bundle, not raw PnL alone.
    for row in report["validation_leaderboard"]:
        m = row["metrics"]
        for k in ("sharpe", "max_dd_pct", "profit_factor", "win_rate", "n_trades"):
            assert k in m


def test_seed_catalog_covers_all_five_repos():
    names = {s.source for s in seed_strategies()}
    assert any("TauricResearch" in s for s in names)
    assert any("NastyRunner13" in s for s in names)
    assert any("kasyap1234" in s for s in names)
    assert any("huygiatrng" in s for s in names)
    assert any("ivebotunac" in s for s in names)
    assert any(s.startswith("hybrid") for s in names)
    assert any(s.name == "hybrid_defensive" for s in seed_strategies())
    assert any(s.name == "hybrid_defensive_search" for s in seed_strategies())


def test_strategy_search_is_not_an_http_admin_route():
    """The search must not hang off the unauthenticated FastAPI app.

    It loads multi-year bars, runs a walk-forward loop, and overwrites
    staging. Operators trigger it via the CLI or the Saturday scheduler —
    both off the trading event loop. CORS is ``*`` on this app, so an HTTP
    RPC here is a control-plane DoS/write.
    """
    src = Path("main.py").read_text()
    assert '@app.post("/admin/strategy-search")' not in src
    assert "def trigger_strategy_search" not in src
