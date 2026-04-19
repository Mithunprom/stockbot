#!/usr/bin/env python3
"""Pipeline B Step-by-Step Test Agent.

Tests each stage of Pipeline B independently:
  Step 1: Database connectivity + data freshness
  Step 2: Feature matrix availability (indicators computed)
  Step 3: Technical scoring (RSI, MACD, Bollinger, etc.)
  Step 4: Fundamental scoring (P/E, earnings, revenue)
  Step 5: Market regime scoring (VIX, SPY momentum)
  Step 6: Sentiment scoring (FinBERT)
  Step 7: Pipeline B composite signal generation
  Step 8: Position sizer gate (cost threshold, conviction)
  Step 9: Alpaca connectivity + order readiness

Usage:
    python scripts/test_pipeline_b_steps.py
    python scripts/test_pipeline_b_steps.py --db-url "postgresql+asyncpg://..."
    python scripts/test_pipeline_b_steps.py --ticker AAPL
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── ANSI colors ──────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
WARN = f"{YELLOW}WARN{RESET}"
SKIP = f"{DIM}SKIP{RESET}"


class StepResult:
    def __init__(self):
        self.checks: list[bool] = []
        self.warnings: int = 0

    def ok(self, msg: str, detail: str = "") -> bool:
        d = f"  ({detail})" if detail else ""
        print(f"    [{PASS}] {msg}{d}")
        self.checks.append(True)
        return True

    def fail(self, msg: str, detail: str = "") -> bool:
        d = f"  ({detail})" if detail else ""
        print(f"    [{FAIL}] {msg}{d}")
        self.checks.append(False)
        return False

    def warn(self, msg: str, detail: str = "") -> None:
        d = f"  ({detail})" if detail else ""
        print(f"    [{WARN}] {msg}{d}")
        self.warnings += 1

    def skip(self, msg: str, detail: str = "") -> None:
        d = f"  ({detail})" if detail else ""
        print(f"    [{SKIP}] {msg}{d}")

    def check(self, condition: bool, msg: str, detail: str = "") -> bool:
        return self.ok(msg, detail) if condition else self.fail(msg, detail)


async def run_tests(db_url: str | None, ticker: str) -> int:
    r = StepResult()

    print(f"\n{BOLD}{'='*65}")
    print(f"  Pipeline B — Step-by-Step Test Agent")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}  |  Ticker: {ticker}")
    print(f"{'='*65}{RESET}")

    # ── Step 1: Database ─────────────────────────────────────────────────────
    print(f"\n{BOLD}Step 1: Database Connectivity & Data{RESET}")
    session_factory = None
    try:
        if db_url:
            os.environ["DATABASE_URL"] = db_url
        from src.data.db import init_db, get_session_factory
        await init_db()
        session_factory = get_session_factory()
        r.ok("Database connected")

        # Check OHLCV freshness
        from sqlalchemy import text
        async with session_factory() as session:
            row = await session.execute(text(
                "SELECT COUNT(*) as n, MAX(time) as latest "
                "FROM ohlcv_1m WHERE time > NOW() - INTERVAL '3 days'"
            ))
            result = row.fetchone()
            bar_count = result[0]
            latest_bar = result[1]
            r.check(bar_count > 0, f"OHLCV bars (last 3d): {bar_count:,}")

            if latest_bar:
                age_h = (datetime.now(timezone.utc) - latest_bar).total_seconds() / 3600
                r.check(age_h < 48, f"Latest bar age: {age_h:.1f}h",
                        "stale — weekend?" if age_h > 24 else "fresh")
            else:
                r.fail("No recent OHLCV bars")

            # Check feature_matrix freshness
            row = await session.execute(text(
                "SELECT COUNT(*) as n, MAX(time) as latest "
                "FROM feature_matrix WHERE time > NOW() - INTERVAL '3 days'"
            ))
            result = row.fetchone()
            feat_count = result[0]
            feat_latest = result[1]
            r.check(feat_count > 0, f"Feature rows (last 3d): {feat_count:,}")

            if feat_latest:
                age_h = (datetime.now(timezone.utc) - feat_latest).total_seconds() / 3600
                r.check(age_h < 48, f"Latest feature age: {age_h:.1f}h")

            # Check for target ticker specifically
            row = await session.execute(text(
                f"SELECT COUNT(*) FROM feature_matrix "
                f"WHERE ticker = '{ticker}' AND time > NOW() - INTERVAL '3 days'"
            ))
            ticker_count = row.scalar()
            r.check(ticker_count > 0, f"Feature rows for {ticker}: {ticker_count}")

    except Exception as e:
        r.fail("Database connection", str(e))

    # ── Step 2: Feature extraction ───────────────────────────────────────────
    print(f"\n{BOLD}Step 2: Feature Matrix — Indicator Values for {ticker}{RESET}")
    feature_row: dict[str, float] = {}
    feature_cols: list[str] = []
    try:
        # Load FFSA feature list
        ffsa_files = sorted(Path("reports/drift").glob("ffsa_*.json"), reverse=True)
        if not ffsa_files:
            ffsa_files = [Path("config/ffsa_features.json")]

        if ffsa_files and ffsa_files[0].exists():
            import json
            with open(ffsa_files[0]) as f:
                ffsa_data = json.load(f)
            feature_cols = [f["feature"] for f in ffsa_data.get("top_features", [])]
            r.ok(f"FFSA features loaded: {len(feature_cols)} features",
                 str(ffsa_files[0].name))
        else:
            r.fail("No FFSA feature list found")

        if session_factory and feature_cols:
            async with session_factory() as session:
                cols_sql = ", ".join(feature_cols[:30])  # safety limit
                row = await session.execute(text(
                    f"SELECT {cols_sql} FROM feature_matrix "
                    f"WHERE ticker = '{ticker}' "
                    f"ORDER BY time DESC LIMIT 1"
                ))
                result = row.fetchone()
                if result:
                    for i, col in enumerate(feature_cols[:30]):
                        feature_row[col] = float(result[i]) if result[i] is not None else 0.0

                    # Show key indicators
                    rsi = feature_row.get("rsi_14", -1)
                    macd = feature_row.get("macd_hist", -999)
                    bb = feature_row.get("bb_pct", -1)
                    vwap = feature_row.get("vwap_dev", -999)
                    adx = feature_row.get("adx", -1)

                    r.ok(f"Latest features fetched for {ticker}")
                    print(f"      RSI={rsi:.1f}  MACD_hist={macd:.4f}  BB%={bb:.2f}  "
                          f"VWAP_dev={vwap:.4f}  ADX={adx:.1f}")
                else:
                    r.fail(f"No feature row found for {ticker}")
    except Exception as e:
        r.fail("Feature extraction", str(e))

    # ── Step 3: Technical scoring ────────────────────────────────────────────
    print(f"\n{BOLD}Step 3: Technical Score (rules-based){RESET}")
    tech_score = 0.0
    try:
        if feature_row:
            from src.models.pipeline_b import _score_technicals
            tech_score = _score_technicals(feature_row)
            r.ok(f"Technical score: {tech_score:+.3f}",
                 "bullish" if tech_score > 0.1 else "bearish" if tech_score < -0.1 else "neutral")

            # Breakdown
            rsi = feature_row.get("rsi_14", 50)
            rsi_signal = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
            macd_hist = feature_row.get("macd_hist", 0)
            macd_signal = "bullish" if macd_hist > 0 else "bearish"
            bb_pct = feature_row.get("bb_pct", 0.5)
            bb_signal = "oversold" if bb_pct < 0.1 else "overbought" if bb_pct > 0.9 else "neutral"
            print(f"      RSI({rsi:.0f})={rsi_signal}  MACD={macd_signal}  BB={bb_signal}")
        else:
            r.skip("No feature data — skipping technical scoring")
    except Exception as e:
        r.fail("Technical scoring", str(e))

    # ── Step 4: Fundamental scoring ──────────────────────────────────────────
    print(f"\n{BOLD}Step 4: Fundamental Score{RESET}")
    fund_score = 0.0
    try:
        from src.data.fundamentals import FundamentalsCache, get_fundamentals
        cache = FundamentalsCache(universe=[ticker])
        await cache.fetch_universe()

        fund_data = get_fundamentals(ticker)
        if fund_data:
            from src.models.pipeline_b import _score_fundamentals
            fund_score = _score_fundamentals(fund_data)
            r.ok(f"Fundamental score: {fund_score:+.3f}")

            pe = fund_data.pe_ratio
            fwd_pe = fund_data.forward_pe
            surprise = fund_data.earnings_surprise_pct
            rev_growth = fund_data.revenue_growth_pct
            print(f"      P/E={pe}  Fwd P/E={fwd_pe}  "
                  f"Earnings surprise={surprise}%  Rev growth={rev_growth}%")
        else:
            r.warn(f"No fundamental data for {ticker}", "yfinance may be rate-limited")
    except Exception as e:
        r.fail("Fundamental scoring", str(e))

    # ── Step 5: Market regime ────────────────────────────────────────────────
    print(f"\n{BOLD}Step 5: Market Regime Score{RESET}")
    regime_score = 0.0
    try:
        from src.data.market_regime import MarketRegimeMonitor, get_market_regime
        if session_factory:
            monitor = MarketRegimeMonitor(session_factory=session_factory)
            await monitor.poll_once()

        regime = get_market_regime()
        regime_score = regime.regime_score
        r.ok(f"Regime score: {regime_score:+.3f}")
        print(f"      VIX={regime.vix:.1f}  SPY_mom={regime.spy_momentum:.4f}  "
              f"QQQ_mom={regime.qqq_momentum:.4f}  Label={regime.label}")

        if regime.vix > 30:
            r.warn("High VIX — signals will be dampened", f"VIX={regime.vix:.1f}")
    except Exception as e:
        r.fail("Market regime", str(e))

    # ── Step 6: Sentiment scoring ────────────────────────────────────────────
    print(f"\n{BOLD}Step 6: Sentiment Score (FinBERT){RESET}")
    sentiment_score = 0.0
    try:
        from src.models.sentiment import SentimentScorer
        scorer = SentimentScorer()
        await scorer.load()
        sentiment_score = await scorer.rolling_sentiment_index(ticker)
        r.ok(f"Sentiment score: {sentiment_score:+.3f}",
             "bullish" if sentiment_score > 0.1 else "bearish" if sentiment_score < -0.1 else "neutral")
    except Exception as e:
        err_str = str(e)
        if "401" in err_str:
            r.warn("HuggingFace API token expired (401)", "sentiment will be 0.0")
        else:
            r.fail("Sentiment scoring", err_str)

    # ── Step 7: Composite signal ─────────────────────────────────────────────
    print(f"\n{BOLD}Step 7: Pipeline B Composite Signal{RESET}")
    ensemble_signal = 0.0
    dir_prob = 0.5
    pred_return = 0.0
    try:
        if feature_row:
            from src.data.fundamentals import FundamentalsCache
            from src.data.market_regime import MarketRegimeMonitor
            from src.data.social_stocktwits import StockTwitsFeed
            from src.models.pipeline_b import PipelineBEngine

            fund_cache = FundamentalsCache(universe=[ticker])
            regime_mon = MarketRegimeMonitor(session_factory=session_factory) if session_factory else None
            social = StockTwitsFeed(universe=[ticker])

            engine = PipelineBEngine(
                fundamentals_cache=fund_cache,
                market_regime=regime_mon,
                social_feed=social,
                sentiment_scorer=None,  # already tested above
            )
            await engine.load()

            sig = await engine.compute_signal(ticker, feature_row)
            ensemble_signal = sig.ensemble_signal
            dir_prob = sig.lgbm_dir_prob
            pred_return = sig.lgbm_pred_return

            r.ok(f"Ensemble signal: {ensemble_signal:+.4f}")
            print(f"      dir_prob={dir_prob:.3f}  pred_return={pred_return:+.5f}")
            print(f"      Weights: tech=30% fund=25% regime=20% sent=20% social=5%")
            print(f"      Breakdown: tech={tech_score:+.3f}  fund={fund_score:+.3f}  "
                  f"regime={regime_score:+.3f}  sent={sentiment_score:+.3f}")

            direction = "LONG" if ensemble_signal > 0.1 else "SHORT" if ensemble_signal < -0.1 else "FLAT"
            strength = abs(ensemble_signal)
            strength_label = "strong" if strength > 0.3 else "moderate" if strength > 0.15 else "weak"
            print(f"      Direction: {direction} ({strength_label}, |signal|={strength:.3f})")
        else:
            r.skip("No feature data — skipping composite signal")
    except Exception as e:
        r.fail("Composite signal generation", str(e))

    # ── Step 8: Position sizer gate ──────────────────────────────────────────
    print(f"\n{BOLD}Step 8: Position Sizer Entry Gate{RESET}")
    try:
        from src.agents.signal_loop import (
            SIZING_COST_THRESHOLD,
            SIZING_MIN_DIR_PROB,
        )
        cost_ok = abs(pred_return) > SIZING_COST_THRESHOLD
        prob_ok = abs(dir_prob - 0.5) > (SIZING_MIN_DIR_PROB - 0.5) if dir_prob != 0.5 else False
        neutral_zone = 0.45 <= dir_prob <= 0.55

        r.check(cost_ok,
                f"Cost gate: |pred_return|={abs(pred_return):.5f} > {SIZING_COST_THRESHOLD}",
                "WOULD ENTER" if cost_ok else "BLOCKED — signal too weak")

        if neutral_zone:
            r.warn(f"Neutral zone: dir_prob={dir_prob:.3f} in [0.45, 0.55]",
                   "signal zeroed by confidence gate")
        else:
            r.check(prob_ok,
                    f"Conviction gate: dir_prob={dir_prob:.3f}",
                    f"min={SIZING_MIN_DIR_PROB}")

        # Compute hypothetical position size
        if cost_ok and not neutral_zone:
            if dir_prob >= 0.80:
                cap_pct = 6.0
            elif dir_prob >= 0.65:
                cap_pct = 4.0
            elif dir_prob >= 0.55:
                cap_pct = 2.0
            else:
                cap_pct = 2.0
            print(f"      Hypothetical size: {cap_pct}% of portfolio "
                  f"(~${100_000 * cap_pct / 100:,.0f} at $100k)")
        else:
            print(f"      Would NOT trade — entry gate blocked")
    except Exception as e:
        r.fail("Position sizer gate", str(e))

    # ── Step 9: Alpaca connectivity ──────────────────────────────────────────
    print(f"\n{BOLD}Step 9: Alpaca Broker Connectivity{RESET}")
    try:
        from src.config import get_settings
        settings = get_settings()
        r.check(settings.alpaca_mode == "paper", "Paper mode confirmed",
                f"mode={settings.alpaca_mode}")

        from src.risk.circuit_breakers import CircuitBreakers
        from src.execution.alpaca import AlpacaOrderRouter
        cb = CircuitBreakers(pipeline_id="test_pipeline_b")
        router = AlpacaOrderRouter(cb)

        account = await router.get_account()
        cash = account.get("cash", 0)
        portfolio = account.get("portfolio_value", 0)
        r.ok(f"Alpaca connected",
             f"cash=${cash:,.0f}  portfolio=${portfolio:,.0f}")

        positions = await router.get_positions()
        r.ok(f"Open positions: {len(positions)}")
        if positions:
            for p in positions[:5]:
                print(f"      {p['ticker']}: {p['qty']} shares @ ${p.get('avg_entry', 0):.2f}")
    except Exception as e:
        r.fail("Alpaca connectivity", str(e))

    # ── Summary ──────────────────────────────────────────────────────────────
    passed = sum(r.checks)
    total = len(r.checks)
    failed = total - passed

    print(f"\n{BOLD}{'='*65}")
    if failed == 0:
        print(f"  {GREEN}All {total} checks passed{RESET}" +
              (f" {YELLOW}({r.warnings} warnings){RESET}" if r.warnings else ""))
    else:
        print(f"  {RED}{failed}/{total} checks FAILED{RESET}" +
              (f"  {YELLOW}({r.warnings} warnings){RESET}" if r.warnings else ""))

    # Signal summary
    if feature_row:
        print(f"\n  {BOLD}Pipeline B Signal for {ticker}:{RESET}")
        print(f"    Ensemble: {ensemble_signal:+.4f}  "
              f"dir_prob: {dir_prob:.3f}  pred_return: {pred_return:+.5f}")
        would_trade = abs(pred_return) > 0.0015 and not (0.45 <= dir_prob <= 0.55)
        trade_label = f"{GREEN}YES{RESET}" if would_trade else f"{RED}NO{RESET}"
        print(f"    Would trade: {trade_label}")

    print(f"{BOLD}{'='*65}{RESET}\n")
    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Test Pipeline B step by step")
    parser.add_argument("--db-url", default=None, help="Database URL override")
    parser.add_argument("--ticker", default="AAPL", help="Ticker to test with")
    args = parser.parse_args()
    sys.exit(asyncio.run(run_tests(args.db_url, args.ticker)))


if __name__ == "__main__":
    main()
