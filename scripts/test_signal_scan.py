#!/usr/bin/env python3
"""Signal Scanner — shows Pipeline B signals for the full universe.

Fetches the latest feature row per ticker, runs the Pipeline B scoring
engine, and displays a ranked table of signals with trade/no-trade flags.

Usage:
    python scripts/test_signal_scan.py
    python scripts/test_signal_scan.py --db-url "postgresql+asyncpg://..."
    python scripts/test_signal_scan.py --top 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ANSI
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


async def run_scan(db_url: str | None, top_n: int) -> int:
    if db_url:
        os.environ["DATABASE_URL"] = db_url

    from src.data.db import init_db, get_session_factory
    from src.data.fundamentals import FundamentalsCache
    from src.data.market_regime import MarketRegimeMonitor, get_market_regime
    from src.data.social_stocktwits import StockTwitsFeed
    from src.models.pipeline_b import PipelineBEngine, _score_technicals

    await init_db()
    sf = get_session_factory()

    # Load universe
    universe_file = Path("config/universe.json")
    if universe_file.exists():
        with open(universe_file) as f:
            universe = json.load(f).get("symbols", [])
    else:
        universe = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA",
                     "AVGO", "AMD", "JPM", "V", "MA", "PLTR", "ARM",
                     "MSTR", "XOM", "CVX", "LLY", "UNH", "COST", "NFLX"]

    # Load FFSA features
    ffsa_files = sorted(Path("reports/drift").glob("ffsa_*.json"), reverse=True)
    if not ffsa_files:
        ffsa_files = [Path("config/ffsa_features.json")]
    with open(ffsa_files[0]) as f:
        ffsa_data = json.load(f)
    feature_cols = [f["feature"] for f in ffsa_data.get("top_features", [])]

    print(f"\n{BOLD}{'='*80}")
    print(f"  Pipeline B Signal Scanner")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}  |  Universe: {len(universe)} tickers")
    print(f"{'='*80}{RESET}")

    # Fetch latest feature row per ticker
    print(f"\n{DIM}Fetching features...{RESET}", end="", flush=True)
    feature_rows: dict[str, dict[str, float]] = {}
    from sqlalchemy import text
    async with sf() as session:
        for ticker in universe:
            cols_sql = ", ".join(feature_cols[:30])
            row = await session.execute(text(
                f"SELECT {cols_sql} FROM feature_matrix "
                f"WHERE ticker = '{ticker}' ORDER BY time DESC LIMIT 1"
            ))
            result = row.fetchone()
            if result:
                feature_rows[ticker] = {
                    col: float(result[i]) if result[i] is not None else 0.0
                    for i, col in enumerate(feature_cols[:30])
                }
    print(f" {len(feature_rows)}/{len(universe)} tickers have data")

    if not feature_rows:
        print(f"{RED}No feature data found. Run the pipeline during market hours.{RESET}")
        return 1

    # Initialize Pipeline B
    print(f"{DIM}Initializing Pipeline B engine...{RESET}", end="", flush=True)
    fund_cache = FundamentalsCache(universe=list(feature_rows.keys()))
    try:
        await fund_cache.fetch_universe()
    except Exception:
        pass
    regime_mon = MarketRegimeMonitor(session_factory=sf)
    try:
        await regime_mon.poll_once()
    except Exception:
        pass
    social = StockTwitsFeed(universe=list(feature_rows.keys()))
    engine = PipelineBEngine(
        fundamentals_cache=fund_cache,
        market_regime=regime_mon,
        social_feed=social,
    )
    await engine.load()
    print(" ready")

    # Compute signals
    print(f"{DIM}Computing signals...{RESET}", end="", flush=True)
    signals = await engine.compute_universe(feature_rows)
    print(f" {len(signals)} signals computed")

    # Get regime for header
    regime = get_market_regime()
    print(f"\n{BOLD}Market Regime:{RESET} {regime.label}  "
          f"VIX={regime.vix:.1f}  SPY_mom={regime.spy_momentum:+.4f}  "
          f"score={regime.regime_score:+.3f}")

    # Cost threshold
    from src.agents.signal_loop import SIZING_COST_THRESHOLD

    # Print ranked table
    print(f"\n{BOLD}{'Rank':<5} {'Ticker':<8} {'Signal':>8} {'Dir':>5} "
          f"{'DirProb':>8} {'PredRet':>9} {'Tech':>7} {'RSI':>5} "
          f"{'MACD':>7} {'ADX':>5} {'Trade?':>7}{RESET}")
    print("-" * 80)

    tradeable = 0
    for i, sig in enumerate(signals[:top_n]):
        ticker = sig.ticker
        row = feature_rows.get(ticker, {})
        tech = _score_technicals(row)
        rsi = row.get("rsi_14", 0)
        macd = row.get("macd_hist", 0)
        adx = row.get("adx", 0)

        would_trade = (abs(sig.lgbm_pred_return) > SIZING_COST_THRESHOLD
                       and not (0.45 <= sig.lgbm_dir_prob <= 0.55))

        if would_trade:
            tradeable += 1

        # Color code signal
        sig_val = sig.ensemble_signal
        if sig_val > 0.15:
            sig_color = GREEN
        elif sig_val < -0.15:
            sig_color = RED
        else:
            sig_color = YELLOW

        trade_flag = f"{GREEN}YES{RESET}" if would_trade else f"{DIM}no{RESET}"
        direction = "LONG" if sig_val > 0.05 else "SHORT" if sig_val < -0.05 else "FLAT"

        print(f"{i+1:<5} {ticker:<8} {sig_color}{sig_val:>+8.4f}{RESET} "
              f"{direction:>5} {sig.lgbm_dir_prob:>8.3f} "
              f"{sig.lgbm_pred_return:>+9.5f} {tech:>+7.3f} "
              f"{rsi:>5.0f} {macd:>+7.4f} {adx:>5.0f} {trade_flag:>7}")

    print("-" * 80)
    print(f"\n{BOLD}Summary:{RESET} {tradeable}/{len(signals)} tickers "
          f"would pass entry gate (cost threshold={SIZING_COST_THRESHOLD})")

    # Top buys and sells
    longs = [s for s in signals if s.ensemble_signal > 0.1
             and abs(s.lgbm_pred_return) > SIZING_COST_THRESHOLD]
    shorts = [s for s in signals if s.ensemble_signal < -0.1
              and abs(s.lgbm_pred_return) > SIZING_COST_THRESHOLD]

    if longs:
        print(f"\n  {GREEN}Top LONG candidates:{RESET} "
              + ", ".join(f"{s.ticker}({s.ensemble_signal:+.3f})" for s in longs[:5]))
    if shorts:
        print(f"  {RED}Top SHORT candidates:{RESET} "
              + ", ".join(f"{s.ticker}({s.ensemble_signal:+.3f})" for s in shorts[:5]))
    if not longs and not shorts:
        print(f"\n  {YELLOW}No strong signals — market may be in neutral zone{RESET}")

    print(f"\n{BOLD}{'='*80}{RESET}\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Pipeline B Signal Scanner")
    parser.add_argument("--db-url", default=None, help="Database URL override")
    parser.add_argument("--top", type=int, default=25, help="Show top N signals")
    args = parser.parse_args()
    sys.exit(asyncio.run(run_scan(args.db_url, args.top)))


if __name__ == "__main__":
    main()
