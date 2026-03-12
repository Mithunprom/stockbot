"""Historical data backfill script.

Fetches OHLCV bars from Alpaca REST API and writes to PostgreSQL.
Handles both equity (StockHistoricalDataClient) and crypto (CryptoHistoricalDataClient).

Usage:
    # Backfill current trading universe (60 days, 1m bars)
    python scripts/backfill.py

    # Backfill top 50 S&P stocks for broad model training
    python scripts/backfill.py --sp500-top50 --days 90

    # Backfill crypto (goes back ~2 years on Alpaca)
    python scripts/backfill.py --crypto --days 180

    # Backfill everything (sp500-top50 + crypto)
    python scripts/backfill.py --sp500-top50 --crypto --days 90

    # Specific tickers
    python scripts/backfill.py --tickers AAPL NVDA JPM --days 30

After backfill, run:
    python scripts/build_features.py   # compute indicators for new tickers
    python scripts/train_models.py --tickers sp500-top50  # retrain
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import _DEFAULT_UNIVERSE
from scripts.train_models import SP500_TOP50
from src.data.alpaca_ws import backfill_crypto, backfill_historical
from src.data.db import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill")

CRYPTO_TICKERS = ["BTC/USD", "ETH/USD", "SOL/USD"]


async def run(
    equity_tickers: list[str],
    crypto_tickers: list[str],
    days: int,
    timeframe: str,
) -> None:
    to_date   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    await init_db()

    if equity_tickers:
        logger.info(
            "Equity backfill: %d tickers | %s → %s | timeframe=%s",
            len(equity_tickers), from_date, to_date, timeframe,
        )
        await backfill_historical(
            tickers=equity_tickers,
            from_date=from_date,
            to_date=to_date,
            timeframe=timeframe,
        )

    if crypto_tickers:
        logger.info(
            "Crypto backfill: %s | %s → %s | timeframe=%s",
            crypto_tickers, from_date, to_date, timeframe,
        )
        await backfill_crypto(
            tickers=crypto_tickers,
            from_date=from_date,
            to_date=to_date,
            timeframe=timeframe,
        )

    total = len(equity_tickers) + len(crypto_tickers)
    logger.info("Backfill complete — %d tickers. Next: python scripts/build_features.py", total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical OHLCV data")
    parser.add_argument("--days",        type=int, default=60,
                        help="Days of history to fetch (default: 60). "
                             "Alpaca IEX free tier: 1m bars go back ~2 years.")
    parser.add_argument("--timeframe",   default="1Min",
                        choices=["1Min", "5Min", "1Hour", "1Day"],
                        help="Bar resolution (default: 1Min)")
    parser.add_argument("--tickers",     nargs="*", default=None,
                        help="Explicit list of equity tickers to backfill")
    parser.add_argument("--sp500-top50", action="store_true",
                        help="Backfill the top 50 S&P 500 stocks (by liquidity) "
                             "for broad model training")
    parser.add_argument("--crypto",      action="store_true",
                        help="Also backfill BTC/USD, ETH/USD, SOL/USD")
    args = parser.parse_args()

    # Build equity ticker list
    if args.sp500_top50:
        equity_tickers = SP500_TOP50
        logger.info("Using SP500_TOP50 (%d tickers)", len(equity_tickers))
    elif args.tickers:
        equity_tickers = args.tickers
    else:
        equity_tickers = [t for t in _DEFAULT_UNIVERSE if "/" not in t]

    crypto_tickers = CRYPTO_TICKERS if args.crypto else []

    asyncio.run(run(
        equity_tickers=equity_tickers,
        crypto_tickers=crypto_tickers,
        days=args.days,
        timeframe=args.timeframe,
    ))


if __name__ == "__main__":
    main()
