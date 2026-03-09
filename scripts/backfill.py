"""Historical data backfill script.

Fetches OHLCV bars from Alpaca REST API (free IEX data) and writes to PostgreSQL.

Usage:
    python scripts/backfill.py                    # last 60 days, 1m bars, all universe
    python scripts/backfill.py --days 30          # last 30 days
    python scripts/backfill.py --timeframe 1Day   # daily bars
    python scripts/backfill.py --tickers AAPL NVDA
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import _DEFAULT_UNIVERSE
from src.data.alpaca_ws import backfill_historical
from src.data.db import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill")


async def run(tickers: list[str], days: int, timeframe: str) -> None:
    to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    logger.info(
        "Backfill: %d tickers | %s → %s | timeframe=%s",
        len(tickers), from_date, to_date, timeframe,
    )

    await init_db()
    await backfill_historical(
        tickers=tickers,
        from_date=from_date,
        to_date=to_date,
        timeframe=timeframe,
    )
    logger.info("Backfill complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical OHLCV data")
    parser.add_argument("--days", type=int, default=60, help="How many days back (default: 60)")
    parser.add_argument(
        "--timeframe",
        default="1Min",
        choices=["1Min", "5Min", "1Hour", "1Day"],
        help="Bar resolution (default: 1Min)",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Tickers to backfill (default: full trading universe)",
    )
    args = parser.parse_args()

    tickers = args.tickers or _DEFAULT_UNIVERSE
    asyncio.run(run(tickers=tickers, days=args.days, timeframe=args.timeframe))


if __name__ == "__main__":
    main()
