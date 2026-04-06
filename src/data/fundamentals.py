"""Daily fundamental data cache via yfinance (free, no API key).

Fetches P/E, forward P/E, earnings surprise, revenue growth, market cap,
price-to-book, and dividend yield once per day per ticker. Cached in memory
with a 24-hour TTL.

Used by Pipeline B to score stocks on value/growth fundamentals.

Pattern follows src/data/options_flow.py: FundamentalsCache with
start()/stop()/set_universe() + get_cached() for signal_loop consumption.
"""

from __future__ import annotations

import asyncio
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

logger = structlog.get_logger(__name__)

BATCH_SIZE = 5
CACHE_TTL_HOURS = 24
POLL_INTERVAL_SECONDS = 3600  # re-check every hour (only re-fetches stale)


@dataclass
class FundamentalData:
    """Per-ticker fundamental snapshot."""

    ticker: str
    pe_ratio: float | None = None
    forward_pe: float | None = None
    earnings_surprise_pct: float | None = None
    revenue_growth_pct: float | None = None
    market_cap: float | None = None
    price_to_book: float | None = None
    dividend_yield: float | None = None
    sector: str = "Unknown"
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "pe_ratio": self.pe_ratio,
            "forward_pe": self.forward_pe,
            "earnings_surprise_pct": self.earnings_surprise_pct,
            "revenue_growth_pct": self.revenue_growth_pct,
            "market_cap": self.market_cap,
            "price_to_book": self.price_to_book,
            "dividend_yield": self.dividend_yield,
            "sector": self.sector,
            "fetched_at": self.fetched_at.isoformat(),
        }


# ─── yfinance fetcher (synchronous — run in thread executor) ─────────────────

def _fetch_fundamentals(ticker: str) -> FundamentalData | None:
    """Fetch fundamental data for a single ticker via yfinance.

    Returns FundamentalData or None on error.
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # Earnings surprise: difference between actual and estimated EPS
        # yfinance doesn't always have this; compute from earnings history
        earnings_surprise = None
        try:
            earnings = stock.earnings_dates
            if earnings is not None and len(earnings) > 0:
                latest = earnings.iloc[0]
                actual = latest.get("Reported EPS")
                estimate = latest.get("EPS Estimate")
                if actual is not None and estimate is not None and estimate != 0:
                    earnings_surprise = ((actual - estimate) / abs(estimate)) * 100
        except Exception:
            pass

        # Revenue growth: YoY quarterly
        revenue_growth = None
        try:
            quarterly = stock.quarterly_financials
            if quarterly is not None and quarterly.shape[1] >= 5:
                if "Total Revenue" in quarterly.index:
                    rev_current = quarterly.loc["Total Revenue"].iloc[0]
                    rev_yoy = quarterly.loc["Total Revenue"].iloc[4]
                    if rev_yoy and rev_yoy != 0:
                        revenue_growth = ((rev_current - rev_yoy) / abs(rev_yoy)) * 100
        except Exception:
            pass

        return FundamentalData(
            ticker=ticker,
            pe_ratio=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            earnings_surprise_pct=earnings_surprise,
            revenue_growth_pct=revenue_growth,
            market_cap=info.get("marketCap"),
            price_to_book=info.get("priceToBook"),
            dividend_yield=info.get("dividendYield"),
            sector=info.get("sector", "Unknown"),
            fetched_at=datetime.now(timezone.utc),
        )

    except Exception as exc:
        logger.warning("yfinance_fundamentals_error", ticker=ticker, error=str(exc))
        return None


# ─── Cache + Poller ──────────────────────────────────────────────────────────

_cache: dict[str, FundamentalData] = {}


def get_fundamentals(ticker: str) -> FundamentalData | None:
    """Return cached fundamental data for a ticker, or None if not yet fetched."""
    return _cache.get(ticker)


class FundamentalsCache:
    """Polls yfinance fundamentals daily for the full universe.

    Usage:
        cache = FundamentalsCache(universe=tickers)
        await cache.start()   # background loop
        data = get_fundamentals("AAPL")
    """

    def __init__(
        self,
        universe: list[str] | None = None,
        poll_interval_seconds: int = POLL_INTERVAL_SECONDS,
    ) -> None:
        self._universe: list[str] = universe or []
        self._poll_interval = poll_interval_seconds
        self._running = False

    def set_universe(self, tickers: list[str]) -> None:
        self._universe = tickers

    async def start(self) -> None:
        self._running = True
        logger.info(
            "fundamentals_cache_started",
            interval=self._poll_interval,
            universe=len(self._universe),
        )
        while self._running:
            try:
                await self._poll_once()
            except Exception as exc:
                logger.error("fundamentals_poll_error", error=str(exc))
            await asyncio.sleep(self._poll_interval)

    async def stop(self) -> None:
        self._running = False

    async def fetch_universe(self) -> dict[str, FundamentalData]:
        """One-shot fetch for all tickers (used by backtest)."""
        await self._poll_once()
        return dict(_cache)

    async def _poll_once(self) -> None:
        if not self._universe:
            return

        loop = asyncio.get_event_loop()
        now = datetime.now(timezone.utc)
        ttl = timedelta(hours=CACHE_TTL_HOURS)
        fetched = 0

        for i in range(0, len(self._universe), BATCH_SIZE):
            batch = self._universe[i: i + BATCH_SIZE]
            # Only fetch stale or missing tickers
            stale = [
                t for t in batch
                if t not in _cache or (now - _cache[t].fetched_at) > ttl
            ]
            if not stale:
                continue

            tasks = [
                loop.run_in_executor(None, _fetch_fundamentals, ticker)
                for ticker in stale
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for ticker, result in zip(stale, results):
                if isinstance(result, FundamentalData):
                    _cache[ticker] = result
                    fetched += 1
                elif isinstance(result, Exception):
                    logger.debug("fundamentals_fetch_failed", ticker=ticker, error=str(result))

            # Brief pause between batches
            await asyncio.sleep(1.0)

        if fetched:
            logger.info("fundamentals_poll_done", fetched=fetched, cached=len(_cache))
