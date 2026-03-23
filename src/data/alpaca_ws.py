"""Alpaca market data WebSocket — free real-time IEX feed.

Replaces polygon_ws.py for ingest. Uses alpaca-py StockDataStream which:
  - Streams minute bars in real time (free IEX feed, no paid plan needed)
  - Covers ~70% of US equity volume (all major exchanges via IEX)
  - Handles reconnect automatically

Data flow:
  bar event → BarAccumulator → write to ohlcv_1m + ohlcv_5m (TimescaleDB)

Historical backfill also uses Alpaca REST (free, same account).

Feed endpoint:
  IEX (free):  wss://stream.data.alpaca.markets/v2/iex
  SIP (paid):  wss://stream.data.alpaca.markets/v2/sip

Usage:
    client = AlpacaDataStreamClient(tickers=["AAPL", "NVDA", ...])
    await client.start()
"""

from __future__ import annotations

import asyncio
import logging
import structlog
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Coroutine

from sqlalchemy.dialects.postgresql import insert

from src.config import get_settings
from src.data.db import OHLCV1m, OHLCV5m, get_session_factory

logger = structlog.get_logger(__name__)

# Free IEX feed — switch to "sip" if upgrading to paid Polygon/Alpaca data
ALPACA_FEED = "iex"


# ─── Bar accumulator (reused from polygon_ws logic) ─────────────────────────

class PartialBar:
    """Accumulates sub-minute ticks into a complete OHLCV bar."""

    def __init__(self, ticker: str, bar_start: datetime, resolution_seconds: int) -> None:
        self.ticker = ticker
        self.bar_start = bar_start
        self.resolution_seconds = resolution_seconds
        self.open = self.high = self.low = self.close = 0.0
        self.volume = self.vwap_sum = self.vwap_vol = 0.0
        self.transactions = 0
        self._init = False

    @property
    def bar_end(self) -> datetime:
        return self.bar_start + timedelta(seconds=self.resolution_seconds)

    @property
    def vwap(self) -> float | None:
        return self.vwap_sum / self.vwap_vol if self.vwap_vol else None

    def ingest_bar(self, bar: dict[str, Any]) -> None:
        """Merge an Alpaca bar event into this accumulator."""
        o = float(bar.get("open", 0))
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))
        v = float(bar.get("volume", 0))
        vw = float(bar.get("vwap", c))
        if not self._init:
            self.open = o
            self.high = h
            self.low = l
            self._init = True
        else:
            self.high = max(self.high, h)
            self.low = min(self.low, l)
        self.close = c
        self.volume += v
        self.vwap_sum += vw * v
        self.vwap_vol += v
        self.transactions += int(bar.get("trade_count", 0))

    def to_row(self) -> dict[str, Any]:
        return {
            "time": self.bar_start,
            "ticker": self.ticker,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "transactions": self.transactions,
        }


class BarAccumulator:
    def __init__(self, resolution_seconds: int) -> None:
        self.resolution_seconds = resolution_seconds
        self._bars: dict[str, PartialBar] = {}

    def _bar_start(self, ts: datetime) -> datetime:
        epoch = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        offset = int((ts - epoch).total_seconds()) // self.resolution_seconds
        return epoch + timedelta(seconds=offset * self.resolution_seconds)

    def ingest(self, bar: dict[str, Any]) -> list[dict[str, Any]]:
        ticker = bar["symbol"]
        ts = _parse_ts(bar.get("timestamp", ""))
        bar_start = self._bar_start(ts)
        completed: list[dict[str, Any]] = []

        existing = self._bars.get(ticker)
        if existing and existing.bar_start != bar_start:
            if existing._init:
                completed.append(existing.to_row())
            del self._bars[ticker]
            existing = None

        if existing is None:
            self._bars[ticker] = PartialBar(ticker, bar_start, self.resolution_seconds)
        self._bars[ticker].ingest_bar(bar)
        return completed


def _parse_ts(ts_str: str) -> datetime:
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc)


# ─── DB writer ────────────────────────────────────────────────────────────────

_WRITE_CHUNK = 3000  # 9 cols × 3000 = 27000 params, safely under PG's 32767 limit


async def _write_bars(table_cls: Any, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    session_factory = get_session_factory()
    for i in range(0, len(rows), _WRITE_CHUNK):
        chunk = rows[i : i + _WRITE_CHUNK]
        async with session_factory() as session:
            stmt = insert(table_cls).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["time", "ticker"],
                set_={
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                    "vwap": stmt.excluded.vwap,
                    "transactions": stmt.excluded.transactions,
                },
            )
            await session.execute(stmt)
            await session.commit()
    logger.debug("alpaca_bars_written: table=%s count=%d", table_cls.__tablename__, len(rows))


# ─── Alpaca data stream client ────────────────────────────────────────────────

def _is_crypto(ticker: str) -> bool:
    return "/" in ticker


class AlpacaDataStreamClient:
    """Streams real-time minute bars from Alpaca's free IEX feed (equities)
    and Alpaca's crypto feed (BTC/USD, ETH/USD, SOL/USD).

    Crypto tickers (those containing "/") are routed to CryptoDataStream;
    equity tickers go to StockDataStream (IEX feed, free).

    Args:
        tickers: List of ticker symbols (mix of equity and crypto is fine).
        feed: "iex" (free) or "sip" (paid). Default: "iex".
    """

    def __init__(self, tickers: list[str], feed: str = ALPACA_FEED) -> None:
        self.tickers = tickers
        self.feed = feed
        self._running = False
        self._acc_1m = BarAccumulator(60)
        self._acc_5m = BarAccumulator(300)
        # Optional callback fired after each completed 1m bar is written to DB.
        # Signature: async (ticker: str, bar_time: datetime) -> None
        self.on_1m_bar: Callable[[str, datetime], Coroutine] | None = None

    @property
    def _equity_tickers(self) -> list[str]:
        return [t for t in self.tickers if not _is_crypto(t)]

    @property
    def _crypto_tickers(self) -> list[str]:
        return [t for t in self.tickers if _is_crypto(t)]

    async def start(self) -> None:
        """Start streaming. Reconnects automatically on disconnect."""
        self._running = True
        tasks = []
        if self._equity_tickers:
            tasks.append(asyncio.create_task(self._stream_equities(), name="equity_stream"))
        if self._crypto_tickers:
            tasks.append(asyncio.create_task(self._stream_crypto(), name="crypto_stream"))
        if tasks:
            await asyncio.gather(*tasks)

    async def stop(self) -> None:
        self._running = False

    async def _stream_equities(self) -> None:
        """Stream equity bars (IEX feed). Reconnects with exponential backoff."""
        backoff = 5
        while self._running:
            try:
                backoff = 5  # reset on successful connection
                await self._stream()
            except Exception as exc:
                logger.error("alpaca_equity_ws_error: %s (retry in %ds)", exc, backoff)
                if self._running:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 300)  # max 5 min

    async def _stream_crypto(self) -> None:
        """Stream crypto bars (Alpaca crypto feed). Reconnects with exponential backoff."""
        backoff = 5
        while self._running:
            try:
                backoff = 5
                await self._stream_crypto_inner()
            except Exception as exc:
                logger.error("alpaca_crypto_ws_error: %s (retry in %ds)", exc, backoff)
                if self._running:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 300)

    async def _stream(self) -> None:
        if not self._equity_tickers:
            return
        try:
            from alpaca.data.live import StockDataStream
        except ImportError:
            raise RuntimeError("alpaca-py not installed: pip install alpaca-py")

        settings = get_settings()
        # alpaca-py requires DataFeed enum, not a plain string
        from alpaca.data.enums import DataFeed
        feed_val = DataFeed(self.feed)   # e.g. "iex" → DataFeed.IEX
        stream = StockDataStream(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            feed=feed_val,
        )

        async def on_bar(bar: Any) -> None:
            await self._handle_bar(bar)

        stream.subscribe_bars(on_bar, *self._equity_tickers)
        logger.info(
            "alpaca_equity_ws_started: feed=%s tickers=%d",
            self.feed, len(self._equity_tickers),
        )
        # Wrap _run_forever with a timeout so tight internal retry loops
        # (e.g. "connection limit exceeded") don't starve the event loop.
        # If it fails within 30s, it's likely a connection limit issue.
        try:
            await asyncio.wait_for(stream._run_forever(), timeout=30)
        except asyncio.TimeoutError:
            pass  # normal: means stream ran for 30s+ (healthy)

    async def _stream_crypto_inner(self) -> None:
        if not self._crypto_tickers:
            return
        try:
            from alpaca.data.live import CryptoDataStream
        except ImportError:
            logger.warning("CryptoDataStream not available — skipping crypto stream")
            return

        settings = get_settings()
        stream = CryptoDataStream(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )

        async def on_bar(bar: Any) -> None:
            await self._handle_bar(bar)

        stream.subscribe_bars(on_bar, *self._crypto_tickers)
        logger.info(
            "alpaca_crypto_ws_started: tickers=%s",
            self._crypto_tickers,
        )
        try:
            await asyncio.wait_for(stream._run_forever(), timeout=30)
        except asyncio.TimeoutError:
            pass

    async def _handle_bar(self, bar: Any) -> None:
        """Process one Alpaca bar event and flush completed bars."""
        # alpaca-py returns Bar objects; convert to dict
        bar_dict = {
            "symbol": getattr(bar, "symbol", ""),
            "timestamp": str(getattr(bar, "timestamp", "")),
            "open": float(getattr(bar, "open", 0)),
            "high": float(getattr(bar, "high", 0)),
            "low": float(getattr(bar, "low", 0)),
            "close": float(getattr(bar, "close", 0)),
            "volume": float(getattr(bar, "volume", 0)),
            "vwap": float(getattr(bar, "vwap", 0) or 0),
            "trade_count": int(getattr(bar, "trade_count", 0) or 0),
        }

        completed_1m: list[dict[str, Any]] = []
        for table_cls, acc in [(OHLCV1m, self._acc_1m), (OHLCV5m, self._acc_5m)]:
            completed = acc.ingest(bar_dict)
            if completed:
                await _write_bars(table_cls, completed)
                if table_cls is OHLCV1m:
                    completed_1m = completed

        # Fire incremental feature computation for each completed 1m bar
        if self.on_1m_bar and completed_1m:
            for row in completed_1m:
                asyncio.create_task(
                    self.on_1m_bar(row["ticker"], row["time"]),
                    name=f"live_features_{row['ticker']}",
                )

    def update_tickers(self, new_tickers: list[str]) -> None:
        """Hot-swap the subscription list (takes effect on next reconnect)."""
        self.tickers = new_tickers
        logger.info("alpaca_ws_tickers_updated: %d tickers", len(new_tickers))


# ─── Historical backfill via Alpaca REST ─────────────────────────────────────

async def backfill_historical(
    tickers: list[str],
    from_date: str,
    to_date: str,
    timeframe: str = "1Min",
    feed: str = ALPACA_FEED,
) -> None:
    """Backfill OHLCV bars from Alpaca historical data API (free).

    Args:
        tickers: Ticker symbols.
        from_date: Start date "YYYY-MM-DD".
        to_date: End date "YYYY-MM-DD".
        timeframe: "1Min", "5Min", "1Hour", "1Day".
        feed: "iex" (free) or "sip" (paid).
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        logger.error("alpaca-py not installed")
        return

    settings = get_settings()
    client = StockHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )

    tf_map = {
        "1Min":  TimeFrame(1, TimeFrameUnit.Minute),
        "5Min":  TimeFrame(5, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day":  TimeFrame(1, TimeFrameUnit.Day),
    }
    alpaca_tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Minute))
    table_cls = OHLCV1m if "Min" in timeframe else OHLCV5m

    # One ticker at a time — multi-ticker requests share the limit across all symbols
    # which means short-changes each ticker. Per-ticker requests get full history.
    start_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(to_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

    for idx, ticker in enumerate(tickers):
        try:
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=alpaca_tf,
                start=start_dt,
                end=end_dt,
                feed=feed,
            )
            bars_response = client.get_stock_bars(req)
            bar_data = bars_response.data if hasattr(bars_response, "data") else bars_response
            rows: list[dict[str, Any]] = []
            for sym, bar_list in bar_data.items():
                for bar in bar_list:
                    rows.append(
                        {
                            "time": bar.timestamp,
                            "ticker": sym,
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": float(bar.volume),
                            "vwap": float(bar.vwap or bar.close),
                            "transactions": int(bar.trade_count or 0),
                        }
                    )
            if rows:
                await _write_bars(table_cls, rows)
                logger.info(
                    "[%d/%d] %s: %d bars written",
                    idx + 1, len(tickers), ticker, len(rows),
                )
            else:
                logger.warning("[%d/%d] %s: no bars returned", idx + 1, len(tickers), ticker)

        except Exception as exc:
            logger.error("alpaca_backfill_error %s: %s", ticker, exc)

        # Brief pause to respect rate limits (200 req/min free tier)
        await asyncio.sleep(0.35)

    logger.info("alpaca_backfill_complete: %d tickers", len(tickers))


async def backfill_crypto(
    tickers: list[str],
    from_date: str,
    to_date: str,
    timeframe: str = "1Min",
) -> None:
    """Backfill crypto OHLCV bars from Alpaca crypto historical API.

    Args:
        tickers: Crypto symbols e.g. ["BTC/USD", "ETH/USD", "SOL/USD"].
        from_date: Start date "YYYY-MM-DD".
        to_date: End date "YYYY-MM-DD".
        timeframe: "1Min", "5Min", "1Hour", "1Day".
    """
    try:
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        logger.error("alpaca-py not installed or CryptoHistoricalDataClient unavailable")
        return

    settings = get_settings()
    client = CryptoHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )

    tf_map = {
        "1Min":  TimeFrame(1, TimeFrameUnit.Minute),
        "5Min":  TimeFrame(5, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day":  TimeFrame(1, TimeFrameUnit.Day),
    }
    alpaca_tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Minute))
    start_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(to_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

    for idx, ticker in enumerate(tickers):
        try:
            req = CryptoBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=alpaca_tf,
                start=start_dt,
                end=end_dt,
            )
            bars_response = client.get_crypto_bars(req)
            bar_data = bars_response.data if hasattr(bars_response, "data") else bars_response
            rows: list[dict[str, Any]] = []
            for sym, bar_list in bar_data.items():
                for bar in bar_list:
                    rows.append({
                        "time": bar.timestamp,
                        "ticker": sym,
                        "open":   float(bar.open),
                        "high":   float(bar.high),
                        "low":    float(bar.low),
                        "close":  float(bar.close),
                        "volume": float(bar.volume),
                        "vwap":   float(bar.vwap or bar.close),
                        "transactions": int(getattr(bar, "trade_count", 0) or 0),
                    })
            if rows:
                await _write_bars(OHLCV1m, rows)
                logger.info(
                    "[%d/%d] %s: %d bars written",
                    idx + 1, len(tickers), ticker, len(rows),
                )
            else:
                logger.warning("[%d/%d] %s: no bars returned", idx + 1, len(tickers), ticker)
        except Exception as exc:
            logger.error("crypto_backfill_error %s: %s", ticker, exc)
        await asyncio.sleep(0.35)
