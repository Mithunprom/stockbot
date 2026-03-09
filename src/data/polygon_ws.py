"""Polygon.io WebSocket client for real-time trade and aggregate data.

Connects to wss://socket.polygon.io/stocks, subscribes to:
  T.*  — individual trades (for VPIN / microstructure)
  A.*  — per-second aggregates (resample → 1m, 5m, 15m bars)

Bars are written to TimescaleDB ohlcv_1m/5m tables.
Handles reconnects, gap detection, and split/dividend adjustments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import structlog
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import websockets
from sqlalchemy.dialects.postgresql import insert
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.data.db import OHLCV1m, OHLCV5m, get_session_factory

logger = structlog.get_logger(__name__)

POLYGON_WS_URL = "wss://socket.polygon.io/stocks"

# Resolutions: (table_class, bar_seconds)
_RESOLUTIONS: list[tuple[Any, int]] = [
    (OHLCV1m, 60),
    (OHLCV5m, 300),
]


# ─── Bar accumulator ──────────────────────────────────────────────────────────

@dataclass
class PartialBar:
    """Accumulates per-second agg events into a complete OHLCV bar."""

    ticker: str
    bar_start: datetime
    resolution_seconds: int

    open: float = 0.0
    high: float = 0.0
    low: float = float("inf")
    close: float = 0.0
    volume: float = 0.0
    vwap_sum: float = 0.0
    vwap_vol: float = 0.0
    transactions: int = 0
    _initialized: bool = field(default=False, repr=False)

    @property
    def bar_end(self) -> datetime:
        return self.bar_start + timedelta(seconds=self.resolution_seconds)

    @property
    def vwap(self) -> float | None:
        return self.vwap_sum / self.vwap_vol if self.vwap_vol else None

    def ingest(self, agg: dict[str, Any]) -> None:
        """Merge a per-second aggregate event into this bar."""
        o, h, l, c, v = (
            float(agg.get("o", 0)),
            float(agg.get("h", 0)),
            float(agg.get("l", 0)),
            float(agg.get("c", 0)),
            float(agg.get("av", agg.get("v", 0))),
        )
        if not self._initialized:
            self.open = o
            self._initialized = True
        self.high = max(self.high, h)
        self.low = min(self.low, l)
        self.close = c
        self.volume += v
        self.vwap_sum += c * v
        self.vwap_vol += v
        self.transactions += int(agg.get("z", 0))

    def to_row(self) -> dict[str, Any]:
        return {
            "time": self.bar_start,
            "ticker": self.ticker,
            "open": self.open,
            "high": self.high,
            "low": min(self.low, self.open),   # guard against inf
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "transactions": self.transactions,
        }


class BarAccumulator:
    """Manages PartialBar instances across tickers and resolutions."""

    def __init__(self, resolution_seconds: int) -> None:
        self.resolution_seconds = resolution_seconds
        self._bars: dict[str, PartialBar] = {}

    def _bar_start(self, ts_ms: int) -> datetime:
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        epoch = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        offset = int((ts - epoch).total_seconds()) // self.resolution_seconds
        return epoch + timedelta(seconds=offset * self.resolution_seconds)

    def ingest(self, agg: dict[str, Any]) -> list[dict[str, Any]]:
        """Process one agg event. Returns completed bars (if any)."""
        ticker = agg["sym"]
        ts_ms = int(agg.get("e", agg.get("s", 0)))
        bar_start = self._bar_start(ts_ms)
        completed = []

        existing = self._bars.get(ticker)
        if existing and existing.bar_start != bar_start:
            # Bar rolled over → emit the completed bar
            if existing._initialized:
                completed.append(existing.to_row())
            del self._bars[ticker]
            existing = None

        if existing is None:
            self._bars[ticker] = PartialBar(
                ticker=ticker,
                bar_start=bar_start,
                resolution_seconds=self.resolution_seconds,
            )

        self._bars[ticker].ingest(agg)
        return completed


# ─── Database writer ──────────────────────────────────────────────────────────

async def _write_bars(table_cls: Any, rows: list[dict[str, Any]]) -> None:
    """Upsert completed bars into TimescaleDB."""
    if not rows:
        return
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = insert(table_cls).values(rows)
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
    logger.debug("bars_written", table=table_cls.__tablename__, count=len(rows))


# ─── Gap detector ─────────────────────────────────────────────────────────────

class GapDetector:
    """Alert when a ticker's bar stream goes silent for too long."""

    MAX_SILENCE_SECONDS = 120

    def __init__(self) -> None:
        self._last_seen: dict[str, datetime] = {}

    def update(self, ticker: str) -> None:
        self._last_seen[ticker] = datetime.now(timezone.utc)

    def check_gaps(self) -> list[str]:
        """Return tickers that haven't been seen in MAX_SILENCE_SECONDS."""
        now = datetime.now(timezone.utc)
        return [
            t
            for t, ts in self._last_seen.items()
            if (now - ts).total_seconds() > self.MAX_SILENCE_SECONDS
        ]


# ─── Polygon WebSocket client ─────────────────────────────────────────────────

class PolygonWebSocketClient:
    """Async WebSocket client for Polygon.io stock data."""

    def __init__(self, tickers: list[str]) -> None:
        self.tickers = tickers
        self._accumulators = {
            res_sec: BarAccumulator(res_sec) for _, res_sec in _RESOLUTIONS
        }
        self._gap_detector = GapDetector()
        self._running = False

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(20),
    )
    async def _connect_and_stream(self) -> None:
        settings = get_settings()
        async with websockets.connect(POLYGON_WS_URL, ping_interval=20) as ws:
            # Authenticate
            auth = await ws.recv()
            logger.info("polygon_ws_connected", message=json.loads(auth))

            await ws.send(json.dumps({"action": "auth", "params": settings.polygon_api_key}))
            auth_resp = json.loads(await ws.recv())
            if not any(m.get("status") == "auth_success" for m in auth_resp):
                raise RuntimeError(f"Polygon auth failed: {auth_resp}")

            # Subscribe to trades (T) and per-second aggs (A) for all tickers
            sub_tickers = ",".join(
                [f"T.{t}" for t in self.tickers] + [f"A.{t}" for t in self.tickers]
            )
            await ws.send(json.dumps({"action": "subscribe", "params": sub_tickers}))
            logger.info("polygon_ws_subscribed", tickers=len(self.tickers))

            async for raw in ws:
                if not self._running:
                    break
                await self._handle_messages(json.loads(raw))

    async def _handle_messages(self, messages: list[dict[str, Any]]) -> None:
        """Dispatch incoming Polygon events to the right handler."""
        for msg in messages:
            ev = msg.get("ev", "")
            if ev == "A":
                await self._handle_agg(msg)
            elif ev == "T":
                self._gap_detector.update(msg.get("sym", ""))
            # "status" events (connected, auth_success) are informational

    async def _handle_agg(self, agg: dict[str, Any]) -> None:
        """Process a per-second aggregate and flush completed bars."""
        ticker = agg.get("sym", "")
        self._gap_detector.update(ticker)

        for table_cls, res_sec in _RESOLUTIONS:
            acc = self._accumulators[res_sec]
            completed = acc.ingest(agg)
            if completed:
                await _write_bars(table_cls, completed)

    async def start(self) -> None:
        self._running = True
        gap_task = asyncio.create_task(self._gap_monitor())
        try:
            await self._connect_and_stream()
        finally:
            self._running = False
            gap_task.cancel()

    async def stop(self) -> None:
        self._running = False

    async def _gap_monitor(self) -> None:
        """Periodically check for silent tickers and log warnings."""
        while self._running:
            await asyncio.sleep(60)
            gaps = self._gap_detector.check_gaps()
            if gaps:
                logger.warning("data_gaps_detected", tickers=gaps)


# ─── Historical backfill (Polygon REST) ───────────────────────────────────────

async def backfill_historical(
    tickers: list[str],
    from_date: str,
    to_date: str,
    resolution: str = "1",
    timespan: str = "minute",
) -> None:
    """Backfill OHLCV via Polygon REST API.

    Args:
        tickers: List of tickers to backfill.
        from_date: ISO date string "YYYY-MM-DD".
        to_date: ISO date string "YYYY-MM-DD".
        resolution: Multiplier (e.g. "1", "5").
        timespan: "minute", "hour", or "day".
    """
    from polygon import RESTClient  # polygon-api-client

    settings = get_settings()
    client = RESTClient(settings.polygon_api_key)

    table_map = {("1", "minute"): OHLCV1m, ("5", "minute"): OHLCV5m, ("1", "hour"): OHLCV1h}
    table_cls = table_map.get((resolution, timespan), OHLCV1d)

    for ticker in tickers:
        rows: list[dict[str, Any]] = []
        try:
            for agg in client.list_aggs(
                ticker,
                int(resolution),
                timespan,
                from_date,
                to_date,
                adjusted=True,
                sort="asc",
                limit=50000,
            ):
                rows.append(
                    {
                        "time": datetime.fromtimestamp(agg.timestamp / 1000, tz=timezone.utc),
                        "ticker": ticker,
                        "open": agg.open,
                        "high": agg.high,
                        "low": agg.low,
                        "close": agg.close,
                        "volume": agg.volume,
                        "vwap": agg.vwap,
                        "transactions": agg.transactions,
                    }
                )
        except Exception as exc:
            logger.error("backfill_error", ticker=ticker, error=str(exc))
            continue

        if rows:
            await _write_bars(table_cls, rows)
            logger.info("backfill_done", ticker=ticker, bars=len(rows))
