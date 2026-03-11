"""Alpaca order router.

Uses alpaca-py REST + WebSocket.
Limit orders within 0.1% of mid-price to reduce slippage.
Waits for order acknowledgment + fill confirmation before next action.
Paper trading endpoint by default; config/paper.yaml switches to live.

SAFETY: Never execute live trades without explicit human confirmation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

from src.config import get_settings
from src.risk.circuit_breakers import CircuitBreakers

logger = structlog.get_logger(__name__)

# Lazy import alpaca-py to avoid import errors if not installed
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed — execution disabled")


# ─── Order dataclass ──────────────────────────────────────────────────────────

@dataclass
class OrderRequest:
    ticker: str
    side: str          # "buy" or "sell"
    qty: float
    limit_price: float | None = None
    reason: str = ""   # for trade log attribution


@dataclass
class OrderResult:
    order_id: str
    ticker: str
    side: str
    qty: float
    filled_qty: float
    filled_avg_price: float | None
    status: str        # "filled" / "partially_filled" / "rejected" / "error"
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    error: str = ""


# ─── Alpaca client wrapper ────────────────────────────────────────────────────

class AlpacaOrderRouter:
    """Routes orders to Alpaca paper or live endpoint.

    Args:
        circuit_breakers: Active CircuitBreakers instance. Orders are rejected
                          if trading is halted.
    """

    LIMIT_OFFSET_PCT = 0.001       # 0.1% from mid-price
    FILL_POLL_INTERVAL = 1.0       # seconds between fill status checks
    FILL_TIMEOUT = 60.0            # seconds before canceling unfilled order

    def __init__(self, circuit_breakers: CircuitBreakers) -> None:
        self._cb = circuit_breakers
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not _ALPACA_AVAILABLE:
                raise RuntimeError("alpaca-py not installed")
            settings = get_settings()
            self._client = TradingClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                paper=settings.alpaca_mode == "paper",
            )
        return self._client

    # ── Order submission ──────────────────────────────────────────────────────

    async def submit_order(self, req: OrderRequest) -> OrderResult:
        """Submit a limit order and wait for fill confirmation.

        Returns OrderResult with fill details.
        """
        if self._cb.is_halted:
            logger.warning(
                "order_rejected_halted",
                ticker=req.ticker,
                reason=self._cb.halt_reason,
            )
            return OrderResult(
                order_id="",
                ticker=req.ticker,
                side=req.side,
                qty=req.qty,
                filled_qty=0,
                filled_avg_price=None,
                status="rejected",
                error=f"Trading halted: {self._cb.halt_reason}",
            )

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self._submit_sync, req)
            return result
        except Exception as exc:
            logger.error("order_submit_error", ticker=req.ticker, error=str(exc))
            return OrderResult(
                order_id="",
                ticker=req.ticker,
                side=req.side,
                qty=req.qty,
                filled_qty=0,
                filled_avg_price=None,
                status="error",
                error=str(exc),
            )

    @staticmethod
    def _is_crypto(ticker: str) -> bool:
        """Return True for crypto symbols (e.g. BTC/USD)."""
        return "/" in ticker

    def _submit_sync(self, req: OrderRequest) -> OrderResult:
        client = self._get_client()
        side = OrderSide.BUY if req.side == "buy" else OrderSide.SELL
        is_crypto = self._is_crypto(req.ticker)

        # Crypto uses GTC (Good Till Canceled) — DAY is rejected for crypto.
        # Equities use DAY so limit orders expire at close.
        tif = TimeInForce.GTC if is_crypto else TimeInForce.DAY

        if req.limit_price:
            order_req = LimitOrderRequest(
                symbol=req.ticker,
                qty=req.qty,
                side=side,
                type=OrderType.LIMIT,
                time_in_force=tif,
                limit_price=round(req.limit_price, 2),
            )
        else:
            order_req = MarketOrderRequest(
                symbol=req.ticker,
                qty=req.qty,
                side=side,
                type=OrderType.MARKET,
                time_in_force=tif,
            )

        order = client.submit_order(order_req)
        logger.info(
            "order_submitted",
            order_id=str(order.id),
            ticker=req.ticker,
            side=req.side,
            qty=req.qty,
            limit_price=req.limit_price,
        )

        # Poll for fill
        import time

        start = time.monotonic()
        while time.monotonic() - start < self.FILL_TIMEOUT:
            updated = client.get_order_by_id(str(order.id))
            status = str(updated.status)
            if status in ("filled", "partially_filled"):
                return OrderResult(
                    order_id=str(order.id),
                    ticker=req.ticker,
                    side=req.side,
                    qty=req.qty,
                    filled_qty=float(updated.filled_qty or 0),
                    filled_avg_price=float(updated.filled_avg_price or 0),
                    status=status,
                    submitted_at=order.submitted_at,
                    filled_at=updated.filled_at,
                )
            elif status in ("canceled", "expired", "rejected"):
                logger.warning("order_not_filled", order_id=str(order.id), status=status)
                return OrderResult(
                    order_id=str(order.id),
                    ticker=req.ticker,
                    side=req.side,
                    qty=req.qty,
                    filled_qty=0,
                    filled_avg_price=None,
                    status=status,
                )
            time.sleep(self.FILL_POLL_INTERVAL)

        # Timeout — cancel the order
        try:
            client.cancel_order_by_id(str(order.id))
        except Exception:
            pass
        return OrderResult(
            order_id=str(order.id),
            ticker=req.ticker,
            side=req.side,
            qty=req.qty,
            filled_qty=0,
            filled_avg_price=None,
            status="timeout",
            error="Fill timeout — order canceled",
        )

    # ── Quote fetching ────────────────────────────────────────────────────────

    async def get_latest_quote(self, ticker: str) -> dict[str, float]:
        """Return latest bid/ask/mid for limit price calculation."""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._get_quote_sync, ticker)
        except Exception as exc:
            logger.error("quote_fetch_error", ticker=ticker, error=str(exc))
            return {"bid": 0.0, "ask": 0.0, "mid": 0.0}

    def _get_quote_sync(self, ticker: str) -> dict[str, float]:
        settings = get_settings()
        if self._is_crypto(ticker):
            from alpaca.data.historical.crypto import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoLatestQuoteRequest
            data_client = CryptoHistoricalDataClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
            )
            req = CryptoLatestQuoteRequest(symbol_or_symbols=ticker)
            quote = data_client.get_crypto_latest_quote(req)[ticker]
        else:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            data_client = StockHistoricalDataClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
            )
            req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quote = data_client.get_stock_latest_quote(req)[ticker]
        bid = float(quote.bid_price or 0)
        ask = float(quote.ask_price or 0)
        mid = (bid + ask) / 2 if bid and ask else 0.0
        return {"bid": bid, "ask": ask, "mid": mid}

    # ── Portfolio ──────────────────────────────────────────────────────────────

    async def get_account(self) -> dict[str, Any]:
        """Return account info: equity, cash, buying_power."""
        loop = asyncio.get_event_loop()
        client = self._get_client()
        account = await loop.run_in_executor(None, client.get_account)
        return {
            "equity": float(account.equity or 0),
            "cash": float(account.cash or 0),
            "buying_power": float(account.buying_power or 0),
            "portfolio_value": float(account.portfolio_value or 0),
        }

    async def get_positions(self) -> list[dict[str, Any]]:
        """Return all open positions."""
        loop = asyncio.get_event_loop()
        client = self._get_client()
        positions = await loop.run_in_executor(None, client.get_all_positions)
        return [
            {
                "ticker": p.symbol,
                "qty": float(p.qty or 0),
                "side": p.side,
                "avg_entry_price": float(p.avg_entry_price or 0),
                "unrealized_pl": float(p.unrealized_pl or 0),
                "unrealized_plpc": float(p.unrealized_plpc or 0),
                "market_value": float(p.market_value or 0),
            }
            for p in positions
        ]
