"""Position manager: tracks open positions, PnL, portfolio heat, and sizing.

Volatility-scaled sizing: size *= (target_vol / realized_vol)
Hard caps: max 25% per position, always ≥ 10% cash buffer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ─── Position dataclass ───────────────────────────────────────────────────────

@dataclass
class Position:
    ticker: str
    side: str                       # "long" or "short"
    qty: float
    avg_entry_price: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_price: float = 0.0

    @property
    def notional(self) -> float:
        return self.qty * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        direction = 1.0 if self.side == "long" else -1.0
        return direction * self.qty * (self.last_price - self.avg_entry_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        direction = 1.0 if self.side == "long" else -1.0
        return direction * (self.last_price - self.avg_entry_price) / self.avg_entry_price


# ─── Position manager ─────────────────────────────────────────────────────────

class PositionManager:
    """Tracks and manages all open positions.

    Args:
        initial_portfolio: Starting portfolio value.
        max_position_pct: Maximum single-position size as % of portfolio.
        cash_buffer_pct: Minimum cash fraction to maintain (default 10%).
        target_daily_vol: Target daily volatility for position sizing (default 1%).
    """

    def __init__(
        self,
        initial_portfolio: float = 100_000.0,
        max_position_pct: float = 0.25,
        cash_buffer_pct: float = 0.10,
        target_daily_vol: float = 0.01,
        broker_sync_enabled: bool = True,
    ) -> None:
        self.portfolio_value = initial_portfolio
        self.max_position_pct = max_position_pct
        self.cash_buffer_pct = cash_buffer_pct
        self.target_daily_vol = target_daily_vol
        self.broker_sync_enabled = broker_sync_enabled

        self._positions: dict[str, Position] = {}
        self._daily_returns: list[float] = []
        self._peak_value = initial_portfolio

    # ── Position tracking ─────────────────────────────────────────────────────

    def open_position(
        self,
        ticker: str,
        side: str,
        qty: float,
        entry_price: float,
    ) -> None:
        """Record a new position opening."""
        self._positions[ticker] = Position(
            ticker=ticker,
            side=side,
            qty=qty,
            avg_entry_price=entry_price,
            last_price=entry_price,
        )
        logger.info(
            "position_opened",
            ticker=ticker,
            side=side,
            qty=qty,
            entry_price=entry_price,
        )

    def close_position(self, ticker: str, exit_price: float) -> float:
        """Record a position closing. Returns realized PnL."""
        pos = self._positions.pop(ticker, None)
        if pos is None:
            logger.warning("close_nonexistent_position", ticker=ticker)
            return 0.0

        pos.last_price = exit_price
        pnl = pos.unrealized_pnl
        logger.info(
            "position_closed",
            ticker=ticker,
            pnl=pnl,
            pnl_pct=pos.unrealized_pnl_pct,
        )
        return pnl

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update last prices for all positions."""
        for ticker, price in prices.items():
            if ticker in self._positions:
                self._positions[ticker].last_price = price

    # ── Portfolio metrics ─────────────────────────────────────────────────────

    @property
    def total_notional(self) -> float:
        return sum(p.notional for p in self._positions.values())

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def portfolio_heat(self) -> float:
        """Fraction of portfolio currently deployed in positions."""
        return self.total_notional / max(self.portfolio_value, 1.0)

    @property
    def available_cash(self) -> float:
        """Cash available for new positions, respecting the cash buffer."""
        min_cash = self.portfolio_value * self.cash_buffer_pct
        invested = self.total_notional
        available = self.portfolio_value - invested - min_cash
        return max(available, 0.0)

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak."""
        self._peak_value = max(self._peak_value, self.portfolio_value)
        return (self._peak_value - self.portfolio_value) / self._peak_value

    def get_positions(self) -> dict[str, dict[str, Any]]:
        return {
            ticker: {
                "side": p.side,
                "qty": p.qty,
                "avg_entry_price": p.avg_entry_price,
                "last_price": p.last_price,
                "notional": p.notional,
                "unrealized_pnl": p.unrealized_pnl,
                "unrealized_pnl_pct": round(p.unrealized_pnl_pct, 4),
            }
            for ticker, p in self._positions.items()
        }

    def record_return(self, step_pnl: float) -> None:
        """Record a step PnL for rolling volatility tracking."""
        self._daily_returns.append(step_pnl)
        if len(self._daily_returns) > 390:   # ~1 trading day at 1m bars
            self._daily_returns = self._daily_returns[-390:]

    # ── Volatility-scaled sizing ──────────────────────────────────────────────

    def compute_position_size(
        self,
        ticker: str,
        recent_returns: list[float] | None = None,
        base_size_pct: float = 0.05,
    ) -> float:
        """Compute volatility-scaled position size as $ notional.

        size_pct = base_size_pct * (target_vol / realized_vol)
        Capped at max_position_pct and available cash.

        Args:
            ticker: Ticker being sized.
            recent_returns: Recent bar returns for realized vol estimate.
                           Falls back to self._daily_returns if None.
            base_size_pct: Base fraction of portfolio before vol scaling.

        Returns:
            Dollar notional to trade.
        """
        returns = recent_returns or self._daily_returns
        if len(returns) < 5:
            realized_vol = self.target_daily_vol   # default to target if insufficient data
        else:
            realized_vol = float(np.std(returns[-20:])) + 1e-9
            # Annualize to daily (1m bars → multiply by sqrt(390))
            realized_vol *= np.sqrt(390)

        # Scale: if realized_vol > target → reduce size; if less → increase (capped)
        vol_scalar = min(self.target_daily_vol / realized_vol, 2.0)   # cap at 2x
        raw_pct = base_size_pct * vol_scalar

        # Apply hard caps
        capped_pct = min(raw_pct, self.max_position_pct)
        max_notional_from_cash = self.available_cash
        target_notional = capped_pct * self.portfolio_value

        return min(target_notional, max_notional_from_cash)

    # ── Sync from broker ─────────────────────────────────────────────────────

    async def sync_from_broker(self, alpaca_router: Any) -> None:
        """Sync position state from Alpaca to catch any fills we missed.

        Skipped when broker_sync_enabled=False (used by Pipeline B in A/B
        testing to prevent both managers claiming the same broker positions).
        """
        if not self.broker_sync_enabled:
            return
        try:
            broker_positions = await alpaca_router.get_positions()
            account = await alpaca_router.get_account()

            self.portfolio_value = account.get("portfolio_value", self.portfolio_value)

            # Rebuild position map from broker state
            self._positions = {}
            for pos in broker_positions:
                self._positions[pos["ticker"]] = Position(
                    ticker=pos["ticker"],
                    side="long" if pos.get("side") == "long" else "short",
                    qty=abs(pos["qty"]),
                    avg_entry_price=pos["avg_entry_price"],
                    last_price=pos["avg_entry_price"],   # will be updated on next price tick
                )

            logger.info(
                "positions_synced",
                count=len(self._positions),
                portfolio=self.portfolio_value,
            )
        except Exception as exc:
            logger.error("position_sync_error", error=str(exc))
