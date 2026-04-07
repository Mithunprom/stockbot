"""A/B Test Runner — orchestrates Pipeline A and Pipeline B in parallel.

Manages two independent signal loops, each with its own PositionManager
and capital allocation. Shared: CircuitBreakers, AlpacaOrderRouter, data feeds.

The runner tracks combined portfolio metrics for safety (circuit breakers
see the aggregate) and provides a combined status endpoint for the dashboard.

Usage:
    runner = ABTestRunner(...)
    await runner.start()  # starts both signal loops
    status = runner.get_combined_status()
    await runner.stop()
"""

from __future__ import annotations

import asyncio
import structlog
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = structlog.get_logger(__name__)


@dataclass
class PipelineStatus:
    """Status snapshot for one pipeline."""

    pipeline_id: str
    portfolio_value: float
    portfolio_heat: float
    n_positions: int
    n_trades_today: int
    daily_pnl_pct: float
    latest_signals: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "portfolio_value": round(self.portfolio_value, 2),
            "portfolio_heat": round(self.portfolio_heat, 4),
            "n_positions": self.n_positions,
            "n_trades_today": self.n_trades_today,
            "daily_pnl_pct": round(self.daily_pnl_pct, 4),
            "latest_signals": self.latest_signals,
        }


class ABTestRunner:
    """Runs Pipeline A and Pipeline B side-by-side.

    Each pipeline has its own:
      - SignalLoop (or SignalLoopB)
      - PositionManager
      - Capital allocation

    Shared across both:
      - CircuitBreakers (combined risk monitoring)
      - AlpacaOrderRouter (single broker account)
      - Data feeds (OHLCV, options flow, news, sentiment)
      - Database (trades table with pipeline_id for attribution)

    IMPORTANT: Both pipelines share one Alpaca account. To prevent
    double-booking positions (Pipeline A buys AAPL, Pipeline B also buys AAPL,
    second sell fails), a ticker exclusion lock is enforced:
      - When Pipeline A opens a position in AAPL, Pipeline B skips AAPL
      - Managed by cross-referencing both PositionManagers before entry
    """

    def __init__(
        self,
        signal_loop_a: Any,       # SignalLoop
        signal_loop_b: Any,       # SignalLoopB
        pos_manager_a: Any,       # PositionManager
        pos_manager_b: Any,       # PositionManager
        circuit_breakers_a: Any,  # CircuitBreakers for Pipeline A
        circuit_breakers_b: Any,  # CircuitBreakers for Pipeline B
    ) -> None:
        self._loop_a = signal_loop_a
        self._loop_b = signal_loop_b
        self._pm_a = pos_manager_a
        self._pm_b = pos_manager_b
        self._cb_a = circuit_breakers_a
        self._cb_b = circuit_breakers_b
        self._task_a: asyncio.Task | None = None
        self._task_b: asyncio.Task | None = None

        # Wire up cross-pipeline ticker exclusion.
        # Each signal loop checks the other's PositionManager before entry.
        self._loop_a._other_pm = pos_manager_b
        self._loop_b._other_pm = pos_manager_a

    async def start(self) -> None:
        """Start both pipeline signal loops as concurrent tasks."""
        logger.info(
            "ab_test_starting",
            pipeline_a_capital=self._pm_a.portfolio_value,
            pipeline_b_capital=self._pm_b.portfolio_value,
        )
        self._task_a = asyncio.create_task(
            self._loop_a.start(), name="signal_loop_pipeline_a"
        )
        self._task_b = asyncio.create_task(
            self._loop_b.start(), name="signal_loop_pipeline_b"
        )
        logger.info("ab_test_started")

    async def stop(self) -> None:
        """Stop both pipelines gracefully."""
        logger.info("ab_test_stopping")
        if self._loop_a:
            await self._loop_a.stop()
        if self._loop_b:
            await self._loop_b.stop()
        for task in (self._task_a, self._task_b):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("ab_test_stopped")

    def get_combined_status(self) -> dict[str, Any]:
        """Return side-by-side pipeline comparison for dashboard."""
        status_a = self._get_pipeline_status("pipeline_a", self._loop_a, self._pm_a)
        status_b = self._get_pipeline_status("pipeline_b", self._loop_b, self._pm_b)

        combined_value = self._pm_a.portfolio_value + self._pm_b.portfolio_value
        combined_heat = (
            (self._pm_a.portfolio_heat * self._pm_a.portfolio_value
             + self._pm_b.portfolio_heat * self._pm_b.portfolio_value)
            / max(combined_value, 1.0)
        )

        return {
            "ab_test_active": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "combined": {
                "portfolio_value": round(combined_value, 2),
                "portfolio_heat": round(combined_heat, 4),
            },
            "pipeline_a": {
                **status_a.to_dict(),
                "halted": self._cb_a.is_halted,
                "halt_reason": self._cb_a.halt_reason,
            },
            "pipeline_b": {
                **status_b.to_dict(),
                "halted": self._cb_b.is_halted,
                "halt_reason": self._cb_b.halt_reason,
            },
        }

    def _get_pipeline_status(
        self,
        pipeline_id: str,
        loop: Any,
        pm: Any,
    ) -> PipelineStatus:
        """Extract status from a pipeline's signal loop and position manager."""
        signals = []
        if loop is not None:
            try:
                raw = loop.get_latest_signals()
                signals = raw[:5] if raw else []
            except Exception:
                pass

        daily_pnl = 0.0
        if hasattr(loop, '_daily_start_value') and loop._daily_start_value > 0:
            daily_pnl = (pm.portfolio_value - loop._daily_start_value) / loop._daily_start_value

        n_trades = 0
        if hasattr(loop, '_sizing_n_trades_today'):
            n_trades = loop._sizing_n_trades_today

        return PipelineStatus(
            pipeline_id=pipeline_id,
            portfolio_value=pm.portfolio_value,
            portfolio_heat=pm.portfolio_heat,
            n_positions=len(pm._positions),
            n_trades_today=n_trades,
            daily_pnl_pct=daily_pnl,
            latest_signals=signals,
        )
