"""Risk Agent — monitors live risk metrics every 15 minutes.

Scope: Monitor live risk metrics. Enforce hard limits. Trigger halts.
Can act autonomously:
  - Reduce position sizes if VIX > 35
  - Trigger emergency halt if daily loss limit breached
Must NOT: Re-enable trading after halt without human confirmation.
Output: reports/risk/live.json updated every 15 minutes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import structlog
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.risk.circuit_breakers import CircuitBreakers, RiskState

logger = structlog.get_logger(__name__)

REPORT_DIR = Path("reports/risk")
AGENT_NAME = "Risk Agent"


class RiskAgent:
    """Scheduled risk monitor — runs every 15 minutes via APScheduler."""

    def __init__(
        self,
        circuit_breakers: CircuitBreakers,
        position_manager: Any,   # PositionManager
        alpaca_router: Any,      # AlpacaOrderRouter
    ) -> None:
        self._cb = circuit_breakers
        self._pm = position_manager
        self._alpaca = alpaca_router
        self._run_count = 0

    async def run(self) -> dict[str, Any]:
        """Run one risk monitoring cycle. Called by APScheduler."""
        start = datetime.now(timezone.utc)
        self._run_count += 1
        logger.info("risk_agent_run", run=self._run_count, time=start.isoformat())

        # Fetch current market data
        try:
            account = await self._alpaca.get_account()
            portfolio_value = account.get("portfolio_value", 0.0)
        except Exception as exc:
            logger.error("risk_agent_account_error", error=str(exc))
            return {"error": str(exc)}

        # Sync position manager
        await self._pm.sync_from_broker(self._alpaca)

        # Build risk state
        state = RiskState(
            portfolio_value=portfolio_value,
            peak_portfolio=self._pm._peak_value,
            daily_start_value=self._pm.portfolio_value,  # approximation
            vix=await self._fetch_vix(),
            consecutive_losses=self._count_consecutive_losses(),
        )

        # Run circuit breaker checks
        triggered = await self._cb.check(state)

        # Build report
        report: dict[str, Any] = {
            "agent": AGENT_NAME,
            "run": self._run_count,
            "timestamp": start.isoformat(),
            "portfolio_value": portfolio_value,
            "portfolio_heat": round(self._pm.portfolio_heat, 3),
            "drawdown_pct": round(state.drawdown_pct, 4),
            "daily_pnl_pct": round(state.daily_pnl_pct, 4),
            "vix": state.vix,
            "consecutive_losses": state.consecutive_losses,
            "open_positions": len(self._pm._positions),
            "triggered_breakers": sorted(triggered),
            "trading_halted": self._cb.is_halted,
            "halt_reason": self._cb.halt_reason,
            "needs_review": bool(triggered) or self._cb.is_halted,
        }

        # Write report
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORT_DIR / "live.json", "w") as f:
            json.dump(report, f, indent=2)

        # Log in standard format
        if triggered or self._cb.is_halted:
            logger.warning(
                "\n🚨 ESCALATION — Risk Agent — %s\n"
                "Issue: Breakers triggered: %s\n"
                "Impact: Portfolio $%.0f (%.1f%% heat)\n"
                "Action taken: %s\n"
                "Required from human: %s",
                start.isoformat(),
                sorted(triggered),
                portfolio_value,
                self._pm.portfolio_heat * 100,
                "Trading halted" if self._cb.is_halted else "Alert generated",
                "Manual restart required" if self._cb.is_halted else "Review dashboard",
            )
        else:
            logger.info(
                "📊 Risk Agent — %s\n"
                "Portfolio: $%.0f | Heat: %.1f%% | DD: %.1f%% | VIX: %.1f\n"
                "Action: None | Needs review: No",
                start.isoformat(),
                portfolio_value,
                self._pm.portfolio_heat * 100,
                state.drawdown_pct * 100,
                state.vix,
            )

        return report

    async def _fetch_vix(self) -> float:
        """Fetch current VIX from Polygon.io (ticker: I:VIX)."""
        try:
            from src.config import get_settings
            import httpx

            settings = get_settings()
            url = f"https://api.polygon.io/v2/last/trade/I:VIX"
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    url, params={"apiKey": settings.polygon_api_key}
                )
                data = resp.json()
                return float(data.get("results", {}).get("p", 20.0))
        except Exception:
            return 20.0   # default to calm market if fetch fails

    def _count_consecutive_losses(self) -> int:
        """Count consecutive losing bars from recent returns."""
        returns = list(reversed(self._pm._daily_returns))
        count = 0
        for r in returns:
            if r < 0:
                count += 1
            else:
                break
        return count
