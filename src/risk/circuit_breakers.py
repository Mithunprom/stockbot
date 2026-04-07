"""Risk circuit breakers — hard limits + automatic halt logic.

| Control            | Threshold            | Action                              |
|--------------------|----------------------|-------------------------------------|
| Daily loss limit   | -3% portfolio        | Halt for day, require manual restart|
| Max drawdown       | -8% from peak        | Pause all trading, alert            |
| Max position       | 25% portfolio        | Reject oversized orders             |
| Consecutive losses | 5 in a row           | Reduce all sizes by 50%             |
| VIX spike          | > 35                 | Switch to cash-only mode            |
| Earnings blackout  | ±2 days from earnings| No new positions                    |
| PDT guard          | 3 round trips/5 days | Alert, block 4th trade if < $25k   |

NEVER disable these controls. If a change is needed, surface to human.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

LIVE_RISK_REPORT = Path("reports/risk/live.json")


# ─── Risk state ───────────────────────────────────────────────────────────────

@dataclass
class RiskState:
    """Snapshot of current risk metrics passed to CircuitBreakers.check()."""

    portfolio_value: float
    peak_portfolio: float
    daily_start_value: float
    vix: float
    consecutive_losses: int
    ticker: str = ""
    earnings_dates: dict[str, date] = field(default_factory=dict)
    pdt_round_trips_5d: int = 0
    account_size: float = 100_000.0
    proposed_position_pct: float = 0.0    # for max position check
    portfolio_heat: float = 0.0           # fraction of portfolio deployed (0–1)

    @property
    def daily_pnl_pct(self) -> float:
        return (self.portfolio_value - self.daily_start_value) / self.daily_start_value

    @property
    def drawdown_pct(self) -> float:
        return (self.peak_portfolio - self.portfolio_value) / self.peak_portfolio


# ─── Circuit breaker result ───────────────────────────────────────────────────

@dataclass
class BreachResult:
    breaker_name: str
    triggered: bool
    value: float
    threshold: float
    action: str
    message: str


# ─── Circuit breakers ─────────────────────────────────────────────────────────

class CircuitBreakers:
    """Evaluates all risk controls and returns the set of triggered breaker names.

    Usage:
        cb = CircuitBreakers()
        triggered = await cb.check(state)
        if triggered:
            await cb.halt_trading(reason=triggered)
    """

    def __init__(self, pipeline_id: str = "") -> None:
        self._pipeline_id = pipeline_id
        self._halted: bool = False
        self._halt_reason: str = ""
        self._halt_time: datetime | None = None

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    async def check(self, state: RiskState) -> set[str]:
        """Run all circuit breaker checks. Returns set of triggered breaker names."""
        triggered: set[str] = set()

        checks = [
            self._check_daily_loss(state),
            self._check_max_drawdown(state),
            self._check_portfolio_heat(state),
            self._check_vix(state),
            self._check_consecutive_losses(state),
            self._check_earnings_blackout(state),
            self._check_pdt(state),
        ]

        for check in checks:
            result = check
            if result.triggered:
                triggered.add(result.breaker_name)
                logger.warning(
                    "circuit_breaker_triggered",
                    breaker=result.breaker_name,
                    value=result.value,
                    threshold=result.threshold,
                    action=result.action,
                    message=result.message,
                )

        # Update live risk report
        await self._write_risk_report(state, triggered)

        return triggered

    def check_position_size(self, proposed_pct: float, max_pct: float = 0.25) -> BreachResult:
        """Synchronous check for oversized orders — call before placing any order."""
        triggered = proposed_pct > max_pct
        return BreachResult(
            breaker_name="max_position",
            triggered=triggered,
            value=proposed_pct,
            threshold=max_pct,
            action="reject_order" if triggered else "allow",
            message=(
                f"Order rejected: {proposed_pct:.1%} > max {max_pct:.1%}"
                if triggered
                else "OK"
            ),
        )

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_daily_loss(self, state: RiskState) -> BreachResult:
        threshold = -0.03
        triggered = state.daily_pnl_pct < threshold
        if triggered and not self._halted:
            self._halt("daily_loss", f"Daily P&L {state.daily_pnl_pct:.2%} < {threshold:.2%}")
        return BreachResult(
            breaker_name="daily_loss",
            triggered=triggered,
            value=state.daily_pnl_pct,
            threshold=threshold,
            action="halt_day" if triggered else "ok",
            message=f"Daily P&L: {state.daily_pnl_pct:.2%}",
        )

    def _check_portfolio_heat(self, state: RiskState) -> BreachResult:
        threshold = 0.80
        triggered = state.portfolio_heat > threshold
        return BreachResult(
            breaker_name="portfolio_heat",
            triggered=triggered,
            value=state.portfolio_heat,
            threshold=threshold,
            action="no_new_entries" if triggered else "ok",
            message=f"Portfolio heat: {state.portfolio_heat:.1%}",
        )

    def _check_max_drawdown(self, state: RiskState) -> BreachResult:
        threshold = 0.08
        triggered = state.drawdown_pct > threshold
        if triggered and not self._halted:
            self._halt("max_drawdown", f"Drawdown {state.drawdown_pct:.2%} > {threshold:.2%}")
        return BreachResult(
            breaker_name="max_drawdown",
            triggered=triggered,
            value=state.drawdown_pct,
            threshold=threshold,
            action="pause_trading" if triggered else "ok",
            message=f"Drawdown: {state.drawdown_pct:.2%}",
        )

    def _check_vix(self, state: RiskState) -> BreachResult:
        threshold = 35.0
        triggered = state.vix > threshold
        return BreachResult(
            breaker_name="vix_spike",
            triggered=triggered,
            value=state.vix,
            threshold=threshold,
            action="cash_only" if triggered else "ok",
            message=f"VIX: {state.vix:.1f}",
        )

    def _check_consecutive_losses(self, state: RiskState) -> BreachResult:
        threshold = 5
        triggered = state.consecutive_losses >= threshold
        return BreachResult(
            breaker_name="consecutive_losses",
            triggered=triggered,
            value=float(state.consecutive_losses),
            threshold=float(threshold),
            action="reduce_sizing_50pct" if triggered else "ok",
            message=f"Consecutive losses: {state.consecutive_losses}",
        )

    def _check_earnings_blackout(self, state: RiskState) -> BreachResult:
        blackout_days = 2
        if not state.ticker or not state.earnings_dates:
            return BreachResult("earnings_blackout", False, 0.0, 0.0, "ok", "No earnings data")

        earnings = state.earnings_dates.get(state.ticker.upper())
        if not earnings:
            return BreachResult("earnings_blackout", False, 0.0, 0.0, "ok", "No earnings for ticker")

        today = datetime.now(timezone.utc).date()
        from datetime import timedelta

        blackout_start = earnings - timedelta(days=blackout_days)
        blackout_end = earnings + timedelta(days=blackout_days)
        triggered = blackout_start <= today <= blackout_end
        return BreachResult(
            breaker_name="earnings_blackout",
            triggered=triggered,
            value=float((earnings - today).days),
            threshold=float(blackout_days),
            action="no_new_positions" if triggered else "ok",
            message=f"{state.ticker} earnings: {earnings} (today: {today})",
        )

    def _check_pdt(self, state: RiskState) -> BreachResult:
        threshold = 3
        pdt_applies = state.account_size < 25_000
        triggered = pdt_applies and state.pdt_round_trips_5d >= threshold
        return BreachResult(
            breaker_name="pdt_guard",
            triggered=triggered,
            value=float(state.pdt_round_trips_5d),
            threshold=float(threshold),
            action="alert_block_4th" if triggered else "ok",
            message=(
                f"PDT: {state.pdt_round_trips_5d} round trips in 5 days"
                f"{' (account < $25k)' if pdt_applies else ''}"
            ),
        )

    # ── Halt management ───────────────────────────────────────────────────────

    def _halt(self, reason: str, message: str) -> None:
        self._halted = True
        self._halt_reason = reason
        self._halt_time = datetime.now(timezone.utc)
        pipeline_label = f" [{self._pipeline_id}]" if self._pipeline_id else ""
        logger.critical(
            "trading_halted",
            pipeline=self._pipeline_id,
            reason=reason,
            message=message,
            requires_human_restart=True,
        )
        # Print escalation to console
        print(
            f"\n🚨 ESCALATION — Risk Agent{pipeline_label} — {self._halt_time.isoformat()}\n"
            f"Issue: {message}\n"
            f"Action taken: Trading halted automatically.\n"
            f"Required from human: Manual restart required after review.\n"
        )

    def resume_trading(self, authorized_by: str) -> None:
        """Re-enable trading after a halt. REQUIRES human authorization.

        This method must never be called autonomously by any sub-agent.
        """
        if not authorized_by:
            raise ValueError("Human authorization required to resume trading")
        self._halted = False
        self._halt_reason = ""
        self._halt_time = None
        logger.info("trading_resumed", authorized_by=authorized_by)

    # ── Risk report ───────────────────────────────────────────────────────────

    async def _write_risk_report(self, state: RiskState, triggered: set[str]) -> None:
        LIVE_RISK_REPORT.parent.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio_value": state.portfolio_value,
            "daily_pnl_pct": round(state.daily_pnl_pct, 4),
            "drawdown_pct": round(state.drawdown_pct, 4),
            "vix": state.vix,
            "consecutive_losses": state.consecutive_losses,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "triggered_breakers": sorted(triggered),
        }
        try:
            with open(LIVE_RISK_REPORT, "w") as f:
                json.dump(report, f, indent=2)
        except Exception as exc:
            logger.error("risk_report_write_error", error=str(exc))
