"""Watchdog Agent — detects broken-bot states, self-heals what's safe, emails the rest.

Scope: Monitor signal-loop liveness, exit-path health, daily-reset state, data
       freshness and circuit breakers. Heal bounded, reversible issues
       autonomously (paper mode). Email the human on anything it heals or
       cannot heal.
Must NOT: Re-enable trading after a circuit-breaker halt (human-only, per
       CLAUDE.md). Take any action in live mode beyond alerting.
Output: reports/watchdog/latest.json + email alerts via the same SMTP path as
       the daily forecast (proven working in prod).
Escalate if: loop dead >5 min in market hours, tick errors accumulating,
       positions stuck past max_hold, CB halted.

Born from the 2026-07-08/09 outage: a NameError in the exit branch crashed
every tick for two sessions and nothing noticed. This agent is the "something
that notices".
"""

from __future__ import annotations

import json
import os
import smtplib
import ssl
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import structlog

logger = structlog.get_logger(__name__)

AGENT_NAME = "Watchdog Agent"
REPORT_PATH = Path("reports/watchdog/latest.json")

TICK_STALE_MINUTES = 5          # market-hours: loop must tick at least this often
TICK_ERROR_CRITICAL = 10        # errors since last check → critical
ZOMBIE_GRACE_BARS = 60          # bars past max_hold before a position is a zombie
EMAIL_DEDUPE_HOURS = 4          # don't re-email an identical issue set within this
ET = ZoneInfo("America/New_York")


class WatchdogAgent:
    """Detect + self-heal + alert. Reads SignalLoop state in-process."""

    def __init__(self, signal_loop: Any, pos_manager: Any, circuit_breakers: Any) -> None:
        self._loop = signal_loop
        self._pm = pos_manager
        self._cb = circuit_breakers
        self._last_error_count = 0
        self._last_email_fingerprint = ""
        self._last_email_at: datetime | None = None
        self.last_report: dict[str, Any] | None = None

    # ── Individual checks ───────────────────────────────────────────────────

    def _market_open(self) -> bool:
        now = datetime.now(ET)
        if now.weekday() >= 5:
            return False
        return (9, 30) <= (now.hour, now.minute) < (16, 0)

    def _check_loop_tick_fresh(self) -> dict[str, Any]:
        last = getattr(self._loop, "last_tick_at", None)
        if last is None:
            return {"name": "loop_tick_fresh", "status": "warn", "healed": False,
                    "detail": "no tick recorded yet (startup?)"}
        age_min = (datetime.now(timezone.utc) - last).total_seconds() / 60
        if age_min > TICK_STALE_MINUTES:
            return {"name": "loop_tick_fresh", "status": "critical", "healed": False,
                    "detail": f"last tick {age_min:.1f} min ago — loop stalled or dead "
                              "(main.py task-watchdog should restart it within 5 min)"}
        return {"name": "loop_tick_fresh", "status": "ok", "healed": False,
                "detail": f"last tick {age_min:.1f} min ago"}

    def _check_tick_errors(self) -> dict[str, Any]:
        count = getattr(self._loop, "tick_error_count", 0)
        new_errors = count - self._last_error_count
        self._last_error_count = count
        if new_errors <= 0:
            return {"name": "tick_errors", "status": "ok", "healed": False,
                    "detail": f"no new tick errors (lifetime {count})"}
        status = "critical" if new_errors >= TICK_ERROR_CRITICAL else "warn"
        return {"name": "tick_errors", "status": status, "healed": False,
                "detail": f"{new_errors} new tick error(s) since last check; "
                          f"last: {getattr(self._loop, 'last_tick_error', '')!r}"}

    def _check_daily_reset(self) -> dict[str, Any]:
        now = datetime.now(ET)
        if not self._market_open() or (now.hour, now.minute) < (9, 35):
            return {"name": "daily_reset", "status": "ok", "healed": False,
                    "detail": "not applicable (outside window)"}
        if getattr(self._loop, "_last_reset_date", None) == now.date():
            return {"name": "daily_reset", "status": "ok", "healed": False,
                    "detail": "reset done for today"}
        # Self-heal: the reset is idempotent and date-guarded
        try:
            self._loop._maybe_reset_daily_value()
        except Exception as exc:
            return {"name": "daily_reset", "status": "critical", "healed": False,
                    "detail": f"reset missing and heal failed: {exc}"}
        healed = getattr(self._loop, "_last_reset_date", None) == now.date()
        return {"name": "daily_reset",
                "status": "ok" if healed else "critical",
                "healed": healed,
                "detail": "daily reset was missing — invoked _maybe_reset_daily_value()"}

    def _check_zombie_positions(self) -> dict[str, Any]:
        from src.agents.signal_loop import SIZING_MAX_HOLD_BARS
        if not self._market_open():
            return {"name": "zombie_positions", "status": "ok", "healed": False,
                    "detail": "not applicable (market closed)"}
        bars_held: dict[str, int] = getattr(self._loop, "_bars_held", {}) or {}
        zombies = [
            t for t, bars in bars_held.items()
            if bars >= SIZING_MAX_HOLD_BARS + ZOMBIE_GRACE_BARS
            and t in self._pm._positions
        ]
        if not zombies:
            return {"name": "zombie_positions", "status": "ok", "healed": False,
                    "detail": "no positions past max_hold grace"}
        return {"name": "zombie_positions", "status": "critical", "healed": False,
                "detail": f"positions stuck past max_hold+{ZOMBIE_GRACE_BARS} bars: "
                          f"{zombies} — exit path may be broken",
                "zombies": zombies}

    async def _heal_zombies(self, zombies: list[str]) -> list[str]:
        """Force-exit zombie positions with a direct market sell (paper only).

        Bypasses the normal exit path entirely — the whole point is that the
        normal path may be the broken component (as on 2026-07-08/09).
        """
        from src.config import get_settings
        if get_settings().alpaca_mode != "paper":
            return []
        if os.environ.get("WATCHDOG_FORCE_EXIT", "true").lower() != "true":
            return []
        healed: list[str] = []
        for ticker in zombies:
            pos = self._pm._positions.get(ticker)
            if pos is None:
                continue
            try:
                from src.execution.alpaca import OrderRequest
                req = OrderRequest(
                    ticker=ticker,
                    side="sell" if pos.side == "long" else "buy",
                    qty=pos.qty,
                    reason=f"watchdog force-exit: stuck {ticker} past max_hold grace",
                )
                result = await self._loop._alpaca.submit_order(req)
                status = getattr(result, "status", "")
                if str(status) in ("filled", "partially_filled", "accepted", "new"):
                    healed.append(ticker)
                    logger.warning("watchdog_force_exit", ticker=ticker, status=str(status))
            except Exception:
                logger.exception("watchdog_force_exit_failed", ticker=ticker)
        return healed

    def _check_circuit_breaker(self) -> dict[str, Any]:
        halted = getattr(self._cb, "_halted", False)
        if not halted:
            return {"name": "circuit_breaker", "status": "ok", "healed": False,
                    "detail": "not halted"}
        # NEVER auto-resume (CLAUDE.md) — escalate only.
        return {"name": "circuit_breaker", "status": "critical", "healed": False,
                "detail": f"TRADING HALTED: {getattr(self._cb, '_halt_reason', '')} — "
                          "human must call resume_trading()"}

    def _check_data_fresh(self) -> dict[str, Any]:
        if not self._market_open():
            return {"name": "data_fresh", "status": "ok", "healed": False,
                    "detail": "not applicable (market closed)"}
        fresh = getattr(self._loop, "_data_fresh", True)
        return {"name": "data_fresh",
                "status": "ok" if fresh else "warn",
                "healed": False,
                "detail": "features fresh" if fresh else
                          "features stale — entries gated, check data pipeline"}

    # ── Run cycle ───────────────────────────────────────────────────────────

    async def run(self, light: bool = False) -> dict[str, Any]:
        """Full check cycle. `light=True` (off-hours) runs liveness only."""
        checks = [self._check_loop_tick_fresh(), self._check_tick_errors()]
        if not light:
            checks.append(self._check_daily_reset())
            zombie_check = self._check_zombie_positions()
            if zombie_check["status"] == "critical":
                healed = await self._heal_zombies(zombie_check.get("zombies", []))
                if healed:
                    zombie_check["healed"] = True
                    zombie_check["detail"] += f" — WATCHDOG FORCE-EXITED: {healed}"
            checks.append(zombie_check)
            checks.append(self._check_circuit_breaker())
            checks.append(self._check_data_fresh())

        worst = "ok"
        for c in checks:
            if c["status"] == "critical":
                worst = "critical"
                break
            if c["status"] == "warn":
                worst = "warn"

        report = {
            "agent": AGENT_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": worst,
            "light_run": light,
            "checks": checks,
            "healed_any": any(c.get("healed") for c in checks),
        }
        self.last_report = report

        try:
            REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
        except Exception:
            logger.exception("watchdog_report_write_failed")

        if worst != "ok" or report["healed_any"]:
            self._maybe_email(report)
        logger.info("watchdog_run", status=worst,
                    issues=[c["name"] for c in checks if c["status"] != "ok"])
        return report

    # ── Alerting ────────────────────────────────────────────────────────────

    def _maybe_email(self, report: dict[str, Any]) -> None:
        """Email on new issues or heals; dedupe identical issue sets for 4h."""
        issues = sorted(
            f"{c['name']}:{c['status']}{':healed' if c.get('healed') else ''}"
            for c in report["checks"] if c["status"] != "ok" or c.get("healed")
        )
        fingerprint = "|".join(issues)
        now = datetime.now(timezone.utc)
        if (fingerprint == self._last_email_fingerprint
                and self._last_email_at is not None
                and now - self._last_email_at < timedelta(hours=EMAIL_DEDUPE_HOURS)):
            return
        lines = [
            f"🚨 ESCALATION — {AGENT_NAME} — {report['timestamp']}",
            f"Overall status: {report['status'].upper()}",
            "",
        ]
        for c in report["checks"]:
            if c["status"] != "ok" or c.get("healed"):
                mark = "🔧 HEALED" if c.get("healed") else ("🔴" if c["status"] == "critical" else "🟡")
                lines.append(f"{mark} {c['name']}: {c['detail']}")
        lines += [
            "",
            "Dashboard: https://stockbot-production-cbde.up.railway.app/dashboard",
            "Diagnostics: https://stockbot-production-cbde.up.railway.app/watchdog",
            "",
            "Required from human: review anything not marked HEALED. "
            "Circuit-breaker halts always require manual resume_trading().",
        ]
        try:
            self._send_email(
                subject=f"[StockBot Watchdog] {report['status'].upper()}: "
                        f"{len(issues)} issue(s)",
                body="\n".join(lines),
            )
            self._last_email_fingerprint = fingerprint
            self._last_email_at = now
        except Exception:
            logger.exception("watchdog_email_failed")

    def _send_email(self, subject: str, body: str) -> None:
        """Same SMTP path as the daily forecast email (proven in prod)."""
        from src.config import get_settings
        s = get_settings()
        if not s.smtp_host or not s.smtp_user or not s.smtp_password:
            logger.warning("watchdog_smtp_not_configured")
            return
        recipients = [r.strip() for r in s.forecast_email_to.split(",") if r.strip()]
        if not recipients:
            return
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = s.forecast_email_from or s.smtp_user
        msg["To"] = ", ".join(recipients)
        msg.set_content(body)
        context = ssl.create_default_context()
        if s.smtp_port == 465:
            with smtplib.SMTP_SSL(s.smtp_host, s.smtp_port, context=context, timeout=30) as srv:
                srv.login(s.smtp_user, s.smtp_password)
                srv.send_message(msg)
        else:
            with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=30) as srv:
                srv.starttls(context=context)
                srv.login(s.smtp_user, s.smtp_password)
                srv.send_message(msg)
