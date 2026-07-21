"""Integrity Sentinel — audits the trade ledger itself, not the strategy.

Scope: Verify that every number downstream systems consume can be recomputed
       from raw fills. Checks: stored pnl_pct vs recomputed pnl/(entry_price
       * shares), zombie open rows the orphan-recovery path can never close,
       Kelly-seed sanity (a poisoned pnl_pct window silently corrupts sizing
       after every redeploy), and DB-vs-broker open-position reconciliation.
Can act autonomously (paper mode, repair=True or INTEGRITY_AUTO_REPAIR=true):
       Rewrite divergent pnl_pct to the recomputed value and close stale open
       rows — after writing a JSON backup of every row it touches.
Must NOT: Touch live config, place orders, or modify any strategy parameter.
Output: reports/integrity/latest.json + dated snapshot + email escalation via
       the same SMTP path as the watchdog.
Escalate if: any divergent pnl_pct, poisoned Kelly window, stale open rows, or
       DB/broker position mismatch.

Born from the 2026-07-20 audit: partially filled exits wrote pnl_pct from the
partial qty (GOOGL logged -15.7% on a -4.0% trade, another row +993%), and
_seed_kelly_from_db() fed those rows straight into position sizing after every
redeploy. The strategy was being throttled by an accounting bug, not by edge.
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

import structlog

logger = structlog.get_logger(__name__)

AGENT_NAME = "Integrity Sentinel"
REPORT_DIR = Path("reports/integrity")

# |stored - expected| above this (absolute, in return units) flags a row.
# 0.005 = half a percentage point — generous vs float noise, tight vs the
# order-of-magnitude corruption this agent exists to catch.
PNL_PCT_TOLERANCE = 0.005
# An open DB row older than this can never be the "most recent open trade"
# the orphan-recovery path matches on — it is a zombie.
STALE_OPEN_DAYS = 5
# No sane single swing trade returns more than ±25%; beyond that the Kelly
# seed window is presumed poisoned by an accounting bug.
KELLY_SANE_ABS_PNL_PCT = 0.25
AUDIT_WINDOW_DAYS = 60
EMAIL_DEDUPE_HOURS = 4


def expected_pnl_pct(pnl: float | None, entry_price: float | None,
                     shares: float | None) -> float | None:
    """Recompute pnl_pct from the row's own entry columns.

    Returns None when the row lacks the inputs (no silent guesses).
    """
    if pnl is None or not entry_price or not shares:
        return None
    notional = abs(entry_price * shares)
    if notional < 1.0:
        return None
    return pnl / notional


def is_divergent(stored: float | None, expected: float | None,
                 tol: float = PNL_PCT_TOLERANCE) -> bool:
    """True when the stored pnl_pct disagrees with the recomputed value."""
    if stored is None or expected is None:
        return False
    return abs(stored - expected) > tol


class IntegrityAgent:
    """Audit + (gated) repair of the trades table. Reports like the watchdog."""

    def __init__(self, session_factory: Any, signal_loop: Any | None = None,
                 alpaca: Any | None = None) -> None:
        self._sf = session_factory
        self._loop = signal_loop
        self._alpaca = alpaca
        self._last_email_fingerprint = ""
        self._last_email_at: datetime | None = None
        self.last_report: dict[str, Any] | None = None

    # ── Individual checks ───────────────────────────────────────────────────

    async def _audit_pnl_pct(self, session: Any) -> dict[str, Any]:
        """Closed trades whose stored pnl_pct can't be recomputed from fills."""
        from sqlalchemy import select

        from src.data.db import Trade

        cutoff = datetime.now(timezone.utc) - timedelta(days=AUDIT_WINDOW_DAYS)
        result = await session.execute(
            select(Trade.id, Trade.ticker, Trade.pnl, Trade.pnl_pct,
                   Trade.entry_price, Trade.shares)
            .where(Trade.exit_time.isnot(None), Trade.exit_time >= cutoff)
        )
        divergent: list[dict[str, Any]] = []
        for tid, ticker, pnl, stored, entry_price, shares in result.all():
            exp = expected_pnl_pct(pnl, entry_price, shares)
            if is_divergent(stored, exp):
                divergent.append({
                    "id": tid, "ticker": ticker,
                    "stored_pnl_pct": round(float(stored), 5),
                    "expected_pnl_pct": round(float(exp), 5),
                })
        if divergent:
            return {"name": "pnl_pct_consistency", "status": "critical",
                    "healed": False, "rows": divergent,
                    "detail": f"{len(divergent)} closed trade(s) in {AUDIT_WINDOW_DAYS}d "
                              "with pnl_pct that cannot be recomputed from "
                              "pnl/(entry_price*shares) — partial-fill exit bug signature"}
        return {"name": "pnl_pct_consistency", "status": "ok", "healed": False,
                "rows": [], "detail": "all stored pnl_pct match recomputation"}

    async def _audit_stale_open_rows(self, session: Any) -> dict[str, Any]:
        """Open DB rows too old for orphan recovery to ever close."""
        from sqlalchemy import select

        from src.data.db import Trade

        cutoff = datetime.now(timezone.utc) - timedelta(days=STALE_OPEN_DAYS)
        result = await session.execute(
            select(Trade.id, Trade.ticker, Trade.entry_time)
            .where(Trade.exit_time.is_(None), Trade.entry_time < cutoff)
        )
        stale = [{"id": tid, "ticker": ticker, "entry_time": str(entry)}
                 for tid, ticker, entry in result.all()]
        if stale:
            return {"name": "stale_open_rows", "status": "critical",
                    "healed": False, "rows": stale,
                    "detail": f"{len(stale)} open row(s) older than {STALE_OPEN_DAYS}d — "
                              "zombie ledger entries; orphan recovery matches newest-first "
                              "and will never close these"}
        return {"name": "stale_open_rows", "status": "ok", "healed": False,
                "rows": [], "detail": "no zombie open rows"}

    async def _audit_kelly_window(self, session: Any) -> dict[str, Any]:
        """Rows inside the Kelly seed window with impossible pnl_pct values."""
        from sqlalchemy import select

        from src.data.db import Trade

        try:
            from src.agents.signal_loop import KELLY_LOOKBACK_DAYS
        except ImportError:
            KELLY_LOOKBACK_DAYS = 10
        cutoff = datetime.now(timezone.utc) - timedelta(days=KELLY_LOOKBACK_DAYS)
        result = await session.execute(
            select(Trade.id, Trade.ticker, Trade.pnl_pct)
            .where(Trade.exit_time.isnot(None), Trade.exit_time >= cutoff,
                   Trade.pnl_pct.isnot(None))
        )
        poisoned = [{"id": tid, "ticker": ticker, "pnl_pct": round(float(p), 5)}
                    for tid, ticker, p in result.all()
                    if abs(p) > KELLY_SANE_ABS_PNL_PCT]
        if poisoned:
            return {"name": "kelly_seed_sanity", "status": "critical",
                    "healed": False, "rows": poisoned,
                    "detail": f"{len(poisoned)} row(s) in the {KELLY_LOOKBACK_DAYS}d Kelly "
                              f"window with |pnl_pct| > {KELLY_SANE_ABS_PNL_PCT} — "
                              "_seed_kelly_from_db() will corrupt sizing on next restart"}
        return {"name": "kelly_seed_sanity", "status": "ok", "healed": False,
                "rows": [], "detail": "Kelly seed window is sane"}

    async def _audit_db_vs_broker(self, session: Any) -> dict[str, Any]:
        """Open DB rows vs actual broker positions (count + ticker set)."""
        from sqlalchemy import select

        from src.data.db import Trade

        if self._alpaca is None:
            return {"name": "db_vs_broker", "status": "ok", "healed": False,
                    "detail": "not applicable (no broker client)"}
        try:
            positions = await self._alpaca.get_positions()
        except Exception as exc:
            return {"name": "db_vs_broker", "status": "warn", "healed": False,
                    "detail": f"broker positions unavailable: {exc}"}
        broker_tickers = {p.get("symbol") for p in positions}
        result = await session.execute(
            select(Trade.ticker).where(Trade.exit_time.is_(None))
        )
        db_tickers = {row[0] for row in result.all()}
        only_db = sorted(db_tickers - broker_tickers)
        only_broker = sorted(broker_tickers - db_tickers)
        if only_db or only_broker:
            return {"name": "db_vs_broker", "status": "critical", "healed": False,
                    "detail": f"ledger/broker mismatch — open in DB only: {only_db}; "
                              f"at broker only: {only_broker}"}
        return {"name": "db_vs_broker", "status": "ok", "healed": False,
                "detail": f"{len(db_tickers)} open position(s) reconcile"}

    # ── Repair (paper mode only, gated) ─────────────────────────────────────

    def _repair_allowed(self, repair: bool) -> bool:
        from src.config import get_settings
        if get_settings().alpaca_mode != "paper":
            return False
        return repair or os.environ.get("INTEGRITY_AUTO_REPAIR", "false").lower() == "true"

    async def _repair(self, session: Any, divergent: list[dict[str, Any]],
                      stale: list[dict[str, Any]]) -> dict[str, Any]:
        """Rewrite bad pnl_pct + close zombie rows, backing up every row first.

        Stale rows are closed with pnl/pnl_pct left NULL: the true exit price
        is unknowable, and NULL keeps them out of the Kelly seed query and the
        profit reports rather than injecting a fabricated zero.
        """
        from sqlalchemy import update

        from src.data.db import Trade

        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        backup_path = REPORT_DIR / f"repair_backup_{stamp}.json"
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(json.dumps(
            {"divergent_pnl_pct": divergent, "stale_open_rows": stale},
            indent=2, default=str))

        for row in divergent:
            await session.execute(
                update(Trade).where(Trade.id == row["id"])
                .values(pnl_pct=row["expected_pnl_pct"])
            )
        now = datetime.now(timezone.utc)
        for row in stale:
            await session.execute(
                update(Trade).where(Trade.id == row["id"])
                .values(exit_time=now, exit_reason="integrity_stale_cleanup")
            )
        await session.commit()

        # Re-seed Kelly from the repaired ledger so sizing recovers without
        # waiting for the next redeploy.
        kelly_reseeded = False
        if self._loop is not None and (divergent or stale):
            try:
                await self._loop._seed_kelly_from_db()
                kelly_reseeded = True
            except Exception:
                logger.exception("integrity_kelly_reseed_failed")

        logger.warning("integrity_repair_applied",
                       pnl_pct_rewritten=len(divergent), stale_closed=len(stale),
                       kelly_reseeded=kelly_reseeded, backup=str(backup_path))
        return {"pnl_pct_rewritten": len(divergent), "stale_closed": len(stale),
                "kelly_reseeded": kelly_reseeded, "backup": str(backup_path)}

    # ── Run cycle ───────────────────────────────────────────────────────────

    async def run(self, repair: bool = False) -> dict[str, Any]:
        """Full audit; optionally repair what the audit found (paper only)."""
        async with self._sf() as session:
            checks = [
                await self._audit_pnl_pct(session),
                await self._audit_stale_open_rows(session),
                await self._audit_kelly_window(session),
                await self._audit_db_vs_broker(session),
            ]

            repair_result: dict[str, Any] | None = None
            divergent = next(c for c in checks if c["name"] == "pnl_pct_consistency")["rows"]
            stale = next(c for c in checks if c["name"] == "stale_open_rows")["rows"]
            if (divergent or stale) and self._repair_allowed(repair):
                repair_result = await self._repair(session, divergent, stale)
                for c in checks:
                    if c["name"] in ("pnl_pct_consistency", "stale_open_rows") \
                            and c["status"] != "ok":
                        c["healed"] = True
                        c["detail"] += " — REPAIRED (backup written)"

        worst = "ok"
        for c in checks:
            if c["status"] == "critical" and not c.get("healed"):
                worst = "critical"
                break
            if c["status"] != "ok":
                worst = "warn"

        report = {
            "agent": AGENT_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": worst,
            "checks": [
                # Row lists can be large — cap what goes in the report.
                {**c, "rows": c.get("rows", [])[:20]} for c in checks
            ],
            "repair": repair_result,
            "healed_any": any(c.get("healed") for c in checks),
        }
        self.last_report = report

        try:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            (REPORT_DIR / "latest.json").write_text(
                json.dumps(report, indent=2, default=str))
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            (REPORT_DIR / f"{day}.json").write_text(
                json.dumps(report, indent=2, default=str))
        except Exception:
            logger.exception("integrity_report_write_failed")

        if worst != "ok" or report["healed_any"]:
            self._maybe_email(report)
        logger.info("integrity_run", status=worst,
                    issues=[c["name"] for c in checks if c["status"] != "ok"])
        return report

    # ── Alerting (same pattern as WatchdogAgent) ────────────────────────────

    def _maybe_email(self, report: dict[str, Any]) -> None:
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
                mark = "🔧 HEALED" if c.get("healed") else (
                    "🔴" if c["status"] == "critical" else "🟡")
                lines.append(f"{mark} {c['name']}: {c['detail']}")
        if report.get("repair"):
            lines.append("")
            lines.append(f"Repair applied: {report['repair']}")
        lines += [
            "",
            "Report: https://stockbot-production-cbde.up.railway.app/integrity",
            "",
            "Required from human: review any unhealed check. A ledger the "
            "sizing engine cannot trust is a trading halt waiting to happen.",
        ]
        try:
            self._send_email(
                subject=f"[StockBot Integrity] {report['status'].upper()}: "
                        f"{len(issues)} issue(s)",
                body="\n".join(lines),
            )
            self._last_email_fingerprint = fingerprint
            self._last_email_at = now
        except Exception:
            logger.exception("integrity_email_failed")

    def _send_email(self, subject: str, body: str) -> None:
        """Same SMTP path as the watchdog/forecast emails (proven in prod)."""
        from src.config import get_settings
        s = get_settings()
        if not s.smtp_host or not s.smtp_user or not s.smtp_password:
            logger.warning("integrity_smtp_not_configured")
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
