"""Integrity Sentinel — audits the trade ledger itself, not the strategy.

Scope: Verify that every number downstream systems consume can be recomputed
       from raw fills. Checks: stored pnl_pct vs recomputed pnl/(entry_price
       * shares), zombie open rows the orphan-recovery path can never close,
       Kelly-seed sanity (a poisoned pnl_pct window silently corrupts sizing
       after every redeploy), and DB-vs-broker open-position reconciliation.
Can act autonomously (paper mode, repair=True or INTEGRITY_AUTO_REPAIR=true):
       Rewrite divergent pnl_pct to the recomputed value, rewrite corrupt
       dollar pnl (and its pnl_pct) to the fill-based truth when the shares
       column is trustworthy, and close stale open rows — after writing a JSON
       backup of every row it touches.
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
DB_BROKER_GRACE_MINUTES = 10   # min age before a DB-only open row is a true orphan
# No sane single swing trade returns more than ±25%; beyond that the Kelly
# seed window is presumed poisoned by an accounting bug.
KELLY_SANE_ABS_PNL_PCT = 0.25
AUDIT_WINDOW_DAYS = 60
# Fill-price vs stored-pnl tolerance: the stored pnl and (exit-entry)*shares
# should agree within this fraction of entry notional. A 5% tolerance absorbs
# legitimate deviations from multi-fill averaging (observed max ~2-3% on clean
# trades) while still catching the partial-fill corruption signature seen in the
# 2026-07-20 audit where MU #93 diverged by 15% and SMCI #83 by 25%.
FILL_PNL_TOLERANCE = 0.05
# Exit rows written after this moment carry a caller-computed pnl_pct from the
# true position notional (v0.4.5 root fix, deployed 2026-07-21 05:20 UTC).
# For those rows a divergence means the entry-side SHARES column is wrong.
TRUSTED_PNL_PCT_SINCE = datetime(2026, 7, 21, 5, 20, tzinfo=timezone.utc)
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


def fill_price_pnl_deviation(
    pnl: float | None,
    exit_price: float | None,
    entry_price: float | None,
    shares: float | None,
) -> float | None:
    """Fraction of entry notional by which stored pnl deviates from fill prices.

    Computes |pnl - (exit_price - entry_price) * shares| / |entry_price * shares|.
    Returns None when any input is missing or the notional is too small.

    A non-zero result is expected for legitimate multi-fill trades (observed
    1-3% from fill averaging). Deviations above FILL_PNL_TOLERANCE (5%)
    are the partial-fill corruption signature (MU #93: 15%, SMCI #83: 25%).
    """
    if pnl is None or exit_price is None or not entry_price or not shares:
        return None
    notional = abs(entry_price * shares)
    if notional < 1.0:
        return None
    fill_pnl = (exit_price - entry_price) * shares
    return abs(float(pnl) - fill_pnl) / notional


def corrected_fill_pnl(
    exit_price: float | None,
    entry_price: float | None,
    shares: float | None,
) -> tuple[float, float] | None:
    """Recompute the true (pnl, pnl_pct) from fill prices and the shares column.

    Used to HEAL fill-price corruption. This is only trustworthy when the
    ``shares`` column is the reliable one — verified because every corrupt row
    seen (XOM #68, MSFT #82, MU #93, SMCI #83) has a ``shares`` count that sits
    inside the sizer's notional cap, while the shares *implied* by the corrupt
    dollar pnl would blow through it (e.g. MU #93 stored pnl implies 13.5 shares
    = $11.6k notional, impossible under the $2.5k cap; the recorded 3 shares =
    $2.6k fits). So the dollar ``pnl`` column is the corrupt side and
    ``(exit - entry) * shares`` is the fill-based truth.

    Returns None when inputs are missing or the notional is dust.
    """
    if exit_price is None or not entry_price or not shares:
        return None
    notional = abs(entry_price * shares)
    if notional < 1.0:
        return None
    pnl = (exit_price - entry_price) * shares
    return pnl, pnl / notional


def is_divergent(stored: float | None, expected: float | None,
                 tol: float = PNL_PCT_TOLERANCE) -> bool:
    """True when the stored pnl_pct disagrees with the recomputed value."""
    if stored is None or expected is None:
        return False
    return abs(stored - expected) > tol


def classify_row(stored: float | None, expected: float | None) -> str | None:
    """Repair action for a closed-trade row, or None to leave it alone.

    "rewrite": stored diverges from a sane recompute → exit-side corruption,
        set stored to the recomputed value.
    "nullify": recompute is itself impossible AND stored is too → entry-side
        corruption (shares/entry columns wrong); no derived pct is
        trustworthy, so NULL it out of the Kelly seed and pct stats.
    None: consistent row — or recompute impossible while stored is sane
        (stored came from the true PM entry notional at exit; keep it).
    """
    if expected is not None and abs(expected) > KELLY_SANE_ABS_PNL_PCT:
        if stored is not None and abs(stored) > KELLY_SANE_ABS_PNL_PCT:
            return "nullify"
        return None
    if is_divergent(stored, expected):
        return "rewrite"
    return None


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
                   Trade.entry_price, Trade.shares, Trade.exit_time)
            .where(Trade.exit_time.isnot(None), Trade.exit_time >= cutoff)
        )
        divergent: list[dict[str, Any]] = []
        for tid, ticker, pnl, stored, entry_price, shares, exit_time in result.all():
            exp = expected_pnl_pct(pnl, entry_price, shares)
            ts = exit_time if exit_time.tzinfo else exit_time.replace(tzinfo=timezone.utc)
            if ts >= TRUSTED_PNL_PCT_SINCE:
                # stored is truth; a divergence indicts the shares column
                if is_divergent(stored, exp) and stored and entry_price:
                    true_shares = round(float(pnl) / (float(stored) * float(entry_price)), 4)
                    divergent.append({
                        "id": tid, "ticker": ticker,
                        "stored_pnl_pct": round(float(stored), 5),
                        "expected_pnl_pct": round(float(exp), 5) if exp is not None else None,
                        "shares_recorded": shares, "shares_implied": true_shares,
                        "action": "reconcile_shares",
                    })
                continue
            action = classify_row(stored, exp)
            if action is not None:
                divergent.append({
                    "id": tid, "ticker": ticker,
                    "stored_pnl_pct": round(float(stored), 5) if stored is not None else None,
                    "expected_pnl_pct": round(float(exp), 5) if exp is not None else None,
                    "action": action,
                })
        if divergent:
            return {"name": "pnl_pct_consistency", "status": "critical",
                    "healed": False, "rows": divergent,
                    "detail": f"{len(divergent)} closed trade(s) in {AUDIT_WINDOW_DAYS}d "
                              "with pnl_pct that cannot be recomputed from "
                              "pnl/(entry_price*shares) — partial-fill exit bug signature"}
        return {"name": "pnl_pct_consistency", "status": "ok", "healed": False,
                "rows": [], "detail": "all stored pnl_pct match recomputation"}

    async def _audit_fill_price_pnl(self, session: Any) -> dict[str, Any]:
        """Closed trades where pnl cannot be reconciled from fill prices.

        The existing pnl_pct check verifies internal consistency:
        pnl_pct = pnl / (entry_price * shares). It passes even when the pnl
        dollar amount itself is corrupted, because both sides of the ratio are
        wrong by the same factor. This check closes that blind spot: for each
        closed trade with a stored exit_price it verifies that
        |pnl - (exit_price - entry_price) * shares| / |entry_price * shares|
        is below FILL_PNL_TOLERANCE.

        Known anomalies motivating this check (pre-v0.4.5 repair window):
          MU #93   pnl=-512 vs fill-implied -114 (15.5% deviation, 3 shares)
          SMCI #83 pnl=-638 vs fill-implied -139 (24.8% deviation, 72 shares)
        Both passed pnl_pct_consistency because the pct was internally consistent
        with the corrupted pnl — not with the actual fill prices.
        """
        from sqlalchemy import select

        from src.data.db import Trade

        cutoff = datetime.now(timezone.utc) - timedelta(days=AUDIT_WINDOW_DAYS)
        result = await session.execute(
            select(Trade.id, Trade.ticker, Trade.pnl, Trade.entry_price,
                   Trade.exit_price, Trade.shares, Trade.exit_time)
            .where(Trade.exit_time.isnot(None), Trade.exit_time >= cutoff,
                   Trade.exit_price.isnot(None))
        )
        anomalous: list[dict[str, Any]] = []
        for tid, ticker, pnl, entry_price, exit_price, shares, _exit_time in result.all():
            dev = fill_price_pnl_deviation(pnl, exit_price, entry_price, shares)
            if dev is None or dev <= FILL_PNL_TOLERANCE:
                continue
            fill_pnl = (float(exit_price) - float(entry_price)) * float(shares)
            corrected = corrected_fill_pnl(exit_price, entry_price, shares)
            # Only offer to auto-heal when the fill-based recompute is itself
            # sane (|pnl_pct| ≤ KELLY_SANE bound). Beyond that the shares column
            # is suspect too, so leave the row for a human (action=None).
            can_heal = (
                corrected is not None
                and abs(corrected[1]) <= KELLY_SANE_ABS_PNL_PCT
            )
            anomalous.append({
                "id": tid,
                "ticker": ticker,
                "stored_pnl": round(float(pnl), 2),
                "fill_price_pnl": round(fill_pnl, 2),
                "deviation_pct": round(dev * 100, 2),
                "entry_price": round(float(entry_price), 4),
                "exit_price": round(float(exit_price), 4),
                "shares": shares,
                "corrected_pnl": round(corrected[0], 2) if corrected else None,
                "corrected_pnl_pct": round(corrected[1], 5) if corrected else None,
                "action": "rewrite_pnl" if can_heal else None,
                "note": ("pnl does not reconcile with (exit-entry)*shares — "
                         "suspect residual partial-fill corruption in pnl column"),
            })
        if anomalous:
            return {
                "name": "fill_price_pnl_consistency",
                "status": "warn",
                "healed": False,
                "rows": anomalous,
                "detail": (
                    f"{len(anomalous)} closed trade(s) in {AUDIT_WINDOW_DAYS}d where "
                    f"|pnl - fill_pnl| > {int(FILL_PNL_TOLERANCE * 100)}% of notional — "
                    "pnl_pct is internally consistent but the dollar pnl itself "
                    "cannot be reconciled from fill prices; investigate before "
                    "trusting cumulative PnL totals"
                ),
            }
        return {
            "name": "fill_price_pnl_consistency",
            "status": "ok",
            "healed": False,
            "rows": [],
            "detail": "all closed trade pnl values reconcile with fill prices",
        }

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
        broker_tickers = {p.get("ticker") or p.get("symbol") for p in positions}
        broker_tickers.discard(None)
        # Grace period: a position entered seconds ago is written to the DB
        # before the broker-position read reflects it. Only rows older than
        # this can be true orphans, never an in-flight entry.
        grace_cutoff = datetime.now(timezone.utc) - timedelta(
            minutes=DB_BROKER_GRACE_MINUTES)
        result = await session.execute(
            select(Trade.id, Trade.ticker, Trade.entry_time)
            .where(Trade.exit_time.is_(None))
        )
        open_rows = result.all()
        db_tickers = {r[1] for r in open_rows}
        only_broker = sorted(broker_tickers - db_tickers)
        # only_db orphans eligible to auto-close: DB-open, not at broker, past grace.
        orphans = [
            {"id": tid, "ticker": tkr, "entry_time": str(et)}
            for tid, tkr, et in open_rows
            if tkr not in broker_tickers
            and (et if et.tzinfo else et.replace(tzinfo=timezone.utc)) < grace_cutoff
        ]
        only_db = sorted({o["ticker"] for o in orphans})
        if only_db or only_broker:
            return {"name": "db_vs_broker", "status": "critical", "healed": False,
                    "orphans": orphans,
                    "detail": f"ledger/broker mismatch — open in DB only (closeable): "
                              f"{only_db}; at broker only (needs human): {only_broker}"}
        return {"name": "db_vs_broker", "status": "ok", "healed": False,
                "orphans": [],
                "detail": f"{len(db_tickers)} open position(s) reconcile"}

    # ── Repair (paper mode only, gated) ─────────────────────────────────────

    def _repair_allowed(self, repair: bool) -> bool:
        from src.config import get_settings
        if get_settings().alpaca_mode != "paper":
            return False
        return repair or os.environ.get("INTEGRITY_AUTO_REPAIR", "false").lower() == "true"

    async def _repair(self, session: Any, divergent: list[dict[str, Any]],
                      stale: list[dict[str, Any]],
                      fill_corrupt: list[dict[str, Any]] | None = None,
                      broker_orphans: list[dict[str, Any]] | None = None
                      ) -> dict[str, Any]:
        """Rewrite bad pnl_pct + dollar pnl + close zombie rows, backing up first.

        Stale rows are closed with pnl/pnl_pct left NULL: the true exit price
        is unknowable, and NULL keeps them out of the Kelly seed query and the
        profit reports rather than injecting a fabricated zero.

        ``fill_corrupt`` rows are the fill-price-reconciliation failures: the
        dollar ``pnl`` column is corrupt (inflated 4-6x by a stale full-position
        qty) while ``shares`` is trustworthy. Each row carries the corrected
        ``pnl``/``pnl_pct`` computed from ``(exit-entry)*shares``; rewriting both
        removes the phantom loss from cumulative PnL and from the Kelly seed
        window. Only rows the audit marked ``action="rewrite_pnl"`` (sane
        recompute) are healed.
        """
        from sqlalchemy import update

        from src.data.db import Trade

        fill_corrupt = fill_corrupt or []
        broker_orphans = broker_orphans or []
        healable_fill = [r for r in fill_corrupt if r.get("action") == "rewrite_pnl"]
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        backup_path = REPORT_DIR / f"repair_backup_{stamp}.json"
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(json.dumps(
            {"divergent_pnl_pct": divergent, "stale_open_rows": stale,
             "fill_corrupt_pnl": healable_fill, "broker_orphans": broker_orphans},
            indent=2, default=str))

        for row in divergent:
            if row.get("action") == "reconcile_shares":
                await session.execute(
                    update(Trade).where(Trade.id == row["id"])
                    .values(shares=row["shares_implied"])
                )
                continue
            if row.get("action") == "nullify":
                # Entry columns corrupt — no derived pct is trustworthy.
                # NULL keeps the dollar pnl and excludes the row from the
                # Kelly seed and pct-based stats instead of fabricating.
                await session.execute(
                    update(Trade).where(Trade.id == row["id"])
                    .values(pnl_pct=None)
                )
            else:
                await session.execute(
                    update(Trade).where(Trade.id == row["id"])
                    .values(pnl_pct=row["expected_pnl_pct"])
                )
        for row in healable_fill:
            # Corrupt dollar pnl → rewrite both pnl and pnl_pct to the
            # fill-based truth (shares is the trustworthy column here).
            await session.execute(
                update(Trade).where(Trade.id == row["id"])
                .values(pnl=row["corrected_pnl"], pnl_pct=row["corrected_pnl_pct"])
            )
        now = datetime.now(timezone.utc)
        for row in stale:
            await session.execute(
                update(Trade).where(Trade.id == row["id"])
                .values(exit_time=now, exit_reason="integrity_stale_cleanup")
            )
        for row in broker_orphans:
            # Broker is the source of truth for whether a position exists.
            # The position closed at the broker without a DB exit write
            # (redeploy lost _open_trade_ids, or a watchdog/broker-side close
            # bypassed the exit path). Close the row; the true exit price is
            # unknowable so pnl/pnl_pct stay NULL (kept out of Kelly + stats).
            await session.execute(
                update(Trade).where(Trade.id == row["id"])
                .values(exit_time=now, exit_reason="integrity_broker_reconcile")
            )
        await session.commit()

        # Re-seed Kelly from the repaired ledger so sizing recovers without
        # waiting for the next redeploy.
        kelly_reseeded = False
        if self._loop is not None and (divergent or stale or healable_fill or broker_orphans):
            try:
                await self._loop._seed_kelly_from_db()
                kelly_reseeded = True
            except Exception:
                logger.exception("integrity_kelly_reseed_failed")

        logger.warning("integrity_repair_applied",
                       pnl_pct_rewritten=len(divergent), stale_closed=len(stale),
                       fill_pnl_rewritten=len(healable_fill),
                       kelly_reseeded=kelly_reseeded, backup=str(backup_path))
        return {"pnl_pct_rewritten": len(divergent), "stale_closed": len(stale),
                "fill_pnl_rewritten": len(healable_fill),
                "broker_orphans_closed": len(broker_orphans),
                "kelly_reseeded": kelly_reseeded, "backup": str(backup_path)}

    # ── Run cycle ───────────────────────────────────────────────────────────

    async def run(self, repair: bool = False) -> dict[str, Any]:
        """Full audit; optionally repair what the audit found (paper only)."""
        async with self._sf() as session:
            checks = [
                await self._audit_pnl_pct(session),
                await self._audit_fill_price_pnl(session),
                await self._audit_stale_open_rows(session),
                await self._audit_kelly_window(session),
                await self._audit_db_vs_broker(session),
            ]

            repair_result: dict[str, Any] | None = None
            divergent = next(c for c in checks if c["name"] == "pnl_pct_consistency")["rows"]
            stale = next(c for c in checks if c["name"] == "stale_open_rows")["rows"]
            broker_orphans = next(
                (c.get("orphans", []) for c in checks if c["name"] == "db_vs_broker"), [])
            fill_corrupt = next(
                c for c in checks if c["name"] == "fill_price_pnl_consistency")["rows"]
            healable_fill = [r for r in fill_corrupt if r.get("action") == "rewrite_pnl"]
            if (divergent or stale or healable_fill or broker_orphans) \
                    and self._repair_allowed(repair):
                repair_result = await self._repair(
                    session, divergent, stale, fill_corrupt, broker_orphans)
                healed_names = {"pnl_pct_consistency", "stale_open_rows"}
                # Only mark the fill check healed if every flagged row was healable.
                if healable_fill and len(healable_fill) == len(fill_corrupt):
                    healed_names.add("fill_price_pnl_consistency")
                # db_vs_broker heals only when the mismatch was purely closeable
                # orphans (no "at broker only" names, which need a human).
                dbc = next(c for c in checks if c["name"] == "db_vs_broker")
                if broker_orphans and "at broker only (needs human): []" in dbc["detail"]:
                    healed_names.add("db_vs_broker")
                for c in checks:
                    if c["name"] in healed_names and c["status"] != "ok":
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
