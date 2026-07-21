"""News Risk Agent — macro event-risk radar (war, crash, systemic shocks).

Scope: Scan broad-market headlines for macro shock events the per-ticker
       FinBERT sentiment pipeline is blind to (it scores company news, not
       geopolitics). Score a macro risk LEVEL, publish it, and email the
       owner the moment it turns high/severe so positions can be reviewed
       before/at the open.
Can act autonomously: NOTHING position-related by default. With
       NEWS_RISK_GATE=block (owner opt-in), the signal loop refuses NEW
       entries while the level is high/severe. It NEVER closes positions —
       closing on headlines is a strategy change that needs the owner (or
       the circuit breaker) per CLAUDE.md.
Output: reports/news_risk/latest.json + GET /news-risk + email escalation.
Escalate if: level high or severe.

Honest latency note: NewsAPI's free tier is not tick-level — expect minutes
of delay, and overnight shocks are caught by the pre-market scans (every 15
minutes from 06:30 ET), not "before the news happens". This is a seatbelt,
not a crystal ball.
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

AGENT_NAME = "News Risk Agent"
REPORT_PATH = Path("reports/news_risk/latest.json")
EMAIL_DEDUPE_HOURS = 4

# Severity keywords — matched case-insensitively against headline+description.
# Levels: 3=severe (assume market dislocation), 2=high (risk-off), 1=elevated.
# Phrases are deliberately specific: "war" alone would fire on "bidding war".
SEVERITY_KEYWORDS: dict[int, tuple[str, ...]] = {
    3: (
        "declares war", "declaration of war", "nuclear", "invasion of",
        "invades", "market crash", "trading halted", "circuit breaker triggered",
        "bank collapse", "bank failure", "sovereign default", "flash crash",
        "state of emergency", "terrorist attack",
    ),
    2: (
        "war breaks out", "goes to war", "at war with", "war with", "warplanes",
        "missile strike", "air strikes", "emergency rate",
        "emergency meeting", "military escalation", "sweeping sanctions",
        "pandemic", "sell-off deepens", "plunges", "contagion",
        "default risk", "capital controls",
    ),
    1: (
        "tariff", "trade war", "escalation", "geopolitical tension", "conflict",
        "selloff", "correction territory", "recession fears", "downgrade",
        "vix spike", "risk-off",
    ),
}
LEVEL_NAMES = {0: "none", 1: "elevated", 2: "high", 3: "severe"}

# The query sent to NewsAPI /everything for macro-shock coverage.
MACRO_QUERY = (
    '"war" OR "invasion" OR "market crash" OR "nuclear" OR "emergency rate" OR '
    '"sanctions" OR "bank failure" OR "sovereign default" OR "trading halted"'
)

# Signal-loop gate cache (module-level so the loop can poll cheaply)
_gate_cache: dict[str, Any] = {"at": None, "level": 0}


def score_headline(title: str, body: str = "") -> tuple[int, list[str]]:
    """Severity level (0-3) for one article + which phrases matched."""
    text = ((title or "") + " " + (body or "")).lower()
    matched: list[str] = []
    level = 0
    for lvl in (3, 2, 1):
        for phrase in SEVERITY_KEYWORDS[lvl]:
            if phrase in text:
                matched.append(phrase.strip())
                level = max(level, lvl)
    return level, matched


def score_articles(articles: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate macro risk from a batch of raw NewsAPI articles.

    A single stray headline can't set high/severe: levels 2-3 require at
    least 2 distinct matching articles (different titles), else they demote
    one level. Level 1 sticks from a single article.
    """
    flagged: list[dict[str, Any]] = []
    for a in articles:
        lvl, matched = score_headline(a.get("title", ""), a.get("description", ""))
        if lvl > 0:
            flagged.append({
                "level": lvl,
                "matched": matched,
                "title": a.get("title", ""),
                "source": (a.get("source") or {}).get("name", ""),
                "published_at": a.get("publishedAt", ""),
                "url": a.get("url", ""),
            })
    flagged.sort(key=lambda f: f["level"], reverse=True)
    top = flagged[0]["level"] if flagged else 0
    if top >= 2:
        n_at_top = len({f["title"] for f in flagged if f["level"] >= top})
        if n_at_top < 2:
            top -= 1  # one lone headline demotes — wire services echo real shocks
    return {
        "level": top,
        "level_name": LEVEL_NAMES[top],
        "n_flagged": len(flagged),
        "n_scanned": len(articles),
        "headlines": flagged[:10],
    }


def current_news_risk_level(max_age_minutes: int = 45) -> int:
    """Signal-loop gate helper: latest published level, 60s file-read cache.

    Fails OPEN (level 0) when the report is missing or stale — a broken news
    feed must never be able to halt trading by itself.
    """
    now = datetime.now(timezone.utc)
    if _gate_cache["at"] and (now - _gate_cache["at"]).total_seconds() < 60:
        return _gate_cache["level"]
    level = 0
    try:
        report = json.loads(REPORT_PATH.read_text())
        ts = datetime.fromisoformat(report["timestamp"])
        if (now - ts).total_seconds() < max_age_minutes * 60:
            level = int(report.get("level", 0))
    except Exception:
        level = 0
    _gate_cache["at"] = now
    _gate_cache["level"] = level
    return level


def news_risk_blocks_entries() -> bool:
    """True only when the owner opted in (NEWS_RISK_GATE=block) AND level >= high."""
    if os.environ.get("NEWS_RISK_GATE", "alert").lower() != "block":
        return False
    return current_news_risk_level() >= 2


class NewsRiskAgent:
    """Fetch macro headlines, score, publish, escalate."""

    def __init__(self) -> None:
        self._last_email_fingerprint = ""
        self._last_email_at: datetime | None = None
        self.last_report: dict[str, Any] | None = None

    async def run(self) -> dict[str, Any]:
        from src.data.news import NewsAPIClient

        articles: list[dict[str, Any]] = []
        try:
            articles = await NewsAPIClient().fetch_recent(
                query=MACRO_QUERY, hours_back=12)
        except Exception as exc:
            logger.warning("news_risk_fetch_failed", error=str(exc))

        scored = score_articles(articles)
        report = {
            "agent": AGENT_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gate_mode": os.environ.get("NEWS_RISK_GATE", "alert").lower(),
            **scored,
        }
        self.last_report = report

        try:
            REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
        except Exception:
            logger.exception("news_risk_report_write_failed")

        if scored["level"] >= 2:
            self._maybe_email(report)
        logger.info("news_risk_run", level=scored["level_name"],
                    flagged=scored["n_flagged"], scanned=scored["n_scanned"])
        return report

    # ── Alerting (same SMTP pattern as watchdog/integrity) ──────────────────

    def _maybe_email(self, report: dict[str, Any]) -> None:
        fingerprint = report["level_name"] + "|" + "|".join(
            sorted(h["title"] for h in report["headlines"][:3]))
        now = datetime.now(timezone.utc)
        if (fingerprint == self._last_email_fingerprint
                and self._last_email_at is not None
                and now - self._last_email_at < timedelta(hours=EMAIL_DEDUPE_HOURS)):
            return
        gate = report["gate_mode"]
        lines = [
            f"🚨 ESCALATION — {AGENT_NAME} — {report['timestamp']}",
            f"Macro risk level: {report['level_name'].upper()} "
            f"({report['n_flagged']} flagged of {report['n_scanned']} scanned)",
            "",
        ]
        for h in report["headlines"][:6]:
            lines.append(f"[L{h['level']}] {h['title']} — {h['source']}")
            lines.append(f"      matched: {', '.join(h['matched'])}")
        lines += [
            "",
            f"Entry gate: {'BLOCKING new entries' if gate == 'block' else 'alert-only (set NEWS_RISK_GATE=block to gate entries)'}",
            "Positions are NEVER auto-closed on news — review and, if needed,",
            "halt via the circuit breaker before the open.",
            "",
            "Report: https://stockbot-production-cbde.up.railway.app/news-risk",
        ]
        try:
            self._send_email(
                subject=f"[StockBot News Risk] {report['level_name'].upper()}: "
                        f"{report['headlines'][0]['title'][:80] if report['headlines'] else ''}",
                body="\n".join(lines),
            )
            self._last_email_fingerprint = fingerprint
            self._last_email_at = now
        except Exception:
            logger.exception("news_risk_email_failed")

    def _send_email(self, subject: str, body: str) -> None:
        from src.config import get_settings
        s = get_settings()
        if not s.smtp_host or not s.smtp_user or not s.smtp_password:
            logger.warning("news_risk_smtp_not_configured")
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
