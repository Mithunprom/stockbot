"""Forecast Email Agent — daily pre-market single-ticker forecast + email.

Scope: Generate a next-session forecast for each configured ticker, persist it
to reports/forecasts/, and email it to the configured recipients.
Must NOT: place trades, modify config, or touch model weights. Read-only w.r.t.
the trading system (it only reads the live /admin/ic report and public prices).
Output: reports/forecasts/{TICKER}_{DATE}.json + one email per run.

Delivery is via SMTP, gated on settings. If SMTP creds or recipients are absent
the agent still writes the JSON forecast and logs a skip — it never raises into
the scheduler.
"""
from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage
from typing import Any

import structlog

from src.analysis.forecast import (
    build_forecast,
    render_text,
    save_forecast,
    subject_line,
)
from src.config import get_settings

logger = structlog.get_logger(__name__)

AGENT_NAME = "Forecast Email Agent"


class ForecastEmailAgent:
    """Builds daily forecasts for configured tickers and emails them."""

    def __init__(self, tickers: list[str] | None = None) -> None:
        settings = get_settings()
        self._tickers = tickers or [
            t.strip().upper()
            for t in settings.forecast_tickers.split(",")
            if t.strip()
        ]

    # ── SMTP ──────────────────────────────────────────────────────────────────

    def _smtp_ready(self) -> tuple[bool, str]:
        s = get_settings()
        if not s.smtp_host or not s.smtp_user or not s.smtp_password:
            return False, "SMTP creds not configured (smtp_host/user/password)"
        if not s.forecast_email_to.strip():
            return False, "no forecast_email_to recipients configured"
        return True, ""

    def _send_email(self, subject: str, body: str) -> None:
        s = get_settings()
        recipients = [r.strip() for r in s.forecast_email_to.split(",") if r.strip()]
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = s.forecast_email_from or s.smtp_user
        msg["To"] = ", ".join(recipients)
        msg.set_content(body)

        context = ssl.create_default_context()
        # Port 465 → implicit TLS; else STARTTLS (587).
        if s.smtp_port == 465:
            with smtplib.SMTP_SSL(s.smtp_host, s.smtp_port, context=context, timeout=30) as srv:
                srv.login(s.smtp_user, s.smtp_password)
                srv.send_message(msg)
        else:
            with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=30) as srv:
                srv.starttls(context=context)
                srv.login(s.smtp_user, s.smtp_password)
                srv.send_message(msg)

    # ── Run ─────────────────────────────────────────────────────────────────

    async def run(self) -> dict[str, Any]:
        """Generate + email a forecast for each configured ticker.

        Returns a summary dict. Never raises — a failure on one ticker is logged
        and the others still run.
        """
        logger.info("forecast_agent_start", tickers=self._tickers)
        ready, skip_reason = self._smtp_ready()
        results: list[dict[str, Any]] = []

        for ticker in self._tickers:
            entry: dict[str, Any] = {"ticker": ticker}
            try:
                fc = build_forecast(ticker)
                path = save_forecast(fc)
                entry.update(
                    target_session=fc["target_session"],
                    direction=fc["direction"],
                    prob_up=fc["prob_up"],
                    saved=str(path),
                )
                if ready:
                    try:
                        self._send_email(subject_line(fc), render_text(fc))
                        entry["emailed"] = True
                        logger.info(
                            "forecast_emailed",
                            ticker=ticker,
                            target=fc["target_session"],
                            direction=fc["direction"],
                        )
                    except Exception as exc:
                        entry["emailed"] = False
                        entry["email_error"] = str(exc)
                        logger.warning("forecast_email_failed", ticker=ticker, error=str(exc))
                else:
                    entry["emailed"] = False
                    entry["email_skipped"] = skip_reason
                    logger.info("forecast_email_skipped", ticker=ticker, reason=skip_reason)
            except Exception as exc:
                entry["error"] = str(exc)
                logger.warning("forecast_build_failed", ticker=ticker, error=str(exc))
            results.append(entry)

        summary = {
            "agent": AGENT_NAME,
            "tickers": self._tickers,
            "smtp_ready": ready,
            "results": results,
        }
        logger.info("forecast_agent_done", **{"n": len(results), "smtp_ready": ready})
        return summary
