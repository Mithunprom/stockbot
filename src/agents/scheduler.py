"""APScheduler setup for all sub-agents.

Activation order (per CLAUDE.md):
  Day 1:  Risk Agent (every 15 min) + Latency Agent (hourly)
  Week 4: Profit Agent (daily) + Model Drift Agent (weekly)
  Month 3+: Opportunity Agent (weekly)
"""

from __future__ import annotations

import asyncio
import logging
import structlog
from datetime import datetime, timezone
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = structlog.get_logger(__name__)


def create_scheduler(
    risk_agent: Any,
    latency_agent: Any,
    profit_agent: Any | None = None,
    screener_agent: Any | None = None,
    critique_agent: Any | None = None,
    retrain_agent: Any | None = None,
    quant_research_agent: Any | None = None,
    mode: str = "paper",
) -> AsyncIOScheduler:
    """Build and configure the APScheduler instance.

    Args:
        risk_agent: RiskAgent instance.
        latency_agent: LatencyAgent instance.
        profit_agent: ProfitAgent instance (optional — enable after Week 4).
        mode: "paper" or "live".

    Returns:
        Configured AsyncIOScheduler (not yet started).
    """
    scheduler = AsyncIOScheduler(timezone="America/New_York")

    # ── Risk Agent: every 15 minutes during market hours ─────────────────────
    scheduler.add_job(
        risk_agent.run,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour="9-16",
            minute="0,15,30,45",
            timezone="America/New_York",
        ),
        id="risk_agent",
        name="Risk Agent (15 min)",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=60,
    )

    # ── Latency Agent: hourly during market hours ─────────────────────────────
    scheduler.add_job(
        latency_agent.run,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour="9-16",
            minute="5",
            timezone="America/New_York",
        ),
        id="latency_agent",
        name="Latency Agent (hourly)",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=300,
    )

    # ── Profit Agent: daily after market close (enable Week 4+) ─────────────
    if profit_agent is not None:
        scheduler.add_job(
            lambda: asyncio.create_task(profit_agent.run(mode=mode)),
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=16,
                minute=30,
                timezone="America/New_York",
            ),
            id="profit_agent",
            name="Profit Agent (daily)",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=600,
        )

    # ── Screener Agent: nightly after market close ────────────────────────────
    if screener_agent is not None:
        scheduler.add_job(
            lambda: asyncio.create_task(screener_agent.run()),
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=18,
                minute=0,
                timezone="America/New_York",
            ),
            id="screener_agent",
            name="Screener Agent (nightly)",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=3600,
        )

    # ── Retrain Agent: daily at 08:00 ET (before market open) ─────────────────
    if retrain_agent is not None:
        scheduler.add_job(
            lambda: asyncio.create_task(retrain_agent.run()),
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=8,
                minute=0,
                timezone="America/New_York",
            ),
            id="retrain_agent",
            name="Retrain Agent (daily pre-market)",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=3600,
        )

    # ── Critique Agent: daily at 17:00 ET (30 min after profit_agent) ─────────
    if critique_agent is not None:
        scheduler.add_job(
            lambda: asyncio.create_task(critique_agent.run()),
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=17,
                minute=0,
                timezone="America/New_York",
            ),
            id="critique_agent",
            name="Critique Agent (daily)",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=1800,
        )

    # ── Quant Research Agent: daily 17:30 ET + weekly Saturday 10:00 ET ───────
    if quant_research_agent is not None:
        # Daily analysis (after profit agent)
        scheduler.add_job(
            lambda: asyncio.create_task(quant_research_agent.run_daily()),
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=17,
                minute=30,
                timezone="America/New_York",
            ),
            id="quant_research_daily",
            name="Quant Research Agent (daily)",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=1800,
        )
        # Weekly deep research (Saturday morning)
        scheduler.add_job(
            lambda: asyncio.create_task(quant_research_agent.run_weekly()),
            trigger=CronTrigger(
                day_of_week="sat",
                hour=10,
                minute=0,
                timezone="America/New_York",
            ),
            id="quant_research_weekly",
            name="Quant Research Agent (weekly)",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=3600,
        )

    logger.info(
        "scheduler_configured",
        jobs=[j.id for j in scheduler.get_jobs()],
    )
    return scheduler
