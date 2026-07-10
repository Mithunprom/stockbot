"""Profit Agent — daily PnL attribution and ensemble weight proposals.

Scope: Analyze trade PnL, re-weight ensemble signals, suggest reward tweaks.
Must NOT: Apply changes directly to live config. Write to staging/profit_suggestions.json.
Output: Daily PnL attribution report + proposed ensemble weights.
Escalate if: Sharpe drops below 1.0 over any rolling 2-week window.
"""

from __future__ import annotations

import json
import logging
import structlog
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select

from src.data.db import Trade, get_session_factory

logger = structlog.get_logger(__name__)

STAGING_PATH = Path("config/staging/profit_suggestions.json")
AGENT_NAME = "Profit Agent"
SHARPE_ALERT_THRESHOLD = 1.0


class ProfitAgent:
    """Analyzes trade performance and proposes ensemble weight adjustments."""

    async def run(self, mode: str = "paper") -> dict[str, Any]:
        """Run daily PnL attribution cycle."""
        start = datetime.now(timezone.utc)
        logger.info("profit_agent_run", mode=mode, time=start.isoformat())

        trades = await self._fetch_recent_trades(mode, days=14)
        if not trades:
            logger.info("profit_agent_no_trades")
            return {"note": "No trades in lookback window"}

        metrics = self._compute_metrics(trades)
        attribution = self._attribute_pnl(trades)
        proposed_weights = self._propose_weights(attribution, metrics)

        report: dict[str, Any] = {
            "agent": AGENT_NAME,
            "timestamp": start.isoformat(),
            "mode": mode,
            "period_days": 14,
            "n_trades": len(trades),
            "metrics": metrics,
            "pnl_attribution": attribution,
            # "ensemble_weights" is the key read by EnsembleWeights.from_staging().
            # "proposed_weights" kept for human-readable context.
            "ensemble_weights": proposed_weights,
            "proposed_weights": proposed_weights,
            "needs_review": metrics.get("sharpe_2w", 999) < SHARPE_ALERT_THRESHOLD,
        }

        # Write to staging (never to live config directly)
        STAGING_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STAGING_PATH, "w") as f:
            json.dump(report, f, indent=2)

        if report["needs_review"]:
            logger.warning(
                "\n🚨 ESCALATION — Profit Agent — %s\n"
                "Issue: Rolling 2-week Sharpe = %.2f (below 1.0 threshold)\n"
                "Impact: Performance degradation detected\n"
                "Action taken: Proposed weight adjustments saved to %s\n"
                "Required from human: Review and approve weight changes",
                start.isoformat(),
                metrics.get("sharpe_2w", 0),
                str(STAGING_PATH),
            )
        else:
            logger.info(
                "📊 Profit Agent — %s\n"
                "Sharpe (2w): %.2f | Win Rate: %.1f%% | PF: %.2f\n"
                "Action: Proposed weights saved to staging. Needs review: %s",
                start.isoformat(),
                metrics.get("sharpe_2w", 0),
                metrics.get("win_rate", 0) * 100,
                metrics.get("profit_factor", 0),
                "Yes" if report["needs_review"] else "No",
            )

        return report

    async def _fetch_recent_trades(
        self, mode: str, days: int = 14
    ) -> list[dict[str, Any]]:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                select(Trade)
                .where(Trade.mode == mode)
                .where(Trade.entry_time >= since)
                .where(Trade.exit_time.is_not(None))
                .order_by(Trade.entry_time)
            )
            rows = result.scalars().all()
        return [
            {
                "pnl": r.pnl or 0.0,
                "pnl_pct": r.pnl_pct or 0.0,
                "transformer_direction": r.transformer_direction,
                "transformer_confidence": r.transformer_confidence,
                "tcn_direction": r.tcn_direction,
                "tcn_confidence": r.tcn_confidence,
                "sentiment_index": r.sentiment_index,
                "ensemble_signal": r.ensemble_signal,
                "entry_time": r.entry_time,
            }
            for r in rows
        ]

    def _compute_metrics(self, trades: list[dict[str, Any]]) -> dict[str, float]:
        returns = np.array([t["pnl_pct"] for t in trades])
        if len(returns) < 2:
            return {}

        sharpe = float(returns.mean() / (returns.std() + 1e-9) * np.sqrt(252))
        win_rate = float((returns > 0).mean())
        gross_profit = float(returns[returns > 0].sum())
        gross_loss = float(abs(returns[returns < 0].sum()))
        profit_factor = gross_profit / max(gross_loss, 1e-9)

        return {
            "sharpe_2w": round(sharpe, 3),
            "win_rate": round(win_rate, 3),
            "profit_factor": round(profit_factor, 3),
            "total_pnl_pct": round(float(returns.sum()), 4),
            "n_trades": len(trades),
        }

    def _attribute_pnl(self, trades: list[dict[str, Any]]) -> dict[str, float]:
        """Compute correlation of each model's signal with trade PnL."""
        attribution: dict[str, float] = {}
        pnl_arr = np.array([t["pnl_pct"] for t in trades])

        for model, key in [
            ("transformer", "transformer_confidence"),
            ("tcn", "tcn_confidence"),
            ("sentiment", "sentiment_index"),
        ]:
            vals = np.array([t.get(key) or 0.0 for t in trades])
            if len(vals) > 2 and vals.std() > 1e-9:
                corr = float(np.corrcoef(vals, pnl_arr)[0, 1])
            else:
                corr = 0.0
            attribution[f"{model}_ic"] = round(corr, 4)

        return attribution

    def _propose_weights(
        self, attribution: dict[str, float], metrics: dict[str, float]
    ) -> dict[str, float]:
        """Propose new ensemble weights based on IC attribution.

        LGBM is the primary model (IC≈0.21 vs IC≈0 for the neural nets) and
        always retains at least a 60% base. The secondary budget (40%) is
        reallocated proportionally among transformer/TCN/sentiment by their
        measured ICs.

        Dead-weight elimination (H4): when transformer_ic and tcn_ic are both
        near zero — as observed when no checkpoints are loaded and every live
        trade records 0.0 confidence for these models — their combined 20%
        weight is redistributed to LGBM (+15%) and sentiment (+5%), reflecting
        the actual signal sources.

        All four weights are returned explicitly so EnsembleWeights.from_staging
        can load them without relying on per-field defaults.
        """
        LGBM_BASE = 0.60
        SECONDARY_BUDGET = 1.0 - LGBM_BASE

        ic_t = max(attribution.get("transformer_ic", 0.0), 0.0)
        ic_tcn = max(attribution.get("tcn_ic", 0.0), 0.0)
        ic_s = max(attribution.get("sentiment_ic", 0.0), 0.0)

        secondary_ic_total = ic_t + ic_tcn + ic_s

        if ic_t < 0.01 and ic_tcn < 0.01:
            # Dead-weight elimination: Transformer and TCN are not contributing.
            # Redistribute their 20% share: LGBM +15%, sentiment +5%.
            return {
                "lgbm": 0.75,
                "transformer": 0.0,
                "tcn": 0.0,
                "sentiment": 0.25,
            }

        if secondary_ic_total < 0.01:
            # Sentiment also flat — return balanced defaults
            return {
                "lgbm": round(LGBM_BASE, 3),
                "transformer": round(SECONDARY_BUDGET * 0.25, 3),
                "tcn": round(SECONDARY_BUDGET * 0.25, 3),
                "sentiment": round(SECONDARY_BUDGET * 0.50, 3),
            }

        # Proportional reweighting of the secondary budget
        w_t = round(min(ic_t / secondary_ic_total * SECONDARY_BUDGET, 0.20), 3)
        w_tcn = round(min(ic_tcn / secondary_ic_total * SECONDARY_BUDGET, 0.20), 3)
        w_s = round(max(SECONDARY_BUDGET - w_t - w_tcn, 0.0), 3)

        return {
            "lgbm": round(LGBM_BASE, 3),
            "transformer": w_t,
            "tcn": w_tcn,
            "sentiment": w_s,
        }
