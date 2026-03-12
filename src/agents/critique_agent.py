"""Claude-powered strategy critique agent.

Runs daily at 17:00 ET (30 min after profit_agent). Reads performance
reports and recent trades, calls Claude claude-sonnet-4-6 to analyze results
and propose specific, actionable strategy improvements.

Output: config/staging/critique_suggestions.json
Escalates if: Sharpe < 1.0 over rolling 2 weeks (surfaces to dashboard)

The agent follows CLAUDE.md rules:
  - Never modifies live config directly — writes to staging only
  - All suggestions require human review before application
  - Reports PnL with Sharpe, drawdown, win rate, profit factor
"""

from __future__ import annotations

import json
import logging
import structlog
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = structlog.get_logger(__name__)

STAGING_DIR = Path("config/staging")
REPORTS_DIR = Path("reports")
OUTPUT_FILE = STAGING_DIR / "critique_suggestions.json"
ESCALATION_SHARPE_THRESHOLD = 1.0


class CritiqueAgent:
    """Daily Claude-powered strategy critique and improvement agent."""

    def __init__(self, session_factory: Any | None = None) -> None:
        self._sf = session_factory

    async def run(self) -> dict[str, Any]:
        """Run one critique cycle. Returns the suggestions dict."""
        logger.info("critique_agent_started")
        STAGING_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Gather metrics from reports + DB
        metrics = await self._gather_metrics()

        # 2. Call Claude API
        suggestions = await self._call_claude(metrics)

        # 3. Write to staging
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metrics_snapshot": metrics,
            "suggestions": suggestions,
            "status": "pending_review",
        }
        OUTPUT_FILE.write_text(json.dumps(output, indent=2))
        logger.info("critique_agent_done", n_suggestions=len(suggestions.get("proposals", [])))

        # 4. Escalate if Sharpe is below gate
        sharpe = metrics.get("sharpe_2w", 0.0)
        if sharpe < ESCALATION_SHARPE_THRESHOLD:
            logger.warning(
                "critique_agent_escalation",
                reason="sharpe_below_gate",
                sharpe_2w=round(sharpe, 3),
                threshold=ESCALATION_SHARPE_THRESHOLD,
            )

        return output

    # ── Metrics gathering ──────────────────────────────────────────────────────

    async def _gather_metrics(self) -> dict[str, Any]:
        """Collect performance data from reports/ and trade DB."""
        metrics: dict[str, Any] = {}

        # Latest risk report
        risk_file = REPORTS_DIR / "risk" / "live.json"
        if risk_file.exists():
            try:
                metrics["risk"] = json.loads(risk_file.read_text())
            except Exception:
                pass

        # Latest profit suggestions (PnL attribution from profit_agent)
        profit_file = STAGING_DIR / "profit_suggestions.json"
        if profit_file.exists():
            try:
                metrics["profit"] = json.loads(profit_file.read_text())
            except Exception:
                pass

        # Latest FFSA drift report (feature importance)
        drift_files = sorted(REPORTS_DIR.glob("drift/ffsa_*.json"), reverse=True)
        if drift_files:
            try:
                drift = json.loads(drift_files[0].read_text())
                metrics["ffsa_ic"] = drift.get("ic_score", 0.0)
                metrics["top_features"] = drift.get("selected_features", [])[:10]
            except Exception:
                pass

        # Recent trades from DB
        if self._sf is not None:
            metrics["recent_trades"] = await self._fetch_recent_trades()

        # Compute rolling Sharpe from trade PnLs
        trades = metrics.get("recent_trades", [])
        metrics["sharpe_2w"] = self._compute_trade_sharpe(trades)
        metrics["win_rate"] = self._compute_win_rate(trades)
        metrics["profit_factor"] = self._compute_profit_factor(trades)
        metrics["n_trades_2w"] = len(trades)

        return metrics

    async def _fetch_recent_trades(self) -> list[dict[str, Any]]:
        """Fetch trades from last 14 days."""
        from sqlalchemy import select

        from src.data.db import Trade

        cutoff = datetime.now(timezone.utc) - timedelta(days=14)
        try:
            async with self._sf() as session:
                result = await session.execute(
                    select(Trade)
                    .where(Trade.entry_time >= cutoff)
                    .where(Trade.pnl.isnot(None))
                    .order_by(Trade.entry_time.desc())
                    .limit(200)
                )
                rows = result.scalars().all()
            return [
                {
                    "ticker": r.ticker,
                    "side": r.side,
                    "entry_time": r.entry_time.isoformat() if r.entry_time else None,
                    "pnl": round(float(r.pnl or 0), 4),
                    "pnl_pct": round(float(r.pnl_pct or 0), 4),
                    "exit_reason": r.exit_reason,
                    "ensemble_signal": round(float(r.ensemble_signal or 0), 4),
                    "transformer_conf": round(float(r.transformer_confidence or 0), 4),
                    "tcn_conf": round(float(r.tcn_confidence or 0), 4),
                }
                for r in rows
            ]
        except Exception as exc:
            logger.warning("critique_trade_fetch_failed", error=str(exc))
            return []

    # ── Stats helpers ──────────────────────────────────────────────────────────

    def _compute_trade_sharpe(self, trades: list[dict]) -> float:
        if len(trades) < 5:
            return 0.0
        import math
        pnls = [t["pnl_pct"] for t in trades if t.get("pnl_pct") is not None]
        if not pnls:
            return 0.0
        mean = sum(pnls) / len(pnls)
        std = (sum((x - mean) ** 2 for x in pnls) / len(pnls)) ** 0.5
        if std < 1e-9:
            return 0.0
        return round(mean / std * math.sqrt(252 * 390), 3)   # annualized

    def _compute_win_rate(self, trades: list[dict]) -> float:
        pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
        if not pnls:
            return 0.0
        return round(sum(1 for p in pnls if p > 0) / len(pnls), 3)

    def _compute_profit_factor(self, trades: list[dict]) -> float:
        pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        return round(gross_profit / gross_loss, 3) if gross_loss > 0 else 0.0

    # ── Claude API call ────────────────────────────────────────────────────────

    async def _call_claude(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Call Claude claude-sonnet-4-6 with the metrics and return structured suggestions."""
        try:
            import anthropic
        except ImportError:
            logger.warning("critique_agent_anthropic_not_installed")
            return {"proposals": [], "error": "anthropic package not installed"}

        from src.config import get_settings
        settings = get_settings()
        if not settings.anthropic_api_key:
            logger.warning("critique_agent_no_api_key")
            return {"proposals": [], "error": "ANTHROPIC_API_KEY not set"}

        # Build a concise metrics summary for the prompt
        recent_trades = metrics.get("recent_trades", [])
        trade_summary = ""
        if recent_trades:
            by_ticker: dict[str, list] = {}
            for t in recent_trades:
                by_ticker.setdefault(t["ticker"], []).append(t["pnl_pct"])
            ticker_lines = []
            for tk, pnls in list(by_ticker.items())[:10]:
                avg = sum(pnls) / len(pnls)
                ticker_lines.append(f"  {tk}: {len(pnls)} trades, avg_pnl={avg:.3%}")
            trade_summary = "\n".join(ticker_lines)

        prompt = f"""You are analyzing a live paper-trading bot that uses:
- Transformer (d_model=128, 5 layers) + TCN (dual-stream 1m/5m) + FinBERT sentiment
- Ensemble weights: Transformer 45%, TCN 35%, Sentiment 20%
- Entry threshold: 0.40 (trending), 0.55 (choppy/high-vol)
- RL agent (PPO, 500k steps, Sharpe=-9.7, not production-ready)
- Position sizing: vol-scaled, max 25% per position, 80% portfolio heat cap

## Performance snapshot (last 14 days)
- Sharpe (2-week): {metrics.get('sharpe_2w', 'N/A')}
- Win rate: {metrics.get('win_rate', 'N/A')}
- Profit factor: {metrics.get('profit_factor', 'N/A')}
- Trade count: {metrics.get('n_trades_2w', 'N/A')}
- FFSA IC score: {metrics.get('ffsa_ic', 'N/A')}
- Top features: {', '.join(metrics.get('top_features', [])[:5])}

## Per-ticker breakdown
{trade_summary or '  No trades recorded yet'}

## Current risk state
{json.dumps(metrics.get('risk', {}), indent=2)[:800]}

## Task
Analyze the above data and return a JSON object with this exact structure:
{{
  "summary": "2-3 sentence assessment of current performance",
  "regime_observations": "What market conditions have we been trading in?",
  "proposals": [
    {{
      "id": "p1",
      "type": "ensemble_weights | threshold | position_sizing | new_feature | model_retraining",
      "description": "Specific change to make",
      "rationale": "Why this change, citing the metrics above",
      "expected_impact": "Expected change in Sharpe/win-rate",
      "priority": "high | medium | low",
      "staging_change": {{
        "parameter": "the config key to change",
        "current_value": "current value",
        "proposed_value": "new value"
      }}
    }}
  ],
  "escalate": true/false,
  "escalation_reason": "Only if escalate=true"
}}

Be specific and cite the actual numbers. Limit to 3-5 proposals. Only suggest changes backed by the data."""

        try:
            loop = __import__("asyncio").get_event_loop()
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

            def _call() -> str:
                msg = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text

            response_text = await loop.run_in_executor(None, _call)

            # Extract JSON from response (Claude may wrap in markdown)
            import re
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"proposals": [], "raw_response": response_text[:500]}

        except Exception as exc:
            logger.error("critique_agent_claude_call_failed", error=str(exc))
            return {"proposals": [], "error": str(exc)}
