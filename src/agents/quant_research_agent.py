"""Quant Research Agent — automated strategy analysis and improvement.

Runs on two schedules:
  1. DAILY (17:30 ET, after Profit Agent): Performance analysis, regime check
  2. WEEKLY (Saturday 10:00 ET): Deep research report with hypothesis testing

Scope:
  - Analyze trade performance, signal quality, regime conditions
  - Run null-signal validation (H2)
  - Compute Kelly-optimal sizing (H4)
  - Generate research reports to reports/research/
  - Propose parameter changes to config/staging/research_proposals.json

Must NOT:
  - Modify config/live.yaml directly
  - Deploy new models without human approval
  - Disable circuit breakers or risk controls

Escalate if:
  - Sharpe < -1.0 over rolling 1-week window
  - Win rate < 30% over 50+ trades
  - Live IC < 0.05 for 5 consecutive days
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from sqlalchemy import select, func

from src.data.db import Trade, FeatureMatrix, get_session_factory

logger = structlog.get_logger(__name__)

REPORT_DIR = Path("reports/research")
STAGING_PATH = Path("config/staging/research_proposals.json")
AGENT_NAME = "Quant Research Agent"


class QuantResearchAgent:
    """Automated quant research: performance forensics + strategy proposals."""

    def __init__(self, signal_loop: Any = None, retrain_agent: Any = None) -> None:
        self._signal_loop = signal_loop
        self._retrain_agent = retrain_agent
        self._sf = get_session_factory()

    # ── Main entry points ─────────────────────────────────────────────────────

    async def run_daily(self) -> dict[str, Any]:
        """Daily analysis — runs after market close.

        Produces: reports/research/daily/YYYY-MM-DD.json
        """
        start = datetime.now(timezone.utc)
        logger.info("quant_research_daily_start")

        trades = await self._fetch_trades(days=7)
        if len(trades) < 5:
            logger.info("quant_research_skip_few_trades", n=len(trades))
            return {"note": "Not enough trades for analysis"}

        report = {
            "agent": AGENT_NAME,
            "type": "daily",
            "timestamp": start.isoformat(),
            "n_trades": len(trades),
            "performance": self._compute_performance(trades),
            "signal_quality": self._analyze_signal_quality(trades),
            "exit_analysis": self._analyze_exits(trades),
            "regime_summary": self._regime_summary(trades),
            "escalations": [],
        }

        # Check escalation triggers
        perf = report["performance"]
        if perf.get("sharpe_1w", 0) < -1.0:
            report["escalations"].append({
                "level": "URGENT",
                "trigger": f"Sharpe={perf['sharpe_1w']:.2f} < -1.0",
                "recommendation": "Review signal model; consider halting new entries",
            })
        if perf.get("win_rate", 1.0) < 0.30 and len(trades) >= 50:
            report["escalations"].append({
                "level": "WARNING",
                "trigger": f"Win rate={perf['win_rate']:.1%} < 30% on {len(trades)} trades",
                "recommendation": "Run H2 null-signal test to validate model edge",
            })

        report["needs_review"] = len(report["escalations"]) > 0

        # Save report
        daily_dir = REPORT_DIR / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        report_path = daily_dir / f"{start.strftime('%Y-%m-%d')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("quant_research_daily_done", path=str(report_path),
                     escalations=len(report["escalations"]))
        return report

    async def run_weekly(self) -> dict[str, Any]:
        """Weekly deep research — runs Saturday morning.

        Produces: reports/research/weekly/YYYY-WW.json
        Includes: null-signal test, Kelly sizing analysis, parameter proposals
        """
        start = datetime.now(timezone.utc)
        logger.info("quant_research_weekly_start")

        trades = await self._fetch_trades(days=14)
        if len(trades) < 20:
            logger.info("quant_research_weekly_skip", n=len(trades))
            return {"note": "Not enough trades for weekly analysis"}

        report = {
            "agent": AGENT_NAME,
            "type": "weekly",
            "timestamp": start.isoformat(),
            "n_trades": len(trades),
            "performance": self._compute_performance(trades),
            "signal_quality": self._analyze_signal_quality(trades),
            "exit_analysis": self._analyze_exits(trades),
            "null_signal_test": self._null_signal_test(trades),
            "kelly_analysis": self._kelly_analysis(trades),
            "parameter_proposals": self._generate_proposals(trades),
            "escalations": [],
        }

        # Save report
        weekly_dir = REPORT_DIR / "weekly"
        weekly_dir.mkdir(parents=True, exist_ok=True)
        week_label = start.strftime("%Y-W%W")
        report_path = weekly_dir / f"{week_label}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Write proposals to staging
        if report["parameter_proposals"]:
            STAGING_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(STAGING_PATH, "w") as f:
                json.dump({
                    "agent": AGENT_NAME,
                    "timestamp": start.isoformat(),
                    "proposals": report["parameter_proposals"],
                    "needs_review": True,
                }, f, indent=2, default=str)

        # Trigger retrain if performance is poor and retrain agent is wired
        if self._retrain_agent is not None:
            should_retrain = (
                not report.get("null_signal_test", {}).get("signal_has_edge", True)
                or report.get("performance", {}).get("sharpe_1w", 0) < 0
                or report.get("performance", {}).get("win_rate", 1.0) < 0.45
            )
            if should_retrain:
                logger.info("quant_research_triggering_retrain",
                            reason="poor weekly performance detected")
                try:
                    retrain_report = await self._retrain_agent.run(force_neural_nets=True)
                    report["retrain_triggered"] = True
                    report["retrain_result"] = retrain_report.get("status", "unknown")
                except Exception as exc:
                    logger.warning("quant_research_retrain_failed", error=str(exc))
                    report["retrain_triggered"] = True
                    report["retrain_result"] = f"failed: {exc}"

        logger.info("quant_research_weekly_done", path=str(report_path))
        return report

    # ── Data fetching ─────────────────────────────────────────────────────────

    async def _fetch_trades(self, days: int = 14) -> list[dict[str, Any]]:
        """Fetch closed trades from the last N days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        try:
            async with self._sf() as session:
                result = await session.execute(
                    select(Trade)
                    .where(Trade.entry_time >= cutoff, Trade.exit_time.isnot(None))
                    .order_by(Trade.exit_time.desc())
                )
                rows = result.scalars().all()
                return [
                    {
                        "ticker": r.ticker,
                        "side": r.side,
                        "entry_time": r.entry_time.isoformat() if r.entry_time else None,
                        "exit_time": r.exit_time.isoformat() if r.exit_time else None,
                        "entry_price": float(r.entry_price or 0),
                        "exit_price": float(r.exit_price or 0),
                        "shares": float(r.shares or 0),
                        "pnl": float(r.pnl or 0),
                        "pnl_pct": float(r.pnl_pct or 0),
                        "exit_reason": r.exit_reason or "unknown",
                        "ensemble_signal": float(r.ensemble_signal or 0),
                        "transformer_confidence": float(r.transformer_confidence or 0),
                        "tcn_confidence": float(r.tcn_confidence or 0),
                        "sentiment_index": float(r.sentiment_index or 0),
                    }
                    for r in rows
                ]
        except Exception as exc:
            logger.warning("quant_research_fetch_failed", error=str(exc))
            return []

    # ── Performance metrics ───────────────────────────────────────────────────

    def _compute_performance(self, trades: list[dict]) -> dict[str, Any]:
        """Core performance metrics."""
        if not trades:
            return {}

        pnls = [t["pnl"] for t in trades]
        pnl_pcts = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001

        # Sharpe (annualized from daily returns approximation)
        if len(pnl_pcts) > 1:
            mean_ret = statistics.mean(pnl_pcts)
            std_ret = statistics.stdev(pnl_pcts)
            sharpe_1w = (mean_ret / max(std_ret, 1e-9)) * (252 ** 0.5)
        else:
            sharpe_1w = 0.0

        # Sortino (only penalize downside)
        downside = [p for p in pnl_pcts if p < 0]
        if len(downside) > 1:
            downside_std = statistics.stdev(downside)
            sortino = (statistics.mean(pnl_pcts) / max(downside_std, 1e-9)) * (252 ** 0.5)
        else:
            sortino = 0.0

        # Max drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cumulative += p
            peak = max(peak, cumulative)
            max_dd = max(max_dd, peak - cumulative)

        return {
            "total_pnl": round(sum(pnls), 2),
            "n_trades": len(trades),
            "win_rate": round(len(wins) / max(len(trades), 1), 4),
            "profit_factor": round(gross_profit / max(gross_loss, 0.01), 3),
            "avg_win": round(statistics.mean(wins), 2) if wins else 0,
            "avg_loss": round(statistics.mean(losses), 2) if losses else 0,
            "win_loss_ratio": round(
                (statistics.mean(wins) if wins else 0) /
                max(abs(statistics.mean(losses)) if losses else 1, 0.01), 3
            ),
            "sharpe_1w": round(sharpe_1w, 3),
            "sortino": round(sortino, 3),
            "max_drawdown": round(max_dd, 2),
            "pnl_mean": round(statistics.mean(pnls), 4),
            "pnl_std": round(statistics.stdev(pnls), 4) if len(pnls) > 1 else 0,
        }

    # ── Signal quality ────────────────────────────────────────────────────────

    def _analyze_signal_quality(self, trades: list[dict]) -> dict[str, Any]:
        """Analyze if signal strength predicts outcomes."""
        if not trades:
            return {}

        # Bin trades by signal strength
        bins = {
            "weak (0.0-0.2)": [],
            "moderate (0.2-0.4)": [],
            "strong (0.4-0.6)": [],
            "very_strong (0.6+)": [],
        }
        for t in trades:
            s = abs(t["ensemble_signal"])
            if s < 0.2:
                bins["weak (0.0-0.2)"].append(t)
            elif s < 0.4:
                bins["moderate (0.2-0.4)"].append(t)
            elif s < 0.6:
                bins["strong (0.4-0.6)"].append(t)
            else:
                bins["very_strong (0.6+)"].append(t)

        quality = {}
        for label, subset in bins.items():
            if not subset:
                continue
            pnls = [t["pnl"] for t in subset]
            wins = sum(1 for p in pnls if p > 0)
            quality[label] = {
                "n": len(subset),
                "win_rate": round(wins / len(subset), 3),
                "avg_pnl": round(statistics.mean(pnls), 4),
                "total_pnl": round(sum(pnls), 2),
            }

        # IC proxy: correlation between signal and pnl
        signals = [t["ensemble_signal"] for t in trades]
        pnls = [t["pnl"] for t in trades]
        if len(signals) > 5:
            ic = float(np.corrcoef(signals, pnls)[0, 1])
            if np.isnan(ic):
                ic = 0.0
        else:
            ic = 0.0

        return {
            "live_ic": round(ic, 4),
            "signal_bins": quality,
            "signal_has_edge": ic > 0.05,
        }

    # ── Exit analysis ─────────────────────────────────────────────────────────

    def _analyze_exits(self, trades: list[dict]) -> dict[str, Any]:
        """Break down performance by exit reason."""
        if not trades:
            return {}

        by_reason: dict[str, list[float]] = defaultdict(list)
        for t in trades:
            by_reason[t["exit_reason"]].append(t["pnl"])

        result = {}
        for reason, pnls in by_reason.items():
            wins = sum(1 for p in pnls if p > 0)
            result[reason] = {
                "count": len(pnls),
                "pct_of_trades": round(len(pnls) / max(len(trades), 1), 3),
                "win_rate": round(wins / max(len(pnls), 1), 3),
                "avg_pnl": round(statistics.mean(pnls), 4),
                "total_pnl": round(sum(pnls), 2),
            }

        return result

    # ── Regime summary ────────────────────────────────────────────────────────

    def _regime_summary(self, trades: list[dict]) -> dict[str, Any]:
        """Summarize what regimes the system traded in."""
        tickers = Counter(t["ticker"] for t in trades)
        return {
            "ticker_distribution": dict(tickers.most_common(10)),
            "n_unique_tickers": len(tickers),
        }

    # ── H2: Null signal test ──────────────────────────────────────────────────

    def _null_signal_test(self, trades: list[dict]) -> dict[str, Any]:
        """Compare real signal performance vs random-shuffled baseline.

        If performance is similar, the signal has no live edge.
        """
        if len(trades) < 20:
            return {"note": "Not enough trades for null test"}

        real_pnls = [t["pnl"] for t in trades]
        real_mean = statistics.mean(real_pnls)
        real_wr = sum(1 for p in real_pnls if p > 0) / len(real_pnls)

        # Monte Carlo: shuffle PnLs 1000x and measure how often random beats real
        n_better = 0
        n_simulations = 1000
        rng = np.random.default_rng(42)
        for _ in range(n_simulations):
            shuffled = rng.permutation(real_pnls)
            # Simulate: if signal was random, half would be flipped
            # Use actual PnLs but randomly flip sign on 50%
            mask = rng.random(len(real_pnls)) > 0.5
            sim_pnls = [p if m else -p for p, m in zip(real_pnls, mask)]
            if statistics.mean(sim_pnls) >= real_mean:
                n_better += 1

        p_value = n_better / n_simulations

        return {
            "real_mean_pnl": round(real_mean, 4),
            "real_win_rate": round(real_wr, 4),
            "null_p_value": round(p_value, 4),
            "signal_has_edge": p_value < 0.05,
            "interpretation": (
                "Signal has statistically significant edge (p<0.05)"
                if p_value < 0.05
                else "Signal performance is NOT distinguishable from random"
            ),
        }

    # ── H4: Kelly analysis ────────────────────────────────────────────────────

    def _kelly_analysis(self, trades: list[dict]) -> dict[str, Any]:
        """Compute Kelly-optimal position sizing from trade history."""
        if len(trades) < 20:
            return {"note": "Not enough trades for Kelly analysis"}

        pnl_pcts = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnl_pcts if p > 0]
        losses = [p for p in pnl_pcts if p < 0]

        if not wins or not losses:
            return {"kelly_fraction": 0, "note": "No wins or no losses"}

        p = len(wins) / len(pnl_pcts)  # win probability
        avg_win = statistics.mean(wins)
        avg_loss = abs(statistics.mean(losses))
        b = avg_win / max(avg_loss, 1e-9)  # win/loss ratio

        # Kelly: f* = (p*b - q) / b  where q = 1-p
        q = 1 - p
        kelly_full = (p * b - q) / max(b, 1e-9)
        kelly_half = max(0, kelly_full / 2)  # half-Kelly for safety

        return {
            "win_probability": round(p, 4),
            "win_loss_ratio": round(b, 4),
            "kelly_full": round(kelly_full, 4),
            "kelly_half": round(kelly_half, 4),
            "recommendation": (
                f"Use {kelly_half:.1%} position size (half-Kelly)"
                if kelly_full > 0
                else "Kelly says DO NOT TRADE — negative expected value"
            ),
            "expected_value_per_trade": round(p * avg_win - q * avg_loss, 6),
        }

    # ── Parameter proposals ───────────────────────────────────────────────────

    def _generate_proposals(self, trades: list[dict]) -> list[dict[str, Any]]:
        """Generate parameter change proposals based on analysis."""
        proposals = []
        perf = self._compute_performance(trades)
        exits = self._analyze_exits(trades)
        kelly = self._kelly_analysis(trades)
        signal = self._analyze_signal_quality(trades)

        # Proposal: Adjust sizing if Kelly says negative EV
        if kelly.get("kelly_full", 0) <= 0:
            proposals.append({
                "id": "reduce_sizing",
                "parameter": "SIZING_ACTION_PCTS",
                "current": "tiny=2%, small=5%, medium=10%, large=20%",
                "proposed": "tiny=1%, small=2%, medium=3%, large=5%",
                "justification": (
                    f"Kelly fraction is {kelly.get('kelly_full', 0):.4f} (negative). "
                    f"Expected value per trade is {kelly.get('expected_value_per_trade', 0):.6f}. "
                    f"Reduce sizing until signal edge is validated."
                ),
                "priority": "P0",
            })

        # Proposal: Tighten entry if signal has no edge
        if not signal.get("signal_has_edge", True):
            proposals.append({
                "id": "raise_entry_threshold",
                "parameter": "SIZING_COST_THRESHOLD",
                "current": "0.0015",
                "proposed": "0.003",
                "justification": (
                    f"Live IC = {signal.get('live_ic', 0):.4f}. "
                    f"Signal is not distinguishable from noise. "
                    f"Raise entry bar to reduce trade count and improve selectivity."
                ),
                "priority": "P1",
            })

        # Proposal: Fix max_hold if most trades time out
        max_hold_exit = exits.get("max_hold", {})
        if max_hold_exit.get("pct_of_trades", 0) > 0.60:
            proposals.append({
                "id": "reduce_max_hold",
                "parameter": "SIZING_MAX_HOLD_BARS",
                "current": "15",
                "proposed": "10",
                "justification": (
                    f"{max_hold_exit['pct_of_trades']:.0%} of trades exit at max_hold. "
                    f"Signal alpha decays before hold period ends. "
                    f"Shorten to capture the edge window."
                ),
                "priority": "P1",
            })

        return proposals
