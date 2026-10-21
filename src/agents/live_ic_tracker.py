"""Live IC Tracker — monitors prediction accuracy in real time.

Tracks every LightGBM prediction alongside actual forward returns to compute
rolling live IC (Spearman rank correlation) and directional accuracy.

Scope: Record predictions, fill actual returns, compute IC, generate reports.
Must NOT: Modify model weights, trading parameters, or execute trades.
Output: reports/ic/YYYY-MM-DD.json (daily), escalation if IC degrades.
Escalate if: Rolling 7d IC < 0.05 for 5 consecutive daily checks.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog
from scipy.stats import spearmanr
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.data.db import PredictionOutcome, get_session_factory

logger = structlog.get_logger(__name__)

REPORT_DIR = Path("reports/ic")
AGENT_NAME = "Live IC Tracker"

# Escalation threshold: 7d rolling IC must stay above this
IC_ESCALATION_THRESHOLD = 0.05
# Number of consecutive days IC must be below threshold to trigger escalation
IC_ESCALATION_CONSECUTIVE_DAYS = 5

# Forward return horizons (in 1-minute bars)
HORIZON_15M = 15
HORIZON_30M = 30

# Rows per fill batch. The job runs 4×/hour, 10:00–16:00 ET (scheduler.py) = 28
# runs/session. Recording covers 75 tickers on every tick, so a session can
# produce ~29,250 rows (75 × 390); 28 × 1200 = 33,600 clears that with modest
# headroom. The old 500 capped out at 14,000 — permanently behind even with
# perfect ordering.
#
# Not set higher: each row costs TWO price lookups, so a batch is ~2,400
# queries, and db_query p95 has spiked to 5.4s on this database before. Keeping
# per-run work bounded matters more than extra headroom.
#
# Raising this alone would have changed nothing — the old query burned the whole
# budget on the same unfillable rows every run. Ordering and retirement are what
# make the budget productive; this just makes it sufficient.
FILL_BATCH_SIZE = 1200

# A prediction with no ohlcv_1m bar this long after the fact will never get one;
# bars do not arrive a day late. Past this age the row is RETIRED (see
# fill_actual_returns) so it stops consuming the batch. Generous on purpose —
# late backfills within the same session must still be allowed to land.
UNFILLABLE_AFTER_HOURS = 24


class LiveICTracker:
    """Tracks live prediction accuracy by comparing LightGBM predictions to
    actual forward returns.

    Lifecycle:
      1. record_prediction() — called from signal loop after each signal.
      2. fill_actual_returns() — scheduled every 15 min to backfill actuals.
      3. generate_report() — scheduled daily after market close.

    The tracker maintains a count of consecutive days with IC below threshold
    for escalation purposes.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._sf = session_factory or get_session_factory()
        self._consecutive_low_ic_days: int = 0
        self._last_report_date: str | None = None
        # Fill-job health state — read by WatchdogAgent._check_ic_fill_fresh().
        # Updated on every run_fill() invocation (whether or not rows were found).
        self.last_fill_at: datetime | None = None
        self.last_fill_result: dict | None = None
        self._consecutive_zero_fill_runs: int = 0  # incremented when attempted>0 but nothing resolved

    # ── Record predictions ────────────────────────────────────────────────────

    async def record_prediction(
        self,
        ticker: str,
        timestamp: datetime,
        pred_return: float,
        dir_prob: float,
        ensemble_signal: float | None = None,
    ) -> None:
        """Store a prediction for later comparison with actual returns.

        Called from the signal loop every time a signal is computed.
        Actual returns are NULL initially and filled later by fill_actual_returns().

        Args:
            ticker: Ticker symbol (e.g. "AAPL").
            timestamp: Bar timestamp when prediction was made.
            pred_return: LightGBM predicted 15-bar forward return.
            dir_prob: LightGBM direction probability P(up).
            ensemble_signal: Weighted ensemble signal value.
        """
        try:
            async with self._sf() as session:
                outcome = PredictionOutcome(
                    ticker=ticker,
                    timestamp=timestamp,
                    pred_return=pred_return,
                    dir_prob=dir_prob,
                    ensemble_signal=ensemble_signal,
                )
                session.add(outcome)
                await session.commit()
        except Exception as exc:
            logger.warning(
                "ic_tracker_record_failed",
                ticker=ticker,
                error=str(exc),
            )

    # ── Fill actual returns ───────────────────────────────────────────────────

    async def fill_actual_returns(self) -> dict[str, Any]:
        """Backfill actual forward returns for unresolved predictions.

        For each unresolved prediction older than 15 minutes:
          - Look up the actual close price at timestamp + 15 bars from ohlcv_1m
          - Compute actual_return = (close_t+15 - close_t) / close_t
          - Update the row with actual_return_15m and filled_at

        Similarly for 30m returns on predictions older than 30 minutes.

        THIS JOB WAS COMPLETELY DEAD, not merely slow. Measured on 2026-10-14:
        three consecutive runs returned `filled_15m: 0, errors: 0`. The batch
        selected rows with `.limit(500)` and NO ORDER BY, so Postgres returned
        the same oldest unfilled rows every run. `_lookup_forward_return` needs
        an ohlcv_1m bar within ±2 min of both t and t+15 and returns None
        otherwise, which left the row untouched — so it was re-selected forever
        and the entire budget was consumed by rows that can never fill. New
        predictions were never reached, and retention deleted them unmeasured.
        `errors: 0` made it look healthy: nothing threw, so nothing alerted.

        That is why live IC read -0.024 against a backtest +0.173. The number
        was not a 3% sample of the signal, it was a FROZEN set filled at some
        point in the past and since aging out. It was never evidence about the
        model.

        Two changes make the budget productive:
          1. `ORDER BY timestamp DESC` — newest first. Recent predictions are
             the ones whose bars exist, and they are also the ones worth
             measuring.
          2. Rows past `UNFILLABLE_AFTER_HOURS` with no bars are RETIRED by
             stamping `filled_at` while leaving `actual_return_15m` NULL, and
             the query skips anything already stamped. Bars do not arrive a day
             late; without this the same rows are retried until retention
             removes them.

        `filled_at` therefore now means "resolved" — filled OR retired. It is
        written in exactly one other place and never read as a filter, so this
        is safe; `actual_return_15m IS NOT NULL` remains the only test for
        "has a measured outcome", which is what every IC query uses.

        The root cause of the missing bars is upstream and NOT fixed here:
        ohlcv_1m is sparse because ENABLE_ALPACA_WS=false is required for
        deploys to pass healthcheck (Alpaca one-connection limit). This change
        stops the backlog from hiding that; it does not create the bars.

        Returns:
            Summary dict with attempted/filled/retired counts. `attempted` is
            reported so "0 filled of 500 attempted" can never again look
            identical to "0 filled of 0 attempted".
        """
        now = datetime.now(timezone.utc)
        cutoff_15m = now - timedelta(minutes=HORIZON_15M + 1)
        cutoff_30m = now - timedelta(minutes=HORIZON_30M + 1)
        retire_before = now - timedelta(hours=UNFILLABLE_AFTER_HOURS)

        filled_15m = 0
        filled_30m = 0
        retired_15m = 0
        attempted_15m = 0
        errors = 0

        try:
            async with self._sf() as session:
                # ── Fill 15m returns ──────────────────────────────────────────
                result = await session.execute(
                    select(PredictionOutcome).where(
                        PredictionOutcome.actual_return_15m.is_(None),
                        PredictionOutcome.filled_at.is_(None),   # skip retired
                        PredictionOutcome.timestamp <= cutoff_15m,
                    )
                    .order_by(PredictionOutcome.timestamp.desc())
                    .limit(FILL_BATCH_SIZE)
                )
                unfilled_15m = result.scalars().all()
                attempted_15m = len(unfilled_15m)

                for pred in unfilled_15m:
                    try:
                        actual_ret = await self._lookup_forward_return(
                            session, pred.ticker, pred.timestamp, HORIZON_15M
                        )
                        if actual_ret is not None:
                            pred.actual_return_15m = actual_ret
                            pred.filled_at = now
                            filled_15m += 1
                        elif pred.timestamp <= retire_before:
                            # No bar will ever arrive this late. Stamp
                            # `filled_at` to retire the row (actual stays NULL,
                            # so it is excluded from every IC query) and stop
                            # it consuming the batch forever.
                            pred.filled_at = now
                            retired_15m += 1
                    except Exception as exc:
                        errors += 1
                        logger.debug(
                            "ic_fill_15m_error",
                            ticker=pred.ticker,
                            ts=pred.timestamp.isoformat(),
                            error=str(exc),
                        )

                # ── Fill 30m returns ──────────────────────────────────────────
                # Retired rows cannot appear here: they keep actual_return_15m
                # NULL, and this query requires it to be set. No separate
                # retirement arm is needed for the 30m leg.
                result = await session.execute(
                    select(PredictionOutcome).where(
                        PredictionOutcome.actual_return_30m.is_(None),
                        PredictionOutcome.actual_return_15m.isnot(None),
                        PredictionOutcome.timestamp <= cutoff_30m,
                    )
                    .order_by(PredictionOutcome.timestamp.desc())
                    .limit(FILL_BATCH_SIZE)
                )
                unfilled_30m = result.scalars().all()

                for pred in unfilled_30m:
                    try:
                        actual_ret = await self._lookup_forward_return(
                            session, pred.ticker, pred.timestamp, HORIZON_30M
                        )
                        if actual_ret is not None:
                            pred.actual_return_30m = actual_ret
                            filled_30m += 1
                    except Exception as exc:
                        errors += 1
                        logger.debug(
                            "ic_fill_30m_error",
                            ticker=pred.ticker,
                            ts=pred.timestamp.isoformat(),
                            error=str(exc),
                        )

                await session.commit()

        except Exception as exc:
            logger.error("ic_fill_actual_returns_failed", error=str(exc))
            return {"error": str(exc)}

        summary = {
            "attempted_15m": attempted_15m,
            "filled_15m": filled_15m,
            "retired_15m": retired_15m,
            "filled_30m": filled_30m,
            "errors": errors,
            "timestamp": now.isoformat(),
        }
        # A dead pipeline used to be indistinguishable from an idle one: both
        # logged nothing and reported filled=0, errors=0. Warn when a full batch
        # was attempted and nothing could be filled — that is the signature of
        # missing ohlcv_1m bars, and it is the only outward sign of it.
        if attempted_15m > 0 and filled_15m == 0:
            logger.warning(
                "ic_fill_yielded_nothing",
                note="full batch attempted, zero filled — ohlcv_1m likely sparse",
                **summary,
            )
        elif filled_15m > 0 or filled_30m > 0 or retired_15m > 0:
            logger.info("ic_actual_returns_filled", **summary)

        # Update fill-job health state for WatchdogAgent.
        resolved = filled_15m + filled_30m + retired_15m
        if attempted_15m > 0 and resolved == 0:
            self._consecutive_zero_fill_runs += 1
        else:
            self._consecutive_zero_fill_runs = 0
        self.last_fill_at = now
        self.last_fill_result = dict(summary, consecutive_zero_fill_runs=self._consecutive_zero_fill_runs)

        return summary

    async def _lookup_forward_return(
        self,
        session: AsyncSession,
        ticker: str,
        pred_timestamp: datetime,
        horizon_bars: int,
    ) -> float | None:
        """Look up actual forward return from ohlcv_1m.

        Finds the close price at pred_timestamp and at pred_timestamp + horizon_bars
        minutes, then computes: (close_future - close_base) / close_base.

        Returns None if either price is not available.
        """
        # Base price: close at prediction time (nearest bar within 2 minutes)
        base_result = await session.execute(
            text("""
                SELECT close FROM ohlcv_1m
                WHERE ticker = :ticker
                  AND time >= :t_start AND time <= :t_end
                ORDER BY ABS(EXTRACT(EPOCH FROM (time - :t_target)))
                LIMIT 1
            """),
            {
                "ticker": ticker,
                "t_start": pred_timestamp - timedelta(minutes=2),
                "t_end": pred_timestamp + timedelta(minutes=2),
                "t_target": pred_timestamp,
            },
        )
        base_row = base_result.fetchone()
        if base_row is None:
            return None
        base_close = float(base_row[0])
        if base_close <= 0:
            return None

        # Future price: close at prediction time + horizon bars
        future_time = pred_timestamp + timedelta(minutes=horizon_bars)
        future_result = await session.execute(
            text("""
                SELECT close FROM ohlcv_1m
                WHERE ticker = :ticker
                  AND time >= :t_start AND time <= :t_end
                ORDER BY ABS(EXTRACT(EPOCH FROM (time - :t_target)))
                LIMIT 1
            """),
            {
                "ticker": ticker,
                "t_start": future_time - timedelta(minutes=2),
                "t_end": future_time + timedelta(minutes=2),
                "t_target": future_time,
            },
        )
        future_row = future_result.fetchone()
        if future_row is None:
            return None
        future_close = float(future_row[0])

        return (future_close - base_close) / base_close

    # ── IC computation ────────────────────────────────────────────────────────

    async def compute_live_ic(self, window_days: int = 7) -> dict[str, Any]:
        """Compute rolling live IC over a specified window.

        IC = Spearman rank correlation between pred_return and actual_return_15m.
        Directional accuracy = % where sign(pred_return) == sign(actual_return_15m).

        Args:
            window_days: Number of days to look back (default 7).

        Returns:
            Dict with ic, dir_acc, n_predictions, period_start, period_end.
            Returns zeroed metrics if insufficient data (< 30 predictions).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

        try:
            async with self._sf() as session:
                result = await session.execute(
                    select(
                        PredictionOutcome.pred_return,
                        PredictionOutcome.actual_return_15m,
                        PredictionOutcome.dir_prob,
                    ).where(
                        PredictionOutcome.actual_return_15m.isnot(None),
                        PredictionOutcome.timestamp >= cutoff,
                    ).order_by(PredictionOutcome.timestamp)
                )
                rows = result.all()
        except Exception as exc:
            logger.error("ic_compute_failed", error=str(exc))
            return {
                "ic": 0.0,
                "dir_acc": 0.5,
                "n_predictions": 0,
                "window_days": window_days,
                "error": str(exc),
            }

        n = len(rows)
        if n < 30:
            return {
                "ic": 0.0,
                "dir_acc": 0.5,
                "n_predictions": n,
                "window_days": window_days,
                "note": f"insufficient data ({n} < 30)",
            }

        pred_returns = [float(r[0]) for r in rows]
        actual_returns = [float(r[1]) for r in rows]

        # Spearman rank correlation (IC)
        ic, p_value = spearmanr(pred_returns, actual_returns)

        # Directional accuracy: sign(pred) matches sign(actual)
        correct = sum(
            1 for p, a in zip(pred_returns, actual_returns)
            if (p > 0 and a > 0) or (p < 0 and a < 0) or (p == 0 and a == 0)
        )
        dir_acc = correct / n

        return {
            "ic": round(float(ic), 4) if ic == ic else 0.0,  # handle NaN
            "ic_p_value": round(float(p_value), 6) if p_value == p_value else 1.0,
            "dir_acc": round(dir_acc, 4),
            "n_predictions": n,
            "window_days": window_days,
        }

    async def get_recent_ic(self, window_days: int = 7) -> dict[str, Any]:
        """Public interface for drift agent and other consumers.

        Convenience wrapper around compute_live_ic().

        Args:
            window_days: Rolling window in days.

        Returns:
            Dict with ic, dir_acc, n_predictions, and metadata.
        """
        return await self.compute_live_ic(window_days=window_days)

    # ── Per-ticker IC breakdown ───────────────────────────────────────────────

    async def _compute_per_ticker_ic(
        self, window_days: int = 7, since: datetime | None = None
    ) -> dict[str, dict[str, Any]]:
        """Compute IC and directional accuracy per ticker.

        Args:
            window_days: Rolling window length.
            since: Optional hard lower bound — predictions made before this are
                excluded even if inside the window. Used by the signal loop to
                avoid judging the current pipeline by a PREVIOUS pipeline's
                prediction record (Pipeline B's June history once blocked
                10/20 tickers for the freshly deployed Pipeline A).

        Returns:
            Dict mapping ticker -> {ic, dir_acc, n} for tickers with >= 20 predictions.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        if since is not None and since > cutoff:
            cutoff = since
        result_map: dict[str, dict[str, Any]] = {}

        try:
            async with self._sf() as session:
                result = await session.execute(
                    select(
                        PredictionOutcome.ticker,
                        PredictionOutcome.pred_return,
                        PredictionOutcome.actual_return_15m,
                    ).where(
                        PredictionOutcome.actual_return_15m.isnot(None),
                        PredictionOutcome.timestamp >= cutoff,
                    ).order_by(
                        PredictionOutcome.ticker,
                        PredictionOutcome.timestamp,
                    )
                )
                rows = result.all()
        except Exception as exc:
            logger.error("ic_per_ticker_failed", error=str(exc))
            return {}

        # Group by ticker
        from collections import defaultdict

        by_ticker: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for ticker, pred_ret, actual_ret in rows:
            by_ticker[ticker].append((float(pred_ret), float(actual_ret)))

        for ticker, pairs in by_ticker.items():
            n = len(pairs)
            if n < 20:
                continue
            preds = [p[0] for p in pairs]
            actuals = [p[1] for p in pairs]
            ic, _ = spearmanr(preds, actuals)
            correct = sum(
                1 for p, a in zip(preds, actuals)
                if (p > 0 and a > 0) or (p < 0 and a < 0)
            )
            result_map[ticker] = {
                "ic": round(float(ic), 4) if ic == ic else 0.0,
                "dir_acc": round(correct / n, 4),
                "n": n,
            }

        return result_map

    # ── Report generation ─────────────────────────────────────────────────────

    async def generate_report(self) -> dict[str, Any]:
        """Generate daily IC report and check for escalation.

        Writes report to reports/ic/YYYY-MM-DD.json.
        Checks if 7d rolling IC < 0.05 for 5+ consecutive days.

        Returns:
            The complete report dict.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = datetime.now(timezone.utc)
        logger.info("ic_tracker_report_start", date=today)

        # Compute metrics for 7d and 30d windows
        ic_7d = await self.compute_live_ic(window_days=7)
        ic_30d = await self.compute_live_ic(window_days=30)
        by_ticker = await self._compute_per_ticker_ic(window_days=7)

        # Check escalation: 7d IC below threshold?
        rolling_ic = ic_7d.get("ic", 0.0)
        n_predictions = ic_7d.get("n_predictions", 0)

        escalation = None
        if n_predictions >= 30:
            if rolling_ic < IC_ESCALATION_THRESHOLD:
                self._consecutive_low_ic_days += 1
            else:
                self._consecutive_low_ic_days = 0

            if self._consecutive_low_ic_days >= IC_ESCALATION_CONSECUTIVE_DAYS:
                escalation = {
                    "type": "low_live_ic",
                    "consecutive_days": self._consecutive_low_ic_days,
                    "current_ic": rolling_ic,
                    "threshold": IC_ESCALATION_THRESHOLD,
                    "message": (
                        f"Live IC has been below {IC_ESCALATION_THRESHOLD} for "
                        f"{self._consecutive_low_ic_days} consecutive days "
                        f"(current 7d IC = {rolling_ic:.4f}). "
                        f"Model may have degraded — consider retraining."
                    ),
                }

        report: dict[str, Any] = {
            "date": today,
            "agent": AGENT_NAME,
            "generated_at": start.isoformat(),
            "rolling_7d_ic": ic_7d.get("ic", 0.0),
            "rolling_7d_dir_acc": ic_7d.get("dir_acc", 0.5),
            "rolling_7d_ic_p_value": ic_7d.get("ic_p_value", 1.0),
            "rolling_30d_ic": ic_30d.get("ic", 0.0),
            "rolling_30d_dir_acc": ic_30d.get("dir_acc", 0.5),
            "rolling_30d_ic_p_value": ic_30d.get("ic_p_value", 1.0),
            "n_predictions_7d": ic_7d.get("n_predictions", 0),
            "n_predictions_30d": ic_30d.get("n_predictions", 0),
            "by_ticker": by_ticker,
            "escalation": escalation,
            "consecutive_low_ic_days": self._consecutive_low_ic_days,
        }

        # Write report to disk
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / f"{today}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Log in CLAUDE.md standard format
        if escalation:
            logger.warning(
                "\n🚨 ESCALATION — %s — %s\n"
                "Issue: %s\n"
                "Impact: Live IC = %.4f (threshold: %.2f) for %d consecutive days\n"
                "Action taken: Report generated at %s\n"
                "Required from human: Review model performance, consider retraining",
                AGENT_NAME,
                today,
                escalation["message"],
                rolling_ic,
                IC_ESCALATION_THRESHOLD,
                self._consecutive_low_ic_days,
                str(report_path),
            )
        else:
            logger.info(
                "📊 %s — %s\n"
                "Summary: 7d IC=%.4f (n=%d), 30d IC=%.4f (n=%d)\n"
                "Dir Acc: 7d=%.1f%%, 30d=%.1f%%\n"
                "Tickers tracked: %d\n"
                "Action: None | Needs review: %s",
                AGENT_NAME,
                today,
                ic_7d.get("ic", 0.0),
                ic_7d.get("n_predictions", 0),
                ic_30d.get("ic", 0.0),
                ic_30d.get("n_predictions", 0),
                ic_7d.get("dir_acc", 0.5) * 100,
                ic_30d.get("dir_acc", 0.5) * 100,
                len(by_ticker),
                "Yes" if escalation else "No",
            )

        self._last_report_date = today
        return report

    # ── Scheduler entry points ────────────────────────────────────────────────

    async def run_fill(self) -> dict[str, Any]:
        """Entry point for APScheduler — fills actual returns.

        Wraps fill_actual_returns() with error handling suitable for
        scheduled execution.
        """
        try:
            return await self.fill_actual_returns()
        except Exception as exc:
            logger.error("ic_tracker_fill_job_failed", error=str(exc))
            return {"error": str(exc)}

    async def run_report(self) -> dict[str, Any]:
        """Entry point for APScheduler — generates daily report.

        Wraps generate_report() with error handling suitable for
        scheduled execution.
        """
        try:
            return await self.generate_report()
        except Exception as exc:
            logger.error("ic_tracker_report_job_failed", error=str(exc))
            return {"error": str(exc)}
