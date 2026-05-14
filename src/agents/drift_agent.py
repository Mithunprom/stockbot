"""Model Drift Agent — monitors model accuracy and feature distributions weekly.

Scope: Monitor model accuracy and feature distributions. Queue retraining.
Must NOT: Deploy retrained models to live. Push to staging/retrain_queue.json.
Output: Weekly drift report to reports/drift/YYYY-WW.json
Escalate if: Any model's PSI > 0.25 or accuracy drop > 5%.
"""

from __future__ import annotations

import asyncio
import json
import logging
import structlog
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select, func
from sqlalchemy.sql import text

from src.data.db import FeatureMatrix, Signal, get_session_factory
from src.features.psi import compute_psi

logger = structlog.get_logger(__name__)

REPORT_DIR = Path("reports/drift")
STAGING_PATH = Path("config/staging/retrain_queue.json")
MODEL_DIR = Path("models/lgbm")
AGENT_NAME = "Model Drift Agent"

# Thresholds from CLAUDE.md
PSI_CRITICAL_THRESHOLD = 0.25
ACCURACY_DROP_THRESHOLD = 0.05  # 5% relative drop in IC or dir_acc

# PSI severity bands (standard industry thresholds)
PSI_STABLE = 0.10
PSI_WARNING = 0.25


class DriftAgent:
    """Scheduled model drift monitor — runs weekly via APScheduler.

    Responsibilities:
        1. Load the active LightGBM model and its training-time metadata.
        2. Fetch recent feature distributions from the feature_matrix table.
        3. Compute PSI for each feature against training-time reference.
        4. Track model prediction accuracy via LiveICTracker.
        5. Generate a weekly drift report with severity classification.
        6. Escalate if PSI > 0.25 or accuracy drops > 5%.
        7. Queue retraining to config/staging/retrain_queue.json if needed.
    """

    def __init__(
        self,
        lookback_days: int = 7,
        live_ic_tracker: Any | None = None,
    ) -> None:
        self._lookback_days = lookback_days
        self._ic_tracker = live_ic_tracker
        self._run_count = 0

    async def run(self) -> dict[str, Any]:
        """Run one drift monitoring cycle. Called by APScheduler weekly."""
        start = datetime.now(timezone.utc)
        self._run_count += 1
        logger.info("drift_agent_run", run=self._run_count, time=start.isoformat())

        # Step 1: Load active model metadata
        model_meta = self._load_model_metadata()
        if model_meta is None:
            logger.error("drift_agent_no_model", msg="No LightGBM model found")
            return {"error": "No LightGBM model checkpoint found"}

        feature_cols: list[str] = model_meta["feature_cols"]
        training_ic: float = model_meta.get("val_ic", 0.0)
        training_dir_acc: float = model_meta.get("val_dir_acc", 0.0)

        # Step 2: Fetch recent feature distributions
        recent_features = await self._fetch_recent_features(
            feature_cols, days=self._lookback_days
        )

        # Step 3: Compute reference distributions from training period
        reference_features = await self._fetch_reference_features(
            feature_cols, days=90
        )

        # Step 4: Compute PSI per feature
        psi_scores: dict[str, float] = {}
        if recent_features is not None and reference_features is not None:
            psi_scores = self._compute_feature_psi(
                reference_features, recent_features, feature_cols
            )

        # Step 5: Track model prediction accuracy
        accuracy_metrics = await self._get_accuracy_metrics()

        # Step 6: Classify drift severity
        severity, escalations = self._classify_drift(
            psi_scores, accuracy_metrics, training_ic, training_dir_acc
        )

        # Step 7: Build report
        iso_week = start.strftime("%Y-W%W")
        report: dict[str, Any] = {
            "agent": AGENT_NAME,
            "run": self._run_count,
            "timestamp": start.isoformat(),
            "iso_week": iso_week,
            "lookback_days": self._lookback_days,
            "model_checkpoint": self._get_best_model_path(),
            "training_metrics": {
                "val_ic": training_ic,
                "val_dir_acc": training_dir_acc,
                "n_features": len(feature_cols),
            },
            "current_metrics": accuracy_metrics,
            "psi_scores": psi_scores,
            "psi_summary": self._summarize_psi(psi_scores),
            "severity": severity,
            "escalations": escalations,
            "retrain_recommended": severity == "critical",
            "needs_review": severity in ("warning", "critical"),
        }

        # Write weekly drift report
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / f"{iso_week}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Queue retraining if critical
        if severity == "critical":
            self._queue_retraining(report, escalations)

        # Log in standard format
        self._log_report(start, report, report_path)

        return report

    # ── Model metadata loading ───────────────────────────────────────────────

    def _load_model_metadata(self) -> dict[str, Any] | None:
        """Load the best LightGBM model's JSON sidecar metadata."""
        best_path = self._get_best_model_path()
        if best_path is None:
            return None

        meta_path = Path(best_path).with_suffix(".json")
        if not meta_path.exists():
            logger.warning("drift_agent_no_sidecar", path=str(meta_path))
            return None

        with open(meta_path) as f:
            return json.load(f)

    def _get_best_model_path(self) -> str | None:
        """Find the best LightGBM checkpoint by IC in filename."""
        candidates = sorted(MODEL_DIR.glob("lgbm_ic_*.pkl"), reverse=True)
        if not candidates:
            return None
        return str(candidates[0])

    # ── Feature distribution fetching ────────────────────────────────────────

    async def _fetch_recent_features(
        self, feature_cols: list[str], days: int = 7
    ) -> np.ndarray | None:
        """Fetch recent feature values from the feature_matrix table.

        Returns:
            np.ndarray of shape (n_rows, n_features) or None if no data.
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)
        return await self._fetch_features_since(feature_cols, since)

    async def _fetch_reference_features(
        self, feature_cols: list[str], days: int = 90
    ) -> np.ndarray | None:
        """Fetch older reference feature values (training-period proxy).

        Uses data from [days_ago - 90, days_ago - 7] as the reference window,
        so it doesn't overlap with the recent window.

        Returns:
            np.ndarray of shape (n_rows, n_features) or None if no data.
        """
        end = datetime.now(timezone.utc) - timedelta(days=self._lookback_days)
        start = end - timedelta(days=days)
        return await self._fetch_features_between(feature_cols, start, end)

    async def _fetch_features_since(
        self, feature_cols: list[str], since: datetime
    ) -> np.ndarray | None:
        """Query feature_matrix for rows since a given timestamp."""
        session_factory = get_session_factory()
        try:
            async with session_factory() as session:
                result = await session.execute(
                    select(FeatureMatrix.features)
                    .where(FeatureMatrix.time >= since)
                    .order_by(FeatureMatrix.time.desc())
                    .limit(50000)  # cap memory usage
                )
                rows = result.scalars().all()
        except Exception as exc:
            logger.error("drift_agent_db_error", error=str(exc), query="recent")
            return None

        if not rows:
            logger.warning("drift_agent_no_recent_data", since=since.isoformat())
            return None

        return self._features_to_array(rows, feature_cols)

    async def _fetch_features_between(
        self, feature_cols: list[str], start: datetime, end: datetime
    ) -> np.ndarray | None:
        """Query feature_matrix for rows between two timestamps."""
        session_factory = get_session_factory()
        try:
            async with session_factory() as session:
                result = await session.execute(
                    select(FeatureMatrix.features)
                    .where(FeatureMatrix.time >= start)
                    .where(FeatureMatrix.time < end)
                    .order_by(FeatureMatrix.time.desc())
                    .limit(50000)
                )
                rows = result.scalars().all()
        except Exception as exc:
            logger.error("drift_agent_db_error", error=str(exc), query="reference")
            return None

        if not rows:
            logger.warning(
                "drift_agent_no_reference_data",
                start=start.isoformat(),
                end=end.isoformat(),
            )
            return None

        return self._features_to_array(rows, feature_cols)

    def _features_to_array(
        self, rows: list[dict[str, Any]], feature_cols: list[str]
    ) -> np.ndarray:
        """Convert JSONB feature rows to a numpy array aligned to feature_cols."""
        matrix = []
        for row in rows:
            values = [float(row.get(col, 0.0) or 0.0) for col in feature_cols]
            matrix.append(values)
        return np.array(matrix, dtype=np.float64)

    # ── PSI computation ──────────────────────────────────────────────────────

    def _compute_feature_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_cols: list[str],
    ) -> dict[str, float]:
        """Compute PSI for each feature column.

        Args:
            reference: (n_ref, n_features) reference distribution.
            current: (n_cur, n_features) current distribution.
            feature_cols: feature names aligned to columns.

        Returns:
            Dict mapping feature name to PSI value.
        """
        psi_scores: dict[str, float] = {}
        for i, col in enumerate(feature_cols):
            ref_col = reference[:, i]
            cur_col = current[:, i]

            # Skip constant columns (PSI undefined)
            if ref_col.std() < 1e-12 and cur_col.std() < 1e-12:
                psi_scores[col] = 0.0
                continue

            try:
                psi_val = compute_psi(ref_col, cur_col, bins=10)
                psi_scores[col] = round(float(psi_val), 6)
            except Exception as exc:
                logger.debug("drift_psi_error", feature=col, error=str(exc))
                psi_scores[col] = 0.0

        return psi_scores

    def _summarize_psi(self, psi_scores: dict[str, float]) -> dict[str, Any]:
        """Summarize PSI scores into counts by severity band."""
        if not psi_scores:
            return {"stable": 0, "warning": 0, "critical": 0, "mean_psi": 0.0}

        values = list(psi_scores.values())
        stable = sum(1 for v in values if v < PSI_STABLE)
        warning = sum(1 for v in values if PSI_STABLE <= v < PSI_WARNING)
        critical = sum(1 for v in values if v >= PSI_WARNING)

        return {
            "stable": stable,
            "warning": warning,
            "critical": critical,
            "mean_psi": round(float(np.mean(values)), 6),
            "max_psi": round(float(np.max(values)), 6),
            "max_psi_feature": max(psi_scores, key=psi_scores.get) if psi_scores else None,
        }

    # ── Accuracy tracking ────────────────────────────────────────────────────

    async def _get_accuracy_metrics(self) -> dict[str, Any]:
        """Get recent model accuracy metrics from the LiveICTracker.

        Returns:
            Dict with recent IC, directional accuracy, and sample size.
        """
        if self._ic_tracker is None:
            return {
                "recent_ic": 0.0,
                "recent_dir_acc": 0.0,
                "n_predictions": 0,
                "window_days": self._lookback_days,
                "error": "no LiveICTracker attached",
            }
        try:
            metrics = await self._ic_tracker.get_recent_ic(
                window_days=self._lookback_days
            )
            return {
                "recent_ic": metrics.get("ic", 0.0),
                "recent_dir_acc": metrics.get("dir_acc", 0.0),
                "n_predictions": metrics.get("n_predictions", 0),
                "window_days": self._lookback_days,
            }
        except Exception as exc:
            logger.warning("drift_agent_ic_tracker_error", error=str(exc))
            return {
                "recent_ic": 0.0,
                "recent_dir_acc": 0.0,
                "n_predictions": 0,
                "window_days": self._lookback_days,
                "error": str(exc),
            }

    # ── Drift classification ─────────────────────────────────────────────────

    def _classify_drift(
        self,
        psi_scores: dict[str, float],
        accuracy_metrics: dict[str, Any],
        training_ic: float,
        training_dir_acc: float,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Classify overall drift severity and generate escalation items.

        Severity levels:
            stable   — no significant drift detected
            warning  — moderate drift, monitor closely
            critical — PSI > 0.25 or accuracy drop > 5%, escalate immediately

        Returns:
            (severity, escalations) tuple.
        """
        escalations: list[dict[str, Any]] = []

        # Check PSI thresholds
        critical_psi_features = {
            feat: psi for feat, psi in psi_scores.items()
            if psi >= PSI_CRITICAL_THRESHOLD
        }
        warning_psi_features = {
            feat: psi for feat, psi in psi_scores.items()
            if PSI_STABLE <= psi < PSI_CRITICAL_THRESHOLD
        }

        if critical_psi_features:
            escalations.append({
                "type": "psi_critical",
                "message": (
                    f"{len(critical_psi_features)} feature(s) exceed PSI threshold "
                    f"of {PSI_CRITICAL_THRESHOLD}"
                ),
                "features": critical_psi_features,
            })

        # Check accuracy drop (relative to training metrics)
        recent_ic = accuracy_metrics.get("recent_ic", 0.0)
        recent_dir_acc = accuracy_metrics.get("recent_dir_acc", 0.0)
        n_predictions = accuracy_metrics.get("n_predictions", 0)

        # Only check accuracy if we have enough predictions
        if n_predictions >= 100:
            ic_drop = (training_ic - recent_ic) / max(abs(training_ic), 1e-9)
            dir_acc_drop = (training_dir_acc - recent_dir_acc) / max(
                training_dir_acc, 1e-9
            )

            if ic_drop > ACCURACY_DROP_THRESHOLD:
                escalations.append({
                    "type": "ic_drop",
                    "message": (
                        f"IC dropped {ic_drop:.1%} from training "
                        f"({training_ic:.4f} -> {recent_ic:.4f})"
                    ),
                    "training_ic": training_ic,
                    "recent_ic": recent_ic,
                    "drop_pct": round(ic_drop, 4),
                })

            if dir_acc_drop > ACCURACY_DROP_THRESHOLD:
                escalations.append({
                    "type": "dir_acc_drop",
                    "message": (
                        f"Directional accuracy dropped {dir_acc_drop:.1%} from training "
                        f"({training_dir_acc:.4f} -> {recent_dir_acc:.4f})"
                    ),
                    "training_dir_acc": training_dir_acc,
                    "recent_dir_acc": recent_dir_acc,
                    "drop_pct": round(dir_acc_drop, 4),
                })

        # Determine overall severity
        if critical_psi_features or any(
            e["type"] in ("ic_drop", "dir_acc_drop") for e in escalations
        ):
            severity = "critical"
        elif warning_psi_features:
            severity = "warning"
        else:
            severity = "stable"

        return severity, escalations

    # ── Retraining queue ─────────────────────────────────────────────────────

    def _queue_retraining(
        self, report: dict[str, Any], escalations: list[dict[str, Any]]
    ) -> None:
        """Write a retraining request to config/staging/retrain_queue.json.

        Must NOT deploy retrained models to live — only queue the request
        for human review.
        """
        request = {
            "requested_by": AGENT_NAME,
            "requested_at": report["timestamp"],
            "reason": "Model drift detected — retraining recommended",
            "severity": report["severity"],
            "escalations": escalations,
            "current_metrics": report["current_metrics"],
            "training_metrics": report["training_metrics"],
            "psi_summary": report["psi_summary"],
            "model_checkpoint": report["model_checkpoint"],
            "status": "pending_review",
        }

        STAGING_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Merge with existing queue (append, don't overwrite)
        queue: list[dict[str, Any]] = []
        if STAGING_PATH.exists():
            try:
                with open(STAGING_PATH) as f:
                    existing = json.load(f)
                if isinstance(existing, list):
                    queue = existing
                else:
                    # Legacy format — wrap in list
                    queue = [existing]
            except (json.JSONDecodeError, Exception):
                queue = []

        queue.append(request)

        with open(STAGING_PATH, "w") as f:
            json.dump(queue, f, indent=2)

        logger.info(
            "drift_agent_retrain_queued",
            path=str(STAGING_PATH),
            n_queue=len(queue),
        )

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log_report(
        self,
        start: datetime,
        report: dict[str, Any],
        report_path: Path,
    ) -> None:
        """Log the drift report in CLAUDE.md communication format."""
        severity = report["severity"]
        psi_summary = report.get("psi_summary", {})
        current = report.get("current_metrics", {})
        training = report.get("training_metrics", {})

        if report.get("escalations"):
            esc_messages = [e["message"] for e in report["escalations"]]
            logger.warning(
                "\n🚨 ESCALATION — %s — %s\n"
                "Issue: %s\n"
                "Impact: Severity=%s | %d critical PSI features | "
                "IC: %.4f (train) -> %.4f (recent)\n"
                "Action taken: Retraining queued to %s\n"
                "Required from human: Review drift report at %s and approve retraining",
                AGENT_NAME,
                start.isoformat(),
                "; ".join(esc_messages),
                severity,
                psi_summary.get("critical", 0),
                training.get("val_ic", 0),
                current.get("recent_ic", 0),
                str(STAGING_PATH),
                str(report_path),
            )
        else:
            logger.info(
                "📊 %s — %s\n"
                "Summary: Severity=%s | Mean PSI=%.4f | Max PSI=%.4f (%s)\n"
                "Metrics: IC %.4f (train) -> %.4f (recent) | "
                "DirAcc %.1f%% (train) -> %.1f%% (recent)\n"
                "Action taken: None | Needs review: %s",
                AGENT_NAME,
                start.isoformat(),
                severity,
                psi_summary.get("mean_psi", 0),
                psi_summary.get("max_psi", 0),
                psi_summary.get("max_psi_feature", "N/A"),
                training.get("val_ic", 0),
                current.get("recent_ic", 0),
                training.get("val_dir_acc", 0) * 100,
                current.get("recent_dir_acc", 0) * 100,
                "Yes" if report["needs_review"] else "No",
            )


# ── Standalone execution ─────────────────────────────────────────────────────


async def main() -> None:
    """Run the drift agent standalone (outside APScheduler)."""
    agent = DriftAgent(lookback_days=7)
    report = await agent.run()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
