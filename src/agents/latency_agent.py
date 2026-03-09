"""Latency Agent — measures and reports pipeline latency hourly.

Scope: Measure and report pipeline latency. Recommend optimizations.
Must NOT: Modify model weights, change trading parameters, execute trades.
Output: reports/latency/YYYY-MM-DD.json
Escalate if: Any stage exceeds 200ms p95 latency.
"""

from __future__ import annotations

import json
import logging
import structlog
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = structlog.get_logger(__name__)

REPORT_DIR = Path("reports/latency")
P95_ALERT_MS = 200.0
AGENT_NAME = "Latency Agent"


class LatencyAgent:
    """Profiles all pipeline stages and writes a latency report."""

    def __init__(self) -> None:
        self._measurements: dict[str, list[float]] = {}

    def record(self, stage: str, latency_ms: float) -> None:
        """Record a latency measurement for a named pipeline stage."""
        if stage not in self._measurements:
            self._measurements[stage] = []
        self._measurements[stage].append(latency_ms)
        # Keep last 100 measurements per stage
        if len(self._measurements[stage]) > 100:
            self._measurements[stage] = self._measurements[stage][-100:]

    async def run(self) -> dict[str, Any]:
        """Run one latency profiling cycle. Returns the report dict."""
        start = datetime.now(timezone.utc)

        # Profile key pipeline stages
        await self._profile_db_query()
        await self._profile_feature_computation()
        await self._profile_model_inference()

        # Build report
        report: dict[str, Any] = {
            "agent": AGENT_NAME,
            "timestamp": start.isoformat(),
            "stages": {},
            "escalations": [],
        }

        for stage, measurements in self._measurements.items():
            if len(measurements) < 2:
                continue
            sorted_ms = sorted(measurements)
            n = len(sorted_ms)
            p95_idx = int(0.95 * n)
            p95 = sorted_ms[p95_idx]
            stats = {
                "count": n,
                "p50_ms": round(statistics.median(sorted_ms), 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(sorted_ms[min(int(0.99 * n), n - 1)], 2),
                "mean_ms": round(statistics.mean(sorted_ms), 2),
                "max_ms": round(max(sorted_ms), 2),
            }
            report["stages"][stage] = stats

            if p95 > P95_ALERT_MS:
                report["escalations"].append(
                    {
                        "stage": stage,
                        "p95_ms": p95,
                        "threshold_ms": P95_ALERT_MS,
                        "message": f"{stage} p95={p95:.0f}ms exceeds {P95_ALERT_MS:.0f}ms threshold",
                    }
                )

        report["needs_review"] = bool(report["escalations"])

        # Write report
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        date_str = start.strftime("%Y-%m-%d")
        report_path = REPORT_DIR / f"{date_str}.json"

        # Merge with existing daily report
        existing: dict[str, Any] = {}
        if report_path.exists():
            with open(report_path) as f:
                existing = json.load(f)
        existing.update(report)
        with open(report_path, "w") as f:
            json.dump(existing, f, indent=2)

        if report["escalations"]:
            logger.warning(
                "\n🚨 ESCALATION — Latency Agent — %s\n"
                "Issue: %d stage(s) exceed 200ms p95\n"
                "Stages: %s\n"
                "Required from human: Review latency report at %s",
                start.isoformat(),
                len(report["escalations"]),
                [e["stage"] for e in report["escalations"]],
                str(report_path),
            )
        else:
            logger.info(
                "📊 Latency Agent — %s\nAll stages within 200ms p95. Needs review: No",
                start.isoformat(),
            )

        return report

    async def _profile_db_query(self) -> None:
        """Profile a representative TimescaleDB query."""
        try:
            from src.data.db import get_session_factory
            from sqlalchemy.sql import text

            sf = get_session_factory()
            t0 = time.perf_counter()
            async with sf() as session:
                await session.execute(text("SELECT 1"))
            self.record("db_query", (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            logger.debug("latency_db_profile_skip", reason=str(exc))

    async def _profile_feature_computation(self) -> None:
        """Profile feature computation on a small synthetic DataFrame."""
        try:
            import numpy as np
            import pandas as pd
            from src.features.indicators import compute_indicators

            n = 200
            idx = pd.date_range("2024-01-01", periods=n, freq="1min")
            df = pd.DataFrame(
                {
                    "open": np.random.uniform(100, 110, n),
                    "high": np.random.uniform(110, 115, n),
                    "low": np.random.uniform(95, 100, n),
                    "close": np.random.uniform(100, 110, n),
                    "volume": np.random.uniform(1e5, 1e6, n),
                },
                index=idx,
            )
            t0 = time.perf_counter()
            compute_indicators(df, shift=True)
            self.record("feature_computation", (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            logger.debug("latency_feature_profile_skip", reason=str(exc))

    async def _profile_model_inference(self) -> None:
        """Profile a mock Transformer forward pass."""
        try:
            import torch
            from src.models.transformer import TransformerSignalModel

            model = TransformerSignalModel()
            model.eval()
            x = torch.randn(1, 60, 30)
            t0 = time.perf_counter()
            with torch.no_grad():
                model(x)
            self.record("transformer_inference", (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            logger.debug("latency_model_profile_skip", reason=str(exc))
