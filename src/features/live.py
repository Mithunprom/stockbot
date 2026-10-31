"""Incremental live feature computation.

Triggered after completed 1m bars are written to ohlcv_1m. Loads the last
WARMUP_BARS of history, runs the indicator pipeline (shift=True to match the
training convention), and upserts the latest feature row per ticker into
feature_matrix — keeping the table fresh so the signal loop always scores on
current features without a manual build_features.py run.

MUST compute the universe in ONE batch (`on_bars`), not per ticker
----------------------------------------------------------------
Some model features are CROSS-SECTIONAL — defined relative to the universe
mean, not derivable from a single ticker's bars:

    rs_1m        ticker 1m return    − universe-mean 1m return
    rs_15m       ticker 15m return   − universe-mean 15m return
    rs_vwap_dev  ticker VWAP dev     − universe-mean VWAP dev

Only `compute_indicators_for_universe` produces them; plain
`compute_indicators` cannot, by construction.

This module used to call the per-ticker function in a loop. `rs_vwap_dev` is one
of the deployed model's 30 features, so it arrived MISSING on every bar and was
zero-filled at scoring time. Nothing errored. The visible symptom was that
prediction magnitudes collapsed — median |pred_return| 0.000438 against a 0.002
entry threshold — so almost nothing cleared the gate and the bot stopped trading
entirely for seven sessions (2026-09-21 → 09-28). Scoring the model offline with
that one feature zeroed reproduced the live distribution almost exactly
(median 0.000384, p90 0.001402), which is how it was identified.

This is the SECOND instance of the same class of bug: a feature whose value
depends on context the serving path does not supply. The first was `obv`
(anchored to the window rather than the session, fixed 2026-09-16). The lesson
generalises — anything computed from more than the current ticker's own bars
must be produced the same way in training and in serving, and
tests/unit/test_feature_serving_parity.py now checks every model feature rather
than a sample.
"""

from __future__ import annotations

import asyncio
import numpy as np
import structlog
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from src.data.db import FeatureMatrix, OHLCV1m, get_session_factory
from src.features.indicators import compute_indicators, compute_indicators_for_universe

logger = structlog.get_logger(__name__)

# Enough bars for longest indicator (EMA-50, VPIN-50) to fully warm up.
# Warmup must be long enough that every feature compute_indicators() produces is
# identical to what the training/backtest path produces over full history.
#
# 300 was not. Measured 2026-09-16 over 320 paired (ticker, bar) samples, 11 of
# the model's 30 features disagreed between the two paths, and the resulting
# predictions had a Spearman rank agreement of only 0.36 with the
# train-consistent ones — which is why production's entries landed at the ~46th
# percentile of the model's true ranking instead of the top decile where the
# edge is. See reports/research/loss_diagnosis_2026-09-16.md.
#
# Warmup alone could not close it (rank agreement plateaued at ~0.55) because
# `obv` was anchored to the first bar of the window; that is fixed separately in
# indicators._obv. With the anchoring fixed, this raised rank agreement to 0.64.
#
# 0.64 was still not enough. Measured 2026-10-27 (sim_live_ic_gap.py):
#   full-history IC  = +0.181   (what training sees)
#   1950-bar IC      = −0.010   (what serving was doing — near-random)
#   rank agreement   =  0.64
# Root cause: 1950 bars = exactly 5 sessions = the ohlcv_1m retention ceiling.
# At the ceiling, every warmup starts from a cold EWM state with zero history
# prior to the window. The mtf_trend_60m feature resamples to 60m bars; with
# 1950/60 = 32.5 resampled periods, EMA(21) still carries ~5% initial-condition
# contamination ((1 − 2/22)^32 ≈ 0.053). That is enough to corrupt rank
# agreement on a feature that is the 3rd-highest-importance predictor.
#
# The fix is to restore ohlcv_1m retention from 7 to 14 days (db._RETENTION_POLICIES)
# so this constant can double without hitting the ceiling. At 3900 bars:
#   3900/60 = 65 resampled 60m periods → (1 − 2/22)^65 ≈ 0.003 (0.3% contamination)
# Expected rank agreement improvement: 0.64 → ~0.85–0.90.
# Expected IC improvement: −0.010 → approaching full-history +0.18.
# Mark as needs_data_run (requires Railway worker to measure IC post-deploy).
#
# INVARIANT: WARMUP_BARS must be ≤ ohlcv_1m retention_days × 390 × 0.9 so the
# warmup never again sit at the ceiling. test_warmup_retention_invariant.py
# enforces this so a future retention cut cannot silently re-introduce the bug.
#
# 3500 bars ≈ 9 trading sessions — 90% of the 14-day retention ceiling (10
# sessions). The 10% margin absorbs market holidays and partial sessions so
# the warmup never sits at the exact ceiling again. At 3500//60 = 58 resampled
# 60m periods, EMA(21) contamination = (1−2/22)^58 ≈ 0.4% — negligible.
WARMUP_BARS = 3500
FFSA_VERSION = "v1"


class LiveFeatureComputer:
    """Computes and upserts features for the universe on each completed 1m bar.

    Prefer `on_bars()` — it computes the whole universe together, which is the
    only way CROSS-SECTIONAL features can be produced (see the module docstring).
    `on_bar()` remains for a single ticker but cannot produce them.

    Args:
        feature_cols: Ordered list of FFSA-selected feature column names.
    """

    def __init__(self, feature_cols: list[str]) -> None:
        self._feature_cols = feature_cols
        self._sf = get_session_factory()
        # Simple per-ticker throttle: skip if a compute is already in flight
        self._in_flight: set[str] = set()
        self._batch_in_flight = False
        # Names whose cross-sectional features could not be produced, so the
        # caller can see degradation instead of inferring it from flat
        # predictions weeks later.
        self.last_batch_missing_cross_sectional: list[str] = []

    async def on_bars(self, bars: dict[str, datetime]) -> None:
        """Compute features for every ticker that just closed a bar, together.

        `bars` is {ticker: bar_time}. Tickers are computed in ONE batch through
        compute_indicators_for_universe so cross-sectional features exist. A
        per-ticker loop cannot produce them — that was the defect that silently
        zero-filled rs_vwap_dev in production for weeks.
        """
        if self._batch_in_flight:
            logger.debug("live_feature_batch_skip_in_flight", n=len(bars))
            return
        self._batch_in_flight = True
        try:
            await self._compute_and_write_batch(bars)
        except Exception as exc:
            logger.warning(
                "live_feature_batch_error", n=len(bars), error=str(exc), exc_info=True,
            )
        finally:
            self._batch_in_flight = False

    async def on_bar(self, ticker: str, bar_time: datetime) -> None:
        """Single-ticker compute.

        WARNING: cannot produce cross-sectional features (rs_*) — they are
        defined relative to the universe mean. Use `on_bars()` for anything the
        model will be scored on. Retained for targeted backfills and tests.
        """
        if ticker in self._in_flight:
            return  # previous compute still running — skip this bar
        self._in_flight.add(ticker)
        try:
            await self._compute_and_write(ticker, bar_time)
        except Exception as exc:
            logger.warning(
                "live_feature_error",
                ticker=ticker,
                bar_time=str(bar_time),
                error=str(exc),
            )
        finally:
            self._in_flight.discard(ticker)

    async def _compute_and_write_batch(self, bars: dict[str, datetime]) -> None:
        """Universe-wide compute → cross-sectional features → one upsert."""
        if not bars:
            return

        # 1. Load warmup history for every ticker (concurrently)
        tickers = sorted(bars)
        loaded = await asyncio.gather(
            *(self._load_ohlcv(t) for t in tickers), return_exceptions=True,
        )
        frames: dict[str, pd.DataFrame] = {}
        for t, df in zip(tickers, loaded):
            if isinstance(df, Exception) or df is None or getattr(df, "empty", True):
                continue
            if len(df) >= 60:
                frames[t] = df

        if len(frames) < 2:
            # Cross-sectional features need a universe. Fall back rather than
            # write rows the model would be scored on with rs_* zeroed.
            logger.warning(
                "live_feature_batch_too_few_tickers",
                n=len(frames),
                note="cross-sectional features unavailable; skipping batch",
            )
            return

        # 2. Universe compute — this is what produces rs_1m / rs_15m / rs_vwap_dev
        results = await asyncio.to_thread(
            compute_indicators_for_universe, frames, True,
        )

        # 3. Build one row per ticker from its latest bar
        rows: list[dict[str, Any]] = []
        missing_cs: list[str] = []
        for ticker, feat_df in results.items():
            if feat_df is None or feat_df.empty:
                continue
            last_row = feat_df.iloc[-1]
            feat_dict = self._row_to_features(feat_df, last_row)
            if not any(c.startswith("rs_") for c in feat_dict):
                missing_cs.append(ticker)
            rows.append({
                "time": last_row.name,
                "ticker": ticker,
                "features": feat_dict,
                "ffsa_version": FFSA_VERSION,
            })

        self.last_batch_missing_cross_sectional = missing_cs
        if missing_cs:
            logger.warning(
                "live_feature_missing_cross_sectional",
                tickers=missing_cs[:10],
                n=len(missing_cs),
            )

        if not rows:
            return

        await self._upsert(rows)
        logger.info(
            "live_feature_batch_written",
            tickers=len(rows),
            cross_sectional_ok=len(rows) - len(missing_cs),
        )

    def _row_to_features(
        self, feat_df: pd.DataFrame, last_row: "pd.Series"
    ) -> dict[str, float | None]:
        """Serialise one feature row, dropping raw OHLCV columns."""
        out: dict[str, float | None] = {}
        for col in feat_df.columns:
            if col in {"open", "high", "low", "close", "volume", "vwap"}:
                continue
            val = last_row[col]
            out[col] = (
                None if not isinstance(val, (int, float, np.floating))
                or not np.isfinite(float(val))
                else round(float(val), 8)
            )
        return out

    async def _upsert(self, rows: list[dict[str, Any]]) -> None:
        async with self._sf() as session:
            stmt = insert(FeatureMatrix).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["time", "ticker"],
                set_={
                    "features": stmt.excluded.features,
                    "ffsa_version": stmt.excluded.ffsa_version,
                },
            )
            await session.execute(stmt)
            await session.commit()

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _compute_and_write(self, ticker: str, bar_time: datetime) -> None:
        # 1. Load the last WARMUP_BARS from ohlcv_1m
        df = await self._load_ohlcv(ticker)
        if df.empty or len(df) < 60:
            logger.debug("live_feature_skip_insufficient_bars", ticker=ticker, n=len(df))
            return

        # 2. Compute indicators — shift=True matches training convention.
        #    Single-ticker path: no cross-sectional features (see on_bar docs).
        feat_df = compute_indicators(df, shift=True)

        # 3. Take only the last row (the just-completed bar)
        last_row = feat_df.iloc[-1]
        feat_dict = self._row_to_features(feat_df, last_row)

        # 4. Upsert into feature_matrix
        await self._upsert([{
            "time": last_row.name,  # DatetimeIndex
            "ticker": ticker,
            "features": feat_dict,
            "ffsa_version": FFSA_VERSION,
        }])

        logger.debug(
            "live_feature_written",
            ticker=ticker,
            bar_time=str(last_row.name),
            n_features=len(feat_dict),
        )

    async def _load_ohlcv(self, ticker: str) -> pd.DataFrame:
        """Load the last WARMUP_BARS 1m bars for a ticker from DB."""
        async with self._sf() as session:
            result = await session.execute(
                select(OHLCV1m)
                .where(OHLCV1m.ticker == ticker)
                .order_by(OHLCV1m.time.desc())
                .limit(WARMUP_BARS)
            )
            rows = list(reversed(result.scalars().all()))

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            [
                {
                    "time": r.time,
                    "open":   float(r.open),
                    "high":   float(r.high),
                    "low":    float(r.low),
                    "close":  float(r.close),
                    "volume": float(r.volume),
                    "vwap":   float(r.vwap or r.close),
                }
                for r in rows
            ]
        )
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time").sort_index()
