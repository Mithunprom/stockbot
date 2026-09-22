"""Durable feature archive — the training set that outlives DB retention.

`feature_matrix` is pruned at 3 days (`db._RETENTION_POLICIES`), tightened after
the 2026-04-13 disk-full incident. `RetrainAgent` trains LightGBM on whatever is
in that table, so every scheduled retrain has been fitting an intraday alpha
model to ~3 days of bars. The result is checkpoints whose validation IC swings
between -0.06 and +0.19 within days of each other; `MIN_VAL_IC = 0.05` then
correctly rejects most of them, which is why production stayed pinned to a
single ageing checkpoint. The original model that worked was trained on ~4.75M
rows spanning months.

The constraint is real — the DB cannot hold months of 1-minute features for a
75-ticker universe. So this module moves the training set OUT of Postgres:
rows are copied to append-only daily files before the prune deletes them, and
training reads back from the archive instead of the live table.

Layout (S3 when AWS_S3_BUCKET is set, local directory otherwise):

    <root>/feature_archive/v1/date=YYYY-MM-DD/features.csv.gz

One file per ET session — ~75 tickers x 390 bars is small enough that
per-ticker partitioning would only add request overhead. gzipped CSV rather
than Parquet deliberately: the whole day is read back at once so columnar
access buys nothing, and it keeps pyarrow (~100MB) out of the Railway image.

Ordering matters: `archive_pending()` must run BEFORE the prune, and the prune
must not delete a day the archive has not confirmed. See
`db.prune_old_data`, which calls this first and refuses to prune on failure.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import structlog
from sqlalchemy import select

logger = structlog.get_logger(__name__)

ARCHIVE_PREFIX = "feature_archive/v1"
LOCAL_ROOT = Path(os.environ.get("FEATURE_ARCHIVE_DIR", "data/feature_archive"))

# Never archive the current session — it is still being written to.
MIN_AGE_DAYS = 1


@dataclass
class ArchiveResult:
    """Outcome of an archive run."""

    days_written: list[str] = field(default_factory=list)
    days_skipped: list[str] = field(default_factory=list)
    rows_written: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {
            "days_written": self.days_written,
            "days_skipped": self.days_skipped,
            "rows_written": self.rows_written,
            "errors": self.errors,
            "ok": self.ok,
        }


# ─── Storage backend ─────────────────────────────────────────────────────────


def _bucket() -> str | None:
    return os.environ.get("AWS_S3_BUCKET") or None


def _s3():
    import boto3

    return boto3.client("s3")


def _key_for(day: date) -> str:
    return f"{ARCHIVE_PREFIX}/date={day.isoformat()}/features.csv.gz"


def exists(day: date) -> bool:
    """True if this session is already archived."""
    bucket = _bucket()
    if bucket:
        try:
            from botocore.exceptions import ClientError

            try:
                _s3().head_object(Bucket=bucket, Key=_key_for(day))
                return True
            except ClientError:
                return False
        except ImportError:
            logger.warning("feature_archive_boto3_missing")
            return False
    return (LOCAL_ROOT / _key_for(day)).exists()


def _write(day: date, df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_csv(buf, index=False, compression="gzip")
    data = buf.getvalue()

    bucket = _bucket()
    if bucket:
        _s3().put_object(Bucket=bucket, Key=_key_for(day), Body=data)
        logger.info("feature_archive_wrote_s3", day=day.isoformat(),
                    rows=len(df), bytes=len(data))
        return

    path = LOCAL_ROOT / _key_for(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    logger.info("feature_archive_wrote_local", day=day.isoformat(),
                rows=len(df), path=str(path))


def _read(day: date) -> pd.DataFrame | None:
    bucket = _bucket()
    if bucket:
        try:
            obj = _s3().get_object(Bucket=bucket, Key=_key_for(day))
            return pd.read_csv(io.BytesIO(obj["Body"].read()), compression="gzip")
        except Exception:
            return None
    path = LOCAL_ROOT / _key_for(day)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, compression="gzip")
    except Exception as exc:                       # pragma: no cover - corrupt file
        logger.warning("feature_archive_read_failed", day=day.isoformat(), error=str(exc))
        return None


# ─── Archiving ───────────────────────────────────────────────────────────────


async def archive_pending(lookback_days: int = 7) -> ArchiveResult:
    """Copy not-yet-archived sessions out of `feature_matrix` into the archive.

    Scans back `lookback_days` (comfortably wider than the 3-day retention, so a
    missed run self-heals on the next pass) and writes any session that is at
    least MIN_AGE_DAYS old and not already present.
    """
    from src.data.db import FeatureMatrix, OHLCV1m, get_session_factory

    result = ArchiveResult()
    session_factory = get_session_factory()
    today = datetime.now(timezone.utc).date()

    for back in range(MIN_AGE_DAYS, lookback_days + 1):
        day = today - timedelta(days=back)
        if exists(day):
            result.days_skipped.append(day.isoformat())
            continue

        start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        try:
            async with session_factory() as session:
                rows = (
                    await session.execute(
                        select(FeatureMatrix.time, FeatureMatrix.ticker,
                               FeatureMatrix.features, FeatureMatrix.ffsa_version)
                        .where(FeatureMatrix.time >= start)
                        .where(FeatureMatrix.time < end)
                        .order_by(FeatureMatrix.ticker, FeatureMatrix.time)
                    )
                ).all()

            if not rows:
                result.days_skipped.append(day.isoformat())
                continue

            records = []
            for ts, ticker, feats, ffsa_version in rows:
                if not isinstance(feats, dict):
                    continue
                records.append({"time": ts, "ticker": ticker,
                                "ffsa_version": ffsa_version, **feats})
            if not records:
                result.days_skipped.append(day.isoformat())
                continue

            df = pd.DataFrame.from_records(records)

            # Carry the close price alongside the features. ohlcv_1m is pruned
            # at 7 days, so without this the archive would hold features that no
            # future training run could ever label — forward_return has to be
            # computable from the archive alone, at whatever horizon a later
            # retrain chooses.
            async with session_factory() as session:
                price_rows = (
                    await session.execute(
                        select(OHLCV1m.time, OHLCV1m.ticker, OHLCV1m.close)
                        .where(OHLCV1m.time >= start)
                        .where(OHLCV1m.time < end)
                    )
                ).all()
            if price_rows:
                prices = pd.DataFrame(
                    [{"time": t, "ticker": tk, "close": float(c)}
                     for t, tk, c in price_rows]
                )
                df = df.merge(prices, on=["time", "ticker"], how="left")
            else:
                df["close"] = pd.NA
                logger.warning("feature_archive_no_prices", day=day.isoformat())
            _write(day, df)
            result.days_written.append(day.isoformat())
            result.rows_written += len(df)

        except Exception as exc:
            logger.error("feature_archive_failed", day=day.isoformat(), error=str(exc))
            result.errors.append(f"{day.isoformat()}: {exc}")

    logger.info("feature_archive_complete", **result.as_dict())
    return result


# ─── Reading back for training ───────────────────────────────────────────────


def load_range(
    start: date,
    end: date,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Load archived features for [start, end] inclusive.

    Returns a long frame with `time`, `ticker` and one column per feature.
    Missing sessions are skipped silently — the archive is best-effort history,
    and a gap should shrink the training set rather than fail the retrain.
    """
    frames: list[pd.DataFrame] = []
    day = start
    while day <= end:
        df = _read(day)
        if df is not None and not df.empty:
            if tickers:
                df = df[df["ticker"].isin(tickers)]
            if not df.empty:
                frames.append(df)
        day += timedelta(days=1)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    return out.sort_values(["ticker", "time"]).reset_index(drop=True)


def available_days() -> list[str]:
    """Sessions currently present in the archive, oldest first."""
    bucket = _bucket()
    if bucket:
        try:
            paginator = _s3().get_paginator("list_objects_v2")
            days: list[str] = []
            for page in paginator.paginate(Bucket=bucket, Prefix=f"{ARCHIVE_PREFIX}/"):
                for obj in page.get("Contents", []):
                    part = obj["Key"].split("date=")
                    if len(part) == 2:
                        days.append(part[1].split("/")[0])
            return sorted(set(days))
        except Exception as exc:
            logger.warning("feature_archive_list_failed", error=str(exc))
            return []

    root = LOCAL_ROOT / ARCHIVE_PREFIX
    if not root.exists():
        return []
    return sorted(p.name.split("date=")[1] for p in root.glob("date=*") if "date=" in p.name)
