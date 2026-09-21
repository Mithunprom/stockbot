"""Seed the feature archive from historical bars.

The archive (src/data/feature_archive.py) only accumulates going forward, and
`train_lgbm` wants ~20 sessions before it will prefer the archive over the
3-day live tables. Waiting a month to retrain is not acceptable when the
current checkpoint is the thing blocking deployment, so this backfills history
directly from the bar feed.

Features are computed over the FULL bar history per ticker and only then split
into sessions, which is the same path training and the backtest use — so the
archive is train-consistent by construction. Do not "optimise" this into a
per-session compute: that reintroduces exactly the windowing skew that
`test_feature_serving_parity` exists to prevent.

Usage:
    python scripts/backfill_feature_archive.py --start 2026-04-15 --end 2026-09-12
    python scripts/backfill_feature_archive.py --start ... --bars-cache /path/to/cache
    python scripts/backfill_feature_archive.py --start ... --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import feature_archive as fa          # noqa: E402
from src.features.indicators import compute_indicators  # noqa: E402

ET = "America/New_York"


def _universe(path: str | None) -> list[str]:
    import json

    p = Path(path) if path else Path("config/universe.json")
    with open(p) as f:
        data = json.load(f)
    return [t for t in data["symbols"] if "/" not in t]


def _alpaca_keys() -> dict[str, str]:
    keys = {k: os.environ[k] for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")
            if os.environ.get(k)}
    if len(keys) < 2 and Path(".env").exists():
        for line in open(".env"):
            if line.startswith(("ALPACA_API_KEY", "ALPACA_SECRET_KEY")) and "=" in line:
                k, v = line.strip().split("=", 1)
                keys.setdefault(k.strip(), v.strip())
    missing = {"ALPACA_API_KEY", "ALPACA_SECRET_KEY"} - keys.keys()
    if missing:
        raise SystemExit(f"missing credentials: {', '.join(sorted(missing))}")
    return keys


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    et = df.index.tz_convert(ET)
    keep = ((et.time >= pd.Timestamp("09:30").time())
            & (et.time <= pd.Timestamp("15:59").time())
            & (et.weekday < 5))
    return df[keep]


def load_bars(ticker: str, start: str, end: str, cache: Path | None) -> pd.DataFrame | None:
    """Bars for one ticker, from a local cache if given, else from Alpaca."""
    if cache is not None:
        p = cache / f"bars_{ticker}.csv.gz"
        if not p.exists():
            return None
        df = pd.read_csv(p, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        keys = _alpaca_keys()
        client = StockHistoricalDataClient(keys["ALPACA_API_KEY"], keys["ALPACA_SECRET_KEY"])
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
            end=datetime.fromisoformat(end).replace(tzinfo=timezone.utc),
            adjustment=Adjustment.ALL,
            feed=DataFeed.IEX,
        )
        got = client.get_stock_bars(req)
        if ticker not in got.data:
            return None
        df = pd.DataFrame([b.model_dump() for b in got.data[ticker]])
        df = df.set_index("timestamp").sort_index()
        df.index = pd.to_datetime(df.index, utc=True)

    df = _rth(df)
    if "vwap" not in df.columns:
        df["vwap"] = df["close"]
    df["vwap"] = df["vwap"].fillna(df["close"])
    return df if len(df) > 400 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (include warmup)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--universe", default=None)
    ap.add_argument("--bars-cache", default=None,
                    help="directory of bars_TICKER.csv.gz to reuse instead of fetching")
    ap.add_argument("--skip-days", type=int, default=5,
                    help="drop the first N sessions — indicator warmup is unreliable there")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cache = Path(args.bars_cache) if args.bars_cache else None
    tickers = _universe(args.universe)
    print(f"universe: {len(tickers)} tickers   {args.start} → {args.end}")
    print(f"source: {'cache ' + str(cache) if cache else 'Alpaca IEX'}\n")

    per_day: dict[date, list[pd.DataFrame]] = {}
    skipped: list[str] = []

    for i, tk in enumerate(tickers, 1):
        bars = load_bars(tk, args.start, args.end, cache)
        if bars is None or bars.empty:
            skipped.append(tk)
            continue

        # full-history compute — matches training/backtest exactly
        feats = compute_indicators(bars, shift=True)
        feats = feats.copy()
        feats["close"] = bars["close"]
        feats["ticker"] = tk
        feats["ffsa_version"] = "v1"
        feats = feats.reset_index().rename(columns={"index": "time", "timestamp": "time"})
        feats["time"] = pd.to_datetime(feats["time"], utc=True)

        sessions = feats["time"].dt.tz_convert(ET).dt.date
        for d, grp in feats.groupby(sessions):
            per_day.setdefault(d, []).append(grp)

        if i % 10 == 0:
            print(f"  {i}/{len(tickers)} tickers ... {len(per_day)} sessions so far")

    days = sorted(per_day)
    if args.skip_days:
        dropped, days = days[:args.skip_days], days[args.skip_days:]
        print(f"\ndropping {len(dropped)} warmup session(s): "
              f"{dropped[0] if dropped else '-'} → {dropped[-1] if dropped else '-'}")

    print(f"\n{len(days)} sessions to write, {len(skipped)} tickers skipped"
          f"{' (' + ', '.join(skipped[:8]) + ')' if skipped else ''}")
    if args.dry_run:
        for d in days[:5]:
            print(f"  would write {d}: {sum(len(g) for g in per_day[d]):,} rows")
        print("  ... (dry run, nothing written)")
        return

    written = rows = 0
    for d in days:
        if fa.exists(d) and not args.overwrite:
            continue
        df = pd.concat(per_day[d], ignore_index=True).sort_values(["ticker", "time"])
        fa._write(d, df)
        written += 1
        rows += len(df)

    print(f"\nwrote {written} sessions, {rows:,} rows")
    print(f"archive now holds {len(fa.available_days())} sessions")


if __name__ == "__main__":
    main()
