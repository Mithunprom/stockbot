"""H44 — per-feature clean IC audit on pipeline-v3 data.

The v0.10.0 OOS measurement showed LightGBM IC = 0.0078 on 30 features, far
below the 2 bps round-trip cost hurdle.  Two competing explanations:

  (A) Features individually have alpha but LightGBM overfits their combination.
  (B) No individual feature has alpha against 15-bar returns at 2 bps costs.

This script settles it by measuring Spearman IC for EVERY computed feature
(not just the 30 in the model) against k-bar forward returns.

If (A): some features will show IC > COST_HURDLE; model architecture needs to
  change (simpler model, stronger regularisation, or rule-based signal on the
  winning feature(s)).
If (B): the signal is simply not in 1-min price/volume data for large caps at
  2 bps costs.  The desk must explore alternative data or universe.

Usage (Railway worker):
    python scripts/feature_ic_audit.py \
        --archive-dir data/features \
        --start-date 2026-10-01 \
        --end-date   2026-10-27 \
        --horizons   5,15,30,60 \
        --out        reports/research/feature_ic_audit_v3.json

Sandboxes (unit tests) pass a small synthetic DataFrame directly via
compute_ic_table(); no archive I/O required.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Cost hurdle: IC × sigma(horizon) must exceed round-trip cost
# ---------------------------------------------------------------------------
ROUND_TRIP_COST_BPS = float(os.environ.get("ROUND_TRIP_COST_BPS", "2.0"))
FORWARD_SIGMA_15BAR_BPS = float(os.environ.get("FORWARD_SIGMA_15BAR_BPS", "59.26"))  # 0.5926%


def _spearman_ic(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation between x and y after dropping NaNs."""
    mask = x.notna() & y.notna()
    n = mask.sum()
    if n < 30:
        return float("nan")
    xr = x[mask].rank()
    yr = y[mask].rank()
    return float(xr.corr(yr))


def _bootstrap_std(x: pd.Series, y: pd.Series, n_boot: int = 500, seed: int = 42) -> float:
    """Bootstrap standard deviation of Spearman IC estimate."""
    rng = np.random.default_rng(seed)
    mask = x.notna() & y.notna()
    xm, ym = x[mask].values, y[mask].values
    n = len(xm)
    if n < 30:
        return float("nan")
    ics = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xr = pd.Series(xm[idx]).rank()
        yr = pd.Series(ym[idx]).rank()
        ics.append(float(xr.corr(yr)))
    return float(np.std(ics))


def compute_ic_table(
    df: pd.DataFrame,
    horizons: Sequence[int] = (5, 15, 30, 60),
    n_boot: int = 200,
) -> pd.DataFrame:
    """Compute per-feature IC table.

    Args:
        df: DataFrame with feature columns AND a 'close' column.  Rows are
            individual (ticker, bar) observations — stack multiple tickers
            before calling.  close must be the RAW (non-shifted) price so
            forward returns can be computed here.
        horizons: list of forward-bar horizons to test.
        n_boot: bootstrap resamples for IC std estimate.

    Returns:
        DataFrame indexed by feature name with columns:
          ic_{h}      — Spearman IC at horizon h
          ic_std_{h}  — bootstrap std of IC at horizon h
          t_stat_{h}  — IC / ic_std (approximate t-statistic)
          n_{h}       — sample size used
          cost_multiple_{h} — |IC| × sigma(h) / cost_bps  (>1.0 clears hurdle)
    """
    if "close" not in df.columns:
        raise ValueError("df must contain a 'close' column for forward return computation")

    result_rows: dict[str, dict] = {}

    # Feature candidates: every column that is numeric, not OHLCV metadata, not a
    # forward return we are about to compute
    ohlcv_cols = {"open", "high", "low", "close", "volume"}
    skip_patterns = ("forward_ret_",)

    feature_cols = [
        c for c in df.columns
        if c not in ohlcv_cols
        and not any(c.startswith(p) for p in skip_patterns)
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    for h in horizons:
        # Annualise sigma: sigma(h) ≈ sigma(15) × sqrt(h/15)
        sigma_bps = FORWARD_SIGMA_15BAR_BPS * np.sqrt(h / 15.0)
        fwd_key = f"forward_ret_{h}"
        df[fwd_key] = df["close"].pct_change(h).shift(-h) * 10_000  # in bps

        for feat in feature_cols:
            row = result_rows.setdefault(feat, {})
            ic = _spearman_ic(df[feat], df[fwd_key])
            std = _bootstrap_std(df[feat], df[fwd_key], n_boot=n_boot)
            n = int((df[feat].notna() & df[fwd_key].notna()).sum())
            t = (ic / std) if (std > 0 and not np.isnan(ic)) else float("nan")
            cost_mult = (abs(ic) * sigma_bps / ROUND_TRIP_COST_BPS) if not np.isnan(ic) else float("nan")
            row[f"ic_{h}"] = round(ic, 6) if not np.isnan(ic) else None
            row[f"ic_std_{h}"] = round(std, 6) if not np.isnan(std) else None
            row[f"t_stat_{h}"] = round(t, 3) if not np.isnan(t) else None
            row[f"n_{h}"] = n
            row[f"cost_multiple_{h}"] = round(cost_mult, 4) if not np.isnan(cost_mult) else None

        # Clean up temporary column
        df.drop(columns=[fwd_key], inplace=True, errors="ignore")

    ic_df = pd.DataFrame(result_rows).T
    ic_df.index.name = "feature"

    # Sort by absolute IC at the primary horizon (15 bars if present, else first)
    primary_h = 15 if 15 in horizons else horizons[0]
    sort_col = f"ic_{primary_h}"
    if sort_col in ic_df.columns:
        ic_df = ic_df.reindex(
            ic_df[sort_col].abs().fillna(0).sort_values(ascending=False).index
        )

    return ic_df


def _load_archive(archive_dir: Path, start: str, end: str) -> pd.DataFrame:
    """Load parquet feature archive files for the given date range.

    Expects files named like ``{archive_dir}/features_YYYYMMDD.parquet`` or
    ``{archive_dir}/{YYYY-MM-DD}/features.parquet``.  Stacks all tickers into
    a single DataFrame with a (ticker, timestamp) MultiIndex.
    """
    frames = []
    for p in sorted(archive_dir.glob("**/*.parquet")):
        # Extract date from path: either filename or parent directory name
        date_str = p.stem.replace("features_", "").replace("_", "-")
        if date_str < start or date_str > end:
            # Also try parent directory name
            parent_date = p.parent.name
            if parent_date < start or parent_date > end:
                continue
        try:
            frame = pd.read_parquet(p)
            frames.append(frame)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not read {p}: {exc}", file=sys.stderr)

    if not frames:
        raise FileNotFoundError(
            f"No parquet files found in {archive_dir} for {start}..{end}.  "
            "Run on Railway worker where the feature archive is populated."
        )

    combined = pd.concat(frames, ignore_index=False)
    print(f"Loaded {len(combined):,} rows from {len(frames)} archive files", file=sys.stderr)
    return combined


def _build_report(ic_df: pd.DataFrame, meta: dict) -> dict:
    """Serialise the IC table as a JSON-able report dict."""
    primary_h = meta.get("primary_horizon_bars", 15)
    cost_col = f"cost_multiple_{primary_h}"
    ic_col = f"ic_{primary_h}"

    above_hurdle = []
    below_hurdle = []
    for feat, row in ic_df.iterrows():
        cm = row.get(cost_col)
        ic = row.get(ic_col)
        entry = {"feature": feat, "ic": ic, "cost_multiple": cm}
        if cm is not None and cm >= 1.0:
            above_hurdle.append(entry)
        else:
            below_hurdle.append(entry)

    n_above = len(above_hurdle)
    n_total = len(ic_df)
    summary = (
        f"{n_above}/{n_total} features clear the {ROUND_TRIP_COST_BPS} bps cost hurdle "
        f"at {primary_h}-bar horizon."
    )
    if n_above == 0:
        conclusion = "EXPLANATION_B: no individual feature clears costs. Need alternative data or universe."
    else:
        conclusion = (
            "EXPLANATION_A: individual features have alpha; LightGBM generalisation failure. "
            "Consider simpler model or rule-based signal on the winning features."
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "summary": summary,
        "conclusion": conclusion,
        "features_above_hurdle": above_hurdle,
        "features_below_hurdle": below_hurdle[:20],  # top-20 near-misses
        "full_table": ic_df.reset_index().to_dict(orient="records"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Per-feature clean IC audit (H44)")
    parser.add_argument("--archive-dir", default="data/features",
                        help="Directory containing parquet feature archive files")
    parser.add_argument("--start-date", default="2026-10-01",
                        help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2026-10-27",
                        help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--horizons", default="5,15,30,60",
                        help="Comma-separated forward bar horizons to test")
    parser.add_argument("--n-boot", type=int, default=500,
                        help="Bootstrap resamples for IC std estimate")
    parser.add_argument("--out", default="reports/research/feature_ic_audit_v3.json",
                        help="Output JSON path")
    args = parser.parse_args(argv)

    horizons = [int(h) for h in args.horizons.split(",")]
    archive_dir = Path(args.archive_dir)

    print(f"Loading feature archive {archive_dir} [{args.start_date}..{args.end_date}]")
    df = _load_archive(archive_dir, args.start_date, args.end_date)

    print(f"Computing IC for {len(df.columns)} columns × {len(horizons)} horizons …")
    ic_df = compute_ic_table(df, horizons=horizons, n_boot=args.n_boot)

    primary_h = 15 if 15 in horizons else horizons[0]
    meta = {
        "hypothesis": "H44",
        "archive_dir": str(archive_dir),
        "date_range": f"{args.start_date}..{args.end_date}",
        "n_rows": len(df),
        "horizons_bars": horizons,
        "primary_horizon_bars": primary_h,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "forward_sigma_15bar_bps": FORWARD_SIGMA_15BAR_BPS,
        "feature_pipeline_version": 3,
        "note": (
            "October 2026 held-out window, never seen during H42 training. "
            "IC is Spearman rank correlation.  cost_multiple = |IC|×sigma(h)/cost_bps; "
            ">1.0 means the feature's expected return exceeds round-trip costs."
        ),
    }

    report = _build_report(ic_df, meta)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{report['summary']}")
    print(f"Conclusion: {report['conclusion']}")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
