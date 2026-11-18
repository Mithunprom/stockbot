"""H43: Multi-horizon IC analysis on clean (pipeline-v3) features.

MOTIVATION
----------
The Nov 16 2026 clean retrain (commit 6501c5e) confirmed that pipeline-v3 features
achieve OOS IC = 0.0078 at F15 — roughly 4× below the 2bps round-trip break-even
threshold (requires IC ≈ 0.034 at F15 sigma = 0.593%).

Three questions remain:
  1. Does the current v3 model (trained on F15) have *any* predictive power at
     longer horizons — i.e., is IC × sigma > 2bps at F30/F60/F390?
  2. If we retrain on a longer-horizon target (F60, F390), does IC improve enough
     to close the edge/cost gap?
  3. What is the MINIMUM IC needed to break even at each horizon?

This script answers all three and reports a go / no-go for each horizon, so the
desk does not need to guess.

VERDICT RULE (Index/Risk Quant veto)
-------------------------------------
  edge_bps  = IC * sigma_bps     (sigma = cross-sectional std of forward return)
  cost_bps  = 2.0                 (round-trip; Alpaca paper-mode spread estimate)
  viable    = edge_bps > cost_bps

A horizon is marked VIABLE only if edge_bps > 2.0 AND n >= 10_000 (noise floor).
Anything else is REJECTED — negative results are results.

USAGE
------
  Requires the research cache (bars + pre-built features from pipeline v3):
      RESEARCH_CACHE=/data/research_cache PYTHONPATH=. python scripts/horizon_ic_clean.py

  Optional flags:
      --train-end   YYYY-MM-DD   last bar of training window (default 2026-09-15)
      --val-end     YYYY-MM-DD   end of validation window   (default 2026-09-30)
      --oos-start   YYYY-MM-DD   start of OOS window        (default 2026-10-01)
      --oos-end     YYYY-MM-DD   end of OOS window           (default 2026-11-01)
      --horizons    int ...       forward bar counts to test (default: 5 15 30 60 390)

The script writes results to reports/research/horizon_ic_clean_<date>.json and
prints a formatted verdict table.

NOTE: marks status as needs_data_run in agent_state.json — set
AGENT_WORKER_ENABLE=true on Railway to execute.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COST_BPS = 2.0  # round-trip cost assumption (bps)
MIN_N = 10_000  # minimum cross-sectional observations for a verdict


def _spearman_ic(pred: np.ndarray, fwd: np.ndarray) -> float:
    """Cross-sectional Spearman rank IC, ignoring NaN pairs."""
    mask = np.isfinite(pred) & np.isfinite(fwd)
    if mask.sum() < 30:
        return float("nan")
    r, _ = spearmanr(pred[mask], fwd[mask])
    return float(r)


def compute_forward_returns(
    bars: dict[str, pd.DataFrame],
    horizons: list[int],
) -> pd.DataFrame:
    """Return a DataFrame with columns (ticker, bar_time, fwd_{h}) for each horizon."""
    rows: list[dict[str, Any]] = []
    for ticker, df in bars.items():
        close = df["close"]
        for h in horizons:
            fwd = close.shift(-h) / close - 1.0
            tmp = fwd.reset_index()
            tmp.columns = ["bar_time", f"fwd_{h}"]
            tmp["ticker"] = ticker
            rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    out = rows[0][["ticker", "bar_time", f"fwd_{horizons[0]}"]].copy()
    for i, h in enumerate(horizons):
        col = f"fwd_{h}"
        merged = pd.concat([r[["ticker", "bar_time", col]] for r in rows if col in r.columns])
        if i == 0:
            out = merged
        else:
            out = out.merge(merged, on=["ticker", "bar_time"], how="outer")
    return out


def run_ic_curve(
    pred_df: pd.DataFrame,  # columns: ticker, bar_time, pred_return
    fwd_df: pd.DataFrame,   # columns: ticker, bar_time, fwd_{h}...
    horizons: list[int],
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
) -> dict[int, dict[str, float]]:
    """Compute IC, sigma, edge_bps, cost_bps per horizon over the OOS window."""
    merged = pred_df.merge(fwd_df, on=["ticker", "bar_time"])
    merged = merged[
        (merged["bar_time"] >= oos_start) & (merged["bar_time"] < oos_end)
    ]
    results: dict[int, dict[str, float]] = {}
    for h in horizons:
        col = f"fwd_{h}"
        if col not in merged.columns:
            continue
        sub = merged[["pred_return", col]].dropna()
        n = len(sub)
        if n < MIN_N:
            results[h] = {"ic": float("nan"), "sigma_bps": float("nan"),
                          "edge_bps": float("nan"), "n": n, "verdict": "INSUFFICIENT_DATA"}
            continue
        ic = _spearman_ic(sub["pred_return"].values, sub[col].values)
        sigma_bps = float(sub[col].std() * 10_000)  # fractional → bps
        edge_bps = ic * sigma_bps
        viable = (edge_bps > COST_BPS) and (n >= MIN_N)
        results[h] = {
            "ic": round(ic, 6),
            "sigma_bps": round(sigma_bps, 3),
            "edge_bps": round(edge_bps, 4),
            "cost_bps": COST_BPS,
            "edge_to_cost_ratio": round(edge_bps / COST_BPS, 4),
            "breakeven_ic": round(COST_BPS / max(sigma_bps, 1e-9), 6),
            "n": n,
            "verdict": "VIABLE" if viable else "REJECTED",
        }
    return results


def train_lgbm_on_horizon(
    feats_df: pd.DataFrame,  # (bar_time, ticker) × feature_cols
    fwd_df: pd.DataFrame,
    horizon: int,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    feature_cols: list[str],
) -> "lgbm.LGBMRegressor":
    """Fit a fresh LightGBM on forward-return target for the given horizon."""
    import lightgbm as lgbm

    col = f"fwd_{horizon}"
    merged = feats_df.merge(fwd_df[["ticker", "bar_time", col]], on=["ticker", "bar_time"])
    train = merged[merged["bar_time"] < train_end].dropna(subset=[col] + feature_cols)
    val = merged[
        (merged["bar_time"] >= train_end) & (merged["bar_time"] < val_end)
    ].dropna(subset=[col] + feature_cols)

    model = lgbm.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        train[feature_cols], train[col],
        eval_set=[(val[feature_cols], val[col])],
        callbacks=[lgbm.early_stopping(50, verbose=False), lgbm.log_evaluation(-1)],
    )
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-end", default="2026-09-15")
    ap.add_argument("--val-end", default="2026-09-30")
    ap.add_argument("--oos-start", default="2026-10-01")
    ap.add_argument("--oos-end", default="2026-11-01")
    ap.add_argument(
        "--horizons", type=int, nargs="+", default=[5, 15, 30, 60, 120, 390],
        help="forward bar counts (1-min bars) to evaluate",
    )
    ap.add_argument(
        "--retrain", action="store_true",
        help="also train a fresh model on each horizon target (not just F15 preds)",
    )
    args = ap.parse_args()

    # +1 day so the named date is inclusive: splits use exclusive < at midnight UTC
    train_end = pd.Timestamp(args.train_end, tz="UTC") + pd.Timedelta(days=1)
    val_end = pd.Timestamp(args.val_end, tz="UTC") + pd.Timedelta(days=1)
    oos_start = pd.Timestamp(args.oos_start, tz="UTC")
    oos_end = pd.Timestamp(args.oos_end, tz="UTC")

    # ------------------------------------------------------------------ load
    from scripts.exit_geometry_study import load_cache  # type: ignore[import]
    bars, feats = load_cache()
    if not bars:
        raise SystemExit("empty cache — run: python scripts/research_backtest.py --phase fetch")

    print(f"Loaded {len(bars)} tickers")

    # ------------------------------------------------------------------ features
    from src.features.indicators import compute_indicators_for_universe  # noqa: E402

    print("Computing v3 features (look-ahead free)...")
    feat_universe = compute_indicators_for_universe(bars, shift=True)

    # Flatten to a single DataFrame with (ticker, bar_time, feature_cols)
    feat_rows: list[pd.DataFrame] = []
    for ticker, df in feat_universe.items():
        tmp = df.reset_index().rename(columns={"index": "bar_time", "timestamp": "bar_time"})
        tmp["ticker"] = ticker
        feat_rows.append(tmp)
    feats_df = pd.concat(feat_rows, ignore_index=True)

    # Identify feature columns (all numeric columns except ticker/bar_time)
    exclude = {"ticker", "bar_time", "open", "high", "low", "close", "volume"}
    feature_cols = [c for c in feats_df.columns if c not in exclude and
                    pd.api.types.is_numeric_dtype(feats_df[c])]
    print(f"Feature columns: {len(feature_cols)}")

    # ------------------------------------------------------------------ forward returns
    print("Computing forward returns...")
    fwd_df = compute_forward_returns(bars, args.horizons)

    # ------------------------------------------------------------------ F15 model preds
    from src.models.lgbm import LGBMSignalModel  # noqa: E402

    print("Loading v3 quarantined model for F15 predictions...")
    quarantine = Path("models/lgbm/quarantine/lgbm_v3_clean.json")
    if not quarantine.exists():
        raise SystemExit(
            "Quarantined v3 model not found — run the Nov 16 clean retrain first"
        )
    # Load via LGBMSignalModel so we get the proper regressor, not the raw pickle dict
    loaded_model = LGBMSignalModel.load(str(quarantine))
    model_feature_cols: list[str] = loaded_model.feature_cols

    # Generate predictions on all OOS bars
    sub = feats_df[feats_df["bar_time"] >= oos_start].dropna(subset=model_feature_cols)
    if sub.empty:
        raise SystemExit("No OOS bars found — check --oos-start")

    sub = sub.copy()
    sub["pred_return"] = loaded_model.regressor.predict(sub[model_feature_cols])
    pred_df = sub[["ticker", "bar_time", "pred_return"]]

    # ------------------------------------------------------------------ IC curve (F15 model)
    print("\n=== IC curve using F15-trained model predictions ===")
    ic_results = run_ic_curve(pred_df, fwd_df, args.horizons, oos_start, oos_end)

    header = f"{'Horizon':>10} {'IC':>10} {'sigma(bps)':>12} {'edge(bps)':>11} {'cost(bps)':>11} {'edge/cost':>10} {'n':>8} {'verdict':>10}"
    print(header)
    print("-" * len(header))
    for h in args.horizons:
        r = ic_results.get(h, {})
        print(
            f"{h:>10}  {r.get('ic', float('nan')):>10.5f}  "
            f"{r.get('sigma_bps', float('nan')):>10.2f}  "
            f"{r.get('edge_bps', float('nan')):>10.4f}  "
            f"{r.get('cost_bps', COST_BPS):>10.1f}  "
            f"{r.get('edge_to_cost_ratio', float('nan')):>10.4f}  "
            f"{r.get('n', 0):>8,}  "
            f"{r.get('verdict', 'N/A'):>10}"
        )

    viable_horizons = [h for h, r in ic_results.items() if r.get("verdict") == "VIABLE"]
    if viable_horizons:
        print(f"\n*** VIABLE horizons (edge > cost): {viable_horizons} ***")
        print("RECOMMENDATION: retrain on the shortest viable horizon.")
    else:
        print("\n*** NO viable horizon found with F15-trained model. ***")
        print("RECOMMENDATION: new alpha sources required (see H43 backlog).")

    # ------------------------------------------------------------------ retrain per horizon
    retrain_results: dict[int, dict[str, float]] = {}
    if args.retrain:
        print("\n=== Retrain IC — fresh model per horizon target ===")
        for h in args.horizons:
            print(f"  Training F{h} model...")
            try:
                model_h = train_lgbm_on_horizon(
                    feats_df, fwd_df, h, train_end, val_end, model_feature_cols
                )
                sub_h = feats_df[feats_df["bar_time"] >= oos_start].dropna(subset=model_feature_cols).copy()
                sub_h["pred_return"] = model_h.predict(sub_h[model_feature_cols])
                pred_df_h = sub_h[["ticker", "bar_time", "pred_return"]]
                r = run_ic_curve(pred_df_h, fwd_df, [h], oos_start, oos_end).get(h, {})
                retrain_results[h] = r
                print(
                    f"  F{h}: IC={r.get('ic', float('nan')):.5f}  "
                    f"edge={r.get('edge_bps', float('nan')):.4f}bps  "
                    f"verdict={r.get('verdict', 'N/A')}"
                )
            except Exception as exc:
                print(f"  F{h}: ERROR — {exc}")
                retrain_results[h] = {"verdict": "ERROR", "error": str(exc)}

    # ------------------------------------------------------------------ save report
    report = {
        "run_date": str(date.today()),
        "train_end": str(train_end.date()),
        "val_end": str(val_end.date()),
        "oos_start": str(oos_start.date()),
        "oos_end": str(oos_end.date()),
        "horizons_tested": args.horizons,
        "cost_bps": COST_BPS,
        "f15_model_ic_curve": ic_results,
        "retrain_ic_curve": retrain_results if args.retrain else "not_run",
        "viable_horizons_f15_model": viable_horizons,
        "conclusion": (
            f"VIABLE at F{viable_horizons[0]}" if viable_horizons
            else "NO_VIABLE_HORIZON — new alpha sources required"
        ),
    }
    out_path = Path("reports/research") / f"horizon_ic_clean_{date.today()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
