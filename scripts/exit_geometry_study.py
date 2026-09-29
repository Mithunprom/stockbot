"""Choose the exit geometry (profit target, stop, trail) from real minute paths.

Why this exists
---------------
v0.8.0 shipped `EXIT_PROFIT_TARGET_MODE` (env-gated, default OFF) with
`STOP_TO_TARGET_RATIO = 0.8` — reward:risk 1.25, which needs a 44.4% win rate
merely to break even before costs. Production has never sustained that: 33.3%
over the M2 window, 27.3% over the first post-fix session. So the geometry would
lose by construction, and the flag must not be enabled until the ratio is set
from evidence.

Two obvious evidence sources are both unusable, which is why this script exists:

  * the research backtest's PF — sim stopped reproducing production in Sep 2026
    (7.27-point PF gap) and fidelity has not been re-established;
  * the live M2 ledger — every one of those trades was selected at the ~46th
    percentile by the train/serve skew bug, so their excursions describe a
    strategy the bot no longer runs. The S3 feature archive has the same defect:
    `src/features/live.py` writes `feature_matrix`, so archived rows before
    2026-09-21 carry v1 (skewed) feature semantics.

So this rebuilds the inputs the honest way — full-history v2 features from real
Alpaca 1m bars, exactly as training computes them — and then asks the only
question that needs no simulator: given an entry the FIXED model ranks in its
own top decile, which barrier does price touch FIRST?

No PnL model and no fill assumptions beyond a cost haircut. Output is expectancy
per trade across a (target, stop) grid, so the geometry is chosen on measured
paths rather than on a ratio someone liked the look of.

Usage:
    RESEARCH_CACHE=/tmp/sb_exitgeom python scripts/exit_geometry_study.py
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(os.environ.get("RESEARCH_CACHE", "/tmp/sb_exitgeom"))
SESSION_BARS = 390          # 09:30-15:59 ET; the session backstop
COST_BPS = 2.0              # round-trip haircut, matches research_backtest


def load_cache() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load per-ticker OHLCV bars and full-history v2 features from the cache."""
    bars, feats = {}, {}
    for p in sorted(CACHE.glob("bars_*.csv.gz")):
        t = p.name[len("bars_"):-len(".csv.gz")]
        fp = CACHE / f"feat_{t}.csv.gz"
        if not fp.exists():
            continue
        b = pd.read_csv(p, parse_dates=["timestamp"]).set_index("timestamp")
        b.index = pd.to_datetime(b.index, utc=True)
        f = pd.read_csv(fp, index_col=0)
        f.index = pd.to_datetime(f.index, utc=True)
        bars[t], feats[t] = b.sort_index(), f.sort_index()
    return bars, feats


def build_entries(feats, model, percentile: float) -> pd.DataFrame:
    """Score every bar, then keep the top `percentile` of each bar's cross-section.

    Ranking WITHIN a timestamp is the point: production's `entry_rank_mean` is a
    cross-sectional percentile (currently 96.9), so entry selection has to be
    reproduced the same way or the study describes a different strategy.
    """
    cols = list(model.feature_cols)
    frames = []
    for t, f in feats.items():
        missing = [c for c in cols if c not in f.columns]
        if missing:
            raise SystemExit(f"{t}: features missing {missing[:6]}")
        X = f[cols].astype(float)
        ok = X.notna().all(axis=1)
        if not ok.any():
            continue
        d = pd.DataFrame(index=f.index[ok])
        d["ticker"] = t
        d["pred"] = model.regressor.predict(X[ok])
        frames.append(d)

    allp = pd.concat(frames).reset_index(names="timestamp")
    rank = allp.groupby("timestamp")["pred"].rank(pct=True) * 100
    # pred > 0 mirrors the live long-only entry gate.
    return allp[(rank >= percentile) & (allp["pred"] > 0)].copy()


def first_touch(o, h, l, c, entry, tp, sl, trail):
    """Bar-by-bar first touch over the session. Returns (return, reason).

    When a single bar's range spans both barriers the STOP is taken first. That
    is the pessimistic assumption and it is applied identically across the grid,
    so it cannot flatter one geometry over another.
    """
    peak = entry
    for i in range(len(c)):
        if l[i] / entry - 1.0 <= -sl:
            return -sl, "stop"
        if h[i] / entry - 1.0 >= tp:
            return tp, "target"
        peak = max(peak, h[i])
        if trail is not None and peak > entry:
            if l[i] / peak - 1.0 <= -trail:
                return max(l[i], peak * (1 - trail)) / entry - 1.0, "trail"
    return (c[-1] / entry - 1.0) if len(c) else 0.0, "session_close"


def run_grid(entries, bars, targets, stop_ratios, trail_ratio):
    cost = COST_BPS / 10_000.0
    # Pre-slice each entry's forward path once; the grid then reuses them.
    paths = []
    for t, ts in entries[["ticker", "timestamp"]].itertuples(index=False):
        b = bars[t]
        fwd = b.loc[b.index > ts]
        # Stay inside the entry's own session — the backstop never carries over.
        fwd = fwd[fwd.index.date == ts.date()].head(SESSION_BARS)
        if len(fwd) < 10:
            continue
        entry_px = b.loc[ts, "close"] if ts in b.index else None
        if entry_px is None or not np.isfinite(entry_px):
            continue
        paths.append((
            float(entry_px),
            fwd["open"].to_numpy(), fwd["high"].to_numpy(),
            fwd["low"].to_numpy(), fwd["close"].to_numpy(),
        ))

    rows = []
    for tgt in targets:
        for sr in stop_ratios:
            res, reasons = [], defaultdict(int)
            trail = tgt * trail_ratio if trail_ratio else None
            for entry_px, o, h, l, c in paths:
                r, why = first_touch(o, h, l, c, entry_px, tgt, tgt * sr, trail)
                res.append(r - cost)
                reasons[why] += 1
            a = np.array(res)
            if not len(a):
                continue
            wins, losses = a[a > 0], a[a <= 0]
            rows.append({
                "target_pct": round(100 * tgt, 2),
                "stop_ratio": sr,
                "RR": round(1 / sr, 2),
                "n": len(a),
                "win_rate": round(100 * len(wins) / len(a), 1),
                "exp_bps": round(10_000 * a.mean(), 2),
                "PF": round(wins.sum() / -losses.sum(), 2) if len(losses) else np.inf,
                "tgt": reasons["target"], "stop": reasons["stop"],
                "trail": reasons["trail"], "close": reasons["session_close"],
            })
    return pd.DataFrame(rows).sort_values("exp_bps", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--since", default=None,
                    help="drop entries before this date (UTC) — use to exclude "
                         "the model's training span and read OOS only")
    ap.add_argument("--checkpoint", default="models/lgbm/lgbm_ic_0.1830.pkl")
    ap.add_argument("--trail-ratio", type=float, default=None)
    ap.add_argument("--out", default="reports/research/exit_geometry.csv")
    args = ap.parse_args()

    from src.models.lgbm import LGBMSignalModel

    bars, feats = load_cache()
    print(f"cache {CACHE}: {len(bars)} tickers")
    if not bars:
        raise SystemExit("no cache — run research_backtest.py --phase fetch/features")

    model = LGBMSignalModel.load(Path(args.checkpoint))
    print(f"model {args.checkpoint} val_ic={model.val_ic:.4f}")

    entries = build_entries(feats, model, args.percentile)
    if args.since:
        entries = entries[entries["timestamp"] >= pd.Timestamp(args.since, tz="UTC")]
    print(f"entries at >= p{args.percentile:g}"
          f"{' since ' + args.since if args.since else ''}: {len(entries):,}")

    grid = run_grid(
        entries, bars,
        targets=[0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.025],
        stop_ratios=[0.25, 0.33, 0.40, 0.50, 0.65, 0.80, 1.00],
        trail_ratio=args.trail_ratio,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    grid.to_csv(args.out, index=False)
    print(f"\n{grid.head(25).to_string(index=False)}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
