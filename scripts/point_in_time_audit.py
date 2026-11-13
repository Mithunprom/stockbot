"""Does any feature at time t use data from after t? Per-feature, decisively.

Borrowed from TauricResearch/TradingAgents' point-in-time discipline — it
enforces "point-in-time integrity across every dated path" and scores backtests
"from the first close after each decision". StockBot has the same requirement
and no harness for it.

WHAT THIS IS NOT
----------------
`sim_live_ic_gap.py` varies how much PAST is available (a trailing warmup
window). That measured flat: rank agreement ~0.83 at 1950 / 3900 / 5850 / 7800
bars, so insufficient history is NOT the cause of the serving gap.

This varies whether the FUTURE is available, which is a different failure and
was never tested. For each scored bar t:

    truncated = bars[:t]        <- everything knowable at t, nothing after
    full      = bars[:end]      <- what research/backtest actually computes on

Both are passed through the SAME `compute_indicators_for_universe`, and the row
AT t is compared. Any disagreement means the research path consumed data from
after t to compute the feature at t. That is look-ahead, and it would inflate
the backtest while being structurally impossible in production — which is the
exact shape of a +0.18 research IC against +0.025 live.

WHY IT CANNOT BE FOOLED
-----------------------
No model, no P&L, no train/validation split. The comparison is a float
difference on identical inputs, and the correct answer is exactly zero. There is
nothing here to overfit, unlike any strategy-level search scored by an evaluator
that is currently ~7x optimistic.

THE OPEN QUESTION THIS SETTLES
------------------------------
H42 names 7 of 30 features that diverge between research and serving:
mtf_trend_15m, mtf_rsi_15m, mtf_confluence, mtf_trend_5m (resample + Wilder EWM)
and orb_dev, orb_vpin_interact, orb_range_atr (session-anchored opening range).
Five are in FFSA's top five. A spot check of `orb_high` on three 09:40 ET bars
found no look-ahead, but three bars is not an answer — the session high often
lands after an early-session bar, which is exactly when
`groupby(date).transform("max")` would leak.

Usage:
    RESEARCH_CACHE=data/research_cache PYTHONPATH=. python scripts/point_in_time_audit.py
    ... --since 2026-10-01 --every 150 --tickers 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exit_geometry_study import load_cache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-01")
    ap.add_argument("--every", type=int, default=150,
                    help="sample one scored bar per N bars")
    ap.add_argument("--tickers", type=int, default=8,
                    help="universe size; the cross-section must stay intact")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    from src.features.indicators import compute_indicators_for_universe
    from src.models.lgbm import LGBMSignalModel

    model = LGBMSignalModel.load()
    cols = list(model.feature_cols)

    bars, _ = load_cache()
    if not bars:
        raise SystemExit("empty cache — see scripts/research_backtest.py --phase fetch")
    keep = sorted(bars)[: args.tickers]
    bars = {t: bars[t] for t in keep}

    since = pd.Timestamp(args.since, tz="UTC")
    common = sorted(set.intersection(
        *[set(b.index[b.index >= since]) for b in bars.values()]))
    sample = common[:: args.every]
    print(f"universe={len(bars)}  scored bars={len(sample)}  features={len(cols)}")
    print(f"comparing feature(t | data<=t)  vs  feature(t | full history)\n")

    full = compute_indicators_for_universe(bars, shift=True)

    diffs: dict[str, list[float]] = {c: [] for c in cols}
    n_rows = 0
    for ts in sample:
        trunc = {t: b[b.index <= ts] for t, b in bars.items()}
        if any(len(v) < 400 for v in trunc.values()):
            continue
        pit = compute_indicators_for_universe(trunc, shift=True)
        for t in bars:
            if t not in pit or ts not in pit[t].index or ts not in full[t].index:
                continue
            a = full[t].loc[ts, cols].astype(float)
            b = pit[t].loc[ts, cols].astype(float)
            n_rows += 1
            for c in cols:
                x, y = a[c], b[c]
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                scale = max(abs(x), abs(y), 1e-9)
                diffs[c].append(abs(x - y) / scale)

    if not n_rows:
        raise SystemExit("no comparable rows — widen --since or lower --every")

    rows = []
    for c in cols:
        d = diffs[c]
        if not d:
            rows.append((c, float("nan"), 0, 0.0))
            continue
        arr = np.array(d)
        rows.append((c, float(arr.max()), int((arr > args.tol).sum()), float(arr.mean())))
    rows.sort(key=lambda r: -(r[1] if r[1] == r[1] else -1))

    print(f"{'feature':26s} {'max rel diff':>13s} {'rows leaking':>13s} {'mean':>10s}")
    leaking = []
    for c, mx, nbad, mean in rows:
        flag = ""
        if mx == mx and mx > args.tol:
            flag = "  <-- LOOK-AHEAD"
            leaking.append(c)
        print(f"{c:26s} {mx:>13.3e} {nbad:>13d} {mean:>10.3e}{flag}")

    print(f"\ncompared {n_rows} (ticker, bar) rows")
    if leaking:
        print(f"\n{len(leaking)} of {len(cols)} features USE FUTURE DATA:")
        for c in leaking:
            print(f"   {c}")
        print("\nThese inflate the backtest and cannot be reproduced in serving.")
    else:
        print("\nNO LOOK-AHEAD. Every feature at t depends only on data <= t.")
        print("The research/serving gap is therefore NOT the backtest cheating —")
        print("look to how serving computes these features, not to what it can see.")

    out = "reports/research/point_in_time_audit.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows, columns=["feature", "max_rel_diff", "rows_leaking", "mean_rel_diff"]
                 ).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
