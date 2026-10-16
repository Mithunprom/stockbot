"""Set PROFIT_TARGET_ANNUAL_VOL_FRAC from real minute paths, not from four trades.

Why this exists
---------------
Production runs `EXIT_PROFIT_TARGET_MODE` with the target at 5% of ANNUAL vol.
On 2026-10-15 — the first clean session after the v0.9.4 wall-clock backstop —
all four trades exited `session_close` and **0 of 4 targets were reachable**:

    SNOW  daily_vol 4.42%   target 3.50%   realized +3.30%
    MU    daily_vol 5.05%   target 4.01%   realized +1.78%
    STX   daily_vol 7.87%   target 6.25%   realized +1.93%
    SKYD  daily_vol 10.67%  target 8.47%   realized +1.12%

`signal_loop._check_sizing_exit` says plainly that a backstop closing most
trades means the target is too far away. But the naive fix is a trap: at a
reachable target SNOW would have capped at ~1.77% instead of +3.30%, turning
$290 into ~$156. Tightening would have made that session WORSE.

n=4 cannot settle it, and the live ledger cannot either — the M2 trades were
selected at the ~46th percentile by the train/serve skew bug, so their
excursions describe a strategy the bot no longer runs.

So this sweeps the target over real minute paths, reusing the machinery of
`exit_mode_ab.py` (same cache, same cross-sectional entry selection, same
first-touch walk) and varying only ONE parameter. Barriers come from the live
`_atr_exits` with the module constant monkeypatched, never reimplemented, so the
sweep cannot flatter a value the production code would not actually produce.

THE SUSPECTED DEFECT, which the sweep is designed to expose
-----------------------------------------------------------
The target is a fraction of annual vol derived from DAILY ATR, which includes
overnight gaps. Since v0.9.4 the position is strictly intraday and can never
capture the gap component, so the target is systematically overstated — and
overstated MORE for volatile names (SKYD 8.47% vs AAPL ~1.6%). If that is the
story, expectancy should improve as the fraction falls and then turn over once
the target starts truncating real winners. The sweep reports the turning point
and the exit mix at each step, so the trade-off is visible rather than asserted.

Reward:risk is held at the production 0.8 stop-to-target ratio throughout, so
this measures target DISTANCE only — the ratio itself was separately cleared in
September (OOS PF 1.62 at top-decile entries) and is not under test here.

Usage:
    RESEARCH_CACHE=/path/to/cache PYTHONPATH=. python scripts/profit_target_sweep.py
    ... --fracs 0.01,0.02,0.03,0.05,0.08 --since 2026-07-15 --percentile 90
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Bare sibling imports, matching exit_mode_ab.py. These modules import each
# other by bare name, so `scripts/` has to be importable — `from scripts.x`
# fails on the inner `from exit_geometry_study import ...`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exit_geometry_study import (  # noqa: E402  (same scripts/ dir)
    COST_BPS, build_entries, load_cache,
)
from exit_mode_ab import daily_vol_map, walk  # noqa: E402

DEFAULT_FRACS = "0.010,0.015,0.020,0.025,0.030,0.040,0.050,0.075,0.100"


def summarise(label: str, res: list[float], reasons: dict[str, int]) -> dict:
    a = np.array(res)
    wins, losses = a[a > 0], a[a <= 0]
    n = len(a)
    return {
        "target_frac": label,
        "n": n,
        "win_rate": round(100 * len(wins) / n, 1) if n else 0.0,
        "exp_bps": round(10_000 * a.mean(), 2) if n else 0.0,
        "PF": round(wins.sum() / -losses.sum(), 2) if len(losses) else np.inf,
        "target": reasons["target"],
        "stop": reasons["stop"],
        "trail": reasons["trail"],
        "timer": reasons["timer"],
        "pct_on_target": round(100 * reasons["target"] / n, 1) if n else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percentile", type=float, default=90.0,
                    help="cross-sectional entry percentile, as in exit_mode_ab")
    ap.add_argument("--since", default="2026-07-15")
    ap.add_argument("--checkpoint", default="models/lgbm/lgbm_ic_0.1830.pkl")
    ap.add_argument("--fracs", default=DEFAULT_FRACS,
                    help="comma-separated PROFIT_TARGET_ANNUAL_VOL_FRAC values")
    args = ap.parse_args()

    import src.agents.signal_loop as sl
    from src.models.lgbm import LGBMSignalModel

    fracs = [float(x) for x in args.fracs.split(",")]

    bars, feats = load_cache()
    if not bars:
        raise SystemExit(
            "empty cache — build it first:\n"
            "  RESEARCH_CACHE=... python scripts/research_backtest.py --phase fetch\n"
            "  RESEARCH_CACHE=... python scripts/research_backtest.py --phase features"
        )
    dv = daily_vol_map(bars)
    print(f"cache: {len(bars)} tickers, daily_vol for {len(dv)}")

    model = LGBMSignalModel.load(Path(args.checkpoint))
    entries = build_entries(feats, model, args.percentile)
    entries = entries[entries["timestamp"] >= pd.Timestamp(args.since, tz="UTC")]
    print(f"entries >= p{args.percentile:g} since {args.since}: {len(entries):,}")

    # Profit-target mode is what production runs; the sweep varies the fraction
    # inside it. Resolve each ladder ONCE per (frac, ticker) from the live
    # `_atr_exits` so the barriers are exactly what production would compute.
    sl.EXIT_PROFIT_TARGET_MODE = True
    cap = sl._effective_hold_bars()
    print(f"hold cap: {cap} bars (profit-target mode)")

    ladders: dict[float, dict[str, tuple[float, float, float]]] = {}
    original = sl.PROFIT_TARGET_ANNUAL_VOL_FRAC
    try:
        for f in fracs:
            sl.PROFIT_TARGET_ANNUAL_VOL_FRAC = f
            ladders[f] = {t: sl._atr_exits(v) for t, v in dv.items()}
    finally:
        sl.PROFIT_TARGET_ANNUAL_VOL_FRAC = original

    cost = COST_BPS / 10_000.0
    acc: dict[float, tuple[list, dict]] = {
        f: ([], defaultdict(int)) for f in fracs
    }

    skipped = 0
    for t, ts_ in entries[["ticker", "timestamp"]].itertuples(index=False):
        if t not in dv:
            skipped += 1
            continue
        b = bars[t]
        fwd = b.loc[b.index > ts_]
        # Same ET session only — the v0.9.4 backstop guarantees nothing carries,
        # so a path that runs past the close is not a path the bot can take.
        fwd = fwd[fwd.index.date == ts_.date()].head(cap)
        if len(fwd) < 10 or ts_ not in b.index:
            skipped += 1
            continue
        entry = float(b.loc[ts_, "close"])
        h, l, c = (fwd["high"].to_numpy(), fwd["low"].to_numpy(),
                   fwd["close"].to_numpy())
        for f in fracs:
            s, tr, tp = ladders[f][t]
            r, why = walk(h, l, c, entry, tp, s, tr, cap)
            acc[f][0].append(r - cost)
            acc[f][1][why] += 1

    rows = [summarise(f"{f:.3f}", *acc[f]) for f in fracs]
    df = pd.DataFrame(rows)
    print(f"\nskipped {skipped:,} entries (no vol / short path)")
    print(f"\n{df.to_string(index=False)}")

    best_exp = df.loc[df["exp_bps"].idxmax()]
    best_pf = df.loc[df["PF"].idxmax()]
    print(f"\nbest expectancy: frac={best_exp['target_frac']} "
          f"({best_exp['exp_bps']} bps, PF {best_exp['PF']}, "
          f"{best_exp['pct_on_target']}% exit on target)")
    print(f"best PF:         frac={best_pf['target_frac']} "
          f"({best_pf['exp_bps']} bps, PF {best_pf['PF']}, "
          f"{best_pf['pct_on_target']}% exit on target)")
    print(f"production now:  frac={original:.3f}")

    out = "reports/research/profit_target_sweep.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
