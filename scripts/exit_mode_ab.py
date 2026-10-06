"""A/B the exit rule PRODUCTION RUNS against the one v0.8.0 ships but disables.

Why this exists
---------------
`scripts/exit_geometry_study.py` showed the profit-target ladder is profitable
(OOS PF 1.62, +23.9 bps/trade). But `EXIT_PROFIT_TARGET_MODE` defaults OFF, so
that ladder is NOT what production runs — and on 2026-09-21, the first session
under the fixed feature pipeline, 10 of 11 trades exited `max_hold` and the
session lost $329.84 despite entries ranking at the 96.9th percentile.

So the open question is not "is the new ladder good" but "is the OLD one the
reason good entries still lost money". This measures both rules on the SAME
entries and the SAME real minute paths, so the comparison isolates the exit
rule and nothing else.

Both ladders are taken from the live `_atr_exits`, with the module flag toggled,
rather than reimplemented here — a reimplementation could flatter either side.

Usage:
    RESEARCH_CACHE=/tmp/sb_exitgeom PYTHONPATH=. python scripts/exit_mode_ab.py
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from exit_geometry_study import (  # noqa: E402  (same scripts/ dir)
    COST_BPS, build_entries, load_cache,
)

SESSION_BARS = 390


def daily_vol_map(bars: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Daily ATR%/price per ticker, the same quantity `_atr_exits` consumes.

    `_daily_vol_for` in production prefers real daily bars; resampling the 1m
    cache to daily is the same construction the research backtest uses.
    """
    out = {}
    for t, b in bars.items():
        d = b.resample("1D").agg(
            {"high": "max", "low": "min", "close": "last"}
        ).dropna()
        if len(d) < 15:
            continue
        prev = d["close"].shift(1)
        tr = pd.concat([
            d["high"] - d["low"],
            (d["high"] - prev).abs(),
            (d["low"] - prev).abs(),
        ], axis=1).max(axis=1)
        out[t] = float((tr.rolling(14).mean() / d["close"]).dropna().mean())
    return out


def walk(h, l, c, entry, tp, sl, ts, max_bars):
    """First-touch over `max_bars`, then exit at that bar's close.

    Same-bar ambiguity resolves to the stop (pessimistic), identically for both
    ladders so it cannot favour either.
    """
    peak = entry
    n = min(len(c), max_bars)
    for i in range(n):
        if l[i] / entry - 1.0 <= -sl:
            return -sl, "stop"
        if h[i] / entry - 1.0 >= tp:
            return tp, "target"
        peak = max(peak, h[i])
        if peak > entry and l[i] / peak - 1.0 <= -ts:
            return max(l[i], peak * (1 - ts)) / entry - 1.0, "trail"
    return (c[n - 1] / entry - 1.0) if n else 0.0, "timer"


def summarise(label, res, reasons):
    a = np.array(res)
    wins, losses = a[a > 0], a[a <= 0]
    return {
        "mode": label,
        "n": len(a),
        "win_rate": round(100 * len(wins) / len(a), 1),
        "exp_bps": round(10_000 * a.mean(), 2),
        "PF": round(wins.sum() / -losses.sum(), 2) if len(losses) else np.inf,
        **{k: reasons[k] for k in ("target", "stop", "trail", "timer")},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--since", default="2026-07-15")
    ap.add_argument("--checkpoint", default="models/lgbm/lgbm_ic_0.1830.pkl")
    args = ap.parse_args()

    import src.agents.signal_loop as sl
    from src.models.lgbm import LGBMSignalModel

    bars, feats = load_cache()
    dv = daily_vol_map(bars)
    print(f"cache: {len(bars)} tickers, daily_vol for {len(dv)}")

    model = LGBMSignalModel.load(Path(args.checkpoint))
    entries = build_entries(feats, model, args.percentile)
    entries = entries[entries["timestamp"] >= pd.Timestamp(args.since, tz="UTC")]
    print(f"entries >= p{args.percentile:g} since {args.since}: {len(entries):,}")

    # Resolve each ladder ONCE per ticker from the live code, flag toggled.
    ladders = {}
    for mode in (False, True):
        sl.EXIT_PROFIT_TARGET_MODE = mode
        bars_cap = sl._effective_hold_bars()
        ladders[mode] = (
            {t: sl._atr_exits(v) for t, v in dv.items()}, bars_cap,
        )
        print(f"  mode={mode}: hold_cap={bars_cap} bars")

    cost = COST_BPS / 10_000.0
    acc = {False: ([], defaultdict(int)), True: ([], defaultdict(int))}

    for t, ts_ in entries[["ticker", "timestamp"]].itertuples(index=False):
        if t not in dv:
            continue
        b = bars[t]
        fwd = b.loc[b.index > ts_]
        fwd = fwd[fwd.index.date == ts_.date()].head(SESSION_BARS)
        if len(fwd) < 10 or ts_ not in b.index:
            continue
        entry = float(b.loc[ts_, "close"])
        h, l, c = (fwd["high"].to_numpy(), fwd["low"].to_numpy(),
                   fwd["close"].to_numpy())
        for mode, (lad, cap) in ladders.items():
            s, tr, tp = lad[t]
            r, why = walk(h, l, c, entry, tp, s, tr, cap)
            acc[mode][0].append(r - cost)
            acc[mode][1][why] += 1

    rows = [
        summarise("PRODUCTION (30-bar timer)", *acc[False]),
        summarise("v0.8.0 profit-target", *acc[True]),
    ]
    df = pd.DataFrame(rows)
    print(f"\n{df.to_string(index=False)}")
    out = "reports/research/exit_mode_ab.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
