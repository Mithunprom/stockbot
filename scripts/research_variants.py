"""Backtest the 'let winners run + bigger size' change set against v0.3.2.

Patches the REAL production position sizer constants (to test ~10%/position)
and the exit Params (higher targets, longer holds, looser reversal), then
compares each variant on the tune leg (May 18-29) and the untouched validation
leg (Jun 1-20). Capital = $98k, no PDT (matches the live account).
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.research_backtest import Params, simulate, load_preds
import src.execution.position_sizer as ps

CAP = 98_000.0
NP = dict(pdt_enabled=False)
LEGS = [("2026-05-18", "2026-05-29", "tune"), ("2026-06-01", "2026-06-20", "val")]


@contextmanager
def sizer_constants(base_min, base_max, buckets, max_pct, upscale_cap=None,
                    target_atr=None):
    """Temporarily patch the production sizer module globals."""
    saved = (ps._BASE_MIN_PCT, ps._BASE_MAX_PCT, ps._KELLY_BUCKETS,
             ps._MAX_NOTIONAL_PCT, ps._MAX_NOTIONAL, ps._ATR_UPSCALE_CAP,
             ps._TARGET_ATR_PCT)
    ps._BASE_MIN_PCT = base_min
    ps._BASE_MAX_PCT = base_max
    ps._KELLY_BUCKETS = buckets
    ps._MAX_NOTIONAL_PCT = max_pct
    ps._MAX_NOTIONAL = max_pct * CAP          # let the % cap bind on a big acct
    if upscale_cap is not None:
        ps._ATR_UPSCALE_CAP = upscale_cap
    if target_atr is not None:
        ps._TARGET_ATR_PCT = target_atr       # raise → bigger positions overall
    try:
        yield
    finally:
        (ps._BASE_MIN_PCT, ps._BASE_MAX_PCT, ps._KELLY_BUCKETS,
         ps._MAX_NOTIONAL_PCT, ps._MAX_NOTIONAL, ps._ATR_UPSCALE_CAP,
         ps._TARGET_ATR_PCT) = saved


# Current production sizer (v0.3.2): base .10-.22, buckets .15/.20/.24, 10% cap
PROD_SIZER = dict(base_min=0.10, base_max=0.22,
                  buckets=[(0.55, 0.65, 0.15), (0.65, 0.80, 0.20), (0.80, 1.01, 0.24)],
                  max_pct=0.10)
# Bigger sizing: push most entries toward the 10% cap, raise upscale so calm
# names reach it; keep some ATR scaling so hyper-vol names stay a touch smaller
BIG_SIZER = dict(base_min=0.16, base_max=0.30,
                 buckets=[(0.55, 0.65, 0.22), (0.65, 0.80, 0.28), (0.80, 1.01, 0.34)],
                 max_pct=0.10, upscale_cap=1.6)
# Flat ~10%: raise target ATR + base so the 10% cap binds for almost every
# name (volatile names lifted toward 10% too) — tests the DD cost of the
# user's literal "10% per stock" with much weaker volatility scaling.
FLAT_SIZER = dict(base_min=0.24, base_max=0.34,
                  buckets=[(0.55, 0.65, 0.30), (0.65, 0.80, 0.33), (0.80, 1.01, 0.36)],
                  max_pct=0.10, upscale_cap=2.0, target_atr=0.030)

# Exit profiles -----------------------------------------------------------------
BASE_EXIT = dict(stop_mult=1.1, trail_mult=0.8, tp_mult=2.2,
                 sl_cap=0.025, ts_cap=0.020, tp_cap=0.050,
                 max_hold_bars=240, stag_bars=240, reversal_bars=3)
# Let winners run: bigger TP target + wider trailing room + looser reversal +
# longer clock, BUT the stop stays tight (cut losers fast — the asymmetry).
RUN_EXIT = dict(stop_mult=1.1, trail_mult=1.6, tp_mult=4.0,
                sl_cap=0.025, ts_cap=0.040, tp_cap=0.090,
                max_hold_bars=780, stag_bars=780, reversal_bars=5)
# Milder "run" — 1-day clock, moderate widening
RUN_MILD = dict(stop_mult=1.1, trail_mult=1.2, tp_mult=3.0,
                sl_cap=0.025, ts_cap=0.030, tp_cap=0.070,
                max_hold_bars=390, stag_bars=390, reversal_bars=4)

VARIANTS = [
    ("V0 prod baseline",        PROD_SIZER, BASE_EXIT),
    ("V1 bigger size only",     BIG_SIZER,  BASE_EXIT),
    ("V2 winners-run only",     PROD_SIZER, RUN_EXIT),
    ("V3 winners-run MILD",     PROD_SIZER, RUN_MILD),
    ("V4 BIG + run",            BIG_SIZER,  RUN_EXIT),
    ("V5 BIG + run MILD",       BIG_SIZER,  RUN_MILD),
    ("V6 FLAT10 + run MILD",    FLAT_SIZER, RUN_MILD),
]


def run() -> None:
    preds = load_preds()
    rows = []
    for name, sz, ex in VARIANTS:
        for s, e, tag in LEGS:
            p = Params(label=f"{name}|{tag}", dead_hi=0.60, **ex, **NP)
            with sizer_constants(**sz):
                r = simulate(preds, p, s, e, capital=CAP)
            r.pop("_trades"); r.pop("exit_reasons")
            # avg notional as % of capital — did sizing actually grow?
            rows.append({**r, "variant": name, "leg": tag})
    df = pd.DataFrame(rows)
    show = ["variant", "leg", "total_return_pct", "sharpe", "max_dd_pct",
            "profit_factor", "win_rate", "n_trades", "avg_notional_pct",
            "avg_win_pct", "avg_loss_pct", "avg_hold_bars"]
    for tag in ("tune", "val"):
        print(f"\n===== {tag.upper()} leg =====")
        print(df[df.leg == tag][show].to_string(index=False))
    df.to_csv("/tmp/stockbot_research/variant_results.csv", index=False)


if __name__ == "__main__":
    run()
