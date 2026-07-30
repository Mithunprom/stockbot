"""Hold-horizon sweep — pick max_hold from data, not from argument.

Question this answers
---------------------
LightGBM is trained on FORWARD_N = 15 (a 15-minute forward return) and live IC
is validated at the same 15-minute horizon (0.086 over n=6,148, p≈0). But
production holds positions for SIZING_MAX_HOLD_BARS = 390 bars, and realised
median hold over the last 30 closes was 1,442 minutes. So the edge is measured
at 15 minutes and harvested 20–95x later.

Live evidence that this costs money: the model calls direction correctly 52.4%
of the time while trades win only 44.4% — 8 points of edge lost between signal
and exit.

This sweep holds every other production parameter fixed and varies ONLY
max_hold_bars, reporting decay of predictive power (IC by horizon) alongside
realised strategy metrics (PF / Sharpe / win rate) at each hold. Two independent
views of the same question, so a hold isn't chosen on one noisy statistic.

Usage
-----
    python scripts/horizon_sweep.py --phase data     # fetch + features + train
    python scripts/horizon_sweep.py --phase ic       # IC decay by horizon
    python scripts/horizon_sweep.py --phase sim      # PF/Sharpe by max_hold
    python scripts/horizon_sweep.py --phase all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Window: out-of-sample leg deliberately covers the losing stretch (Jul 13–28)
# so a candidate hold has to survive the regime that actually hurt us, not a
# friendlier earlier one. Set BEFORE importing research_backtest — that module
# reads these at import time.
os.environ.setdefault("RESEARCH_CACHE", "/tmp/stockbot_research_horizon")
os.environ.setdefault("RESEARCH_BARS_START", "2026-04-20")   # indicator warmup
os.environ.setdefault("RESEARCH_BARS_END", "2026-07-28")
os.environ.setdefault("RESEARCH_TRAIN_END", "2026-06-20")
os.environ.setdefault("RESEARCH_VAL_END", "2026-06-30")
os.environ.setdefault("RESEARCH_TUNE_END", "2026-07-10")
os.environ.setdefault("RESEARCH_VAL_LEG_START", "2026-07-13")
os.environ.setdefault("RESEARCH_VAL_LEG_END", "2026-07-28")

import scripts.research_backtest as rb  # noqa: E402

CACHE = rb.CACHE
LIVE_UNIVERSE = CACHE / "universe_live.json"


def _universe_live() -> list[str]:
    """Production's real universe (73 names), not the stale 20 in the repo."""
    if LIVE_UNIVERSE.exists():
        with open(LIVE_UNIVERSE) as f:
            return [t for t in json.load(f)["symbols"] if "/" not in t]
    return rb.universe()


# research_backtest's phases call universe() internally; point them at live.
rb.universe = _universe_live  # type: ignore[assignment]


# ─── Production baseline ─────────────────────────────────────────────────────
# Mirrors live v0.5.5 (src/agents/signal_loop.py + position_sizer.py) as of
# 2026-07-29. The Params defaults in research_backtest are STALE (tp_mult 3.0,
# max_hold 1170, max_pos 4, heat 0.60) and would have compared against a
# strategy that hasn't run in months.
PROD_PARAMS = rb.Params(
    stop_mult=1.1,          # SIZING_STOP_LOSS_DVOL_MULT
    trail_mult=1.2,         # SIZING_TRAILING_DVOL_MULT
    tp_mult=1.5,            # SIZING_TAKE_PROFIT_DVOL_MULT  (v0.5.3: was 3.0)
    sl_floor=0.010,
    sl_cap=0.100,
    ts_floor=0.008,
    ts_cap=0.100,
    tp_floor=0.015,         # v0.5.3: was 0.020
    tp_cap=0.135,           # v0.5.3: was 0.200
    max_hold_bars=390,      # SIZING_MAX_HOLD_BARS — the variable under test
    stag_bars=390,          # SIZING_STAGNATION_BARS (== max hold: inert)
    stag_pnl=0.004,
    catastrophic_mult=2.0,  # CATASTROPHIC_STOP_MULT
    dead_lo=0.40,           # live dir_prob_dead_zone = [0.40, 0.60]
    dead_hi=0.60,
    max_pos=6,              # v0.3.5: 6 slots
    heat_ceiling=0.75,      # v0.3.5: 75%
    trades_per_day=6,
    dyn_thresh_pct=92,      # live threshold_mode = dynamic_percentile
    pdt_enabled=True,       # account < $25k
    cost_bps=2.0,
    label="prod_v0.5.5",
)

# Horizons: the trained/validated edge (15), a small multiple (30/60), a half
# session (195), production today (390), and 3 days (1170) to show the bleed.
HORIZONS = [15, 30, 60, 120, 195, 390, 1170]


# ─── Phase: IC decay ─────────────────────────────────────────────────────────

def phase_ic() -> pd.DataFrame:
    """Spearman IC and directional accuracy of the SAME predictions at each
    horizon. Isolates signal decay from exit-rule effects."""
    from scipy.stats import spearmanr

    preds = pd.read_csv(CACHE / "preds_oos.csv.gz", parse_dates=["time"])
    rows = []
    for h in HORIZONS:
        per_ticker = []
        for ticker, g in preds.groupby("ticker"):
            g = g.sort_values("time").reset_index(drop=True)
            fwd = g["close"].pct_change(h).shift(-h)
            ok = g["pred_return"].notna() & fwd.notna()
            if ok.sum() < 30:
                continue
            ic = spearmanr(g.loc[ok, "pred_return"], fwd[ok]).statistic
            dir_acc = float(
                ((g.loc[ok, "pred_return"] > 0) == (fwd[ok] > 0)).mean()
            )
            if np.isfinite(ic):
                per_ticker.append((ic, dir_acc, int(ok.sum())))
        if not per_ticker:
            continue
        ics = [x[0] for x in per_ticker]
        rows.append({
            "horizon_bars": h,
            "mean_ic": round(float(np.mean(ics)), 4),
            "median_ic": round(float(np.median(ics)), 4),
            "pct_tickers_positive_ic": round(
                100.0 * float(np.mean([i > 0 for i in ics])), 1),
            "dir_acc": round(float(np.mean([x[1] for x in per_ticker])), 4),
            "n_obs": int(sum(x[2] for x in per_ticker)),
            "n_tickers": len(per_ticker),
        })
    return pd.DataFrame(rows)


# ─── Phase: simulate ─────────────────────────────────────────────────────────

def phase_sim() -> pd.DataFrame:
    """Full strategy sim at each max_hold, all else fixed at production.

    Run under BOTH PDT regimes, because PDT is what decides whether a short
    hold is even reachable:

      pdt_on  — account < $25k (today: ~$97k paper, but the live account is
                $9.9k). research_backtest defers any same-day non-stop exit
                (`if not allowed: continue`), so a 15-bar max_hold fires, gets
                deferred to the next session, and the REALISED hold is ~333
                bars no matter what the parameter says. Short holds are
                unreachable by construction, which is why 15/30/60/120 return
                byte-identical metrics.
      pdt_off — account >= $25k. max_hold is actually honoured, so this is the
                only regime where the horizon question can be answered.
    """
    preds = pd.read_csv(CACHE / "preds_oos.csv.gz", parse_dates=["time"])
    rows = []
    for pdt in (True, False):
        for h in HORIZONS:
            p = replace(PROD_PARAMS, max_hold_bars=h, stag_bars=h,
                        pdt_enabled=pdt, label=f"hold_{h}_pdt_{'on' if pdt else 'off'}")
            # OOS leg only — the stretch that actually lost money.
            m = rb.simulate(preds, p, rb.VAL_LEG_START, rb.VAL_LEG_END)
            m["max_hold_bars"] = h
            m["pdt"] = "on" if pdt else "off"
            rows.append(m)
    df = pd.DataFrame(rows)
    front = ["pdt", "max_hold_bars", "n_trades", "profit_factor", "sharpe",
             "win_rate", "total_return_pct", "max_dd_pct", "avg_hold_bars"]
    return df[[c for c in front if c in df.columns]
              + [c for c in df.columns if c not in front]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["data", "ic", "sim", "all"])
    args = ap.parse_args()

    if args.phase in ("data", "all"):
        print(f"universe: {len(_universe_live())} tickers")
        print(f"window: {rb.BARS_START} → {rb.BARS_END} "
              f"(OOS leg {rb.VAL_LEG_START} → {rb.VAL_LEG_END})")
        print("→ fetch"); rb.phase_fetch()
        print("→ features"); rb.phase_features()
        print("→ train"); rb.phase_train()

    if args.phase in ("ic", "all"):
        ic = phase_ic()
        print("\n=== IC DECAY BY HORIZON (same predictions, OOS) ===")
        print(ic.to_string(index=False))
        ic.to_csv(CACHE / "horizon_ic.csv", index=False)

    if args.phase in ("sim", "all"):
        sim = phase_sim()
        print("\n=== STRATEGY METRICS BY MAX_HOLD (prod params, OOS leg) ===")
        print(sim.to_string(index=False))
        sim.to_csv(CACHE / "horizon_sim.csv", index=False)


if __name__ == "__main__":
    main()
