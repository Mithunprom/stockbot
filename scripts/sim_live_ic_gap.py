"""Is the sim/live gap signal DECAY or train/serve SKEW? They have opposite fixes.

The gap
-------
The backtest claims OOS IC15 +0.173 (and the selected checkpoint records
val_ic 0.2063). Live measures **+0.025** over n=132,340 filled predictions at
p~0 — real, significant, and about a seventh of the claim. PF is 0.82 live
against 1.44 in sim. That gap has blocked all parameter tuning since the
2026-09-16 loss diagnosis.

Two hypotheses, and the fix depends entirely on which holds:

  DECAY / OPTIMISTIC VALIDATION — the model genuinely scores ~0.02 on recent
    data, and +0.17/+0.21 came from an easier period or from selecting the
    best-scoring checkpoint out of twelve. Fix: retrain, honest walk-forward,
    stop selecting on val_ic.

  TRAIN/SERVE SKEW — the model still scores ~0.17 when features are computed
    the research way, but the LIVE feature path produces different inputs. Fix:
    find the diverging feature. This already happened once (`rs_vwap_dev`
    zero-filled in serving, v0.8.2) and cost 1.3 points of return per trade
    while every health check stayed green.

The test
--------
Score the EXACT production checkpoint over real minute bars, computing features
the research way, and measure IC the same way the live tracker does — Spearman
over (pred_return, forward 15-bar return) across all bars, not top-decile
entries. Then compare like with like on the overlapping window.

This isolates the question because the only thing that differs between the two
numbers is WHERE THE FEATURES CAME FROM. Same model file, same horizon, same
statistic, same tickers, same calendar window.

Deliberately NOT measured here: PF, costs, exits, sizing. Those are downstream
of signal quality and cannot explain an IC gap.

Usage:
    RESEARCH_CACHE=/path/to/cache PYTHONPATH=. python scripts/sim_live_ic_gap.py
    ... --since 2026-09-27 --horizon 15
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exit_geometry_study import load_cache  # noqa: E402

# The live tracker's horizon. src/agents/live_ic_tracker.py HORIZON_15M and
# scripts/train_lgbm.py FORWARD_N are both 15 — the model's validated horizon.
DEFAULT_HORIZON = 15


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Forward `horizon`-bar return, same construction as the training target.

    `train_lgbm.py` uses `close.pct_change(N).shift(-N)`. Matching it exactly
    matters: a half-bar misalignment is itself a plausible cause of an IC gap,
    so this must not be re-derived a different way here.
    """
    return close.pct_change(horizon).shift(-horizon)


def ic_of(preds: np.ndarray, actuals: np.ndarray) -> tuple[float, float, int]:
    ok = np.isfinite(preds) & np.isfinite(actuals)
    p, a = preds[ok], actuals[ok]
    if len(p) < 30:
        return float("nan"), float("nan"), len(p)
    rho, pval = spearmanr(p, a)
    return float(rho), float(pval), len(p)


def dir_acc(preds: np.ndarray, actuals: np.ndarray) -> float:
    ok = np.isfinite(preds) & np.isfinite(actuals)
    p, a = preds[ok], actuals[ok]
    if not len(p):
        return float("nan")
    return float(np.mean((p > 0) == (a > 0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-27",
                    help="start of the window the live 30d IC covers")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--checkpoint", default=None,
                    help="default: whatever production's _select_checkpoint picks")
    args = ap.parse_args()

    from src.models.lgbm import LGBMSignalModel

    ckpt = Path(args.checkpoint) if args.checkpoint else LGBMSignalModel._select_checkpoint()
    model = LGBMSignalModel.load(ckpt)
    print(f"checkpoint      {ckpt.name}")
    print(f"  recorded      val_ic={model.val_ic:.4f} train_ic={model.train_ic:.4f} "
          f"dir_acc={model.val_dir_acc:.4f} n_features={len(model.feature_cols)}")
    print(f"  train/val     {model.train_ic / max(model.val_ic, 1e-9):.1f}x "
          f"(a large ratio is an overfitting signature)")

    bars, feats = load_cache()
    if not bars:
        raise SystemExit(
            "empty cache — build it first:\n"
            "  RESEARCH_CACHE=... python scripts/research_backtest.py --phase fetch\n"
            "  RESEARCH_CACHE=... python scripts/research_backtest.py --phase features"
        )
    print(f"cache           {len(bars)} tickers")

    cols = list(model.feature_cols)
    since = pd.Timestamp(args.since, tz="UTC")

    per_ticker: list[dict] = []
    all_p: list[np.ndarray] = []
    all_a: list[np.ndarray] = []
    all_p_full: list[np.ndarray] = []
    all_a_full: list[np.ndarray] = []

    for t, f in sorted(feats.items()):
        missing = [c for c in cols if c not in f.columns]
        if missing:
            print(f"  {t}: SKIPPED — features missing {missing[:4]}")
            continue
        b = bars[t]
        fwd = forward_return(b["close"], args.horizon)

        idx = f.index.intersection(fwd.index)
        X = f.loc[idx, cols].astype(float)
        ok = X.notna().all(axis=1)
        X, idx2 = X[ok], idx[ok]
        if len(X) < 100:
            continue
        pred = np.asarray(model.regressor.predict(X), dtype=float)
        act = fwd.loc[idx2].to_numpy(dtype=float)

        all_p_full.append(pred)
        all_a_full.append(act)

        m = idx2 >= since
        if m.sum() >= 30:
            rho, pv, n = ic_of(pred[m], act[m])
            per_ticker.append({"ticker": t, "ic": rho, "p": pv, "n": n,
                               "dir_acc": dir_acc(pred[m], act[m])})
            all_p.append(pred[m])
            all_a.append(act[m])

    if not all_p:
        raise SystemExit(f"no rows at or after {args.since} — widen the cache window")

    P, A = np.concatenate(all_p), np.concatenate(all_a)
    Pf, Af = np.concatenate(all_p_full), np.concatenate(all_a_full)

    rho_w, pv_w, n_w = ic_of(P, A)
    rho_f, pv_f, n_f = ic_of(Pf, Af)

    print()
    print("SIM IC — research-computed features, production checkpoint")
    print(f"  live-overlap window (>= {args.since}): "
          f"IC={rho_w:+.4f}  p={pv_w:.3g}  dir_acc={dir_acc(P, A):.4f}  n={n_w:,}")
    print(f"  full cache window:                     "
          f"IC={rho_f:+.4f}  p={pv_f:.3g}  dir_acc={dir_acc(Pf, Af):.4f}  n={n_f:,}")

    df = pd.DataFrame(per_ticker).sort_values("ic")
    if not df.empty:
        print(f"\n  per-ticker over the window: {len(df)} tickers, "
              f"median IC {df['ic'].median():+.4f}, "
              f"{100 * (df['ic'] > 0).mean():.0f}% positive")
        print(f"  worst: {', '.join(f'{r.ticker} {r.ic:+.3f}' for r in df.head(4).itertuples())}")
        print(f"  best:  {', '.join(f'{r.ticker} {r.ic:+.3f}' for r in df.tail(4).itertuples())}")

    print()
    print("VERDICT")
    print(f"  claimed (checkpoint val_ic) : {model.val_ic:+.4f}")
    print(f"  sim, recent real data       : {rho_w:+.4f}")
    print("  live (prod /admin/ic/report): +0.0252   (7d, n=132,340, p~0)")
    if not np.isfinite(rho_w):
        print("  -> inconclusive: too little overlapping data")
    elif abs(rho_w - 0.0252) < 0.04:
        print("  -> SIM REPRODUCES LIVE. The feature path is NOT the problem;")
        print("     the claimed IC did not survive out of sample. Fix the model")
        print("     and the checkpoint-selection bias, not the serving code.")
    elif rho_w > 0.10:
        print("  -> SIM DOES NOT REPRODUCE LIVE. Same model + same window scores")
        print("     far better on research features => TRAIN/SERVE SKEW in the")
        print("     live feature path. Diff feature-by-feature next.")
    else:
        print("  -> PARTIAL: sim sits between the claim and live. Both effects")
        print("     are likely present; size each before fixing either.")

    out = "reports/research/sim_live_ic_gap.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
