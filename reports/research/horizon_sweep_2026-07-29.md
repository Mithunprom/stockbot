# Hold-Horizon Sweep — 2026-07-29

**Question:** LightGBM is trained on `FORWARD_N = 15` and live IC is validated at
15 minutes (0.086, n=6,148, p≈0), but production holds to
`SIZING_MAX_HOLD_BARS = 390`. Is the horizon mismatch costing money?

**Answer: yes, decisively.** Production sits on the far side of a cliff.

Tooling: `scripts/horizon_sweep.py`. Baseline `PROD_PARAMS` mirrors live v0.5.5
(the `Params` defaults in `research_backtest.py` are stale — `tp_mult=3.0`,
`max_hold=1170`, `max_pos=4`, `heat=0.60`).

Data: 73-ticker live universe, 1m bars 2026-04-20 → 07-28, train ≤ 06-20,
val ≤ 06-30, OOS 07-01 → 07-27 (432,350 rows). OOS IC15 = 0.1630.

---

## 1. Signal decay — the mechanism

Same predictions, forward return measured at each horizon:

| horizon (bars) | mean IC | median IC | % tickers IC>0 | dir acc |
|---|---|---|---|---|
| 15 | **0.1690** | 0.1744 | **100.0%** | 55.5% |
| 30 | 0.1243 | 0.1222 | 98.6% | 54.3% |
| 60 | 0.0857 | 0.0902 | 93.2% | 53.2% |
| 120 | 0.0527 | 0.0549 | 71.2% | 52.2% |
| 195 | 0.0384 | 0.0298 | 64.4% | 51.8% |
| **390 — PRODUCTION** | **0.0044** | **−0.0182** | **46.6%** | **50.7%** |
| 1170 | −0.0487 | −0.0446 | 30.1% | 49.6% |

At the production hold the edge is **statistically zero**: mean IC 0.0044,
median *negative*, and fewer than half the tickers positive. At 15 bars the
edge is strong and **universal — 100% of 73 tickers**.

## 2. Strategy metrics by max_hold

PDT does **not** bind: `signal_loop.py:1533` bypasses the PDT gate at
`portfolio_value >= $25,000` and the account is at $96.9k. So the PDT-off rows
are the operative ones.

**Window B — Jul 13–27 (the leg that actually lost money):**

| max_hold | n | PF | win rate | return | max DD |
|---|---|---|---|---|---|
| 15 | 66 | **6.45** | 81.8% | +3.25% | **0.25%** |
| 30 | 66 | 4.55 | 77.3% | +3.71% | 0.44% |
| 60 | 66 | 3.96 | 74.2% | +4.07% | 0.53% |
| 120 | 66 | 3.36 | 77.3% | +4.10% | 0.86% |
| 195 | 65 | 4.11 | 67.7% | **+6.00%** | 0.91% |
| **390 — PROD** | 52 | **0.54** | 40.4% | **−4.29%** | **5.58%** |
| 1170 | 25 | 1.09 | 44.0% | +1.21% | 4.97% |

**Window A — Jul 1–10 (held out, never used to pick anything):**

| max_hold | n | PF | win rate | return | max DD |
|---|---|---|---|---|---|
| 15 | 42 | **5.67** | 76.2% | +2.10% | **0.28%** |
| 30 | 42 | 3.67 | 69.0% | +2.15% | 0.51% |
| 195 | 41 | 1.42 | 58.5% | +1.73% | 3.61% |
| 390 | 33 | 1.14 | 60.6% | +1.07% | 3.91% |
| 1170 | 18 | 0.64 | 33.3% | +0.69% | 4.93% |

Direction replicates. Production's 40.4% simulated win rate closely matches the
**44.4% realised** win rate on the live ledger (n=117) — the sim is reproducing
the actual failure, not an idealised one.

## 3. Cost sensitivity

Short holds mean higher turnover, so the result was stress-tested against a 10×
cost assumption (per-side bps; production baseline 2.0):

| bps | PF@15 (A / B) | PF@390 (A / B) |
|---|---|---|
| 2 | 5.67 / 6.45 | 1.14 / 0.54 |
| 5 | 4.80 / 5.32 | 1.11 / 0.60 |
| 10 | 3.67 / 3.85 | 1.07 / 0.55 |
| 20 | 2.12 / 1.99 | 0.98 / 0.75 |

The advantage survives at 20 bps. Not a costs artifact.

---

## Recommendation

`SIZING_MAX_HOLD_BARS: 390 → 30` (with 15 as the more aggressive option).

Why 30 rather than 15: PF 3.67/4.55 across both windows with DD ≤ 0.51%, while
carrying slightly more return than 15 and staying further from the
per-trade-cost floor. 195 posts the best single return (+6.00%) but only in
Window B — it degrades to PF 1.42 in Window A, so it is not robust.

## Caveats — read before deploying

1. **Sharpe figures in the raw CSVs are inflated.** They annualize a ~12-day
   window (values of 10–21 are not forward estimates). Judge on PF and DD.
2. **Small samples:** n = 33–66 per cell, 2 windows, one regime (July 2026).
3. **Contradicts an earlier finding.** Memory (2026-06-11) records *"late
   entries + 1-day hold"* and *"multi-day holds = beta bleed"*. The multi-day
   half agrees; the 1-day half does not. That earlier work was done on a $9.9k
   account where PDT genuinely blocked short holds — the constraint, not the
   edge, likely drove it.
4. **PDT returns below $25k.** At the $9.9k live account, a 30-bar hold is a day
   trade and the PDT deferral reappears (the sim shows this: with PDT on, holds
   of 15–195 collapse to an identical ~333-bar realised hold). This fix is
   paper-viable now but is **conditional on account size** for live.
5. Not deployed. No production parameter changed.
