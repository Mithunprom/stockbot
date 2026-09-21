# Loss Diagnosis — 2026-09-16

**Question:** why is the bot losing money, and is `max_hold` the cause?

**Short answer:** `max_hold` is not the cause. The signal is healthy (OOS IC15 =
0.173). The losses come from the layer between prediction and fill. But the
headline finding is a **fidelity failure**: the backtest no longer reproduces
production, so no parameter change can be justified from it yet.

Data: live ledger `/trades` (236 closed, 2026-05-14 → 09-11) + a fresh
walk-forward backtest (75-ticker live universe, 1m bars 2026-04-15 → 09-12,
train ≤ 07-14, OOS 07-15 → 09-12, 803,562 OOS rows).

---

## 1. Scoreboard — the M2 gate failed

| Window | n | WR | PF | Net |
|---|---|---|---|---|
| M2 (v0.6.0, exits ≥ Aug 6) | 111 | 45.0% | **0.86** | **−$794.59** |
| Since Aug 19 | 42 | 35.7% | 0.25 | −$2,561.64 |

The M2 gate was PF ≥ 1.2 at n ≥ 100. At n=111, PF=0.86 — **reached and failed.**
The last committed report (W34 addendum, Aug 19) showed n=68 / PF 1.80; that has
fully decayed.

Per-trade statistics:

| Sample | Mean/trade | t-stat |
|---|---|---|
| M2 (n=111) | −0.059% | **−0.50** (indistinguishable from zero) |
| Since Aug 19 (n=42) | −0.548% | −2.94 ¹ |

¹ Breakpoint chosen post hoc after inspecting the data — significance inflated.
Treat as suggestive only.

## 2. `max_hold` is a tautology, not a cause

111/111 M2 exits are labelled `max_hold` because the SL/trail/TP barriers are
unreachable by construction: `_atr_exits()` derives thresholds from **daily**
sigma while the trade lives 30 bars. √(30/390) = 0.277, so the nominal 1.1σ stop
is really ~3.97σ. An exit label attached to 100% of trades carries zero
explanatory variance — it cannot separate the 50 winners from the 61 losers.

Sizing the prize: a correctly-scaled 1.5σ stop would have saved **+$76 across 68
trades** (W34 addendum). The 30-bar timer is doing its job. Fixing the exit
label would not have saved this money.

## 3. The signal is healthy — this is the important one

Fresh model, hard Jul 14 train cutoff, tested on Jul 15 → Sep 12:

```
OOS IC15 = 0.1728    dir_acc = 0.5576    n = 803,562 rows
```

IC decay by horizon (same predictions, OOS):

| horizon | mean IC | % of 75 tickers with IC>0 | dir acc |
|---|---|---|---|
| **15** | **0.1733** | **100.0%** | 0.5555 |
| 30 ← production | 0.1246 | 97.3% | 0.5402 |
| 60 | 0.0910 | 97.3% | 0.5307 |
| 195 | 0.0581 | 80.0% | 0.5217 |
| 390 | 0.0340 | 69.3% | 0.5157 |
| 1170 | −0.0085 | 38.7% | 0.5033 |

This independently replicates `horizon_sweep_2026-07-29.md` on a later, harsher
regime. The model forecasts a 15-bar return (`train_lgbm.py:35 FORWARD_N = 15`);
production holds 30 (`signal_loop.py:177`). Holding 2× the forecast horizon
gives up ~28% of the IC. The comment on that line ("== the model's validated
15–30m horizon") is wrong: the target is exactly 15.

**The signal is not the problem, and staleness is not the problem** — a model
frozen at Jul 14 still scores IC 0.173 across Jul 15 – Sep 12.

## 4. THE BLOCKER — the backtest no longer reproduces production

Same window (Aug 21 – Sep 12), same `max_hold=30`, same universe, PDT off
(account ~$100k, so PDT does not bind):

| | n | WR | PF | Return |
|---|---|---|---|---|
| **Simulation** | 90 | 72.2% | 6.54 | **+5.78%** |
| **Production** | 30 | 33.3% | 0.27 | **−1.49%** |

**A 7.27-point gap at the parameter production actually ran.** The sim says the
current configuration should have made money; it lost.

This matters more than any tuning result. In July the sim was explicitly
validated against reality — "production's 40.4% simulated win rate closely
matches the 44.4% realised" (`horizon_sweep_2026-07-29.md` §2). That fidelity is
now gone (72.2% sim vs 33.3% live). Until the gap is explained, **every
parameter recommendation derived from this backtest is unsafe**, including
`max_hold=15`.

Two leads:

- **Adverse selection by the gates.** Production took 30 of the sim's ~90
  opportunities — a 33% subset — and turned PF 6.54 into PF 0.27. Taking a third
  of a profitable population should not invert its sign. The entry gates (Kelly
  probation probe, dynamic percentile threshold, per-ticker IC block) are
  selecting *worse* than random.
- **Train/serve skew.** Offline OOS IC15 = 0.173; live IC15 on the same horizon
  = **0.082** (n=3,495, p=1e-6, `/admin/ic/report`). Production is realising
  roughly half the model's offline edge. Some is sample/regime, but the gap is
  large and specific.

## 4b. ADDENDUM — narrowing the gap (same day, later)

Follow-up work on §4. Two claims tightened, one retracted.

**PROVEN — production selects at the median of the model's own ranking.** For
each live entry in the validation leg, its percentile rank among all tickers at
that exact minute, using the offline model:

| Statistic | Value |
|---|---|
| Mean percentile of names production bought | **46.0** |
| In the model's top decile | **0 of 26** |
| In the bottom half | **11 of 26** |
| Mean offline `pred_return` of bought names | +0.000285 (≈ zero) |

**PROVEN — that selection level reproduces the loss.** Same universe, same entry
bars, same 30-bar hold; only the selected percentile varies:

| Selection rule | n | Mean return | WR | PF |
|---|---|---|---|---|
| Top decile (what the sim does) | 90 | +1.667% | 86.7% | 10.55 |
| 70–90th pct | 90 | +0.246% | 57.8% | 1.97 |
| **46th pct (what production does)** | 90 | **−0.291%** | **33.3%** | **0.34** |
| **Production, actual** | 30 | −0.491% | **33.3%** | 0.27 |

Win rate matches to the decimal. The fidelity gap in §4 is therefore explained
**at the level of selection quality**: the sim picks the top decile, production
picks the median, and the median has a negative edge. The remaining question is
only *why* production's selection lands there.

**RETRACTED — it is not ensemble dilution.** An earlier draft argued production
ranks entries on `ensemble_signal` (LGBM 0.60 + Transformer 0.10 + TCN 0.10 +
Sentiment 0.20, where Transformer/TCN carry IC≈0 and TCN is not even loaded).
That is wrong: `_execute_entries` (`signal_loop.py:1676-1680`) already sorts by
`abs(lgbm_pred_return)`. Ensemble dilution is real and still worth fixing — in a
live sample, sentiment supplied ~34% of signal magnitude, and `feat/rnd-H4-dead-
weight-renorm` addresses it — but it is **not** the mechanism behind the 46th-
percentile selection.

**OPEN — train/serve skew is the leading remaining hypothesis, not yet proven.**
Production's live `pred_return` cross-section (`/signals`, 50 tickers,
2026-09-16 19:59Z) has mean **+0.00314**. Against offline predictions at the
same bar-of-day (19:59 UTC, 34 days):

| | Offline EOD | Production |
|---|---|---|
| Cross-sectional mean | +0.000589 (sd 0.000816) | **+0.00314** |
| Days at/above that level | 1 of 34 | — |
| z-score | — | **+3.13** |

A systematically long-biased prediction surface would explain why 111/111 trades
are `buy`. **But this is a single cross-section and z=+3.13 is a ~1-in-34 event —
a genuinely strong up-day produces the same reading.** Production's max
prediction (+0.0204) sits comfortably inside the offline EOD range (max ever
+0.0458), so magnitudes are not obviously corrupted. Not established.

**Decisive test, not yet run:** compute features for the same (ticker, bar) with
both `src/features/live.py` (incremental) and `compute_indicators_for_universe`
(batch) and diff them column by column. That distinguishes feature skew from
gate-driven filtering. Until it is run, the mechanism behind §4b is *unknown*,
and the §4 prohibition on parameter tuning still stands.

## 5. Concentration — three correlation guards, none binding

All 111 M2 trades are `side: "buy"`. Entries fire 2 per bar at 9:40/9:41/9:42
ET, so the book is ~6 correlated longs opened in three minutes and held 30.
13 of 29 multi-entry baskets had every leg finish the same way.

| Guard | Why it does not bind |
|---|---|
| `MAX_ENTRIES_PER_TICK = 2` | Spreads the burst across adjacent minutes. A 1-minute boundary is not a diversification interval. |
| `MAX_POSITIONS_PER_SECTOR = 2` | `SECTOR_MAP` covers 24 tickers; the universe is 75. 34 traded names default to a shared `"other"` bucket. |
| `_SECTOR_CAP_PCT` | Same broken lookup. |

Unmapped names include KLAC, INTC, LRCX, QCOM, AMAT, MRVL, LITE, COHR, CIEN,
STX, TER — all semiconductor/photonics. Aug 19 held four simultaneous semi longs
(KLAC, INTC, MU, WDC); the guard counted two. **−$1,237 in 31 minutes.**

Trades on unmapped tickers: **−$1,203**. On mapped tickers: **+$408**.

## 6. Negative result — entry timing is NOT the cause

An earlier read of profit-factor-by-hour suggested the 9:40 burst was the
problem. Tested properly on per-trade returns, it is not:

| | n | mean/trade |
|---|---|---|
| 09:xx entries | 69 | −0.0476% |
| 10:00+ entries | 42 | −0.0784% |

Later entries are *worse*. Permutation test (20,000 shuffles): **p = 0.55**;
Welch t = −0.13. No effect. PF-by-hour was misleading because PF is
dollar-weighted and a few large trades dominated it.

Excluding the four concentration-blowup days, the 09:xx mean is **+0.44%**.
The "bad hour" was the correlated-basket effect in disguise. **Do not retime
entries.** Fix concentration instead.

## 7. Structural defect — the retrain pipeline cannot work

```python
# src/data/db.py:446
("feature_matrix", "time", 3),   # 3 DAYS
```

`RetrainAgent` runs daily at 08:00 ET and trains LightGBM on whatever is in
`feature_matrix` — i.e. **3 days of data**, for a model originally trained on
~4.75M rows spanning months. The result is unstable checkpoints
(`lgbm_ic_-0.0630`, `lgbm_ic_-0.0179`, two days apart in May), which
`MIN_VAL_IC = 0.05` then correctly rejects, pinning production to the Jul 22
checkpoint.

There is no working path to a fresh model. Not currently costing money (§3 shows
the frozen model is fine), but it means the system cannot adapt. Retention was
tightened after the Apr 13 disk-full incident, so the fix must archive out of
Postgres, not grow it.

## 8. Current state

The bot **self-halted** and has not traded since Sep 11:

```
kelly_fraction         = -0.6661
kelly_mode             = probation
tickers_probe_eligible = []     # probation + empty ⇒ ALL entries blocked
```

The governor worked. Note `kelly_entries_blocked: False` in `/diagnostics` is a
**hardcoded literal** (`signal_loop.py:876`) and does not reflect real state —
a reporting bug. This is also the third Kelly-probation deadlock (cf. 06-11,
06-30): probation needs IC-eligible tickers, the IC cache is empty, so nothing
can probe its way out. Currently protective.

---

## Recommendation — ordered, and deliberately not "ship max_hold=15"

1. **Close the sim/live fidelity gap (§4) before changing any parameter.**
   Reconcile the sim's ~90 opportunities against production's 30 for
   Aug 21 – Sep 12, trade by trade, and attribute each rejection to the gate
   that caused it. Until PF 6.54 vs 0.27 is explained, tuning is fitting noise.
2. **Fix the correlation guards (§5).** Fail-closed sector resolution; make the
   burst limit time-based, not tick-based. Pure correctness, defensible alone.
3. **Fix the exit-barrier units (§2)** as a *disaster* stop (~2.5–3σ of the hold
   window), not a trading stop. 1.1σ would fire on 16%+ of the book and truncate
   the edge.
4. **Then, and only then,** revisit `max_hold` 30 → 15 with a sim that has
   re-earned its credibility.
5. **Rebuild the training-data path (§7)** — archive features to S3/parquet,
   retrain on a rolling multi-month window, promote on OOS IC.

**Do not** retime entries (§6). **Do not** resume trading until 1–3 land.

---

*Method notes: offline IC uses per-ticker walk-forward with indicators shifted 1
bar. Simulated Sharpe figures annualise a ~15-day window and are not forward
estimates — judge on PF and DD. Sim n=90 vs live n=30 across only 4 live trading
days; the live sample is ~4 independent day-bets, so its variance is enormous
regardless of the above.*
