# Exit geometry — measured on real minute paths (2026-09-23)

**Verdict: the v0.8.0 exit geometry is sound. Do not change the ratios. The
binding requirement is entry quality (`entry_rank_mean`), not the exit ladder.**

## What prompted this

v0.8.0 shipped `EXIT_PROFIT_TARGET_MODE` (env-gated, default OFF) with
`STOP_TO_TARGET_RATIO = 0.8` — reward:risk 1.25, which needs a 44.4% win rate to
break even before costs. Production's observed win rates were 33.3% (M2 window)
and 27.3% (first post-fix session), which implies break-even R:R of 2.00–2.66.
On that arithmetic the geometry looked like it lost money by construction.

**That reasoning was wrong, and the error is worth recording.** It applied a win
rate measured on entries the train/serve skew bug selected at the ~46th
percentile to a geometry that will run on top-decile entries. Those are different
strategies. At the entry quality the bot now achieves (`entry_rank_mean = 96.9`),
the measured win rate is ~55%, and R:R 1.25 clears break-even comfortably.

## Method

Neither obvious evidence source was usable:

* **the research backtest's PF** — sim stopped reproducing production in Sep 2026
  (7.27-point PF gap); fidelity has not been re-established;
* **the live M2 ledger and the S3 feature archive** — `src/features/live.py`
  writes `feature_matrix`, so every archived row before 2026-09-21 carries v1
  (skewed) feature semantics. Only two sessions are true v2.

So the inputs were rebuilt the honest way and the question narrowed to one that
needs no simulator:

1. Alpaca IEX 1m bars, 20 tickers, 2026-06-01 → 09-22 (`--phase fetch`).
2. Full-history v2 features, computed exactly as training computes them
   (`--phase features`) — not the live incremental path.
3. Score every bar with the v2 checkpoint `lgbm_ic_0.1830.pkl` (val IC 0.1830).
4. Keep the top decile of **each bar's cross-section**, mirroring how
   `entry_rank_mean` is defined, plus the live `pred > 0` long-only gate.
5. For each entry, walk the forward path bar by bar on real high/low and record
   which barrier price touches **first**.

Reproduce with:

```
RESEARCH_CACHE=/tmp/sb_exitgeom PYTHONPATH=. \
  python scripts/exit_geometry_study.py --percentile 90 \
    --since 2026-07-15 --trail-ratio 0.9
```

## Result — out of sample (entries from 2026-07-15, n = 42,472 paths)

Trailing stop at 0.9 × target, as v0.8.0 ships it.

| target | stop_ratio | R:R | win rate | exp (bps) | PF |
|-------:|-----------:|----:|---------:|----------:|---:|
| 2.5% | **0.80** | **1.25** | **54.7%** | **23.92** | **1.62** |
| 2.5% | 0.65 | 1.54 | 54.3% | 23.75 | 1.63 |
| 2.5% | 0.50 | 2.00 | 52.8% | 23.17 | 1.63 |
| 2.5% | 0.40 | 2.50 | 50.6% | 22.35 | 1.63 |
| 2.5% | 0.25 | 4.00 | 45.1% | 21.05 | 1.70 |
| 2.0% | 0.80 | 1.25 | 53.6% | 20.76 | 1.59 |
| 1.5% | 0.80 | 1.25 | 52.3% | 17.80 | 1.59 |

Three things the grid says:

* **Every cell is profitable** — expectancy 13–24 bps, PF 1.59–1.71. The signal
  has real edge once entries are top-decile.
* **Expectancy is nearly flat in `stop_ratio`** (23.92 → 21.05 across a 3.2x
  change in R:R). The stop placement is not where the money is; the *target
  size* is. Tighter stops do buy PF (1.62 → 1.70) at the cost of win rate, so
  `stop_ratio` 0.4–0.5 is a defensible choice if the M3 gate is PF-shaped —
  but it is a preference, not a correction.
* **OOS ≥ in-sample** (23.92 vs 21.40 bps on the full window), so the result is
  not a training-span artifact.

## Caveats — read before quoting these numbers

* **20 tickers, not the 75-name universe.** Liquid large caps; the unmapped
  tail is unrepresented.
* **These are not the bot's trades.** Entries here are *every* top-decile bar
  (42k paths). Live entries additionally pass the cost threshold, the dir_prob
  dead zone, sector caps and a 6/day limit — roughly 6 trades a day. Per-trade
  expectancy should carry; the trade count will not.
* **Costs are a flat 2 bps** round trip. Real slippage on the thinner names
  will exceed that.
* Same-bar ambiguity resolves to the **stop** (pessimistic), applied uniformly
  across the grid so it cannot favour one geometry.
* **The whole result is conditional on `entry_rank_mean` staying ≥ 85.** At the
  46th percentile the Sep 16 diagnosis measured −0.291%/trade at PF 0.34. Entry
  quality is the load-bearing variable; the exit ladder is second order.

## What this does not explain

The first post-fix session (2026-09-21) still lost money: n=11, WR 27.3%,
PF 0.59, −$329.84, which drove `kelly_fraction` to −0.2699 and tripped the hard
block. n=11 is far too small to contradict a 42,472-path measurement, and one
trade (STX, −$408) dominates it. But it is not yet explained, and the honest
reading is that the exit geometry is now cleared as a suspect — not that the
system is proven profitable.
