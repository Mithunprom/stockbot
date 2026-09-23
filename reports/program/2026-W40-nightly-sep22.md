# Nightly R&D Desk Review — 2026-09-22

**Desk personas active:** Senior Staff Engineer, Principal MLE, Hedge Fund Strategist,
Index/Risk Quant (veto), Integrity Sentinel, Principal Skeptic, TPM, PM

---

## Status Summary

| Item | Value |
|------|-------|
| Version deployed | v0.7.3 |
| M3 measurement window | ACTIVE — n=10 (Day 1) |
| Watchdog | OK — last tick 2026-09-21T22:05 UTC |
| Integrity | ALL GREEN — last run 2026-09-21T21:25 UTC |
| Kelly mode | **HARD-BLOCKED** (fraction=-0.2699, threshold=-0.25) |
| entry_rank_mean | **96.9** ✅ (≥85 threshold confirmed) |
| Open positions | 0 |
| CODE RED | ACTIVE — Railway worker week 15 |

---

## M3 Day 1 — September 21, 2026

### Trade Ledger (all 10 closed trades)

| ID | Ticker | Entry (UTC) | Exit (UTC) | PnL | PnL% | Exit Reason | Ensemble |
|----|--------|-------------|------------|-----|------|-------------|----------|
| 238 | HOOD | 13:40:07 | 14:12:03 | -$68.21 | -0.56% | max_hold | +0.593 |
| 239 | COIN | 13:41:03 | 14:13:06 | -$142.25 | -1.18% | max_hold | +0.634 |
| 240 | STX | 13:41:04 | 14:12:05 | **-$408.33** | -3.36% | **stop_loss** | +0.459 |
| 241 | TSLA | 13:42:24 | 14:14:04 | +$35.65 | +0.29% | max_hold | +0.429 |
| 242 | LMT | 13:43:07 | 14:14:02 | -$51.55 | -0.43% | max_hold | +0.445 |
| 243 | INTC | 17:01:03 | 17:32:09 | -$74.13 | -1.08% | max_hold | +0.125 |
| 244 | ARM | 17:02:03 | 17:33:02 | -$14.04 | -0.21% | max_hold | +0.311 |
| 245 | MSTR | 17:21:03 | 17:52:03 | -$31.57 | -0.71% | max_hold | +0.181 |
| 246 | **SNDK** | 17:34:02 | 18:05:04 | +$6.49 | +0.08% | max_hold | **-0.069** ⚠️ |
| 247 | COIN | 17:51:02 | 18:22:02 | -$8.10 | -0.15% | max_hold | +0.353 |

**M3 n=10 | 2W/8L | Win rate: 20% | PF: 0.053 | Net: -$757.54**

> Index/Risk Quant veto note: **n=10 is noise — do not read strategy signal from this.**
> Gate is PF≥1.2 @ n≥100. The n=10 window is also structurally contaminated
> by the H15 double-batch defect (see below). No hypothesis conclusions drawn.

---

## Key Findings

### ✅ Finding 1: Train/Serve Skew Fix Confirmed

`entry_rank_mean = 96.9` at `/diagnostics` after 5 measured fills (n=5 — minimum
threshold for reporting). This is the most important signal from today: the core
root-cause fix (OBV session-anchoring, WARMUP_BARS 300→1950) is holding.

- v0.6.x production: ~46th percentile (entries were random relative to model ranking)
- v0.7.0 Day 1: **96.9th percentile** — bot is entering at top-decile model predictions

This validates the theory. Continue monitoring as n grows.

### ✅ Finding 2: Exit Barriers Are Live

STX (ID 240) exited via `stop_loss` at -3.36% (entry 891.99, exit 862.01). This is
the first stop-loss exit in M3 and confirms the H14 ATR-scaled barriers are functioning.
Under v0.6.x all 111 exits were via `max_hold` (barriers were dead code). The barriers
are now active.

> Execution Trader note: STX moved -3.36% intraday in 30 minutes — this is a genuine
> event, not a tight stop miscalibration. The barrier fired correctly.

### 🚨 Finding 3: H15 Double-Batch Confirmed (Critical)

**10 trades fired vs SIZING_MAX_TRADES_PER_DAY=6.**

Trade batches:
- Batch 1: IDs 238–242 (13:40–13:43 UTC) — 5 trades, market open
- Batch 2: IDs 243–247 (17:01–17:51 UTC) — 5 more trades, 3.5 hours later

Diagnostics at EOD shows `n_trades_today: 5` — confirming the counter was reset by a
mid-session restart and only the second batch is counted. This is the exact H15 defect:
"mid-day restarts reset `_sizing_n_trades_today` to 0, granting a second allotment."

**PR #35 (H15) is 39 days stale and unmerged.** The double-batch on M3 Day 1:
- Added 5 extra trades (4 losses including the $408 STX loss)
- Pushed Kelly into hard-block territory (-0.2699 ≤ -0.25)
- Contaminated the M3 measurement window

> TPM flag: PR #35 caused measurable harm in the second occurrence (first: Aug 10
> +15 trades; today: +5 trades + Kelly hard-block). Every day unmerged is a daily
> risk of a second allotment on any redeploy.

### 🚨 Finding 4: SNDK Negative-Ensemble Entry (Second Anomaly)

SNDK (ID 246) entered LONG with `ensemble_signal = -0.069`. The composite ensemble
(LGBM 75% + sentiment 25%) was net bearish while the model's individual scores
(pred_return, dir_prob) passed their gates.

This is the second instance of the JNJ anomaly (ID 163, Aug 2026, ensemble=-0.0238).
H18a (PR #41) blocks on `ensemble_signal ≤ 0`. It is still open with freeze
classification pending.

**H28 (tonight):** Added `ensemble_direction_ok` field and `signal_direction_anomalies`
count to `/diagnostics` so this situation is visible before trades execute.

> Principal Skeptic: SNDK was a winner (+$6.49). This does not vindicate the entry.
> A negative-ensemble entry that wins is still a broken gate — it means the model
> said "bearish" and the price moved favorably by chance. H18a classification
> remains necessary.

### 🚨 Finding 5: Kelly Hard-Blocked

The 10 trades on Sep 21 (2W/8L) drove Kelly from 0.0 (inactive) to -0.2699.
The `KELLY_HARD_BLOCK_THRESHOLD = -0.25` is breached.

**`kelly_entries_blocked = true`** — verified at `/diagnostics`. Entries (including
probes) are correctly prevented. The hard block fires at Gate 2b in
`_sizing_entry_gate_open()`, before the probation probe path at Gate 3.

Recovery path: Sep 21 trades roll off the 10-day lookback window around Oct 1.
With 0 new trades in the window (entries blocked), Kelly will naturally recover.

> Hedge Fund Strategist: Do NOT attempt to accelerate Kelly recovery by enabling
> probes or weakening the hard block. The block exists precisely for this scenario.
> Let it clear naturally.

---

## Code Shipped Tonight: H28

**Hypothesis:** Add ensemble direction consistency to `/diagnostics` so negative-ensemble
entry candidates are flagged before they trade.

**Changes:**
- `main.py`: Added to each `signal_gate_analysis` entry:
  - `ensemble_direction_ok: bool` — True when `ensemble_signal > 0`
  - `"ensemble_direction_inconsistent"` added to `blocked_by` list when applicable
- `main.py`: Added top-level diagnostics fields:
  - `signal_direction_anomalies: int` — count of passing-gate signals with negative ensemble
  - `kelly_hard_blocked: bool` — mirrors `kelly_entries_blocked`, named to distinguish from probation
- `tests/unit/test_h28_diagnostics.py`: 5 tests, all pass

**Validation:** No data run required (diagnostic-only). 5/5 unit tests green.
Freeze-exempt: does not change entry gate logic.

---

## Open PRs Requiring Owner Action (ranked by urgency)

| Priority | PR | Hypothesis | Status | Days Stale |
|----------|-----|------------|--------|-----------|
| 🔴 CRITICAL | #35 | H15 persist n_trades_today | OPEN | **39d** |
| 🔴 CRITICAL | #51–54 | H24 Kelly boundary cluster | OPEN | 21d |
| 🔴 CRITICAL | #56–57 | H25 Kelly systemic guard | OPEN | 14d |
| 🟠 HIGH | #31 or #32 | H13 halt-aware exits | OPEN | **46d** |
| 🟠 HIGH | #41 | H18a ensemble direction gate | OPEN | 34d (classify) |
| 🟡 MEDIUM | #62 | H27 sector map (may be superseded by v0.7.0 fail-closed) | OPEN | 4d |
| 🟡 MEDIUM | #43 | H18b SPY session gate | OPEN | 32d (classify) |
| 🔵 LOW | #28 | H28 this PR | NEW | 0d |

> TPM: PR #35 is the single highest-leverage unblock. One merge eliminates the
> double-batch structural contamination of M3 AND removes the risk of Kelly
> hard-blocks from excess trades on restarts.

---

## CODE RED Status

| Criterion | Status |
|-----------|--------|
| Integrity clean 5 days | ✅ MET (Jul 28 2026; continuing clean Sep 21) |
| Kelly window sane | 🚨 HARD-BLOCKED — fraction -0.2699, recovery ~Oct 1 |
| ≥1 hypothesis at data_run | ❌ BLOCKED — Railway worker week 15 |

Exit criteria not met. CODE RED continues.

---

## Tomorrow's Priority

1. Owner merges PR #35 (H15) before market open — prevents another double-batch
2. Monitor Kelly recovery at `/diagnostics`
3. Watch `entry_rank_mean` continue building sample (currently n=5 measured fills)
4. No new strategy hypotheses — freeze intact, M3 measurement window contaminated
   until H15 merges and Kelly recovers (~Oct 1)
