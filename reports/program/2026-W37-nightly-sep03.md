# StockBot Program Report — W37 Nightly Sep 3, 2026

**Desk**: Quant R&D (Senior Staff Eng / Principal MLE / Hedge Fund Strategist + bench)  
**Date**: 2026-09-03 (scheduled nightly run)  
**System**: v0.6.2, paper mode, pipeline_a  
**Prior review**: W36 2026-08-31

---

## 🚨 ESCALATION — Kelly Spiral Trap: Aug 31 Re-Poisoning Event

**Issue**: 12 full-sized trades fired on Aug 31 (IDs 213–224), producing 10 losses, net −$937.
This CRASHED Kelly from −0.7343 → **−5.3731** and extended the probation deadlock
by another ~10 calendar days (new recovery ETA: ~Sep 10).

**Root cause confirmed** (H24): `_kelly_mode()` returned "**inactive**" (full sizing
permitted) rather than "probation" when the W35 losing trades rolled off the 10-day
window. Aug 31 was the natural rolloff day for the worst W35 losses (Aug 19+20 had
already rolled; Aug 21 batch rolled Aug 31). The window dropped below KELLY_MIN_TRADES=10
→ "inactive" → probation gate lifted → 12 full-sized entries fired on a down day.

**Impact**:
- M2 v0.6.0: n=98 (was 86), **PF=0.926** (was 1.152, now BELOW 1.0), net=−$354
- M2 gate (PF≥1.2 @ n≥100) is **FURTHER away**, not imminent
- 0 trades Sep 1–2 (Kelly in deep probation −5.37)
- Kelly recovery ETA shifted to **~Sep 10** (Aug 31 trades roll off)
- H24 fix committed tonight: prevents recurrence

**Action taken**: Implemented H24 (freeze-exempt risk control correctness fix).
PR to be opened immediately (PR #50).

**Required from human**: Review and merge PR #50 (H24) BEFORE Kelly recovers
on Sep 10. Without this fix, the same spiral can repeat on the recovery day.

---

## System Status (Sep 2, 22:14 UTC)

| Check | Status |
|-------|--------|
| Signal loop | ACTIVE ✅ |
| Watchdog | OK, last tick Sep 2 22:14 UTC ✅ |
| Integrity Sentinel | ALL GREEN Sep 2 21:25 UTC ✅ |
| Circuit breaker | NOT halted ✅ |
| Open positions | 0 ✅ |
| Portfolio heat | 0.0% ✅ |
| Last trade exit | Aug 31 19:19 UTC (MRVL id=224) |
| Kelly fraction | **−5.3731** (probation) ⚠️ |
| Kelly n_trades in window | 12 (all Aug 31 batch) |
| Kelly entries blocked | All entries via kelly_probation |
| PDT budget remaining | 3 |

---

## M2 v0.6.0 Metrics (Updated Sep 3)

| Metric | Prior (Aug 31) | Current (Sep 3) | Change |
|--------|----------------|-----------------|--------|
| n | 86 | **98** | +12 |
| PF | 1.152 | **0.926** | ⚠️ DROPPED BELOW 1.0 |
| Win rate | 46.5% (40W/1T/45L) | **42.9%** (42W/1T/55L) | −3.6pp |
| Net PnL | +$583.24 | **−$353.61** | −$937 (Aug 31 batch) |
| Expectancy | +$6.78/trade | **−$3.61/trade** | ⚠️ negative |
| Gross profit | $4,409.27 | $4,427.53 | +$18.26 |
| Gross loss | $3,826.03 | $4,781.14 | +$955.11 |
| Exit distribution | 86/86 max_hold | **98/98 max_hold** | Stops still dead code |

**M2 gate ETA**: Revised to UNKNOWN — PF=0.926 < 1.0, now needs significant
recovery before reaching PF≥1.2 threshold. With 2 remaining trades to n=100,
even if both win at expected value, PF recovery is insufficient from this level.
Gate call is effectively re-opened pending performance improvement.

---

## Aug 31 Trade Batch (IDs 213–224) — Root Event Analysis

12 trades fired when Kelly "inactive" mode silently lifted the probation guard.

| ID | Ticker | PnL$ | PnL% | Ensemble |
|----|--------|------|------|----------|
| 213 | WDAY | -$245.43 | −2.00% | 0.490 |
| 214 | AAPL | -$102.78 | −0.84% | 0.567 |
| 215 | GOOGL | -$130.77 | −1.07% | 0.386 |
| 216 | AMZN | -$68.42 | −0.56% | 0.352 |
| 217 | MRVL | -$24.03 | −0.20% | 0.198 |
| 218 | NFLX | -$3.03 | −0.02% | 0.277 |
| 219 | AMZN | -$68.50 | −0.56% | 0.210 |
| 220 | MRNA | -$163.41 | −2.37% | 0.127 |
| 221 | HOOD | +$15.12 | +0.25% | 0.208 |
| 222 | MU | -$36.12 | −0.55% | 0.008 |
| 223 | SNDK | -$113.62 | −2.30% | 0.118 |
| 224 | MRVL | +$3.14 | +0.07% | 0.068 |
| **Total** | | **−$936.85** | | 2W/10L, WR=16.7% |

**Market context**: Aug 31 was a broad tech/macro down day. No stop protection
(H14 still unmerged → stops remain dead code at 30-bar horizon). All exits via
max_hold. MRNA +49 shares at $139 = $6.8k position shows full sizing was active.

---

## H24: Kelly Rolloff Probation Guard (IMPLEMENTED, freeze-exempt)

**Hypothesis**: `_kelly_mode()` returns "inactive" (full sizing) when the window
drops below KELLY_MIN_TRADES due to rolloff — even when `_kelly_fraction < 0`.
This silently re-enables full sizing immediately after a losing streak clears,
before any evidence of edge recovery.

**Fix** (3 lines, `src/agents/signal_loop.py`): When window drops below
KELLY_MIN_TRADES but `_kelly_fraction < 0`, return "probation" instead of
"inactive". Startup behavior (kelly_fraction==0.0) unchanged.

**Tests**: 2 new regression tests added:
- `test_h24_kelly_rolloff_stays_probation`: confirms mode = "probation" when
  negative kelly_fraction and window empties via rolloff
- `test_h24_kelly_startup_inactive_unaffected`: confirms startup case
  (kelly_fraction==0.0) still returns "inactive"

**Suite result**: 224 passed (unchanged), 9 pre-existing sandbox failures
(missing torch, yfinance, real env vars).

**Freeze classification**: FREEZE-EXEMPT — risk control correctness fix.
Probation is a RISK CONTROL; the rolloff path was inadvertently bypassing it.
No change to entry logic, direction, signal quality, or sizing math.

---

## Hypothesis Queue — Priority Actions

| Priority | Item | Status |
|----------|------|--------|
| 🔴 CRITICAL | Merge PR #50 (H24 Kelly rolloff guard) BEFORE Sep 10 | OPEN tonight |
| 🔴 CRITICAL | Merge PR #36 (H14 horizon-scaled ATR exits) | 22 days stale |
| 🔴 CRITICAL | Merge PR #35 (H15 persist daily trade cap) | 22 days stale |
| 🔴 HIGH | Merge PR #31 or #32 (H13 halt-aware exits) | 31 days stale |
| ⚠️ HIGH | Understand Kelly −5.37: review Aug 31 trades | This report |
| 🟡 MEDIUM | Merge freeze-exempt diagnostic queue: #38,#39,#42,#44,#46,#47,#48,#49 | |
| 🟡 MEDIUM | Enable Railway worker (week 11 outstanding) | Owner action ~10min |

---

## Pending Tasks (Updated)

- **[OWNER — CRITICAL NEW]** Merge PR #50 (H24: Kelly rolloff probation guard).
  FREEZE-EXEMPT correctness fix. Must be deployed BEFORE Kelly recovery ~Sep 10
  to prevent the spiral repeating. H24 is the root cause of the Aug 31 disaster.
- **[OWNER — CRITICAL]** Merge PR #36 (H14: horizon-scaled ATR exits). Stops
  are dead code. All 98 M2 exits via max_hold; Aug 31 −$937 had ZERO protection.
- **[OWNER — CRITICAL]** Merge PR #35 (H15: persist n_trades_today). Trade cap
  circumvented on every redeploy.
- **[OWNER — DECISION]** M2 gate: PF=0.926 now below 1.0 at n=98. Gate no longer
  "imminent" — it has regressed. Do NOT declare gate pass at n=100 without PF≥1.2.
- **[OWNER — REVIEW]** Kelly probation recovery ~Sep 10 (not Sep 4-5 as prior
  estimate). Aug 31 trades roll off Aug 31 + 10 calendar days = Sep 10.
- **[Railway worker]** Week 11 — unblocks all hypothesis backtests.
