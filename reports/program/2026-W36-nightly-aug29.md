# 📊 Nightly Desk — 2026-08-29 (W36)

**Generated:** 2026-08-29 (automated nightly run)
**Bot Version:** v0.6.2 (paper trading mode)
**Strategy Freeze:** ACTIVE since Aug 2

---

## Summary

Kelly probation deadlock continues — 0 trades taken Aug 25-28. Aug 19 semiconductor
losses dominate the 10-day Kelly window. Natural rolloff expected ~Sep 2 (10 trading
days after Aug 19). H23 (Kelly rolloff diagnostics) implemented tonight to surface
this schedule in the diagnostics endpoint.

M2 gate stalled at n≈90 / PF≈1.1. Gate requires n=100 / PF≥1.2. Neither will
advance until Kelly probation clears or the IC gate logic is revisited.

---

## Live System Status

| Metric | Value |
|--------|-------|
| Version | v0.6.2 (unchanged) |
| Mode | Paper trading |
| Watchdog | OK — last tick Aug 28 22:49 UTC |
| Open positions | 0 |
| Portfolio heat | 0.0% |
| Trades since Aug 21 | 0 |
| Integrity sentinel | ALL GREEN (Aug 28 22:25 UTC) |
| Signal loop | Active |

---

## Kelly Governor — PROBATION DEADLOCK

| Metric | Value |
|--------|-------|
| kelly_fraction | -0.7343 ⚠️ |
| kelly_mode | probation |
| kelly_n_trades (10d window) | 18 |
| probe_eligible tickers | 0 of 5 candidates (IC < 0.05 in 30d window for all) |
| Trades blocked today | 5 signals, all suppressed by probation |

**Root cause:** Aug 19 semiconductor selloff produced 4 large losses in a single day
(INTC -$371, KLAC -$328, WDC -$318, MU -$220 = -$1,222 total). These entries drag
kelly_fraction to -0.7343, triggering probation. In probation, probe entries require
per-ticker IC ≥ 0.05 over the 30-day window — but the 30-day window itself contains
insufficient trades (all entries blocked by the same probation) → IC never accumulates
→ all 5 probe candidates remain blocked → perpetual deadlock.

**Expected relief:** Aug 19 losses are 10 calendar days (≈10 trading days) old.
They will age off the lookback window ~Sep 2, at which point Kelly will recalculate
from remaining outcomes. If the remaining 14 trades are net-positive, Kelly returns
to normal mode and entries resume.

**H23 impact:** The new `kelly_probation_clears_earliest`/`kelly_probation_clears_latest`
fields now tell the owner exactly this, surfaced in the `/diagnostics` endpoint.

---

## M2 Gate — v0.6.0

| Metric | Value |
|--------|-------|
| n | ≈90 (STALLED since Aug 21) |
| PF | ≈1.1 ⚠️ BELOW GATE (1.2) |
| Trades to gate | ≈10 |
| All exits via | max_hold (100%) |
| ETA | Unknown — depends on Kelly probation clearing |

Prior ETA (Aug 24) was Aug 25-26. That gate call did not fire because Kelly probation
blocked all entries Aug 25-28. New ETA: ~Sep 3-5 (assuming probation clears Sep 2 and
3-5 trades/day resumes at the prior rate).

---

## Tonight's Work — H23

**Hypothesis:** Kelly window loss-rolloff diagnostics. During probation, the owner has
no visibility into *when* the deadlock will resolve. Answering "when do the Aug 19
losses age off?" requires reading the in-process `_sizing_recent_outcomes` list, which
is not exposed externally. H23 adds this as a pure-diagnostic field.

**Implementation:**
- Added `_compute_kelly_rolloff(outcomes, lookback_days) -> dict` to `src/agents/signal_loop.py`
  as a module-level function (freeze-exempt: pure computation, no trading path impact)
- Wired into `get_portfolio_summary()` via `**_compute_kelly_rolloff(...)`
- New fields in `/diagnostics` response:
  - `kelly_window_negative_count`: how many losses are in the current window
  - `kelly_probation_clears_earliest`: when the oldest loss ages off (ISO timestamp)
  - `kelly_probation_clears_latest`: when the newest loss ages off (ISO timestamp)

**Tests:** 3 new unit tests in `tests/unit/test_signal_loop.py`:
- `test_h23_kelly_rolloff_all_positive` — no losses → all fields zero/None
- `test_h23_kelly_rolloff_single_loss` — single loss ages off at ts + lookback_days
- `test_h23_kelly_rolloff_multiple_losses` — earliest/latest correctly mapped

**Test suite result:** 63 passed, 3 skipped, 0 failures (full `tests/unit` suite)

**Governance:** Freeze-exempt — zero change to entry/exit logic, sizing, or strategy.
Pure diagnostic transparency.

---

## Open PRs (15 total)

| PR | ID | Priority | Status |
|----|-----|----------|--------|
| #36 | H14 | HIGH | Horizon-scaled ATR exits — freeze-exempt, merge anytime |
| #35 | H15 | HIGH | Persist daily trade cap — freeze-exempt, merge anytime |
| #31/#32 | H13 | HIGH | Halt-aware exits — choose one variant |
| #43 | H18b | HIGH-CLASSIFY | SPY session gate — FREEZE CLASSIFICATION NEEDED |
| #41 | H18a | MEDIUM | Ensemble direction gate — classify vs H12/PR #29 |
| #42 | H19 | MEDIUM | SECTOR_MAP completeness — freeze-exempt |
| #38 | H16 | MEDIUM-CLASSIFY | Session-boundary cap — classify: safety fix or strategy? |
| #39 | H17 | LOW | Diagnostic short-block fix — freeze-exempt |
| #44 | H18c | LOW | Kelly window by-day — freeze-exempt diagnostics |
| TBD | H23 | LOW | Kelly rolloff diagnostics — freeze-exempt (tonight) |

Plus DRAFT PRs: #7 (H1), #10 (H2/H6), #11 (H7), #26 (H11), #29 (H12) — all awaiting Railway.

---

## Owner Action Items

| Priority | Action | Effort |
|----------|--------|--------|
| **CRITICAL** | Enable Railway worker: `AGENT_WORKER_ENABLE=true` in Railway dashboard | ~10 min |
| **URGENT** | Merge PR #36 (H14 horizon-scaled exits) — stops are dead code for all 30-bar positions | review |
| **URGENT** | Merge PR #35 (H15 persist daily trade cap) — cap circumventable on restart | review |
| **HIGH** | Choose and merge PR #31 or #32 (H13 halt-aware exits) | review |
| **HIGH** | Classify PR #43 (H18b SPY session gate) — strategy change vs freeze-exempt | decision |
| **MEDIUM** | Merge PR #42 (H19 SECTOR_MAP completeness) | review |

---

## Single Bottleneck (TPM)

Railway worker — week 11. Enables M3, M4, and CODE RED exit. ~10 minutes.
Until it's enabled: strategy is frozen, validation is stalled, and every hypothesis
that needs historical data waits indefinitely. The desk has 15 open PRs and 13
blocked hypotheses. The bottleneck is not code; it's one env var in a dashboard.
