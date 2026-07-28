# StockBot Program Roadmap

Maintained by the TPM persona (Program Office weekly review + nightly desk).
Last updated: 2026-07-28 (Day 5/5 integrity clean-streak confirmation).

## 🔴 CODE RED — declared 2026-07-20 by owner (CFO)

**Trigger:** trailing-7d PF 0.30 (n=17) AND a confirmed ledger defect: partial-fill
exits wrote corrupted `pnl_pct`, and `_seed_kelly_from_db()` re-injected those rows into
position sizing after every redeploy. Full declaration: `reports/program/CODE-RED-2026-07-20.md`.

**Posture (manifesto governance rule F):** integrity first, validation throughput
second, new alpha queued.

**Exit criteria (all three required):**

| Criterion | Status |
|-----------|--------|
| Integrity Sentinel clean 5 consecutive trading days | ✅ **MET** — ALL GREEN at 08:25 UTC Jul 28. 5 consecutive clean days (Jul 24–28). |
| Kelly window verified sane | ✅ Sentinel confirmed ok |
| ≥1 hypothesis reaches data_run | ❌ BLOCKED — Railway worker not yet enabled (4 weeks) |

**Repair summary (complete):** PRs #14, #15, #17, #20 healed all 5 corrupt rows.
Fill-corrected T30 PF = 0.864 (was 0.60 stored). Integrity criterion **MET** as of Jul 28.
CODE RED cannot exit until Railway worker unblocks the `data_run` criterion.

## North Star

Stable ops → measured edge → paper-trading gate → client product.
No stage skips a gate. PnL is never reported without Sharpe/PF/WR/DD/n.

**M2 WARNING:** n=32 frozen-config trades closed at PF=0.741. At current trajectory
(~5 trades/day), n=100 arrives ~mid-August. PF≥1.2 is not on track. Owner decision
needed before that gate closes (see W31 report Decision #3).

## Milestones

| ID | Milestone | Gate | Status |
|----|-----------|------|--------|
| M1 | Outage-free operations | 2 weeks w/o critical watchdog event | 🟡 IN PROGRESS — ~17 days (Jul 10–27 est.); live-snapshot relay restored cloud visibility (PR #16) |
| M2 | Measured edge on frozen config | PF ≥ 1.2 at n ≥ 100 closed trades on v0.4.4 | 🔴 OFF TRACK — n=32 @ PF=0.741 (Jul 13–23). W31 sub-window PF=1.42 (n=13, noise). **n=100 ~mid-August on current pace.** |
| M3 | H1 cross-sectional validation | Backtest improves BOTH tune + hold-out legs | 🔴 BLOCKED — Railway worker not enabled. H1 draft PR #7 ready. |
| M4 | H5/H3/H2/H4 validation (data runs) | Same walk-forward standard | 🔴 BLOCKED — H4 at 17 days (3.4× flag), all others also flagged. |
| M5 | Paper-trading gate | Sharpe ≥ 1.5, DD ≤ 8%, 3 months | ⚪ NOT STARTED — depends on M2 |
| M6 | Client/commercial track | M5 + registration/partner decision | ⚪ NOT STARTED |

## Current Single Bottleneck (TPM)

**Railway worker service** (`python agent_worker.py`, env `AGENT_WORKER_ENABLE=true`
+ Alpaca paper keys). Blocks M3, M4, and CODE RED exit (data_run criterion).
Outstanding **3 consecutive weeks** (since W29, Jul 15). ~10 minutes in Railway dashboard.

All 7+ hypothesis validations blocked behind this single action.

## Freeze Status (TPM-enforced)

- **Strategy FROZEN at v0.4.4** since 2026-07-13. Intact through W31 (2026-07-27).
  No strategy merges. No risk control changes.
- Bug fixes, infra, monitoring always exempt.
- Deployed since freeze: v0.4.5–v0.4.17 (monitoring, bug fixes), v0.5.0–v0.5.3 (CODE RED
  repairs, TP fix, heat cap fix), v0.5.4 (durable risk state + halt alerting, PR #24),
  v0.5.5 (peak equity reconciliation fix — `max(stored, broker, current)` on every start).
- **⚠️ v0.5.0 CLASSIFICATION PENDING (owner decision):** `dff72eb` promoted a retrained
  LGBM model alongside a staleness-trap bug fix. Staleness fix = exempt. Model promotion =
  ambiguous (bug fix or strategy change?). If strategy change, M2 clock resets to v0.5.0
  deploy. Owner must confirm classification.
- Draft PRs queued (freeze intact): #7 (H1), #9 (H5 phase), #10 (H2+H6 phases), #11 (H7)

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Railway worker not running — all validation + CODE RED exit blocked | **CRITICAL** | Week 3; ~10 min owner action in Railway dashboard |
| v0.5.0 LGBM retrain classification | **HIGH (NEW)** | Owner must confirm bug-fix vs strategy-change; affects M2 clock |
| M2 trajectory: PF 0.741 @ n=32, gate at n≥100 | **HIGH** | ~13 more trading days to n=100; PF<1.2 on current path; owner decision needed |
| Hypothesis accumulation without validation | **HIGH** | 8 hypotheses queued, 0 data runs in 3 weeks |
| H4 at 17 days in needs_data_run | **MED** | 3.4× flag threshold; Railway is only fix |
| Attribution blur — compound v0.4.4 | **MED** | Need Railway isolated backtests |
| Long-only assumption landmines | **LOW** | v0.5.x fixes addressed TP and heat; audit remaining paths |

## Decision Log

- 2026-07-28: Integrity exit criterion MET (Day 5/5 clean). v0.5.4 durable risk state
  deployed (max_drawdown was 0.00% anchored; true DD 6.66%). v0.5.5 peak equity fix deployed.
  CODE RED still active — Railway worker (data_run criterion) outstanding 4 weeks.
- 2026-07-27: W31 review — M2 off-track (PF 0.741@n=32); v0.5.0 LGBM classification
  flagged; M2 owner decision needed before n=100 (~mid-August); CODE RED Day 4/5
- 2026-07-20: CODE RED declared; Integrity Sentinel + Principal Skeptic onboarded
- 2026-07-13: Hard freeze at v0.4.4; TPM+PM personas onboarded
- 2026-07-15: W29 weekly review — freeze confirmed; 3 owner decisions escalated
- 2026-07-20: W30 weekly review — 403 CRITICAL (day 8); Railway worker CRITICAL (week 2)
- 2026-07-10: PR-only governance; risk controls never weakened (manifesto)
- 2026-07-11: H5 must be signal-conditional (unconditional 3-day holds = −493bps, June backtest)
- 2026-07-13: Owner approved H5+H3 deploy ahead of data runs (paper = lab)
