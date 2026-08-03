# StockBot Program Roadmap

Maintained by the TPM persona (Program Office weekly review + nightly desk).
Last updated: 2026-08-03 (v0.6.0 major — max_hold 390→30; M2 clock reset).

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

**M2 RESET (2026-08-02):** v0.6.0 changed max_hold 390→30 bars — a confirmed strategy
change (research: IC at 390 bars = 0.004, edge gone; backtest PF@30 = 4.55 vs 0.54 @390).
Prior M2 stats (PF=0.636@n=35) measured a broken strategy and are retired. New M2 window
starts from v0.6.0 deploy. Bot still HALTED; no new trades until halt is lifted.

## Milestones

| ID | Milestone | Gate | Status |
|----|-----------|------|--------|
| M1 | Outage-free operations | 2 weeks w/o critical watchdog event | 🟡 IN PROGRESS — ~17 days (Jul 10–27 est.); live-snapshot relay restored cloud visibility (PR #16) |
| M2 | Measured edge on new config | PF ≥ 1.2 at n ≥ 100 closed trades on v0.6.0 | 🔄 RESET — v0.6.0 (max_hold 390→30) is a strategy change; prior n=35@PF=0.636 retired. New window starts Aug 2. n=100 ~mid-October (bot halted; no trades until lift). |
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
- Deployed since original freeze: v0.4.5–v0.4.17 (monitoring, bug fixes), v0.5.0–v0.5.3
  (CODE RED repairs, TP fix, heat cap fix), v0.5.4 (durable risk state + halt alerting),
  v0.5.5 (peak equity reconciliation), v0.6.0 (max_hold 390→30 **STRATEGY CHANGE**),
  v0.6.1 (resume-persistence fix).
- **v0.6.0 resets the strategy freeze baseline to Aug 2, 2026.** Prior M2 window closed
  with invalid data. New M2 window: v0.6.0 (max_hold=30, stagnation=30, extensions=0).
- **⚠️ v0.5.0 CLASSIFICATION MOOT:** v0.6.0 supersedes both the frozen config and the
  v0.5.0 LGBM question. The M2 clock has reset.
- Draft PRs queued: #7 (H1), #9 (H5 phase), #10 (H2+H6 phases), #11 (H7).
  H5 (hold extension) is now disabled in prod (MAX_HOLD_EXTENSIONS=0); backtest phase
  needs re-scoping under the 30-bar horizon before merit can be assessed.

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Railway worker not running — CODE RED exit + all hypothesis validation blocked | **CRITICAL** | Week 5; ~10 min owner action in Railway dashboard |
| Stale open rows: MSCI/AMAT/TSLA >5d open, bot halted | **CRITICAL** | Sentinel critical since Aug 2; v0.6.0 max_hold fix deployed but halt blocks processing; will resolve at first run post-lift |
| Bot HALTED — no exits firing, 6 positions open | **HIGH** | max_drawdown CB since Jul 28; owner must call /admin/resume-trading (v0.6.1 will persist it) |
| M2 reset — new n=100 window not started (bot halted) | **HIGH** | No new trades until halt lifted; target ~mid-October |
| H5 (hold extension) disabled by v0.6.0 | **MED** | MAX_HOLD_EXTENSIONS=0; backtest phase needs re-scoping for 30-bar horizon |
| Hypothesis accumulation without validation | **MED** | 8+ hypotheses queued, 0 data runs in 5 weeks (Railway worker) |
| Sentinel snapshot stale — Railway may not be publishing | **MED** | Last snapshot 07:25 UTC Aug 2; possible service restart gap after v0.6.0 deploy |

## Decision Log

- 2026-08-02: v0.6.0 MAJOR — max_hold 390→30 (research: IC@390=0.004, edge gone 26x past
  training horizon). PF@30=4.55 vs 0.54@390 OOS. M2 clock reset. v0.6.1: resume-persistence
  fix. Sentinel CRITICAL: stale_open_rows (MSCI/AMAT/TSLA >5d). Bot still halted.
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
