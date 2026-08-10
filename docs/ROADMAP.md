# StockBot Program Roadmap

Maintained by the TPM persona (Program Office weekly review + nightly desk).
Last updated: 2026-08-10 W33 review (v0.6.2 deployed; MSCI zombie resolved; halt lifted; M2 n=16; Railway worker week 7).

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
| ≥1 hypothesis reaches data_run | ❌ BLOCKED — Railway worker not yet enabled (7 weeks) |

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
| M1 | Outage-free operations | 2 weeks w/o critical watchdog event | 🟡 IN PROGRESS — ~31 days (Jul 10–Aug 10); watchdog clean; no outages |
| M2 | Measured edge on new config | PF ≥ 1.2 at n ≥ 100 closed trades on v0.6.0 | 🟡 IN PROGRESS — n=16, PF=2.02 (noise — n too small). Bot resumed Aug 6. ETA mid-Sep at ~5 trades/day. |
| M3 | H1 cross-sectional validation | Backtest improves BOTH tune + hold-out legs | 🔴 BLOCKED — Railway worker not enabled. H1 draft PR #7 ready. Week 7. |
| M4 | H5/H3/H2/H4 validation (data runs) | Same walk-forward standard | 🔴 BLOCKED — H4 at 31 days (6× flag), 13 hypotheses total blocked. Week 7. |
| M5 | Paper-trading gate | Sharpe ≥ 1.5, DD ≤ 8%, 3 months | ⚪ NOT STARTED — depends on M2 |
| M6 | Client/commercial track | M5 + registration/partner decision | ⚪ NOT STARTED |

## Current Single Bottleneck (TPM)

**Railway worker service** (`python agent_worker.py`, env `AGENT_WORKER_ENABLE=true`
+ Alpaca paper keys). Blocks M3, M4, and CODE RED exit (data_run criterion).
Outstanding **7 consecutive weeks** (since W29, Jul 15). ~10 minutes in Railway dashboard.

All 13 hypothesis validations (H0–H12 + H13) blocked behind this single action.

Secondary bottleneck resolved: halt lifted Aug 6, MSCI zombie closed Aug 10 (v0.6.2). M2 is now
accumulating at n=16. Remaining structural risk: halt-aware exits (PRs #31/#32) not yet merged.

## Freeze Status (TPM-enforced)

- **Strategy FROZEN at v0.6.0** since 2026-08-02. No strategy merges until n=100 closed
  trades on v0.6.0 (max_hold=30) or a walk-forward-validated backtest justifies an exception.
- Bug fixes, infra, monitoring always exempt.
- v0.4.4 freeze (Jul 13) superseded by v0.6.0 (Aug 2): a governance-compliant exception
  backed by two OOS backtest windows (IC horizon sweep, PF@30=4.55 OOS + 3.67 holdout).
- v0.6.0 deployed: max_hold 390→30, MAX_HOLD_EXTENSIONS 2→0, SIZING_STAGNATION_BARS 390→30.
  224 tests. STRATEGY CHANGE — M2 clock resets to Aug 2, 2026.
- v0.6.1 deployed: resume-persistence fix (not a strategy change).
- **⚠️ v0.5.0 CLASSIFICATION MOOT:** v0.6.0 supersedes both frozen configs. M2 clock
  resets from v0.6.0 regardless. Owner should log formally but urgency is low.
- Draft PRs queued: #7 (H1), #9 (H5 phase needs re-scope for 30-bar), #10 (H2+H6),
  #11 (H7 needs re-scope: stagnation=30=max_hold now), #26 (H11), #29 (H12).
- H5 (hold extension) disabled in prod (MAX_HOLD_EXTENSIONS=0); needs re-evaluation
  under 30-bar horizon.
- H10 (shorter max-hold) SUPERSEDED by v0.6.0 — max_hold=30 is live.
- H7 (stagnation exits) needs re-scoping: stagnation_bars now equals max_hold (both 30).
- Non-draft PRs #25 (PROD_PARAMS) and #27 (H9+H10) ready to merge.

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Railway worker not running — CODE RED exit + 13 hypothesis validations blocked | **CRITICAL** | Week 7; ~10 min owner action in Railway dashboard |
| Halt-aware exits not merged (PRs #31/#32) — halt could block exits if CB fires again | **HIGH** | Owner: choose #31 or #32 and merge; v0.6.2 fixed universe bug but halt-path still blocks exits |
| M2 n=16 — PF=2.02 driven by 1 trade (MSTR +$504, 43% of gross profit) | **MED** | Monitor trajectory as n grows; do not act on signal until n≥50 |
| Portfolio_heat blind spot sentinel check pending (PR #33) | **MED** | v0.6.2 fixes root cause; PR #33 adds ongoing detection — merge soon |
| H5 + H7 need re-scoping for 30-bar horizon | **MED** | Backtest phases target wrong regime; re-scope before Railway runs |
| Hypothesis accumulation without validation | **MED** | 13 hypotheses queued, 0 data runs in 7 weeks (Railway worker) |
| Non-draft PRs aging without merge (#25, #27) | **MED** | #25 safe to merge (infra); #27 safe to merge (probe floor + bug fix) |
| **RESOLVED** Bot HALTED — ✅ Halt lifted Aug 6 | — | — |
| **RESOLVED** MSCI zombie (id 119, 14d open) — ✅ Closed Aug 10 via v0.6.2 | — | — |
| **RESOLVED** Sentinel CRITICAL stale_open_rows — ✅ Clearing after MSCI exit | — | — |

## Decision Log

- 2026-08-10: W33 review — halt lifted (Aug 6), MSCI zombie closed (Aug 10) via v0.6.2 (universe-rotation exit bug). v0.6.2 revealed portfolio_heat blind spot ($12.6k / 12.9% deployed and invisible during zombie). M2 v0.6.0: n=16, PF=2.02 (noise). Railway worker week 7 — sole CODE RED exit blocker. Decisions outstanding: Railway worker (CRITICAL), halt-aware exits PRs #31/#32 (HIGH), PR #25/#27 merge (MED).
- 2026-08-03: W32 review — bot halted (Jul 28), M2 reset (v0.6.0), Railway week 6. Sentinel
  CRITICAL (stale_open_rows all 6 positions). H10 marked superseded by v0.6.0. H5/H7 flagged
  for re-scoping. 2 CRITICAL owner decisions outstanding: lift halt + Railway worker.
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
