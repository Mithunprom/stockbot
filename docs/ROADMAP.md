# StockBot Program Roadmap

Maintained by the TPM persona (Program Office weekly review + nightly desk).
Last updated: 2026-08-13 nightly review (v0.6.2 deployed; M2 n=37; PRs #35/#36 open; Railway worker week 8).

## 🔴 CODE RED — declared 2026-07-20 by owner (CFO)

**Trigger:** trailing-7d PF 0.30 (n=17) AND a confirmed ledger defect: partial-fill
exits wrote corrupted `pnl_pct`, and `_seed_kelly_from_db()` re-injected those rows into
position sizing after every redeploy. Full declaration: `reports/program/CODE-RED-2026-07-20.md`.

**Posture (manifesto governance rule F):** integrity first, validation throughput
second, new alpha queued.

**Exit criteria (all three required):**

| Criterion | Status |
|-----------|--------|
| Integrity Sentinel clean 5 consecutive trading days | ✅ **MET** — ALL GREEN at 08:25 UTC Jul 28. Continuing clean: ALL GREEN Aug 12 21:25 UTC. |
| Kelly window verified sane | ✅ OK — kelly_seed_sanity confirmed; mode: normal, fraction=0.2349 |
| ≥1 hypothesis reaches data_run | ❌ BLOCKED — Railway worker not yet enabled (8 consecutive weeks since Jul 15) |

**Repair summary (complete):** PRs #14, #15, #17, #20 healed all 5 corrupt rows.
Fill-corrected T30 PF = 0.864 (was 0.60 stored). Integrity criterion **MET** as of Jul 28.
CODE RED cannot exit until Railway worker unblocks the `data_run` criterion.

## North Star

Stable ops → measured edge → paper-trading gate → client product.
No stage skips a gate. PnL is never reported without Sharpe/PF/WR/DD/n.

**M2 RESET (2026-08-02):** v0.6.0 changed max_hold 390→30 bars — a confirmed strategy
change (research: IC at 390 bars = 0.004, edge gone; backtest PF@30 = 4.55 vs 0.54 @390).
Prior M2 stats (PF=0.636@n=35) measured a broken strategy and are retired. New M2 window
starts from v0.6.0 deploy. Bot resumed Aug 6.

**M2 status (Aug 13):** n=37, PF=1.87, WR=47.2%, net +$947.95. Still noise at n=37.
All 37 exits via max_hold — stops are miscalibrated for 30-bar horizon (H14 PR #36 addresses).
ETA n=100: mid-Sep at ~5 trades/day.

## Milestones

| ID | Milestone | Gate | Status |
|----|-----------|------|--------|
| M1 | Outage-free operations | 2 weeks w/o critical watchdog event | 🟡 IN PROGRESS — ~34 days (Jul 10–Aug 13); watchdog clean; no outages |
| M2 | Measured edge on new config | PF ≥ 1.2 at n ≥ 100 closed trades on v0.6.0 | 🟡 IN PROGRESS — n=37, PF=1.87 (noise — n too small). Bot active Aug 6+. ETA mid-Sep at ~5 trades/day. |
| M3 | H1 cross-sectional validation | Backtest improves BOTH tune + hold-out legs | 🔴 BLOCKED — Railway worker not enabled. H1 draft PR #7 ready. Week 8. |
| M4 | H5/H3/H2/H4 validation (data runs) | Same walk-forward standard | 🔴 BLOCKED — Railway worker not enabled. 13 hypotheses total blocked. Week 8. |
| M5 | Paper-trading gate | Sharpe ≥ 1.5, DD ≤ 8%, 3 months | ⚪ NOT STARTED — depends on M2 |
| M6 | Client/commercial track | M5 + registration/partner decision | ⚪ NOT STARTED |

## Current Single Bottleneck (TPM)

**Railway worker service** (`python agent_worker.py`, env `AGENT_WORKER_ENABLE=true`
+ Alpaca paper keys). Blocks M3, M4, and CODE RED exit (data_run criterion).
Outstanding **8 consecutive weeks** (since W29, Jul 15). ~10 minutes in Railway dashboard.

All 13 hypothesis validations (H0–H12 + H13) blocked behind this single action.

Secondary bottleneck: PRs #35, #36, and #31/#32 are open freeze-exempt correctness fixes
that should be merged promptly — they fix live defects (miscalibrated stops, circumvented
trade cap, blocked exits during halt).

## Freeze Status (TPM-enforced)

- **Strategy FROZEN at v0.6.0** since 2026-08-02. No strategy merges until n=100 closed
  trades on v0.6.0 (max_hold=30) or a walk-forward-validated backtest justifies an exception.
- Bug fixes, infra, monitoring always exempt.
- **PR #36 (H14: horizon-scaled ATR exits)** — FREEZE-EXEMPT. Corrects _atr_exits() stops
  that are 3.6× too wide for 30-bar holds. All 37 v0.6.0 exits via max_hold because
  SL/trail/TP floors never fire. Merge recommended.
- **PR #35 (H15: persist n_trades_today)** — FREEZE-EXEMPT. Fixes risk-control enforcement:
  daily trade cap resets to 0 on restarts, circumventing the cap. Aug 10 had 15 trades vs
  cap=6. Merge recommended.
- v0.6.0 deployed: max_hold 390→30, MAX_HOLD_EXTENSIONS 2→0, SIZING_STAGNATION_BARS 390→30.
  224 tests. STRATEGY CHANGE — M2 clock resets to Aug 2, 2026.
- v0.6.1 deployed: resume-persistence fix (not a strategy change).
- v0.6.2 deployed: universe-rotation zombie fix — _owned_tickers now persists positions
  regardless of screener membership. Revealed portfolio_heat blind spot ($12.6k / 12.9%
  deployed and invisible during MSCI zombie).
- Draft PRs queued: #7 (H1), #9 (H5 phase needs re-scope for 30-bar), #10 (H2+H6),
  #11 (H7 needs re-scope: stagnation_bars=30=max_hold now), #26 (H11), #29 (H12).
- H5 (hold extension) disabled in prod (MAX_HOLD_EXTENSIONS=0); needs re-evaluation
  under 30-bar horizon.
- H7 (stagnation exits) needs re-scoping: stagnation_bars now equals max_hold (both 30).
- Non-draft PRs #25 (PROD_PARAMS) and #27 (H9+H10) ready to merge.

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Railway worker not running — CODE RED exit + 13 hypothesis validations blocked | **CRITICAL** | Week 8; ~10 min owner action in Railway dashboard |
| PR #36 (H14) not merged — SL/trail/TP stops dead code for all 30-bar positions | **HIGH** | All 37 exits via max_hold; stops never fire; merge PR #36 |
| PR #35 (H15) not merged — daily trade cap reset on every redeploy | **HIGH** | Aug 10: 15 trades vs cap=6 due to v0.6.2 restart; merge PR #35 |
| Halt-aware exits not merged (PRs #31/#32) — halt could block exits if CB fires again | **HIGH** | Owner: choose #31 or #32 and merge |
| JNJ anomaly — ID 163 BUY with ensemble_signal=-0.0238 (negative ensemble on long) | **MED** | Monitor for recurrence; H12 (PR #29) would block this class |
| M2 n=37 — PF=1.87 driven by small sample; MSTR +$504 still outsized | **MED** | Monitor trajectory as n grows; do not act on signal until n≥50 |
| PR merge conflict — PRs #35/#36 modified agent_state.json from Aug 10 baseline | **MED** | Merge chore/rnd-log-aug13 first; then #35/#36 code-only changes merge cleanly |
| H5 + H7 need re-scoping for 30-bar horizon | **MED** | Backtest phases target wrong regime; re-scope before Railway runs |
| Hypothesis accumulation without validation | **MED** | 13 hypotheses queued, 0 data runs in 8 weeks (Railway worker) |
| Non-draft PRs aging without merge (#25, #27, #33) | **MED** | All safe to merge (infra/monitoring, freeze-exempt) |
| **RESOLVED** Bot HALTED — ✅ Halt lifted Aug 6 | — | — |
| **RESOLVED** MSCI zombie (id 119, 14d open) — ✅ Closed Aug 10 via v0.6.2 | — | — |
| **RESOLVED** Sentinel CRITICAL stale_open_rows — ✅ ALL GREEN Aug 12 21:25 UTC | — | — |

## Decision Log

- 2026-08-13: Nightly review — M2 v0.6.0: n=37 (PF=1.87, WR=47.2%, net +$947.95). All 37 exits via max_hold — root cause: SL/trail/TP floors are 3.6× too wide for 30-bar holds (H14 in PR #36 addresses). Trailing-30: PF=2.03, WR=43.3%. Integrity Sentinel ALL GREEN Aug 12. Kelly mode: normal, fraction=0.2349. JNJ anomaly: ID 163 BUY with ensemble_signal=-0.0238. PRs #35 (H15) and #36 (H14) open, freeze-exempt. Merge order: chore/rnd-log-aug13 first (avoids conflict), then #35/#36. Railway worker week 8 — sole CODE RED exit blocker.
- 2026-08-10: W33 review — halt lifted (Aug 6), MSCI zombie closed (Aug 10) via v0.6.2 (universe-rotation exit bug). v0.6.2 revealed portfolio_heat blind spot ($12.6k / 12.9% deployed and invisible during zombie). M2 v0.6.0: n=16, PF=2.02 (noise). Railway worker week 7 — sole CODE RED exit blocker. Decisions outstanding: Railway worker (CRITICAL), halt-aware exits PRs #31/#32 (HIGH), PR #25/#27 merge (MED).
- 2026-08-03: W32 review — bot halted (Jul 28), M2 reset (v0.6.0), Railway week 6. Sentinel CRITICAL (stale_open_rows all 6 positions). H10 marked superseded by v0.6.0. H5/H7 flagged for re-scoping. 2 CRITICAL owner decisions outstanding: lift halt + Railway worker.
- 2026-08-02: v0.6.0 MAJOR — max_hold 390→30 (research: IC@390=0.004, edge gone 26x past training horizon). PF@30=4.55 vs 0.54@390 OOS. M2 clock reset. v0.6.1: resume-persistence fix. Sentinel CRITICAL: stale_open_rows (MSCI/AMAT/TSLA >5d). Bot still halted.
- 2026-07-28: Integrity exit criterion MET (Day 5/5 clean). v0.5.4 durable risk state deployed (max_drawdown was 0.00% anchored; true DD 6.66%). v0.5.5 peak equity fix deployed. CODE RED still active — Railway worker (data_run criterion) outstanding 4 weeks.
- 2026-07-27: W31 review — M2 off-track (PF 0.741@n=32); v0.5.0 LGBM classification flagged; CODE RED Day 4/5
- 2026-07-20: CODE RED declared; Integrity Sentinel + Principal Skeptic onboarded
- 2026-07-13: Hard freeze at v0.4.4; TPM+PM personas onboarded
- 2026-07-15: W29 weekly review — freeze confirmed; 3 owner decisions escalated
- 2026-07-20: W30 weekly review — 403 CRITICAL (day 8); Railway worker CRITICAL (week 2)
- 2026-07-10: PR-only governance; risk controls never weakened (manifesto)
- 2026-07-11: H5 must be signal-conditional (unconditional 3-day holds = −493bps, June backtest)
- 2026-07-13: Owner approved H5+H3 deploy ahead of data runs (paper = lab)
