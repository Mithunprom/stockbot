# StockBot Program Roadmap

Maintained by the TPM persona (Program Office weekly review + nightly desk).
Last updated: 2026-08-31 W36 program review (v0.6.2 deployed; M2 n=86 PF=1.152 STALLED — Kelly probation -0.7343, 0 trades W36; PRs #35/#36/#38–#49 open; Railway worker week 11).

## 🔴 CODE RED — declared 2026-07-20 by owner (CFO)

**Trigger:** trailing-7d PF 0.30 (n=17) AND a confirmed ledger defect: partial-fill
exits wrote corrupted `pnl_pct`, and `_seed_kelly_from_db()` re-injected those rows into
position sizing after every redeploy. Full declaration: `reports/program/CODE-RED-2026-07-20.md`.

**Posture (manifesto governance rule F):** integrity first, validation throughput
second, new alpha queued.

**Exit criteria (all three required):**

| Criterion | Status |
|-----------|--------|
| Integrity Sentinel clean 5 consecutive trading days | ✅ **MET** — ALL GREEN at 08:25 UTC Jul 28. Continuing clean: ALL GREEN Aug 24 13:25 UTC. |
| Kelly window verified sane | ✅ OK — kelly_seed_sanity confirmed; mode: normal, fraction=0.2349 |
| ≥1 hypothesis reaches data_run | ❌ BLOCKED — Railway worker not yet enabled (10 consecutive weeks since Jul 15) |

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

**M2 status (Aug 31):** n=86, PF=1.152, WR=46.5% (40W/1T/45L), net +$583.24, exp $6.78/trade.
⚠️ PF BELOW GATE (1.2). M2 STALLED — Kelly fraction -0.7343 (inactive/probation), 0 new trades
W36 (Aug 25-31, 7 trading days). Gate ETA revised to ~Sep 8-10 (pending Kelly recovery ~Sep 4-5).
All 86 exits via max_hold — stops remain dead code for 30-bar horizon (H14 PR #36 unmerged).
W35 (Aug 17-21, 37 trades, IDs 176-212): 15W/22L, PF=0.535, net -$1,165.60. Aug 19 sector
selloff (INTC -$371, KLAC -$328, WDC -$318, MU -$220) = -$1,222 in one day drove Kelly negative.
W36 (Aug 25-31): 0 trades — Kelly probation deadlock. Probes also blocked (H21 PR #47 open).

## Milestones

| ID | Milestone | Gate | Status |
|----|-----------|------|--------|
| M1 | Outage-free operations | 2 weeks w/o critical watchdog event | 🟡 IN PROGRESS — ~52 days (Jul 10–Aug 31); watchdog clean; no outages |
| M2 | Measured edge on new config | PF ≥ 1.2 at n ≥ 100 closed trades on v0.6.0 | 🟡 IN PROGRESS — n=86, PF=1.152 ⚠️ BELOW GATE. STALLED — Kelly probation -0.7343, 0 trades W36. ETA n=100 ~Sep 8-10. Stops still dead code. |
| M3 | H1 cross-sectional validation | Backtest improves BOTH tune + hold-out legs | 🔴 BLOCKED — Railway worker not enabled. H1 draft PR #7 ready. Week 10. |
| M4 | H5/H3/H2/H4 validation (data runs) | Same walk-forward standard | 🔴 BLOCKED — Railway worker not enabled. 13+ hypotheses total blocked. Week 10. |
| M5 | Paper-trading gate | Sharpe ≥ 1.5, DD ≤ 8%, 3 months | ⚪ NOT STARTED — depends on M2 |
| M6 | Client/commercial track | M5 + registration/partner decision | ⚪ NOT STARTED |

## Current Single Bottleneck (TPM)

**Railway worker service** (`python agent_worker.py`, env `AGENT_WORKER_ENABLE=true`
+ Alpaca paper keys). Blocks M3, M4, and CODE RED exit (data_run criterion).
Outstanding **11 consecutive weeks** (since W29, Jul 15). ~10 minutes in Railway dashboard.

All 15+ hypothesis validations (H0–H12+) blocked behind this single action.

**Secondary bottleneck (NEW W36): Kelly probation deadlock.** Kelly fraction -0.7343 (inactive)
blocked ALL entries for 7 consecutive trading days (Aug 22–31). Even probe entries appear blocked
(H21 PR #47). M2 cannot mature until Kelly recovers (~Sep 4–5 window rolloff). No code change
required for natural recovery, but probes must also unblock for accumulation to resume.

Tertiary: PRs #35, #36, #31/#32 are freeze-exempt correctness fixes (19d / 28d stale) with direct
live-risk impact. PRs #43/#41 require freeze classification before merge. Open PR backlog: 20.

## Freeze Status (TPM-enforced)

- **Strategy FROZEN at v0.6.0** since 2026-08-02. No strategy merges until n=100 closed
  trades on v0.6.0 (max_hold=30) or a walk-forward-validated backtest justifies an exception.
- Bug fixes, infra, monitoring always exempt.
- **PR #36 (H14: horizon-scaled ATR exits)** — FREEZE-EXEMPT. Corrects _atr_exits() stops
  that are 3.6× too wide for 30-bar holds. All 49 v0.6.0 exits via max_hold because
  SL/trail/TP floors never fire. Merge recommended.
- **PR #35 (H15: persist n_trades_today)** — FREEZE-EXEMPT. Fixes risk-control enforcement:
  daily trade cap resets to 0 on restarts, circumventing the cap. Aug 10–12 had excess trades.
  Merge recommended.
- **PR #38 (H16: ENTRY_WINDOW_ET 15:30→15:28 ET)** — FREEZE-EXEMPT CANDIDATE. Prevents
  entries whose 30-bar max_hold exit lands at market close. Owner to classify before merge.
- **PR #39 (H17: diagnostic short-block)** — FREEZE-EXEMPT. Diagnostics-only; no trade logic.
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
| Railway worker not running — CODE RED exit + 15+ hypothesis validations blocked | **CRITICAL** | Week 11; ~10 min owner action in Railway dashboard |
| Kelly probation deadlock — 0 trades in 7 trading days (Aug 22–31), M2 stalled | **CRITICAL** | NEW — Kelly -0.7343; natural rolloff ~Sep 4–5; probes also blocked (H21 open). M2 gate ~Sep 8–10. |
| PR #36 (H14) not merged — SL/trail/TP stops dead code, no protection when trades resume | **CRITICAL** | 19d stale; upgraded from HIGH; merge immediately so stops live when Kelly recovers |
| PR #35 (H15) not merged — daily trade cap reset on every redeploy | **HIGH** | 19d stale; active structural gap; merge PR #35 |
| Halt-aware exits not merged (PRs #31/#32) — halt could block exits if CB fires again | **HIGH** | 28d stale; owner must choose #31 or #32 and merge |
| PR #43 (H18b SPY session gate) freeze classification outstanding | **HIGH** | Strategy change candidate; must not merge during M2 freeze without classification |
| PR #41 (H18a Gate 5c) vs H12 (PR #29) overlap — two ensemble-floor PRs | **HIGH** | Disambiguation needed; risk of conflicting strategy changes if merged without resolution |
| 20 open PRs with 0 merged in W36 — hypothesis debt compounding | **MED** | Worsening; all infra/diagnostics freeze-exempt PRs safe to merge now |
| Kelly probes blocked — root cause unknown (H21 PR #47 investigates) | **MED** | NEW — if probes stay blocked post-window-rolloff, M2 accumulation cannot resume |
| Hypothesis accumulation without validation | **MED** | 15+ queued, 0 data runs in 11 weeks (Railway worker) |
| H5 + H7 need re-scoping for 30-bar horizon | **MED** | Re-scope before Railway runs |
| **RESOLVED** Bot HALTED — ✅ Halt lifted Aug 6 | — | — |
| **RESOLVED** MSCI zombie (id 119, 14d open) — ✅ Closed Aug 10 via v0.6.2 | — | — |
| **RESOLVED** Sentinel CRITICAL stale_open_rows — ✅ ALL GREEN continuing Aug 31 | — | — |

## Decision Log

- 2026-08-31: W36 program review — M2 v0.6.0: n=86 UNCHANGED (0 new trades W36, Kelly probation -0.7343). W36 (Aug 25-31): 0 closed trades — Kelly probation deadlock (was +0.2349 at W35). Probes also blocked per H21 PR #47. M2 gate ETA revised to ~Sep 8–10 (Kelly window rolls off ~Sep 4–5). Integrity Sentinel ALL GREEN Aug 31 12:25 UTC. Watchdog OK, v0.6.2, signal_loop_active, 0 errors, 0 open positions. Diagnostics: 5 tickers pass gates (CRM/WDAY/ARM/AMD/COIN) but Kelly inactive blocks sizing. No strategy code shipped — freeze intact. New PRs: #46 (H20 Kelly ETA), #47 (H21 probe_ic_debug), #48 (H22 MNST sector fix), #49 (H23 Kelly rolloff schedule). Total open PRs: 20 (was 16). Railway worker week 11 — sole CODE RED exit blocker. PR #36 (H14) and #35 (H15) now 19d stale; upgraded H14 to CRITICAL (stops must be live when Kelly recovers). Report: reports/program/2026-W36.md.
- 2026-08-24: W35 program review — M2 v0.6.0: n=86 (PF=1.152 ⚠️ BELOW GATE, WR=46.5% 40W/1T/45L, net +$583.24, exp $6.78/trade). W35 (IDs 176-212, 37 trades): 15W/22L, PF=0.535, net -$1,165.60. Aug 19 sector selloff: INTC -$371/KLAC -$328/WDC -$318/MU -$220 = -$1,222 in one day, all max_hold exits. Ex-Aug 19, W35 ≈ breakeven (+$56). Integrity Sentinel ALL GREEN Aug 24 13:25 UTC. Watchdog OK, 0 open positions. No strategy code shipped — freeze intact at v0.6.2. New PRs #41 (Gate 5c/H18 ensemble), #42 (H19 SECTOR_MAP), #43 (H18 SPY session gate — FREEZE CLASSIFICATION NEEDED), #44 (H18 Kelly diagnostics). PR #43 requires freeze classification before merge (strategy change candidate). Railway worker week 10 — sole CODE RED exit blocker. M2 gate IMMINENT: ≈14 trades remaining, ETA Aug 25–26.
- 2026-08-17: W34 program review — M2 v0.6.0: n=49 (PF=2.33, WR=51.0%, net +$1,748.84, exp $35.69/trade, 25W/23L/1T). All 49 exits via max_hold — H14 (PR #36) still unmerged, stops dead code. New trades IDs 164–175 (Aug 13–14): 8W/4L, net +$800.89, PF=4.54; CIEN +$358 (35% sub-period gross). Integrity Sentinel ALL GREEN Aug 17 13:25 UTC. Watchdog OK, 0 open positions. New PRs: #38 (H16 session-boundary cap, needs freeze classification), #39 (H17 diagnostic fix). No strategy code shipped — freeze intact. Railway worker week 9 — sole CODE RED exit blocker. All 13 hypotheses blocked.
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
