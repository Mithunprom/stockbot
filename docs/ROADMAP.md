# StockBot Program Roadmap

Maintained by the TPM persona (Program Office weekly review + nightly desk).
Last updated: 2026-10-05 W41 program review (v0.9.0 deployed; M3 clock CONFIRMED Sep 23 — EXIT_PROFIT_TARGET_MODE=ON confirmed by live take_profit exit; n=18 Sep 23+; PF=5.11 noise/halt-contaminated; Kelly epoch Sep 29 owner-authorized; 9 commits W41 addressing cascading incidents; SMTP credential dead; Railway worker week 16).

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

**M2 RETIRED (Sep 21):** n=110, PF=0.857, WR=44.5% (49W/1T/60L), net -$804.19. RETIRED — was
measuring a bot with corrupted feature serving (OBV session-anchoring bug; entries at ~46th
pctile of model's own ranking vs top decile). v0.7.0 fixed root cause. M2 data is an infra
failure case study, NOT strategy evidence.

**M3 clock confirmed Sep 23** (v0.8.0 strategy config). W40 Decision #3 resolved by live data: LITE (ID 255) exited via `take_profit` Oct 2 — EXIT_PROFIT_TARGET_MODE=ON confirmed. Prior n=11 (Sep 21 trades) are retired from M3 (different strategy config). **W41 update:** n=18 closed (Sep 23 window); PF=5.11 (NOISE — halt-contaminated: 11/18 exits via integrity_broker_reconcile; COHR/MSTR/LITE gains were held overnight through halt, not strategy exits). 6 open positions Oct 5. Kelly epoch re-seed Sep 29 (owner-authorized). 9 commits W41: v0.8.2 (rs_vwap_dev serving fix), v0.8.3 (Kelly epoch), v0.8.4 (retrain; freeze classification pending), v0.8.5–v0.8.9 (infra/correctness), v0.9.0 (H13 halt-aware exits — live incident Sep 30).

## Milestones

| ID | Milestone | Gate | Status |
|----|-----------|------|--------|
| M1 | Outage-free operations | 2 weeks w/o critical watchdog event | 🟡 IN PROGRESS — ~73 days (Jul 10–Sep 21); watchdog OK; no outages or halts |
| M2 | Measured edge (v0.6.0) | PF ≥ 1.2 at n ≥ 100 on v0.6.0 | 🏁 **RETIRED** — n=110, PF=0.857; was measuring broken feature serving pipeline. v0.7.0 deployed Sep 21. |
| M3 | Measured edge (v0.8.0+) | PF ≥ 1.2 at n ≥ 100 | 🟡 n=18 (Sep 23 clock confirmed); PF=5.11 NOISE (halt-contaminated); Kelly epoch Sep 29; 6 open positions Oct 5; v0.8.4 retrain freeze class pending. |
| M4 | H1/H5/H3/H2/H4 validation (data runs) | Walk-forward backtest standard | 🔴 BLOCKED — Railway worker not enabled. 15+ hypotheses total blocked. Week **15**. |
| M5 | Paper-trading gate | Sharpe ≥ 1.5, DD ≤ 8%, 3 months | ⚪ NOT STARTED — depends on M3 |
| M6 | Client/commercial track | M5 + registration/partner decision | ⚪ NOT STARTED |

## Current Single Bottleneck (TPM)

**Railway worker service** (`python agent_worker.py`, env `AGENT_WORKER_ENABLE=true`
+ Alpaca paper keys). Blocks M4 and CODE RED exit (data_run criterion).
Outstanding **16 consecutive weeks** (since W29, Jul 15). ~10 minutes in Railway dashboard.

All 15+ hypothesis validations (H0–H12+) blocked behind this single action.

**RESOLVED Sep 29–Oct 1:** Kelly hard-block resolved via owner-authorized epoch re-seed. H13 (halt stranding positions) resolved in v0.9.0 Oct 1 (41 days after PRs #31/#32 opened).

**NEW CRITICAL: SMTP credential expired** — all CRITICAL alert emails (CB halts, integrity escalations) silently failing. Duration unknown. Owner action required.

**SECOND BOTTLENECK: ~35 open PRs; direct-push governance breach W41.**
v0.8.3 (Kelly epoch) and v0.8.4 (retrain) pushed directly to main without PRs. Per manifesto governance, all strategy/risk-control changes require PRs. Emergency posture appropriate for v0.9.0 (live halt); not appropriate for the retrain. Structural Kelly guards (PRs #35, #51–57) still unmerged.

## Freeze Status (TPM-enforced)

- **Strategy frozen at v0.8.0 config** (EXIT_PROFIT_TARGET_MODE=ON) since 2026-09-23. M3 clock confirmed Sep 23. No strategy merges until n=100 closed trades on v0.8.0+ or a walk-forward-validated backtest justifies an exception.
- **W40 Decision #3 RESOLVED:** EXIT_PROFIT_TARGET_MODE=ON confirmed by live `take_profit` exit (LITE ID 255, Oct 2). M3 clock reset to Sep 23. n=18 closed.
- **⚠️ W41: v0.8.4 (Sep 29) model retrain — freeze classification pending.** Either way, M3 clock stays Sep 23 (v0.8.0 is the earlier reset). Owner must classify: freeze-exempt correctness fix (closes mismatch v0.8.2 opened) or strategy change.
- Bug fixes, infra, monitoring always exempt.
- **PR #35 (H15: persist n_trades_today)** — FREEZE-EXEMPT. Daily trade cap resets to 0 on
  restarts, circumventing the cap. 38 days stale. Merge recommended.
- **H24 cluster (PRs #51, #52, #53, #54)** — FREEZE-EXEMPT. Kelly boundary structural fixes.
  Kelly now "inactive" (fraction=0.0) — less urgent than Sep 20-21 scenario, but structural
  fix still owed before Kelly builds under M3. Merge recommended.
- **H25 cluster (PRs #56, #57)** — FREEZE-EXEMPT. Systemic Kelly guard.
- **PR #65 (H28: entry-spread timer)** — FREEZE-EXEMPT. Risk-control enhancement.
- **SUPERSEDED by v0.7.0 — CLOSE RECOMMENDED:**
  - PR #36 (H14 ATR exits) — incorporated in v0.7.0
  - PR #62 (H27 sector map) — superseded by fail-closed sector resolution
  - PRs #42/#48 (H19/H22 sector completeness) — superseded by fail-closed sector resolution
- v0.7.0 deployed Sep 21: feature pipeline v2 (OBV session-anchored, WARMUP_BARS 300→1950),
  feature version guard, feature archive + backfill, retrained model (OOS IC15=0.1795),
  fail-closed sector guards, working exit barriers, functional Kelly block, entry_rank monitor.
  STRATEGY CHANGE — M3 clock resets to Sep 21, 2026.
- v0.7.1: entry_rank fields surfaced at /diagnostics.
- v0.7.2: diagnostics fix (entry_rank export).
- v0.7.3: forecast email opt-in.
- Draft PRs queued: #7 (H1), #9 (H5 phase needs re-scope), #10 (H2+H6), #11 (H7 re-scope),
  #26 (H11), #29 (H12). All blocked on Railway worker for data runs.
- Non-draft PRs #25 (PROD_PARAMS) and #27 (H9+H10) — may be superseded by v0.7.0; triage.

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|-----------|
| SMTP credential expired — CRITICAL alert emails silently failing | **CRITICAL — NEW** | v0.8.6 revealed; duration unknown; CB halt Sep 30 fired — owner may not have received email; unresolved |
| RetrainAgent hot-swapping models daily at 08:00 ET (4-month backlog now clearing) | **HIGH — NEW** | v0.8.9 fixed FFSA path; model can now change without deploy; MIN_VAL_IC gate + version guard are the only checks |
| GOOGL partial-fill record error (same class as CODE RED) | **HIGH — NEW** | Sep 29 ledger: 5 shares vs 35.4 real; resolved by v0.8.9 broker reconciliation; root cause unclear |
| v0.8.4 retrain freeze classification pending | **MEDIUM** | M3 clock stays Sep 23 either way; owner must classify |
| PR governance breach — v0.8.3/v0.8.4 pushed direct-to-main | **MEDIUM — NEW** | Risk-control and model changes should be PRs; emergency posture was valid for v0.9.0 only; recommend reinstate PR discipline |
| H24 cluster (PRs #51–#54) — Kelly boundary structural fix not merged | **HIGH** | Kelly epoch resolved immediate block; structural risk returns as Kelly rebuilds; merge recommended |
| H25 cluster (PRs #56–#57) — systemic Kelly guard not merged | **HIGH** | Freeze-exempt; merge recommended |
| PR #35 (H15) — daily cap circumventable on restart | **HIGH** | Still unmerged; double-batch confirmed Sep 21; merge recommended |
| Railway worker not running — CODE RED exit + 16+ validations blocked | **CRITICAL** | Week 16; ~10 min owner action |
| ~35 open PRs | **HIGH** | Direct-push habit makes queue management harder; triage needed |
| PR #43 (H18b SPY gate) freeze classification outstanding | **MED** | Strategy change candidate; must not merge during M3 freeze |
| PR #41 (H18a Gate 5c) vs H12 (PR #29) overlap | **MED** | Disambiguation needed before either merges |
| H5 + H7 need re-scoping for 30-bar horizon | **MED** | Re-scope before Railway runs |
| Hypothesis accumulation without validation | **MED** | 15+ queued, 0 data runs in 16 weeks (Railway worker) |
| **RESOLVED** H13 halt strands positions — ✅ v0.9.0 Oct 1 | — | — |
| **RESOLVED** rs_vwap_dev cross-sectional serving — ✅ v0.8.2 Sep 29 | — | — |
| **RESOLVED** Kelly hard-block -0.2699 — ✅ v0.8.3 epoch re-seed (owner-authorized) | — | — |
| **RESOLVED** RetrainAgent FFSA failure — ✅ v0.8.9 Sep 30 | — | — |
| **RESOLVED** EXIT_PROFIT_TARGET_MODE ambiguity — ✅ ON confirmed by take_profit exit | — | — |
| **RESOLVED** Train/serve skew — ✅ v0.7.0 Sep 21 | — | — |
| **RESOLVED** PR #36 (H14 dead stops) — ✅ incorporated in v0.7.0 | — | — |
| **RESOLVED** Kelly block hardcoded False — ✅ fixed in v0.7.0 | — | — |
| **RESOLVED** M2 gate — ✅ RETIRED (measuring broken infra) | — | — |
| **RESOLVED** Bot HALTED — ✅ Halt lifted Aug 6 | — | — |
| **RESOLVED** MSCI zombie (id 119, 14d open) — ✅ Closed Aug 10 via v0.6.2 | — | — |
| **RESOLVED** Sentinel CRITICAL stale_open_rows — ✅ ALL GREEN continuing | — | — |

## Decision Log

- 2026-10-05: W41 program review — M3 clock CONFIRMED Sep 23 (EXIT_PROFIT_TARGET_MODE=ON; LITE ID 255 take_profit Oct 2 confirms live). n=18 closed (PF=5.11 noise; 11/18 exits integrity_broker_reconcile; halt-recovery contaminated — do not read as edge). 9 commits W41: v0.8.2 (rs_vwap_dev serving fix, root cause 7-session zero-trades), v0.8.3 (Kelly epoch owner-authorized), v0.8.4 (retrain OOS IC15=0.1839; freeze class pending), v0.8.5–v0.8.9 (infra/correctness), v0.9.0 (H13 halt-aware exits — live incident Sep 30, fixed Oct 1). SMTP credential expired — all alert emails dead for unknown duration. Integrity ALL GREEN Oct 5 13:25 UTC. Watchdog OK. 6 open positions. Railway worker week 16. 4 owner decisions needed. Report: reports/program/2026-W41.md.
- 2026-09-28: W40 program review — M3 Day 1 (Sep 21): n=11, entry_rank_mean=96.9 ✅ (skew fix confirmed), first stop_loss exit STX ✅ (barriers working), SNDK negative-ensemble 2nd confirmed, H15 double-batch 11 vs 6. Kelly hard-blocked EOD Sep 21 (fraction=-0.2699). Sep 22–28: 0 trades. v0.8.0 deployed Sep 23: profit-target exits (env-gated) + Kelly probe freeze fix (KELLY_PROBE_MIN_N 300→30). v0.8.1 diagnostics fix. PRs #68–71 opened (all diagnostics, freeze-exempt). Kelly rolloff ~Oct 1 — PRs #35/#51–57 must land first. Railway worker week 15. 6 owner decisions needed. Report: reports/program/2026-W40.md.
- 2026-09-21: W39 program review — ✅ v0.7.0 DEPLOYED (merged Sep 21 05:45 UTC). ROOT CAUSE DIAGNOSED AND FIXED: train/serve skew (OBV session-anchoring, WARMUP_BARS 300→1950). Entries were at ~46th pctile of model's own ranking vs top decile (measured). v0.7.0: feature pipeline v2, version guard, feature archive (99 sessions), retrained model (OOS IC15=0.1795), fail-closed sector guards (24→57 tickers), working exit barriers, functional Kelly block, entry_rank monitor. M2 RETIRED (was measuring broken infra). M3 starts today n=0. Kelly: -0.6661 → inactive 0.0 (Sep 20-21 rolloff). Integrity ALL GREEN Sep 21. W39: 0 new trades. Railway worker week 14. 6 owner decisions needed. Report: reports/program/2026-W39.md.
- 2026-09-14: W38 program review — 🔴 M2 DECLINING: n=110, PF=0.857 (was 0.926). Sep 10-11: 12 new trades 7W/5L, net -$449.58. Aug 31 losses rolled off Kelly window Sep 10 (calendar basis); bot resumed trading without H24 guard live. Sep 11: LITE -$428 at ensemble=0.191 with zero stop protection (H14 33d stale, H12 PR #29 not merged). Kelly -0.6661 (probation, recovering from -5.37). H24 deadline Sep 12 MISSED — PRs #51-54 still open; next window transition ~Sep 20-21. H25 cluster PRs #56-57 added (systemic Kelly guard: min unique days + cluster-day cap). All 110 exits via max_hold. ~30 open PRs, 0 code merges in 2 weeks. Railway worker week 13. TPM recommendation: reset M2 gate after merging H14+H15+H24+H25. Report: reports/program/2026-W38.md.
- 2026-09-07: W37 program review — 🚨 M2 GATE MISSED. Aug 31 structural defect: Kelly window-empty condition fired 12 full-size entries (2 batches of 6 = H15 daily-cap-reset signature; all max_hold = H14 stops dead). Result: 2W/10L, net -$937.85. M2 PF collapsed 1.152→0.926 (below 1.0), net +$583→-$354, T30 PF=0.182 (worst on record). Kelly crashed -0.7343→-5.3731 (deep probation). Sep 1–5: 0 new trades. Sep 7: watchdog OK, fresh ticks, 0 open positions. Desk shipped 4 H24 PRs (#51–#54) addressing Kelly boundary defect — all freeze-exempt, must merge before ~Sep 12. Total open PRs: 30 (was 20). PR #35/#36 now 26d stale — their absence caused real harm Aug 31. Railway worker week 12. Owner gate-posture decision required: PF=0.926@n=98 is a gate failure; continue window or reset after structural fixes. Report: reports/program/2026-W37.md.
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
