# StockBot Program Roadmap

Maintained by the TPM persona (Program Office weekly review + nightly desk).
Last updated: 2026-07-20 (W30 weekly review).

## North star
Stable ops → measured edge → paper-trading gate → client product.
No stage skips a gate. PnL is never reported without Sharpe/PF/WR/DD/n.

## Milestones

| ID | Milestone | Gate | Status |
|----|-----------|------|--------|
| M1 | Outage-free operations | 2 weeks w/o critical watchdog event | 🟡 IN PROGRESS — streak est. ~10 days (Jul 10–20). **⚠️ UNVERIFIABLE** — endpoints 403 since Jul 12 (day 8); watchdog unreadable second week |
| M2 | Measured edge on frozen config | PF ≥ 1.2 at n ≥ 100 closed trades on v0.4.4 | 🟡 IN PROGRESS — n ≈ 23 est.; endpoints 403 prevent confirmation. **CONFIG FROZEN** |
| M3 | H1 cross-sectional validation | Backtest improves BOTH tune + hold-out legs | 🔴 BLOCKED — needs Railway worker service (owner action). H1 draft PR #7 ready. H5-phase PR #9 also queued. |
| M4 | H5/H3/H2/H4 validation (data runs) | Same walk-forward standard | 🔴 BLOCKED — same bottleneck as M3. H4 at 10 days (2× flag), H2 at 9 days, H5/H3 at 6 days. |
| M5 | Paper-trading gate | Sharpe ≥ 1.5, DD ≤ 8%, 3 months | ⚪ NOT STARTED — depends on M2 |
| M6 | Client/commercial track | M5 + registration/partner decision | ⚪ NOT STARTED — /track page live as groundwork |

## Current single bottleneck (TPM)
**Railway worker service** (`python agent_worker.py`, env `AGENT_WORKER_ENABLE=true`
+ Alpaca keys). Blocks M3 and M4 — all 7 pending hypothesis validations (H1–H5 deployed,
H6/H7 in draft). Owner action, ~10 minutes in the Railway dashboard.
Outstanding since W29 (Jul 15) — two weeks with no resolution.

**Secondary blocker (now CRITICAL — monitoring):** Railway endpoints return HTTP 403 from
the Claude sandbox proxy for **8 consecutive days** (Jul 12–20). M1 and M2 measurement is
blind. Independent of the worker issue — both need resolution. Recommend external uptime
monitor (UptimeRobot/Pingdom) to restore M1/M2 visibility independent of the sandbox.

## Freeze status (TPM-enforced)
- **FROZEN at v0.4.4** since 2026-07-13. Strategy merges paused until M2
  matures or a validated backtest justifies an exception.
- Always exempt: bug fixes, infra, monitoring, docs.
- Draft PRs queue freely (currently: PR #7 / H1 — correctly held in draft
  pending its data run).

## Risk register
| Risk | Severity | Mitigation |
|------|----------|-----------|
| Railway endpoint 403 — monitoring blind | **CRITICAL** (↑ 07-20) | Day 8; north star unmeasurable. External uptime monitor needed. Owner action |
| Railway worker not running — zero validations | **CRITICAL** (↑ 07-20) | H4 at 10 days (2× flag), 0/5 deployed hypotheses validated. Owner action, ~10 min |
| Hypothesis accumulation without validation | HIGH (NEW 07-20) | 7 in queue, building on unvalidated layers. PM recommends pausing new implementation |
| Attribution blur — 5 hypotheses merged w/o individual data runs | HIGH | v0.4.4 is a compound treatment. Cleared only when Railway isolates each hypothesis |
| H4 stuck in needs_data_run 10 days | MED | Double the 5-day flag threshold; Railway worker is the only path |
| Mid-sample config churn resets M2 clock | MED (was HIGH) | ✅ mitigated 07-13: hard freeze at v0.4.4; held through W30 |
| Single-operator deploys | MED | ✅ closed 07-13: auto-deploy on merge |
| Long-only assumption landmines | MED | v0.3.7 exit-side fix + regression tests; audit remaining paths during M2 |
| Alpaca paper account external resets | MED | /track + history now read broker directly; keep baseline notes in agent_state |

## Decision log
- 2026-07-10: PR-only governance; risk controls never weakened (manifesto)
- 2026-07-11: H5 must be signal-conditional (unconditional 3-day holds = −493bps, June backtest)
- 2026-07-13: owner approved H5+H3 deploy ahead of data runs (paper = lab)
- 2026-07-14: hard freeze at v0.4.4; TPM+PM personas onboarded
- 2026-07-15: W29 weekly review — freeze confirmed intact; 3 decisions escalated to owner (Railway worker, endpoint 403, compound baseline acceptance)
- 2026-07-16–18: R&D desk added H6 (signal persistence, PR #10) and H7 (stagnation exit + PROD_PARAMS, PR #11) in DRAFT while Railway still offline — all correctly held pending backtest; freeze intact
- 2026-07-20: W30 weekly review — Railway endpoint 403 escalated to CRITICAL (day 8); Railway worker escalated to CRITICAL (day 10+ for H4); PM recommends pausing new hypothesis work pending first data run; freeze confirmed intact second consecutive week
