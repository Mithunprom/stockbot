# StockBot Program Roadmap

Maintained by the TPM persona (Program Office weekly review + nightly desk).
Last updated: 2026-07-15 (W29 weekly review).

## North star
Stable ops → measured edge → paper-trading gate → client product.
No stage skips a gate. PnL is never reported without Sharpe/PF/WR/DD/n.

## Milestones

| ID | Milestone | Gate | Status |
|----|-----------|------|--------|
| M1 | Outage-free operations | 2 weeks w/o critical watchdog event | 🟡 IN PROGRESS — streak started 2026-07-10 (5 days). **⚠️ UNVERIFIABLE** — endpoints 403 since Jul 12; watchdog unreadable |
| M2 | Measured edge on frozen config | PF ≥ 1.2 at n ≥ 100 closed trades on v0.4.4 | 🟡 IN PROGRESS — n ≈ 15 est. (07-14); endpoints 403 prevent confirmation. **CONFIG FROZEN** |
| M3 | H1 cross-sectional validation | Backtest improves BOTH tune + hold-out legs | 🔴 BLOCKED — needs Railway worker service (owner action). H1 draft PR #7 ready. |
| M4 | H5/H3 validation (data runs) | Same walk-forward standard | 🔴 BLOCKED — same bottleneck as M3. H4 now at 5-day flag threshold (Jul 10). |
| M5 | Paper-trading gate | Sharpe ≥ 1.5, DD ≤ 8%, 3 months | ⚪ NOT STARTED — depends on M2 |
| M6 | Client/commercial track | M5 + registration/partner decision | ⚪ NOT STARTED — /track page live as groundwork |

## Current single bottleneck (TPM)
**Railway worker service** (`python agent_worker.py`, env `AGENT_WORKER_ENABLE=true`
+ Alpaca keys). Blocks M3 and M4 — every pending hypothesis validation.
Owner action, ~10 minutes in the Railway dashboard.

**Secondary blocker (monitoring):** Railway endpoints return HTTP 403 from the Claude
sandbox proxy for 4 consecutive days (Jul 12–15). M1 and M2 measurement is blind.
Independent of the worker issue — both need resolution.

## Freeze status (TPM-enforced)
- **FROZEN at v0.4.4** since 2026-07-13. Strategy merges paused until M2
  matures or a validated backtest justifies an exception.
- Always exempt: bug fixes, infra, monitoring, docs.
- Draft PRs queue freely (currently: PR #7 / H1 — correctly held in draft
  pending its data run).

## Risk register
| Risk | Severity | Mitigation |
|------|----------|-----------|
| Railway endpoint 403 — monitoring blind | HIGH | ⚠️ NEW 07-15: 4 days running; M1 unverifiable, M2 unreadable. Owner: check Railway dashboard + consider external uptime monitor |
| Attribution blur — 4 hypotheses merged w/o individual data runs | HIGH | ⚠️ NEW 07-15: v0.4.4 is a compound treatment. Cleared only when Railway backtests isolate each hypothesis |
| Validation bottleneck (no worker) | HIGH | Owner action pending >5 days; H4 at 5-day flag threshold |
| Mid-sample config churn resets M2 clock | MED (was HIGH) | ✅ mitigated 07-13: hard freeze at v0.4.4 |
| Single-operator deploys | MED | ✅ closed 07-13: auto-deploy on merge |
| Long-only assumption landmines | MED | v0.3.7 exit-side fix + regression tests; audit remaining paths during M2 |
| Alpaca paper account external resets | MED | /track + history now read broker directly; keep baseline notes in agent_state |

## Decision log
- 2026-07-10: PR-only governance; risk controls never weakened (manifesto)
- 2026-07-11: H5 must be signal-conditional (unconditional 3-day holds = −493bps, June backtest)
- 2026-07-13: owner approved H5+H3 deploy ahead of data runs (paper = lab)
- 2026-07-14: hard freeze at v0.4.4; TPM+PM personas onboarded
- 2026-07-15: W29 weekly review — freeze confirmed intact; 3 decisions escalated to owner (Railway worker, endpoint 403, compound baseline acceptance)
