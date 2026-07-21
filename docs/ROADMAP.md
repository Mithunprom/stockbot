# StockBot Program Roadmap

Maintained by the TPM persona (Program Office weekly review + nightly desk).
Last updated: 2026-07-20 (CODE RED declared).

## 🔴 CODE RED — declared 2026-07-20 by owner (CFO)

**Trigger:** trailing-7d PF 0.30 (n=17) AND a confirmed ledger defect: partial-fill
exits wrote corrupted `pnl_pct` (e.g. −15.7% stored on a −4.0% trade, one row at
+993%), and `_seed_kelly_from_db()` re-injected those rows into position sizing
after every redeploy. Two zombie open rows (MU/WDC, 2026-06-12) sat unclosable
by orphan recovery. Full declaration: `reports/program/CODE-RED-2026-07-20.md`.

**Posture (manifesto governance rule F):** integrity first, validation throughput
second, new alpha queued. Personas 10 (Integrity Sentinel) + 11 (Principal
Skeptic) onboarded; `src/agents/integrity_agent.py` audits the ledger hourly.

**Exit criteria:** integrity report clean 5 consecutive trading days + Kelly
window verified sane + ≥1 hypothesis reaches data_run. Exit is logged here.

## North star
Stable ops → measured edge → paper-trading gate → client product.
No stage skips a gate. PnL is never reported without Sharpe/PF/WR/DD/n.

## Milestones

| ID | Milestone | Gate | Status |
|----|-----------|------|--------|
| M1 | Outage-free operations | 2 weeks w/o critical watchdog event | 🟡 IN PROGRESS — streak started 2026-07-10 (v0.3.5 fixes + watchdog live) |
| M2 | Measured edge on frozen config | PF ≥ 1.2 at n ≥ 100 closed trades on v0.4.4 | 🟡 IN PROGRESS — n=15 as of 07-14; clock restarted at v0.4.4 (07-13). **CONFIG FROZEN** |
| M3 | H1 cross-sectional validation | Backtest improves BOTH tune + hold-out legs | 🔴 BLOCKED — needs Railway worker service (owner action) |
| M4 | H5/H3 validation (data runs) | Same walk-forward standard | 🔴 BLOCKED — same bottleneck as M3 |
| M5 | Paper-trading gate | Sharpe ≥ 1.5, DD ≤ 8%, 3 months | ⚪ NOT STARTED — depends on M2 |
| M6 | Client/commercial track | M5 + registration/partner decision | ⚪ NOT STARTED — /track page live as groundwork |

## Current single bottleneck (TPM)
**Railway worker service** (`python agent_worker.py`, env `AGENT_WORKER_ENABLE=true`
+ Alpaca keys). Blocks M3 and M4 — every pending hypothesis validation.
Owner action, ~10 minutes in the Railway dashboard.

## Freeze status (TPM-enforced)
- **FROZEN at v0.4.4** since 2026-07-13. Strategy merges paused until M2
  matures or a validated backtest justifies an exception.
- Always exempt: bug fixes, infra, monitoring, docs.
- Draft PRs queue freely (currently: PR #7 / H1 — correctly held in draft
  pending its data run).

## Risk register
| Risk | Severity | Mitigation |
|------|----------|-----------|
| Mid-sample config churn resets M2 clock | HIGH (realized 4× last week) | TPM freeze rule above |
| Single-operator deploys | MED | ✅ closed 07-13: auto-deploy on merge |
| Long-only assumption landmines | MED | v0.3.7 exit-side fix + regression tests; audit remaining paths during M2 |
| Validation bottleneck (no worker) | HIGH | Owner action pending |
| Alpaca paper account external resets | MED | /track + history now read broker directly; keep baseline notes in agent_state |

## Decision log
- 2026-07-20: CODE RED declared (ledger defect + PF 0.30/7d); Integrity Sentinel
  + Principal Skeptic personas onboarded; v0.4.5 fix + repair (freeze-exempt: bug/monitoring)
- 2026-07-10: PR-only governance; risk controls never weakened (manifesto)
- 2026-07-11: H5 must be signal-conditional (unconditional 3-day holds = −493bps, June backtest)
- 2026-07-13: owner approved H5+H3 deploy ahead of data runs (paper = lab)
- 2026-07-14: hard freeze at v0.4.4; TPM+PM personas onboarded
