"""H40: Machine-readable halt-lift readiness checklist.

Returns a structured go/no-go block for /diagnostics so the owner can see the
exact gate state before deciding to call /admin/resume-trading.

Pure stdlib — no structlog/numpy/pandas. Callable from tests directly.
"""
from __future__ import annotations

from typing import Any


def halt_lift_readiness(
    *,
    cb_halted: bool,
    kelly_fraction: float,
    kelly_n_trades: int,
    kelly_mode: str,
    n_open_positions: int,
    n_ic_blocked: int,
    email_alerts_enabled: bool,
) -> dict[str, Any]:
    """Return a structured go/no-go checklist for lifting the CB halt.

    `can_resume_safely` is the single bool the owner acts on.
    `blockers` lists every remaining gate so the owner knows what to fix first.
    If `cb_halted` is False the block still runs — it answers "is the system
    healthy enough to resume IF halted again?" which is useful after lift too.
    """
    blockers: list[dict[str, str]] = []
    requirements_met: list[dict[str, str]] = []

    # ── Requirements met ─────────────────────────────────────────────────────
    if kelly_n_trades == 0 or kelly_mode == "inactive":
        requirements_met.append({
            "id": "KELLY_HARD_BLOCK_RELEASED",
            "detail": (
                f"Kelly window is empty (n={kelly_n_trades}, mode={kelly_mode}). "
                "The Oct 29-30 losses have rolled off the 10-day lookback window. "
                "Hard block is no longer active."
            ),
        })
    else:
        # Kelly still active — only a requirement if fraction is in safe range
        if kelly_fraction >= 0.0:
            requirements_met.append({
                "id": "KELLY_POSITIVE",
                "detail": (
                    f"Kelly fraction={kelly_fraction:.4f} (non-negative). "
                    f"n={kelly_n_trades} trades in window."
                ),
            })

    if n_open_positions == 0:
        requirements_met.append({
            "id": "ZERO_OPEN_POSITIONS",
            "detail": "No open positions. Resume starts from a clean slate.",
        })

    # ── Blockers ─────────────────────────────────────────────────────────────
    if not email_alerts_enabled:
        blockers.append({
            "id": "EMAIL_ALERTS_DISABLED",
            "severity": "HIGH",
            "detail": (
                "EMAIL_ALERTS_ENABLED=false. If the bot re-halts after resume, "
                "the owner receives NO halt notification. Set "
                "EMAIL_ALERTS_ENABLED=true before resuming so the safety net "
                "is active."
            ),
        })

    if n_ic_blocked >= 10:
        blockers.append({
            "id": "IC_CONTAMINATION_HIGH",
            "severity": "HIGH",
            "detail": (
                f"{n_ic_blocked} tickers are IC-blocked. These may include "
                "contaminated outcomes from Oct 8-16 (IC fill job was dead). "
                "Merge H38 (IC_OUTCOMES_EPOCH env var) to clear contaminated "
                "IC blocks before resuming."
            ),
        })
    elif n_ic_blocked > 0:
        blockers.append({
            "id": "IC_BLOCKS_PRESENT",
            "severity": "MEDIUM",
            "detail": (
                f"{n_ic_blocked} tickers are IC-blocked. Review before resuming."
            ),
        })

    can_resume_safely = len(blockers) == 0

    if not cb_halted:
        summary = (
            "Bot is not halted. "
            + (
                "System is healthy."
                if can_resume_safely
                else (
                    f"{len(blockers)} issue(s) to address: "
                    + ", ".join(b["id"] for b in blockers)
                )
            )
        )
    elif can_resume_safely:
        summary = (
            "All halt-lift gates clear. Ready for owner to call "
            "/admin/resume-trading if decision is to lift."
        )
    else:
        blocker_ids = ", ".join(b["id"] for b in blockers)
        summary = (
            f"NOT ready to resume: {len(blockers)} blocker(s) active — "
            f"{blocker_ids}. Resolve before lifting halt."
        )

    return {
        "can_resume_safely": can_resume_safely,
        "blockers": blockers,
        "requirements_met": requirements_met,
        "summary": summary,
    }
