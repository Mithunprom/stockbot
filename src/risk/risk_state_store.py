"""Durable risk state — the counters every circuit breaker measures against.

Railway's filesystem is ephemeral and the worker rebuilds on every push, so
in-memory risk counters were silently reset by routine deploys. Concretely,
before this module existed:

    PositionManager._peak_value    = initial_portfolio   (position_manager.py)
    SignalLoop._daily_start_value  = current value       (signal_loop.py)
    SignalLoop._consecutive_losses = 0                   (signal_loop.py)

Nothing persisted them. A redeploy therefore re-anchored "peak" and "daily
start" to the CURRENT (already drawn-down) equity and zeroed the loss streak,
which meant:

  - max_drawdown (-8% from peak) measured from a peak that moved with us down,
    so it could effectively never fire;
  - daily_loss (-3%) forgot the day's loss (the 2026-07-27 -$3,733 session
    reported +$731 after the v0.5.3 deploy that evening);
  - consecutive_losses (5 → halve sizing) reset to 0 mid-streak.

Worse, `_halted` itself was in-memory: a deploy silently RE-ENABLED trading
after a halt, which CLAUDE.md forbids without human confirmation.

State lives in the existing `app_state` KV table, keyed per pipeline, and is
restored on startup. Every function here fails OPEN (logs and returns None /
does nothing) — a database hiccup must never prevent the bot from starting or
block a tick. The cost of failing open is a reset counter, which is exactly
the pre-existing behaviour; the cost of failing closed would be an outage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)

# Bump when the payload shape changes incompatibly; unknown versions are ignored
# on load so a rollback can't crash on a newer row it doesn't understand.
STATE_VERSION = 1


def _key(pipeline_id: str) -> str:
    return f"risk_state:{pipeline_id or 'default'}"


@dataclass
class RiskStateSnapshot:
    """Risk counters that must survive a restart.

    `daily_start_date` is the ET trading date `daily_start_value` belongs to.
    On restore we only trust the value when it matches today, so an overnight
    restart doesn't carry yesterday's baseline into a new session (the 09:30
    reset owns that transition).
    """

    peak_value: float
    daily_start_value: float
    daily_start_date: str          # ISO date, America/New_York
    consecutive_losses: int
    halted: bool = False
    halt_reason: str = ""
    halt_time: str | None = None   # ISO timestamp, UTC
    version: int = STATE_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "RiskStateSnapshot | None":
        data = json.loads(raw)
        if data.get("version") != STATE_VERSION:
            logger.warning("risk_state_version_mismatch", found=data.get("version"),
                           expected=STATE_VERSION)
            return None
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})


async def save_risk_state(pipeline_id: str, snap: RiskStateSnapshot) -> None:
    """Upsert the risk snapshot. Fails open — never raises into the tick loop."""
    from sqlalchemy.dialects.postgresql import insert as _pg_insert

    from src.data.db import AppState, get_session_factory

    try:
        now = datetime.now(timezone.utc)
        payload = snap.to_json()
        sf = get_session_factory()
        async with sf() as session:
            stmt = _pg_insert(AppState).values(
                key=_key(pipeline_id), value=payload, updated_at=now,
            ).on_conflict_do_update(
                index_elements=["key"],
                set_={"value": payload, "updated_at": now},
            )
            await session.execute(stmt)
            await session.commit()
    except Exception as exc:
        logger.warning("risk_state_save_failed", error=str(exc), pipeline=pipeline_id)


async def load_risk_state(pipeline_id: str) -> RiskStateSnapshot | None:
    """Read the last persisted snapshot, or None when absent/unreadable."""
    from sqlalchemy import select as _sel

    from src.data.db import AppState, get_session_factory

    try:
        sf = get_session_factory()
        async with sf() as session:
            row = (await session.execute(
                _sel(AppState.value).where(AppState.key == _key(pipeline_id))
            )).first()
        if row is None:
            return None
        return RiskStateSnapshot.from_json(row[0])
    except Exception as exc:
        logger.warning("risk_state_load_failed", error=str(exc), pipeline=pipeline_id)
        return None


def et_today_iso() -> str:
    """Today's date in America/New_York — the market's notion of 'today'."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:                                     # pragma: no cover
        from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()
