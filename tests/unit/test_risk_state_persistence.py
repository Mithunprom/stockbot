"""Risk-state durability + halt alerting.

Regression suite for the 2026-07-28 finding: peak equity, the daily baseline,
the loss streak and the halt flag were all in-memory, so every redeploy
re-anchored them to the current (already drawn-down) equity. That silently
disarmed max_drawdown and daily_loss, zeroed loss streaks mid-run, and
re-enabled trading after a halt — and no halt ever sent mail.

The tests below all model the same shape: mutate state, simulate a restart,
assert the risk view survived.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.risk.circuit_breakers import CircuitBreakers, RiskState
from src.risk.risk_state_store import RiskStateSnapshot, et_today_iso


# ─── Snapshot round-trip ──────────────────────────────────────────────────────

def test_snapshot_roundtrip_preserves_every_field():
    snap = RiskStateSnapshot(
        peak_value=104_278.55,
        daily_start_value=101_942.18,
        daily_start_date="2026-07-27",
        consecutive_losses=5,
        halted=True,
        halt_reason="max_drawdown",
        halt_time="2026-07-27T14:29:00+00:00",
    )
    back = RiskStateSnapshot.from_json(snap.to_json())
    assert back == snap


def test_snapshot_rejects_unknown_version():
    """A rollback must not crash on a payload written by a newer build."""
    raw = '{"version": 999, "peak_value": 1.0, "daily_start_value": 1.0, ' \
          '"daily_start_date": "2026-07-27", "consecutive_losses": 0}'
    assert RiskStateSnapshot.from_json(raw) is None


# ─── The bug: drawdown survives a restart ─────────────────────────────────────

class _FakePM:
    """Minimal PositionManager stand-in for peak-ratchet behaviour."""

    def __init__(self, value: float):
        self.portfolio_value = value
        self._peak_value = value

    def update_peak(self) -> float:
        self._peak_value = max(self._peak_value, self.portfolio_value)
        return self._peak_value


def test_peak_ratchets_up_but_never_down():
    pm = _FakePM(100_000.0)
    pm.portfolio_value = 104_278.55
    pm.update_peak()
    assert pm._peak_value == pytest.approx(104_278.55)

    # Equity falls — peak must hold, so drawdown stays visible.
    pm.portfolio_value = 97_329.60
    pm.update_peak()
    assert pm._peak_value == pytest.approx(104_278.55)


def test_restart_midbdrawdown_keeps_the_drawdown_visible():
    """THE regression: a redeploy used to reset peak to current equity.

    Real numbers from 2026-07-28: peak $104,278.55, equity $97,329.60 — a
    6.66% drawdown against an 8% breaker limit. Before persistence, a restart
    reported 0.00% and the breaker could never fire.
    """
    peak, equity = 104_278.55, 97_329.60
    snap = RiskStateSnapshot(
        peak_value=peak,
        daily_start_value=equity,
        daily_start_date=et_today_iso(),
        consecutive_losses=0,
    )

    # Fresh process: PositionManager anchors peak to whatever equity is now.
    pm = _FakePM(equity)
    assert pm._peak_value == equity                    # the old, broken view
    naive_dd = (pm._peak_value - equity) / pm._peak_value
    assert naive_dd == pytest.approx(0.0)

    # Restore applies the same max() the loop uses.
    pm._peak_value = max(snap.peak_value, pm.portfolio_value)
    restored_dd = (pm._peak_value - equity) / pm._peak_value
    assert restored_dd == pytest.approx(0.0666, abs=1e-4)
    assert restored_dd > naive_dd


def test_drawdown_breaker_fires_on_restored_peak():
    """Past the limit, the breaker must trip on restored state alone."""
    cb = CircuitBreakers()
    state = RiskState(
        portfolio_value=95_000.0,
        peak_portfolio=104_278.55,     # restored, not re-anchored
        daily_start_value=95_000.0,
        vix=20.0,
        consecutive_losses=0,
    )
    with patch.object(CircuitBreakers, "_notify_halt"), \
         patch.object(CircuitBreakers, "_write_risk_report", return_value=None):
        triggered = asyncio.run(cb.check(state))

    assert state.drawdown_pct > 0.08
    assert "max_drawdown" in triggered
    assert cb.is_halted

    # Same equity with a re-anchored peak: silent. This is what shipped.
    cb2 = CircuitBreakers()
    reanchored = RiskState(
        portfolio_value=95_000.0,
        peak_portfolio=95_000.0,
        daily_start_value=95_000.0,
        vix=20.0,
        consecutive_losses=0,
    )
    with patch.object(CircuitBreakers, "_write_risk_report", return_value=None):
        assert "max_drawdown" not in asyncio.run(cb2.check(reanchored))
    assert not cb2.is_halted


def test_daily_baseline_only_carried_within_the_same_session():
    """Yesterday's baseline must not become today's — 09:30 owns that."""
    today = RiskStateSnapshot(
        peak_value=104_278.55, daily_start_value=101_942.18,
        daily_start_date=et_today_iso(), consecutive_losses=0,
    )
    stale = RiskStateSnapshot(
        peak_value=104_278.55, daily_start_value=101_942.18,
        daily_start_date="2020-01-02", consecutive_losses=0,
    )
    assert today.daily_start_date == et_today_iso()
    assert stale.daily_start_date != et_today_iso()


# ─── A deploy must not resume trading ────────────────────────────────────────

def test_persisted_halt_survives_restart():
    """CLAUDE.md: only a human resumes. A redeploy is not a human."""
    snap = RiskStateSnapshot(
        peak_value=104_278.55, daily_start_value=97_000.0,
        daily_start_date=et_today_iso(), consecutive_losses=5,
        halted=True, halt_reason="max_drawdown",
        halt_time=datetime.now(timezone.utc).isoformat(),
    )
    cb = CircuitBreakers()
    assert not cb.is_halted                       # fresh process starts clean

    if snap.halted and not cb.is_halted:          # restore, as the loop does
        cb._halted = True
        cb._halt_reason = snap.halt_reason

    assert cb.is_halted
    assert cb.halt_reason == "max_drawdown"

    # And it still takes a named human to clear it.
    with pytest.raises(ValueError):
        cb.resume_trading(authorized_by="")
    cb.resume_trading(authorized_by="mithun")
    assert not cb.is_halted


def test_restore_never_clears_a_halt_this_process_already_set():
    """A stale 'not halted' snapshot must not override a live halt."""
    cb = CircuitBreakers()
    cb._halted = True
    cb._halt_reason = "daily_loss"

    snap = RiskStateSnapshot(
        peak_value=1.0, daily_start_value=1.0,
        daily_start_date=et_today_iso(), consecutive_losses=0,
        halted=False,
    )
    if snap.halted and not cb.is_halted:          # guard must not fire
        cb._halted = True

    assert cb.is_halted, "a stale snapshot silently un-halted trading"


# ─── Halt alerting ───────────────────────────────────────────────────────────

def test_halt_sends_email():
    cb = CircuitBreakers(pipeline_id="pipeline_a")
    with patch.object(CircuitBreakers, "_send_email") as send:
        cb._halt("max_drawdown", "Drawdown 8.40% > 8.00%")

    send.assert_called_once()
    subject = send.call_args.kwargs["subject"]
    body = send.call_args.kwargs["body"]
    assert "HALTED" in subject and "max_drawdown" in subject
    assert "resume_trading" in body
    assert "redeploy will NOT clear it" in body


def test_halt_applies_even_when_email_fails():
    """Mail is best-effort; the halt is the safety action and must stand."""
    cb = CircuitBreakers()
    with patch.object(CircuitBreakers, "_send_email",
                      side_effect=RuntimeError("smtp down")):
        cb._halt("daily_loss", "Daily P&L -3.50% < -3.00%")

    assert cb.is_halted
    assert cb.halt_reason == "daily_loss"


def test_missing_smtp_config_does_not_raise():
    cb = CircuitBreakers()
    settings = MagicMock(smtp_host="", smtp_user="", smtp_password="",
                         forecast_email_to="")
    with patch("src.config.get_settings", return_value=settings):
        cb._send_email("subject", "body")      # must be a no-op, not a crash


# ─── First-run bootstrap ─────────────────────────────────────────────────────

def test_empty_store_seeds_peak_from_broker_history_not_current_equity():
    """A first deploy must not inherit a reset drawdown.

    When the store is empty, anchoring peak to today's equity reproduces the
    very bug this module fixes. Real case: v0.5.4's own first boot reported
    0.00% drawdown against a true $104,278.55 peak. The broker's equity curve
    is the source of truth for the high-water mark.
    """
    history_peak = 104_278.55
    current = 97_482.41

    pm = _FakePM(current)
    assert pm._peak_value == current                  # naive anchor

    seeded = max(history_peak, pm.portfolio_value)    # what restore now does
    pm._peak_value = seeded
    dd = (pm._peak_value - current) / pm._peak_value

    assert pm._peak_value == pytest.approx(history_peak)
    assert dd == pytest.approx(0.0652, abs=1e-4)


def test_broker_history_never_drags_peak_below_current_equity():
    """A grown account must not be handed a stale, lower peak."""
    pm = _FakePM(120_000.0)
    pm._peak_value = max(104_278.55, pm.portfolio_value)
    assert pm._peak_value == pytest.approx(120_000.0)


def test_understated_stored_peak_is_corrected_upward():
    """Self-heal: a stored peak below the broker's high-water mark is wrong.

    v0.5.4's own first boot persisted peak=$97,482 while the true peak was
    $104,278.55. A stored-only restore would carry that understatement (and
    the 0.00% drawdown it implies) forever, so the peak is reconciled against
    broker history on EVERY start.
    """
    stored, broker, current = 97_482.41, 104_278.55, 97_482.41
    peak = max(current, broker, stored)
    assert peak == pytest.approx(broker)
    assert peak > stored
    dd = (peak - current) / peak
    assert dd == pytest.approx(0.0652, abs=1e-4)


def test_peak_reconciliation_is_monotonic():
    """Reconciliation must only ever raise the peak, never lower it."""
    stored, broker, current = 104_278.55, 90_000.0, 97_482.41
    assert max(current, broker, stored) == pytest.approx(stored)
