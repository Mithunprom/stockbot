"""H40 — halt_lift_readiness() unit tests.

Tests the pure-stdlib halt_lift.py module that produces the go/no-go block
for /diagnostics. No numpy/pandas/structlog needed.

As of 2026-11-11 the live state is:
  cb_halted=True, kelly_fraction=0.0, kelly_n_trades=0, kelly_mode=inactive,
  n_open_positions=0, n_ic_blocked=19, email_alerts_enabled=False
  → can_resume_safely=False (blockers: EMAIL_ALERTS_DISABLED, IC_CONTAMINATION_HIGH)
"""
from __future__ import annotations

import pytest

from src.risk.halt_lift import halt_lift_readiness


# ── helpers ──────────────────────────────────────────────────────────────────

def _base(**overrides):
    """Minimal safe state: Kelly recovered, no positions, emails on, no IC blocks."""
    defaults = dict(
        cb_halted=True,
        kelly_fraction=0.0,
        kelly_n_trades=0,
        kelly_mode="inactive",
        n_open_positions=0,
        n_ic_blocked=0,
        email_alerts_enabled=True,
    )
    defaults.update(overrides)
    return halt_lift_readiness(**defaults)


def _ids(result, field):
    return [item["id"] for item in result[field]]


# ── return-shape contract ─────────────────────────────────────────────────────

def test_result_has_required_keys():
    r = _base()
    assert set(r) >= {"can_resume_safely", "blockers", "requirements_met", "summary"}


def test_can_resume_safely_is_bool():
    assert isinstance(_base()["can_resume_safely"], bool)


def test_blockers_is_list():
    assert isinstance(_base()["blockers"], list)


def test_requirements_met_is_list():
    assert isinstance(_base()["requirements_met"], list)


def test_summary_is_nonempty_str():
    r = _base()
    assert isinstance(r["summary"], str) and len(r["summary"]) > 10


# ── happy path: fully clear ───────────────────────────────────────────────────

def test_all_clear_gives_can_resume_true():
    r = _base()
    assert r["can_resume_safely"] is True


def test_all_clear_no_blockers():
    r = _base()
    assert r["blockers"] == []


def test_all_clear_summary_mentions_ready():
    r = _base()
    assert "ready" in r["summary"].lower() or "clear" in r["summary"].lower()


# ── Kelly recovery recorded as a requirement_met ─────────────────────────────

def test_kelly_inactive_recorded_as_requirement_met():
    r = _base(kelly_n_trades=0, kelly_mode="inactive")
    assert "KELLY_HARD_BLOCK_RELEASED" in _ids(r, "requirements_met")


def test_kelly_window_empty_recorded():
    r = _base(kelly_n_trades=0, kelly_mode="inactive")
    met = {item["id"]: item for item in r["requirements_met"]}
    assert "n=0" in met["KELLY_HARD_BLOCK_RELEASED"]["detail"]


def test_kelly_recovered_does_not_produce_blocker():
    r = _base(kelly_n_trades=0, kelly_mode="inactive")
    assert all(b["id"] != "KELLY_HARD_BLOCK" for b in r["blockers"])


# ── zero open positions is a requirement_met ─────────────────────────────────

def test_zero_positions_recorded_as_requirement():
    r = _base(n_open_positions=0)
    assert "ZERO_OPEN_POSITIONS" in _ids(r, "requirements_met")


def test_open_positions_do_not_add_requirement():
    r = _base(n_open_positions=3)
    assert "ZERO_OPEN_POSITIONS" not in _ids(r, "requirements_met")


# ── email alerts disabled → HIGH blocker ─────────────────────────────────────

def test_email_disabled_adds_blocker():
    r = _base(email_alerts_enabled=False)
    assert "EMAIL_ALERTS_DISABLED" in _ids(r, "blockers")


def test_email_disabled_blocks_resume():
    r = _base(email_alerts_enabled=False)
    assert r["can_resume_safely"] is False


def test_email_disabled_severity_is_high():
    r = _base(email_alerts_enabled=False)
    b = next(x for x in r["blockers"] if x["id"] == "EMAIL_ALERTS_DISABLED")
    assert b["severity"] == "HIGH"


def test_email_enabled_no_email_blocker():
    r = _base(email_alerts_enabled=True)
    assert "EMAIL_ALERTS_DISABLED" not in _ids(r, "blockers")


# ── IC contamination ─────────────────────────────────────────────────────────

def test_ic_blocked_19_adds_high_blocker():
    r = _base(n_ic_blocked=19)
    assert "IC_CONTAMINATION_HIGH" in _ids(r, "blockers")


def test_ic_blocked_19_severity_is_high():
    r = _base(n_ic_blocked=19)
    b = next(x for x in r["blockers"] if x["id"] == "IC_CONTAMINATION_HIGH")
    assert b["severity"] == "HIGH"


def test_ic_blocked_10_threshold_exact():
    # Exactly at threshold → HIGH
    r = _base(n_ic_blocked=10)
    assert "IC_CONTAMINATION_HIGH" in _ids(r, "blockers")


def test_ic_blocked_9_gives_medium():
    r = _base(n_ic_blocked=9)
    assert "IC_BLOCKS_PRESENT" in _ids(r, "blockers")
    b = next(x for x in r["blockers"] if x["id"] == "IC_BLOCKS_PRESENT")
    assert b["severity"] == "MEDIUM"


def test_ic_blocked_zero_no_ic_blocker():
    r = _base(n_ic_blocked=0)
    ids = _ids(r, "blockers")
    assert "IC_CONTAMINATION_HIGH" not in ids
    assert "IC_BLOCKS_PRESENT" not in ids


def test_ic_detail_mentions_count():
    r = _base(n_ic_blocked=19)
    b = next(x for x in r["blockers"] if x["id"] == "IC_CONTAMINATION_HIGH")
    assert "19" in b["detail"]


# ── live state as of 2026-11-11 ───────────────────────────────────────────────

def test_live_state_nov11_cannot_resume():
    """The actual live state: Kelly recovered but email off and 19 IC-blocked."""
    r = halt_lift_readiness(
        cb_halted=True,
        kelly_fraction=0.0,
        kelly_n_trades=0,
        kelly_mode="inactive",
        n_open_positions=0,
        n_ic_blocked=19,
        email_alerts_enabled=False,
    )
    assert r["can_resume_safely"] is False


def test_live_state_nov11_has_two_blockers():
    r = halt_lift_readiness(
        cb_halted=True,
        kelly_fraction=0.0,
        kelly_n_trades=0,
        kelly_mode="inactive",
        n_open_positions=0,
        n_ic_blocked=19,
        email_alerts_enabled=False,
    )
    assert len(r["blockers"]) == 2


def test_live_state_nov11_has_two_requirements_met():
    r = halt_lift_readiness(
        cb_halted=True,
        kelly_fraction=0.0,
        kelly_n_trades=0,
        kelly_mode="inactive",
        n_open_positions=0,
        n_ic_blocked=19,
        email_alerts_enabled=False,
    )
    ids = _ids(r, "requirements_met")
    assert "KELLY_HARD_BLOCK_RELEASED" in ids
    assert "ZERO_OPEN_POSITIONS" in ids


def test_live_state_summary_mentions_not_ready():
    r = halt_lift_readiness(
        cb_halted=True,
        kelly_fraction=0.0,
        kelly_n_trades=0,
        kelly_mode="inactive",
        n_open_positions=0,
        n_ic_blocked=19,
        email_alerts_enabled=False,
    )
    assert "NOT" in r["summary"] or "not" in r["summary"].lower()


# ── not-halted branch ─────────────────────────────────────────────────────────

def test_not_halted_clean_state_still_reports_healthy():
    r = _base(cb_halted=False)
    assert r["can_resume_safely"] is True
    assert "not halted" in r["summary"].lower()


def test_not_halted_with_blocker_reports_issue():
    r = _base(cb_halted=False, email_alerts_enabled=False)
    assert "not halted" in r["summary"].lower()
    assert r["can_resume_safely"] is False


# ── summary content checks ────────────────────────────────────────────────────

def test_blocker_ids_appear_in_summary_when_blocked():
    r = _base(email_alerts_enabled=False, n_ic_blocked=19)
    assert "EMAIL_ALERTS_DISABLED" in r["summary"]
    assert "IC_CONTAMINATION_HIGH" in r["summary"]


def test_summary_mentions_resume_path_when_clear():
    r = _base()
    assert "resume-trading" in r["summary"] or "resume" in r["summary"].lower()
