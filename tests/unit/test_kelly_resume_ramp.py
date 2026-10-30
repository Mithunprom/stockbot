"""`inactive` must throttle, not permit full risk.

On 2026-10-15 the hard block's 10-day window drained, the sample fell under
KELLY_MIN_TRADES, and `_kelly_mode()` returned "inactive" — a state that applied
NO throttle whatsoever. On 2026-10-16 the loop opened all six slots at ~$12.7k
each on the day it knew least about its own edge and lost $2,147.89 (MRNA −$737,
STX −$520, NOW −$395, AMD −$323, HPE −$279, UNH +$107). Kelly then measured
−0.651 and re-blocked.

`inactive` was the MOST permissive mode in the governor despite carrying the
LEAST information. CLAUDE.md: "The bot should earn the right to larger positions
over time through demonstrated performance, not assume them."

Every timestamp here is relative to `now`. `_prune_kelly_window` drops outcomes
older than KELLY_LOOKBACK_DAYS, so an absolute date makes these tests pass for
ten days and then fail on a date with no code change behind it — that exact time
bomb was planted once already in test_record_forced_exit.py and detonated.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.agents import signal_loop as sl


def _loop(n_outcomes: int, pnl_pct: float = 0.004, entries_today: int = 0):
    """A SignalLoop stub carrying `n_outcomes` fresh outcomes in the window."""

    class _L:
        pass

    loop = _L()
    now = datetime.now(timezone.utc)
    loop._sizing_recent_outcomes = [
        (now - timedelta(minutes=5 + i), pnl_pct) for i in range(n_outcomes)
    ]
    loop._sizing_n_trades_today = entries_today
    loop._kelly_fraction = 0.0
    loop._kelly_min_trades = sl.KELLY_MIN_TRADES
    # `_kelly_size_cap` scales off the portfolio, so the stub needs a manager.
    class _PM:
        portfolio_value = 100_000.0
    loop._pm = _PM()
    for name in ("_prune_kelly_window", "_kelly_mode", "_kelly_entries_blocked",
                 "_kelly_ramp_entry_cap", "_kelly_size_cap",
                 "_kelly_probe_notional"):
        setattr(loop, name, getattr(sl.SignalLoop, name).__get__(loop))
    return loop


# ── The ramp exists and scales with evidence ─────────────────────────────────

def test_the_ramp_applies_while_the_sample_rebuilds():
    loop = _loop(n_outcomes=0)
    assert loop._kelly_mode() == "inactive"
    assert loop._kelly_ramp_entry_cap() is not None


@pytest.mark.parametrize("n,expected", [
    (0, 1), (1, 1), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3), (8, 4), (9, 4),
])
def test_the_cap_scales_with_closed_outcomes(n, expected):
    """One extra entry per KELLY_RAMP_OUTCOMES_PER_SLOT outcomes, floor 1."""
    assert _loop(n_outcomes=n)._kelly_ramp_entry_cap() == expected


def test_the_cap_is_monotone_in_the_sample_size():
    """More evidence may never buy LESS allowance."""
    caps = [_loop(n_outcomes=n)._kelly_ramp_entry_cap() for n in range(0, 10)]
    assert caps == sorted(caps)


# ── The Oct-16 scenario, pinned ──────────────────────────────────────────────

def test_at_n8_the_oct16_batch_is_not_permitted():
    """The specific regression: n=8, inactive, must NOT allow 6 entries."""
    loop = _loop(n_outcomes=8)
    assert loop._kelly_mode() == "inactive"
    cap = loop._kelly_ramp_entry_cap()
    assert cap < 6, f"n=8 still permits {cap} entries — Oct 16 could recur"
    assert cap == 4


def test_the_ramp_closes_the_gate_once_its_budget_is_spent():
    """The gate reads `_sizing_n_trades_today`, the same counter Gate 1 uses."""
    loop = _loop(n_outcomes=8, entries_today=4)
    cap = loop._kelly_ramp_entry_cap()
    assert loop._sizing_n_trades_today >= cap   # 5th entry of the day refused


def test_entries_are_probe_sized_while_inactive():
    """Rationing the COUNT alone was not enough.

    Each Oct-16 entry was ~$12.7k, so even a 4-entry cap would have risked
    ~$50k on an unmeasured edge. The notional cap is what makes the resume
    probe-like.

    Asserts BEHAVIOUR, not source text. The previous version matched the
    strings `_kmode in ("probation", "inactive")` and
    `KELLY_PROBATION_NOTIONAL`; both became misleading on 2026-10-28 when the
    cap moved into `_kelly_size_cap()` and the constant became a fraction — and
    the substring check would still have passed off a COMMENT mentioning the old
    name. A source-text assertion that passes for the wrong reason is worse than
    no assertion.
    """
    loop = _loop(n_outcomes=0)
    cap = loop._kelly_size_cap()
    assert cap is not None
    # A probe is ~1.25% of portfolio, nowhere near the ~$12.5k per-slot size.
    assert 500.0 < cap < 2_000.0


# ── Invariant: never looser than before ──────────────────────────────────────

@pytest.mark.parametrize("n", list(range(0, 12)))
def test_the_ramp_can_only_subtract_entries(n):
    """Bounded above by the cap Gate 1 already enforces, for every n."""
    cap = _loop(n_outcomes=n)._kelly_ramp_entry_cap()
    if cap is not None:
        assert cap <= sl.SIZING_MAX_TRADES_PER_DAY


def test_the_ramp_does_not_touch_normal_or_probation():
    """Those modes have their own gates; the ramp must return None there."""
    normal = _loop(n_outcomes=sl.KELLY_MIN_TRADES)
    normal._kelly_fraction = 0.25
    assert normal._kelly_mode() == "normal"
    assert normal._kelly_ramp_entry_cap() is None

    probation = _loop(n_outcomes=sl.KELLY_MIN_TRADES)
    probation._kelly_fraction = -0.05
    assert probation._kelly_mode() == "probation"
    assert probation._kelly_ramp_entry_cap() is None


# ── Invariant: the hard block still wins ─────────────────────────────────────

def test_the_hard_block_still_fires_and_takes_precedence():
    """The ramp must not become a route around the negative-edge block."""
    loop = _loop(n_outcomes=sl.KELLY_MIN_TRADES)
    loop._kelly_fraction = sl.KELLY_HARD_BLOCK_THRESHOLD - 0.01
    assert loop._kelly_entries_blocked() is True
    assert loop._kelly_ramp_entry_cap() is None   # not inactive → no ramp

    import inspect
    src = inspect.getsource(sl.SignalLoop._sizing_entry_gate_open)
    assert src.index("_kelly_entries_blocked") < src.index("_kelly_ramp_entry_cap")


def test_the_block_threshold_was_not_touched():
    assert sl.KELLY_HARD_BLOCK_THRESHOLD == -0.25


# ── Invariant: no deadlock ───────────────────────────────────────────────────

def test_the_ramp_always_permits_forward_progress():
    """A cap of 0 at any n would brick the governor permanently."""
    for n in range(0, sl.KELLY_MIN_TRADES):
        assert _loop(n_outcomes=n)._kelly_ramp_entry_cap() >= 1


def test_the_sample_can_reach_a_measurable_state_against_the_draining_window():
    """The ramp must outrun KELLY_LOOKBACK_DAYS, not plateau under the bar.

    This subsystem has frozen the bot twice (2026-05-27, 2026-06-26). Simulate
    the accumulation: each session adds up to `cap` closed outcomes. A flat
    1/day would add ~1 per session against a 10-day window also draining ~1 per
    session once steady — it could sit just below KELLY_MIN_TRADES forever. The
    n//2 shape accelerates instead.
    """
    n = 0
    sessions_traded = 0
    for _ in range(sl.KELLY_LOOKBACK_DAYS):
        if n >= sl.KELLY_MIN_TRADES:
            break
        n += _loop(n_outcomes=n)._kelly_ramp_entry_cap()
        sessions_traded += 1
    assert n >= sl.KELLY_MIN_TRADES, (
        f"sample stalled at {n} after {sl.KELLY_LOOKBACK_DAYS} sessions"
    )
    # Accumulation is 1,2,3,4,6,9,13 — measurable on session 7, comfortably
    # inside the 10-day window so the oldest outcomes have not yet aged out.
    assert sessions_traded == 7, f"took {sessions_traded} sessions"
    assert sessions_traded < sl.KELLY_LOOKBACK_DAYS


# ── Observability ────────────────────────────────────────────────────────────

def test_the_ramp_is_exposed_at_diagnostics():
    """`/diagnostics` is a WHITELIST in main.py — a field is not automatic."""
    main_src = open("main.py").read()
    assert '"kelly_ramp_entry_cap": summary.get("kelly_ramp_entry_cap")' in main_src
    loop_src = open("src/agents/signal_loop.py").read()
    assert '"kelly_ramp_entry_cap": self._kelly_ramp_entry_cap()' in loop_src


# ── H13: exits must be untouched ─────────────────────────────────────────────

def test_the_ramp_is_an_entry_gate_only():
    """Per H13, an entry throttle may never affect exits or strand a position."""
    import inspect

    exit_src = inspect.getsource(sl.SignalLoop._check_sizing_exit)
    assert "_kelly_ramp_entry_cap" not in exit_src
    assert "ramp" not in exit_src.lower()

    gate_src = inspect.getsource(sl.SignalLoop._sizing_entry_gate_open)
    assert "_kelly_ramp_entry_cap" in gate_src


# ── Size ramp: notional scales with evidence, no 10x cliff ───────────────────
#
# Added 2026-10-28 after the owner asked why a $99k account was buying $1,200 of
# AMD. Two real defects behind that: the probe was a hardcoded $1,200 rather
# than a share of the portfolio, and the step out of `inactive` was a 10x cliff
# (four entries at ~$1,200, then $3,949 and $6,045 once the sample crossed
# KELLY_MIN_TRADES mid-session).

def _full_slot(pv: float = 100_000.0) -> float:
    """The per-slot notional the normal path would allow."""
    return pv * sl.PORTFOLIO_HEAT_CEILING / sl.MAX_OPEN_POSITIONS


def test_probe_size_scales_with_the_portfolio():
    """The old $1,200 constant meant 1.2% at $99k but 6% at $20k.

    A risk budget whose meaning changes with account size is a magic number,
    not a policy.
    """
    # Proportionality holds ABOVE the floor. The floor binds below
    # KELLY_PROBATION_NOTIONAL_FLOOR / FRAC (~$48k), deliberately — that is the
    # small-account guard, not a scaling bug, and asserting strict
    # proportionality through it is what my first version of this test got
    # wrong.
    floor_pv = sl.KELLY_PROBATION_NOTIONAL_FLOOR / sl.KELLY_PROBATION_NOTIONAL_FRAC
    small, large = _loop(n_outcomes=0), _loop(n_outcomes=0)
    small._pm.portfolio_value = 100_000.0
    large._pm.portfolio_value = 400_000.0
    assert small._pm.portfolio_value > floor_pv
    assert large._kelly_probe_notional() == pytest.approx(
        4 * small._kelly_probe_notional())
    for pv in (100_000.0, 250_000.0, 400_000.0):
        loop = _loop(n_outcomes=0)
        loop._pm.portfolio_value = pv
        assert loop._kelly_probe_notional() / pv == pytest.approx(
            sl.KELLY_PROBATION_NOTIONAL_FRAC)


def test_probe_size_reproduces_the_old_behaviour_at_the_current_portfolio():
    """At the 2026-10-28 portfolio this is a refactor, not a size change."""
    loop = _loop(n_outcomes=0)
    loop._pm.portfolio_value = 99_074.43
    assert loop._kelly_probe_notional() == pytest.approx(1238.43, abs=1.0)


def test_a_tiny_account_can_still_place_a_probe():
    """Below the floor the probe would fall under the $500 entry minimum and
    silently stop the window refreshing — re-creating the deadlock."""
    loop = _loop(n_outcomes=0)
    loop._pm.portfolio_value = 5_000.0
    assert loop._kelly_probe_notional() >= 500.0


def test_the_size_ramp_is_monotone_and_continuous():
    """No cliff: each step up is a fraction of the span, not the whole of it."""
    caps = []
    for n in range(0, sl.KELLY_MIN_TRADES):
        loop = _loop(n_outcomes=n)
        caps.append(loop._kelly_size_cap())
    assert caps == sorted(caps), "more evidence bought less size"
    span = _full_slot() - caps[0]
    biggest_step = max(b - a for a, b in zip(caps, caps[1:]))
    assert biggest_step < 0.5 * span, (
        f"largest single step {biggest_step:.0f} is a cliff, not a ramp")


def test_the_ramp_reaches_full_size_exactly_as_inactive_ends():
    """The discontinuity at the mode boundary is what this removes."""
    loop = _loop(n_outcomes=sl.KELLY_MIN_TRADES - 1)
    assert loop._kelly_mode() == "inactive"
    near_full = loop._kelly_size_cap()
    assert near_full == pytest.approx(_full_slot(), rel=0.12)


@pytest.mark.parametrize("n", list(range(0, 14)))
def test_the_size_cap_can_only_reduce_risk(n):
    """The load-bearing invariant, asserted directly rather than implied.

    Previously this property lived in the count bound plus a `min()` and was
    never stated, so an edit replacing the `min()` with an assignment would have
    passed the suite.
    """
    loop = _loop(n_outcomes=n)
    cap = loop._kelly_size_cap()
    if cap is not None:
        assert cap <= _full_slot() + 1e-6, f"n={n} cap {cap} exceeds a full slot"


def test_probation_stays_flat_and_does_not_ramp():
    """probation has a sample and it says the edge is non-positive.

    There is nothing to ramp toward, so size must stay at the probe — only
    `inactive` (rebuilding, no verdict yet) earns size back.
    """
    loop = _loop(n_outcomes=sl.KELLY_MIN_TRADES)
    loop._kelly_fraction = -0.05
    assert loop._kelly_mode() == "probation"
    assert loop._kelly_size_cap() == pytest.approx(loop._kelly_probe_notional())


def test_normal_mode_has_no_size_cap():
    """The sizer's own output stands once the edge is measured positive."""
    loop = _loop(n_outcomes=sl.KELLY_MIN_TRADES)
    loop._kelly_fraction = 0.25
    assert loop._kelly_mode() == "normal"
    assert loop._kelly_size_cap() is None


def test_the_oct28_session_would_have_ramped_not_jumped():
    """The session that prompted this: $1,200 x4 then $3,949, $6,045.

    Under the ramp the first four entries already scale with the sample
    instead of sitting flat at the probe floor.
    """
    pv = 99_074.43
    caps = []
    for n in (0, 1, 2, 3):
        loop = _loop(n_outcomes=n)
        loop._pm.portfolio_value = pv
        caps.append(loop._kelly_size_cap())
    assert caps[0] < caps[-1], "first four entries still flat — no ramp applied"
    assert all(c <= pv * sl.PORTFOLIO_HEAT_CEILING / sl.MAX_OPEN_POSITIONS for c in caps)
