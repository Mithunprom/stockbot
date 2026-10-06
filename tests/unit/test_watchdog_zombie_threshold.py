"""The watchdog must not force-exit positions the ladder is still managing.

Between 2026-10-01 and 2026-10-05, 14 of 18 exits were `integrity_broker_reconcile`
rather than ladder exits. Cause: `_check_zombie_positions` compared against the raw
`SIZING_MAX_HOLD_BARS` (30) while production runs `EXIT_PROFIT_TARGET_MODE=true`,
where the ladder deliberately holds for the full session (390 bars) and the timer
is only a backstop. Every ordinary position became a "zombie" at 90 bars and was
force-sold with no ledger write. Entries at 09:40 ET, kills at 11:10 ET — exactly
30+60 bars. The two genuine ladder exits in the window both fired under 90 bars.

The whole suite stayed green because `EXIT_PROFIT_TARGET_MODE` defaults OFF in
code, so every existing test exercised the one mode where the two constants agree.
That is the gap these tests close: the mode PRODUCTION runs is now covered.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.agents import signal_loop as sl
from src.agents.watchdog_agent import ZOMBIE_GRACE_BARS, WatchdogAgent


def _agent() -> WatchdogAgent:
    loop = MagicMock()
    loop._bars_held = {}
    pm = MagicMock()
    pm._positions = {}
    cb = MagicMock()
    cb._halted = False
    agent = WatchdogAgent(signal_loop=loop, pos_manager=pm, circuit_breakers=cb)
    agent._market_open = lambda: True  # type: ignore[method-assign]
    agent._send_email = MagicMock()  # type: ignore[method-assign]
    return agent


@pytest.fixture
def profit_target(monkeypatch):
    """The mode production actually runs."""
    monkeypatch.setattr(sl, "EXIT_PROFIT_TARGET_MODE", True)


# ── The regression ────────────────────────────────────────────────────────────

def test_ordinary_position_is_not_a_zombie_in_profit_target_mode(profit_target):
    """90 bars is a normal mid-session hold here, not a stuck position."""
    agent = _agent()
    agent._pm._positions = {"COHR": MagicMock()}
    agent._loop._bars_held = {"COHR": sl.SIZING_MAX_HOLD_BARS + ZOMBIE_GRACE_BARS + 1}
    assert agent._check_zombie_positions()["status"] == "ok"


def test_a_position_past_the_session_is_still_a_zombie(profit_target):
    """The check must keep working — past the full session + grace is genuinely stuck."""
    agent = _agent()
    agent._pm._positions = {"COHR": MagicMock()}
    agent._loop._bars_held = {"COHR": sl.FULL_SESSION_BARS + ZOMBIE_GRACE_BARS + 1}
    check = agent._check_zombie_positions()
    assert check["status"] == "critical"
    assert check["zombies"] == ["COHR"]


def test_timer_mode_behaviour_is_unchanged(monkeypatch):
    """With the ladder off, the threshold stays at max_hold + grace."""
    monkeypatch.setattr(sl, "EXIT_PROFIT_TARGET_MODE", False)
    agent = _agent()
    agent._pm._positions = {"GOOGL": MagicMock()}
    agent._loop._bars_held = {"GOOGL": sl.SIZING_MAX_HOLD_BARS + 5}
    assert agent._check_zombie_positions()["status"] == "ok"
    agent._loop._bars_held = {"GOOGL": sl.SIZING_MAX_HOLD_BARS + ZOMBIE_GRACE_BARS + 1}
    assert agent._check_zombie_positions()["status"] == "critical"


def test_threshold_tracks_the_effective_hold_cap(profit_target):
    """Pin the relationship, not the literal — the next mode must not re-break it."""
    agent = _agent()
    agent._pm._positions = {"X": MagicMock()}
    cap = sl._effective_hold_bars()
    agent._loop._bars_held = {"X": cap + ZOMBIE_GRACE_BARS - 1}
    assert agent._check_zombie_positions()["status"] == "ok"
    agent._loop._bars_held = {"X": cap + ZOMBIE_GRACE_BARS}
    assert agent._check_zombie_positions()["status"] == "critical"


# ── A force-exit must leave a record ─────────────────────────────────────────

def _filled(price=101.0, qty=10.0):
    r = MagicMock()
    r.status = "filled"
    r.filled_avg_price = price
    r.filled_qty = qty
    r.filled_at = datetime(2026, 10, 5, 15, 10, tzinfo=timezone.utc)
    return r


def test_force_exit_is_booked_with_its_own_reason(monkeypatch):
    """It must be distinguishable from a dropped exit in the ledger.

    Reusing a ladder reason — or writing nothing, as before — is what made the
    October failure unreadable: a force-exit and a lost exit both surfaced as
    `integrity_broker_reconcile`.
    """
    agent = _agent()
    pos = MagicMock()
    pos.qty = 10.0
    pos.side = "long"
    agent._pm._positions = {"COHR": pos}
    monkeypatch.setenv("WATCHDOG_FORCE_EXIT", "true")

    booked: dict = {}

    async def fake_record(**kwargs):
        booked.update(kwargs)

    async def fake_submit(req):
        return _filled()

    agent._loop._alpaca.submit_order = fake_submit
    agent._loop.record_forced_exit = fake_record

    with patch("src.config.get_settings") as gs:
        gs.return_value.alpaca_mode = "paper"
        healed = asyncio.run(agent._heal_zombies(["COHR"]))

    assert healed == ["COHR"]
    assert booked["exit_reason"] == "watchdog_force_exit"
    assert booked["fill_price"] == 101.0
    assert booked["ticker"] == "COHR"


def test_an_unfilled_ack_is_left_unpriced(monkeypatch):
    """`accepted`/`new` carry no fill price — guessing one corrupted the ledger before."""
    agent = _agent()
    pos = MagicMock()
    pos.qty = 10.0
    pos.side = "long"
    agent._pm._positions = {"COHR": pos}
    monkeypatch.setenv("WATCHDOG_FORCE_EXIT", "true")

    called = []

    async def fake_record(**kwargs):
        called.append(kwargs)

    async def fake_submit(req):
        r = MagicMock()
        r.status = "accepted"
        return r

    agent._loop._alpaca.submit_order = fake_submit
    agent._loop.record_forced_exit = fake_record

    with patch("src.config.get_settings") as gs:
        gs.return_value.alpaca_mode = "paper"
        healed = asyncio.run(agent._heal_zombies(["COHR"]))

    assert healed == ["COHR"]      # the heal still counts
    assert called == []            # but nothing is priced


def test_bookkeeping_failure_does_not_strand_remaining_zombies(monkeypatch):
    """The order already filled. A write error must not abort the heal loop.

    This is the load-bearing one: if recording raised, the first ticker's
    failure would leave every later zombie open — turning an observability bug
    back into a position-management bug.
    """
    agent = _agent()
    for t in ("COHR", "MSTR"):
        pos = MagicMock()
        pos.qty = 10.0
        pos.side = "long"
        agent._pm._positions[t] = pos
    monkeypatch.setenv("WATCHDOG_FORCE_EXIT", "true")

    async def fake_submit(req):
        return _filled()

    async def boom(**kwargs):
        raise RuntimeError("db down")

    agent._loop._alpaca.submit_order = fake_submit
    agent._loop.record_forced_exit = boom

    with patch("src.config.get_settings") as gs:
        gs.return_value.alpaca_mode = "paper"
        healed = asyncio.run(agent._heal_zombies(["COHR", "MSTR"]))

    assert healed == ["COHR", "MSTR"]
