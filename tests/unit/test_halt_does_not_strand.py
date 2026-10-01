"""H13 — a circuit-breaker halt must block entries but never block exits.

`_act_on_signal` began with an unconditional `if self._cb.is_halted: return
False`. The per-tick exit loop reaches exits THROUGH that same function (see
_tick, which calls _act_on_signal for every held ticker), so the early return
killed exits along with entries: a halt froze the book instead of de-risking it.

Observed live 2026-09-30 — `max_drawdown` halted the bot with six positions
open at ~70% heat and +$794 unrealised, and no code path could close them. The
weekly reviews had carried this as "HIGH: halt strands positions" since W32,
unmerged for 41 days.

A breaker exists to stop taking NEW risk. An exit reduces risk, so blocking it
inverts the control.
"""
from __future__ import annotations

import inspect

from src.agents import signal_loop as sl


def _act_on_signal_source() -> str:
    return inspect.getsource(sl.SignalLoop._act_on_signal)


def test_halt_guard_is_conditional_on_having_no_position():
    """REGRESSION: the guard must not be an unconditional early return."""
    src = _act_on_signal_source()
    assert "if self._cb.is_halted and not has_position:" in src, (
        "the halt guard is not position-aware — an unconditional "
        "`if self._cb.is_halted: return False` strands open positions"
    )
    assert "if self._cb.is_halted:\n            return False" not in src, (
        "unconditional halt return reintroduced; exits would be blocked"
    )


def test_position_is_resolved_before_the_halt_guard():
    """`has_position` must be computed first, or the guard cannot use it."""
    src = _act_on_signal_source()
    pos_at = src.index("has_position = ")
    guard_at = src.index("if self._cb.is_halted")
    assert pos_at < guard_at, (
        "has_position is assigned after the halt guard — the guard would "
        "reference it before it exists"
    )


def test_exit_check_is_reachable_while_halted():
    """The exit call must sit after the guard, and every halt return between
    function entry and that exit call must be position-scoped.

    Matching on code lines only — the explanatory comment above the guard
    quotes the old unconditional form, so a naive substring search finds the
    comment instead of the statement.
    """
    src = _act_on_signal_source()
    code = [ln for ln in src.split("\n") if not ln.lstrip().startswith("#")]

    guard_lines = [i for i, ln in enumerate(code) if "self._cb.is_halted" in ln]
    exit_lines = [i for i, ln in enumerate(code) if "_check_sizing_exit(" in ln]
    assert guard_lines and exit_lines
    assert guard_lines[0] < exit_lines[0], "exit check precedes the halt guard"

    for i in guard_lines:
        if i < exit_lines[0]:
            assert "not has_position" in code[i], (
                f"halt return at code line {i} is unconditional and would "
                f"block the exit check: {code[i].strip()!r}"
            )


def test_entries_remain_blocked_while_halted():
    """De-risking must not become a licence to open new positions.

    The entry branch is reached only when has_position is False, which is
    exactly the case the guard still refuses while halted.
    """
    src = _act_on_signal_source()
    assert "if self._cb.is_halted and not has_position:\n            return False" in src


def test_held_branch_cannot_open_a_position():
    """Belt and braces: the has_position branch must exit or hold, not enter.

    If a future edit let the held branch place an entry order, the new guard
    would be admitting orders during a halt.
    """
    src = _act_on_signal_source()
    held_start = src.index("if has_position:")
    held_end = src.index("else:", held_start)
    held_block = src[held_start:held_end]
    for forbidden in ("self._sizer.compute(", "_sizing_entry_gate_open("):
        assert forbidden not in held_block, (
            f"{forbidden} appears in the held-position branch — a halted book "
            f"could place an entry"
        )
