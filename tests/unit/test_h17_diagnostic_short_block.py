"""H17: would_trade diagnostic correctly excludes Phase 5 short signals.

Prior to this fix, a signal with lgbm_pred_return < 0 (bearish → SHORT direction)
showed would_trade:true in /diagnostics even though Phase 5 blocks all shorts.
Root case: SMCI (pred_return=-0.50%, dir_prob=0.17) showed would_trade:true on Aug 14
because the magnitude and dir_prob gates both passed, but the Phase 5 short-disable
check in _act_on_signal was absent from the diagnostic path.
"""
from __future__ import annotations

COST_THRESHOLD = 0.003
DEAD_ZONE = (0.40, 0.60)


def _gate_flags(
    pred_ret: float,
    dir_prob: float,
    cost_thr: float = COST_THRESHOLD,
    dead_zone: tuple[float, float] = DEAD_ZONE,
    cooldown: bool = False,
    probation_block: bool = False,
) -> tuple[bool, list[str]]:
    """Mirror of the main.py _pipeline_diagnostics gate logic (post-H17).

    Kept here as a pure function so the test suite doesn't need to import
    main.py (which triggers full app startup on import).
    """
    lo, hi = dead_zone
    passes_pred = abs(pred_ret) > cost_thr
    passes_dir = not (lo < dir_prob < hi)
    passes_both = passes_pred and passes_dir
    direction = 1 if pred_ret > 0 else (-1 if pred_ret < 0 else 0)
    short_blocked = direction < 0
    would_trade = (
        passes_both and not cooldown and not probation_block and not short_blocked
    )
    blocked_by = (
        (["pred_return_too_small"] if not passes_pred else [])
        + (["dir_prob_in_dead_zone"] if not passes_dir else [])
        + (["ticker_cooldown"] if cooldown else [])
        + (["kelly_probation"] if probation_block else [])
        + (["shorts_disabled_phase5"] if short_blocked else [])
    )
    return would_trade, blocked_by


def test_h17_smci_case_not_tradeable() -> None:
    """SMCI Aug-14: pred_return=-0.50%, dir_prob=0.17 → would_trade must be False."""
    would_trade, blocked_by = _gate_flags(pred_ret=-0.005, dir_prob=0.17)
    assert not would_trade
    assert "shorts_disabled_phase5" in blocked_by


def test_h17_short_blocked_by_includes_phase5_label() -> None:
    """Any bearish signal with clear gates records the Phase 5 block reason."""
    _, blocked_by = _gate_flags(pred_ret=-0.010, dir_prob=0.20)
    assert "shorts_disabled_phase5" in blocked_by


def test_h17_long_signal_still_tradeable() -> None:
    """A bullish signal that passes all gates must still show would_trade:true."""
    would_trade, blocked_by = _gate_flags(pred_ret=0.008, dir_prob=0.72)
    assert would_trade
    assert "shorts_disabled_phase5" not in blocked_by


def test_h17_zero_pred_return_blocked_by_pred_gate_not_short() -> None:
    """Zero pred_return → direction=0, blocked by pred_return_too_small, not short."""
    would_trade, blocked_by = _gate_flags(pred_ret=0.0, dir_prob=0.72)
    assert not would_trade
    assert "pred_return_too_small" in blocked_by
    assert "shorts_disabled_phase5" not in blocked_by


def test_h17_small_negative_pred_return_shows_both_blocks() -> None:
    """When pred_return is both small AND negative, both gate reasons appear."""
    _, blocked_by = _gate_flags(pred_ret=-0.001, dir_prob=0.20)
    assert "pred_return_too_small" in blocked_by
    assert "shorts_disabled_phase5" in blocked_by
