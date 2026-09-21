"""Entry-rank monitor — the live detector for train/serve skew.

The v0.6.x window lost money because filled entries sat at the ~46th percentile
of the model's own cross-sectional ranking rather than the top decile. Nothing
reported it: the model loaded, every feature computed, the health checks were
green, and the only visible symptom was the P&L two months later.

This monitor turns that into one number per fill. These tests pin the maths and,
more importantly, that a 46th-percentile book reads as UNHEALTHY.
"""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from src.agents.signal_loop import ENTRY_RANK_HEALTHY_PCTILE, SignalLoop


def _sig(ticker: str, pred: float) -> SimpleNamespace:
    return SimpleNamespace(ticker=ticker, lgbm_pred_return=pred)


class _Loop:
    """Minimal stand-in exposing just the recorder under test."""

    def __init__(self) -> None:
        self._entry_ranks = deque(maxlen=60)

    _record_entry_rank = SignalLoop._record_entry_rank


def _candidates(n: int = 20):
    # pred_return spread 0.001 .. 0.020
    return [_sig(f"T{i}", 0.001 * (i + 1)) for i in range(n)]


def test_top_pick_scores_near_100():
    loop = _Loop()
    cands = _candidates()
    loop._record_entry_rank(cands[-1], cands)          # strongest signal
    assert loop._entry_ranks[0] >= 95


def test_median_pick_scores_near_50():
    """REGRESSION: this is what production was doing for two months."""
    loop = _Loop()
    cands = _candidates()
    loop._record_entry_rank(cands[len(cands) // 2], cands)
    assert 40 <= loop._entry_ranks[0] <= 60
    assert loop._entry_ranks[0] < ENTRY_RANK_HEALTHY_PCTILE, (
        "a median-ranked entry must not read as healthy — that is precisely "
        "the failure this monitor exists to catch"
    )


def test_weakest_pick_scores_near_zero():
    loop = _Loop()
    cands = _candidates()
    loop._record_entry_rank(cands[0], cands)
    assert loop._entry_ranks[0] <= 5


def test_ranking_uses_absolute_magnitude():
    """Shorts rank on conviction too — a strong negative is a strong signal."""
    loop = _Loop()
    cands = [_sig("A", -0.02), _sig("B", 0.001), _sig("C", 0.002),
             _sig("D", -0.001), _sig("E", 0.0005), _sig("F", 0.0001)]
    loop._record_entry_rank(cands[0], cands)
    assert loop._entry_ranks[0] >= 80


def test_too_few_candidates_records_nothing():
    """A 3-name cross-section yields a meaningless percentile."""
    loop = _Loop()
    cands = [_sig("A", 0.01), _sig("B", 0.02), _sig("C", 0.03)]
    loop._record_entry_rank(cands[-1], cands)
    assert len(loop._entry_ranks) == 0


def test_recorder_never_raises_on_bad_input():
    """It runs inside the fill path — it must never be able to break an entry."""
    loop = _Loop()
    bad = [_sig("A", None), _sig("B", "x"), _sig("C", 0.01)]
    loop._record_entry_rank(_sig("A", None), bad)      # must not raise
    loop._record_entry_rank(_sig("Z", 0.01), [])       # must not raise


def test_window_is_bounded():
    loop = _Loop()
    cands = _candidates()
    for _ in range(200):
        loop._record_entry_rank(cands[-1], cands)
    assert len(loop._entry_ranks) == 60


@pytest.mark.parametrize("pick,healthy", [(-1, True), (10, False), (0, False)])
def test_healthy_threshold_discriminates(pick, healthy):
    loop = _Loop()
    cands = _candidates()
    loop._record_entry_rank(cands[pick], cands)
    got = loop._entry_ranks[0] >= ENTRY_RANK_HEALTHY_PCTILE
    assert got is healthy
