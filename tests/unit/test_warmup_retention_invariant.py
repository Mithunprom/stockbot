"""WARMUP_BARS must never sit at or above the ohlcv_1m retention ceiling.

History
-------
2026-10-27 (sim_live_ic_gap.py): serving IC was measured at −0.010 against
+0.181 for full-history features. Rank agreement 0.64. Root cause: WARMUP_BARS
was 1950, which equals exactly 5 RTH sessions — also the ohlcv_1m retention
window at the time (7 calendar days ≈ 5 trading sessions). Every warmup started
from a cold EWM with zero prior history, which left mtf_trend_60m carrying ~5%
initial-condition contamination from its first bar ((1 − 2/22)^32 ≈ 0.053).

The fix (H36): ohlcv_1m retention restored to 14 days; WARMUP_BARS doubled to
3900. This brings the 60m contamination to ≈ 0.3% and (expected) pushes rank
agreement toward 0.85–0.90.

The tests here lock in the invariant so a future retention cut cannot silently
recreate the same bug. They are intentionally forward-looking: they assert
PROPERTIES of the constants, not their literal values, so they survive a
justified change to either constant as long as the relationship holds.

Design note
-----------
These tests do NOT import src.* modules to avoid pulling in numpy/pandas/torch.
They read the literal constant values from the source file text (AST-safe grep)
so they can run in a minimal environment — the same reason test_kelly_resume_ramp
is the only test that imports signal_loop: that test is worth the deps; these
tests should work in CI without them.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path


# ── Extract constants from source without full imports ───────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent  # stockbot/


def _read_constant_int(filepath: Path, name: str) -> int:
    """Extract a top-level integer constant from a Python source file."""
    source = filepath.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            return node.value.value
    raise ValueError(f"{name} not found as integer constant in {filepath}")


def _read_ohlcv1m_retention_days() -> int:
    """Extract ohlcv_1m retention days from _RETENTION_POLICIES in db.py."""
    source = (ROOT / "src" / "data" / "db.py").read_text()
    # Find the tuple ("ohlcv_1m", "time", N)
    match = re.search(r'\("ohlcv_1m",\s*"time",\s*(\d+)\)', source)
    if not match:
        raise ValueError("ohlcv_1m retention policy not found in db.py")
    return int(match.group(1))


def _ema_contamination(span: int, n_periods: int) -> float:
    """Fraction of the initial-condition value remaining after `n_periods`.

    alpha = 2/(span+1); after n periods weight of initial value = (1-alpha)^n.
    """
    alpha = 2.0 / (span + 1)
    return (1.0 - alpha) ** n_periods


SESSIONS_PER_WEEK = 5
BARS_PER_SESSION = 390
CALENDAR_DAYS_PER_WEEK = 7
MARGIN = 0.90  # WARMUP_BARS must not exceed 90% of available bars at retention


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_warmup_bars_fits_within_retention_window():
    """WARMUP_BARS ≤ 90% of the bars ohlcv_1m can actually hold.

    Fail means WARMUP_BARS is at or near the retention ceiling — the same
    condition that produced IC = −0.010 in Oct 2026. Fix: raise ohlcv_1m
    retention or lower WARMUP_BARS. The 10% margin covers tickers with
    slightly fewer stored bars (weekend gaps, late-arriving data).
    """
    warmup = _read_constant_int(ROOT / "src" / "features" / "live.py", "WARMUP_BARS")
    retention_days = _read_ohlcv1m_retention_days()
    sessions = (retention_days / CALENDAR_DAYS_PER_WEEK) * SESSIONS_PER_WEEK
    ceiling = int(sessions * BARS_PER_SESSION * MARGIN)

    assert warmup <= ceiling, (
        f"WARMUP_BARS={warmup} exceeds {MARGIN:.0%} of the ohlcv_1m retention "
        f"ceiling ({retention_days}d → {ceiling} bars). Raise ohlcv_1m retention "
        "or lower WARMUP_BARS. See reports/research/sim_live_ic_gap.py."
    )


def test_warmup_bars_above_minimum_for_ema50():
    """WARMUP_BARS gives < 1% contamination for the longest standalone indicator (EMA-50)."""
    warmup = _read_constant_int(ROOT / "src" / "features" / "live.py", "WARMUP_BARS")
    contamination = _ema_contamination(span=50, n_periods=warmup)
    assert contamination < 0.01, (
        f"EMA(50) contamination = {contamination:.4%} at WARMUP_BARS={warmup}. "
        "Increase WARMUP_BARS."
    )


def test_old_warmup_bars_had_insufficient_mtf60m_convergence():
    """Regression anchor: 1950 bars left > 3% EMA(21)/60m contamination.

    This test PASSES (asserts the problem existed), confirming the math matches
    the Oct 2026 measurement. If it starts failing, check the formula.
    """
    old_warmup = 1950
    n_60m = old_warmup // 60
    contamination = _ema_contamination(span=21, n_periods=n_60m)
    assert contamination > 0.03, (
        f"Old warmup (1950 → {n_60m} 60m periods) shows {contamination:.4%} "
        "contamination; expected > 3%. Check formula."
    )


def test_new_warmup_bars_achieves_sub_half_pct_mtf60m_contamination():
    """Current WARMUP_BARS gives < 0.5% contamination for mtf_trend_60m EMA(21)."""
    warmup = _read_constant_int(ROOT / "src" / "features" / "live.py", "WARMUP_BARS")
    n_60m = warmup // 60
    contamination = _ema_contamination(span=21, n_periods=n_60m)
    assert contamination < 0.005, (
        f"mtf_trend_60m contamination = {contamination:.4%} at "
        f"WARMUP_BARS={warmup} ({n_60m} 60m periods). Expected < 0.5%."
    )


def test_ohlcv1m_retention_at_least_14_days():
    """ohlcv_1m retention ≥ 14 days after the H36 restore."""
    days = _read_ohlcv1m_retention_days()
    assert days >= 14, (
        f"ohlcv_1m retention={days}d is below the 14-day minimum restored by H36. "
        "A retention cut to < 14d will put WARMUP_BARS back at the ceiling."
    )
