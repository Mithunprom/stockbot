"""Unit tests for scripts/feature_ic_audit.py (H44).

All tests use synthetic DataFrames — no archive I/O, no Railway required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.feature_ic_audit import _spearman_ic, _bootstrap_std, compute_ic_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Return a synthetic OHLCV + feature DataFrame."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-01 09:31", periods=n, freq="1min")
    close = 100 + rng.standard_normal(n).cumsum() * 0.1
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.uniform(-0.001, 0.001, n)),
            "high": close * (1 + rng.uniform(0, 0.002, n)),
            "low": close * (1 - rng.uniform(0, 0.002, n)),
            "close": close,
            "volume": rng.integers(1_000, 10_000, n).astype(float),
        },
        index=idx,
    )
    return df


# ---------------------------------------------------------------------------
# _spearman_ic
# ---------------------------------------------------------------------------

def test_spearman_ic_perfect_positive():
    s = pd.Series(range(100), dtype=float)
    assert _spearman_ic(s, s) == pytest.approx(1.0, abs=1e-6)


def test_spearman_ic_perfect_negative():
    s = pd.Series(range(100), dtype=float)
    assert _spearman_ic(s, -s) == pytest.approx(-1.0, abs=1e-6)


def test_spearman_ic_uncorrelated_near_zero():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.standard_normal(1000))
    y = pd.Series(rng.standard_normal(1000))
    assert abs(_spearman_ic(x, y)) < 0.1


def test_spearman_ic_nan_for_small_n():
    x = pd.Series(range(20), dtype=float)
    y = pd.Series(range(20), dtype=float)
    assert np.isnan(_spearman_ic(x, y))


def test_spearman_ic_handles_nans():
    # Pattern produces 2 valid pairs per cycle; need ≥30 so use 20 cycles
    x = pd.Series([1.0, 2.0, np.nan, 4.0] * 20)
    y = pd.Series([1.0, 2.0, 3.0, np.nan] * 20)
    ic = _spearman_ic(x, y)
    # Should compute on the double-valid subset (40 pairs); not raise
    assert not np.isnan(ic)


# ---------------------------------------------------------------------------
# _bootstrap_std
# ---------------------------------------------------------------------------

def test_bootstrap_std_positive():
    rng = np.random.default_rng(42)
    x = pd.Series(rng.standard_normal(200))
    y = x + rng.standard_normal(200) * 0.5
    std = _bootstrap_std(x, y, n_boot=100)
    assert std > 0


def test_bootstrap_std_nan_for_small_n():
    x = pd.Series(range(20), dtype=float)
    y = pd.Series(range(20), dtype=float)
    assert np.isnan(_bootstrap_std(x, y, n_boot=50))


# ---------------------------------------------------------------------------
# compute_ic_table
# ---------------------------------------------------------------------------

def test_compute_ic_table_returns_dataframe():
    df = _make_df(600)
    df["signal_a"] = df["close"].pct_change(1)
    df["signal_b"] = df["volume"] / df["volume"].rolling(20).mean()
    result = compute_ic_table(df, horizons=[15], n_boot=50)
    assert isinstance(result, pd.DataFrame)
    assert "ic_15" in result.columns
    assert "cost_multiple_15" in result.columns
    assert "n_15" in result.columns


def test_compute_ic_table_excludes_ohlcv_cols():
    df = _make_df(600)
    df["my_feature"] = df["close"].pct_change(5)
    result = compute_ic_table(df, horizons=[15], n_boot=50)
    for ohlcv_col in ("open", "high", "low", "close", "volume"):
        assert ohlcv_col not in result.index


def test_compute_ic_table_sorted_by_abs_ic():
    """Rows should be sorted by |ic_{primary_h}| descending."""
    rng = np.random.default_rng(7)
    df = _make_df(800)
    # Create a strong signal (correlated with 15-bar forward return)
    fwd = df["close"].pct_change(15).shift(-15)
    df["strong_signal"] = fwd + rng.standard_normal(len(df)) * 0.001
    df["weak_signal"] = rng.standard_normal(len(df))
    result = compute_ic_table(df, horizons=[15], n_boot=50)
    ics = result["ic_15"].abs().dropna()
    if len(ics) > 1:
        assert ics.iloc[0] >= ics.iloc[-1]


def test_compute_ic_table_requires_close():
    df = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="close"):
        compute_ic_table(df, horizons=[5], n_boot=10)


def test_compute_ic_table_multiple_horizons():
    df = _make_df(600)
    df["feat"] = df["close"].diff(1)
    result = compute_ic_table(df, horizons=[5, 15, 30], n_boot=50)
    for h in (5, 15, 30):
        assert f"ic_{h}" in result.columns
        assert f"cost_multiple_{h}" in result.columns


def test_compute_ic_table_cost_multiple_formula():
    """cost_multiple = |IC| × sigma(h) / cost_bps.  Verify the formula numerically."""
    import scripts.feature_ic_audit as aud
    rng = np.random.default_rng(99)
    df = _make_df(1000)
    fwd = df["close"].pct_change(15).shift(-15)
    df["signal"] = fwd + rng.standard_normal(len(df)) * 0.01
    result = compute_ic_table(df, horizons=[15], n_boot=20)
    row = result.loc["signal"]
    ic = row["ic_15"]
    cost_mult = row["cost_multiple_15"]
    if ic is not None and cost_mult is not None:
        expected = abs(ic) * aud.FORWARD_SIGMA_15BAR_BPS / aud.ROUND_TRIP_COST_BPS
        assert abs(cost_mult - expected) < 0.01


def test_compute_ic_table_does_not_modify_input():
    """compute_ic_table should not leave forward_ret_* columns in the input df."""
    df = _make_df(300)
    df["feat"] = df["close"].diff()
    before_cols = set(df.columns)
    compute_ic_table(df, horizons=[5, 15], n_boot=20)
    assert set(df.columns) == before_cols


def test_gap_pct_is_a_candidate_feature():
    """gap_pct (overnight gap) must appear in the IC table as a testable feature."""
    df = _make_df(600)
    # Simulate gap_pct as it is computed in indicators.py
    df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    result = compute_ic_table(df, horizons=[15], n_boot=20)
    assert "gap_pct" in result.index, "gap_pct should be tested as an alpha candidate"
