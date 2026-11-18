"""Unit tests for scripts/horizon_ic_clean.py — no data dependency."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.horizon_ic_clean import (
    COST_BPS,
    MIN_N,
    _spearman_ic,
    compute_forward_returns,
    run_ic_curve,
)


class TestSpearmanIC:
    def test_perfect_positive_correlation(self) -> None:
        x = np.arange(100, dtype=float)
        assert abs(_spearman_ic(x, x) - 1.0) < 1e-6

    def test_perfect_negative_correlation(self) -> None:
        x = np.arange(100, dtype=float)
        assert abs(_spearman_ic(x, -x) + 1.0) < 1e-6

    def test_nan_pairs_ignored(self) -> None:
        # pattern has 2 valid pairs per repeat of 4; need >= 30 valid pairs → 20 repeats
        x = np.array([1.0, 2.0, np.nan, 4.0] * 20)
        y = np.array([1.0, 2.0, 3.0, np.nan] * 20)
        # Only positions where BOTH are finite count (40 valid pairs)
        ic = _spearman_ic(x, y)
        assert np.isfinite(ic)

    def test_insufficient_data_returns_nan(self) -> None:
        x = np.arange(10, dtype=float)
        assert np.isnan(_spearman_ic(x, x))

    def test_uncorrelated_noise(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(size=10_000)
        y = rng.normal(size=10_000)
        ic = _spearman_ic(x, y)
        assert abs(ic) < 0.05  # noise IC should be near zero


class TestComputeForwardReturns:
    def _make_bars(self, n: int = 50) -> dict[str, pd.DataFrame]:
        idx = pd.date_range("2026-10-01", periods=n, freq="1min", tz="UTC")
        return {
            "AAPL": pd.DataFrame({"close": np.linspace(100, 110, n)}, index=idx),
            "MSFT": pd.DataFrame({"close": np.linspace(200, 190, n)}, index=idx),
        }

    def test_columns_exist_for_each_horizon(self) -> None:
        bars = self._make_bars()
        df = compute_forward_returns(bars, [5, 15])
        assert "fwd_5" in df.columns
        assert "fwd_15" in df.columns

    def test_fwd_1_matches_simple_return(self) -> None:
        """fwd_1 at bar t = close[t+1]/close[t] - 1."""
        bars = {
            "AAPL": pd.DataFrame(
                {"close": [100.0, 102.0, 101.0]},
                index=pd.date_range("2026-10-01", periods=3, freq="1min", tz="UTC"),
            )
        }
        df = compute_forward_returns(bars, [1])
        aapl = df[df["ticker"] == "AAPL"].set_index("bar_time")
        t0 = pd.Timestamp("2026-10-01 00:00", tz="UTC")
        assert abs(aapl.loc[t0, "fwd_1"] - 0.02) < 1e-9

    def test_last_bar_is_nan_at_horizon(self) -> None:
        bars = self._make_bars(n=20)
        df = compute_forward_returns(bars, [5])
        last_bars = df[df["ticker"] == "AAPL"].nlargest(5, "bar_time")
        # Last 5 bars should all be NaN for fwd_5
        assert last_bars["fwd_5"].isna().all()


class TestRunICCurve:
    def _make_pred_and_fwd(
        self, ic_target: float = 0.05, n: int = 15_000
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic pred/fwd DataFrames with controlled IC."""
        rng = np.random.default_rng(99)
        tickers = ["A", "B", "C", "D", "E"]
        idx = pd.date_range("2026-10-01", periods=n // len(tickers), freq="1min", tz="UTC")
        rows_pred, rows_fwd = [], []
        for t in tickers:
            noise = rng.normal(size=len(idx))
            signal = rng.normal(size=len(idx))
            # fwd = ic_target * signal + sqrt(1 - ic^2) * noise (rank approx)
            fwd = ic_target * signal + np.sqrt(max(1 - ic_target**2, 0)) * noise
            rows_pred.append(
                pd.DataFrame({"ticker": t, "bar_time": idx, "pred_return": signal})
            )
            rows_fwd.append(
                pd.DataFrame({"ticker": t, "bar_time": idx, "fwd_15": fwd})
            )
        return pd.concat(rows_pred, ignore_index=True), pd.concat(rows_fwd, ignore_index=True)

    def test_viable_when_edge_exceeds_cost(self) -> None:
        # Manufacture IC ≈ 0.5, sigma ≈ 60bps → edge ≈ 30bps >> 2bps
        pred_df, fwd_df = self._make_pred_and_fwd(ic_target=0.5)
        # Inflate fwd to give large sigma
        fwd_df["fwd_15"] *= 0.006  # 0.6% per unit → ~60bps sigma
        oos_start = pd.Timestamp("2026-10-01", tz="UTC")
        oos_end = pd.Timestamp("2027-01-01", tz="UTC")
        result = run_ic_curve(pred_df, fwd_df, [15], oos_start, oos_end)
        assert result[15]["verdict"] == "VIABLE"

    def test_rejected_when_ic_near_zero(self) -> None:
        pred_df, fwd_df = self._make_pred_and_fwd(ic_target=0.0078)
        fwd_df["fwd_15"] *= 0.0059  # realistic sigma
        oos_start = pd.Timestamp("2026-10-01", tz="UTC")
        oos_end = pd.Timestamp("2027-01-01", tz="UTC")
        result = run_ic_curve(pred_df, fwd_df, [15], oos_start, oos_end)
        assert result[15]["verdict"] == "REJECTED"

    def test_insufficient_data_marked_correctly(self) -> None:
        pred_df = pd.DataFrame({
            "ticker": ["A"] * 100,
            "bar_time": pd.date_range("2026-10-01", periods=100, freq="1min", tz="UTC"),
            "pred_return": np.linspace(0, 1, 100),
        })
        fwd_df = pred_df.copy().rename(columns={"pred_return": "fwd_15"})
        oos_start = pd.Timestamp("2026-10-01", tz="UTC")
        oos_end = pd.Timestamp("2027-01-01", tz="UTC")
        result = run_ic_curve(pred_df, fwd_df, [15], oos_start, oos_end)
        assert result[15]["verdict"] == "INSUFFICIENT_DATA"

    def test_breakeven_ic_field_populated(self) -> None:
        pred_df, fwd_df = self._make_pred_and_fwd(ic_target=0.1)
        fwd_df["fwd_15"] *= 0.006
        oos_start = pd.Timestamp("2026-10-01", tz="UTC")
        oos_end = pd.Timestamp("2027-01-01", tz="UTC")
        result = run_ic_curve(pred_df, fwd_df, [15], oos_start, oos_end)
        assert "breakeven_ic" in result[15]
        assert result[15]["breakeven_ic"] > 0

    def test_cost_constant_matches_module_constant(self) -> None:
        pred_df, fwd_df = self._make_pred_and_fwd(ic_target=0.1)
        fwd_df["fwd_15"] *= 0.006
        oos_start = pd.Timestamp("2026-10-01", tz="UTC")
        oos_end = pd.Timestamp("2027-01-01", tz="UTC")
        result = run_ic_curve(pred_df, fwd_df, [15], oos_start, oos_end)
        assert result[15]["cost_bps"] == COST_BPS

    def test_multiple_horizons_returned(self) -> None:
        pred_df, fwd_df = self._make_pred_and_fwd(ic_target=0.1)
        fwd_df = fwd_df.rename(columns={"fwd_15": "fwd_15"})
        fwd_df["fwd_30"] = fwd_df["fwd_15"] * 1.4
        fwd_df["fwd_60"] = fwd_df["fwd_15"] * 2.0
        oos_start = pd.Timestamp("2026-10-01", tz="UTC")
        oos_end = pd.Timestamp("2027-01-01", tz="UTC")
        result = run_ic_curve(pred_df, fwd_df, [15, 30, 60], oos_start, oos_end)
        assert set(result.keys()) == {15, 30, 60}
