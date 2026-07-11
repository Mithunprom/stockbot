"""Unit tests for market_regime — H2: SPY 20-day MA overlay.

Tests cover:
  - _fetch_spy_ma20(): happy path, SPY above/below MA, insufficient bars, error
  - MarketRegimeSnapshot: field defaults, to_dict() completeness
  - _classify_regime(): existing VIX + momentum logic
  - market_regime_scalar value correctness
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

import pandas as pd

from src.data.market_regime import (
    MarketRegimeSnapshot,
    MA20_BELOW_SCALAR,
    MA20_LOOKBACK_DAYS,
    MA20_MIN_BARS,
    _fetch_spy_ma20,
    _classify_regime,
    get_market_regime,
)


# ─── Helper: build a fake yfinance DataFrame ────────────────────────────────

def _make_spy_df(closes: list[float]) -> pd.DataFrame:
    """Return a minimal yfinance-style DataFrame with a Close column."""
    return pd.DataFrame({"Close": closes})


# ─── _fetch_spy_ma20 tests ─────────────────────────────────────────────────
# yfinance is imported INSIDE _fetch_spy_ma20, so we patch yfinance.download
# directly rather than src.data.market_regime.yf.

def test_spy_above_20d_ma_returns_full_scalar():
    """When SPY close > 20d MA, scalar must be 1.0 and above_20d_ma True.

    21 bars: tail(20) = [100.0]*19 + [110.0], MA20 = (19*100+110)/20 = 100.5.
    110.0 > 100.5, so above_20d_ma=True.
    """
    closes = [100.0] * 20 + [110.0]
    df = _make_spy_df(closes)

    with patch("yfinance.download", return_value=df):
        result = _fetch_spy_ma20()

    assert result["spy_above_20d_ma"] is True
    assert result["market_regime_scalar"] == 1.0
    assert result["spy_price"] == pytest.approx(110.0)
    assert result["spy_ma20"] == pytest.approx(100.5)  # tail(20) includes the 110 bar


def test_spy_below_20d_ma_returns_half_scalar():
    """When SPY close < 20d MA, scalar must equal MA20_BELOW_SCALAR (0.5).

    21 bars: tail(20) = [100.0]*19 + [85.0], MA20 = (19*100+85)/20 = 99.25.
    85.0 < 99.25, so above_20d_ma=False.
    """
    closes = [100.0] * 20 + [85.0]
    df = _make_spy_df(closes)

    with patch("yfinance.download", return_value=df):
        result = _fetch_spy_ma20()

    assert result["spy_above_20d_ma"] is False
    assert result["market_regime_scalar"] == pytest.approx(MA20_BELOW_SCALAR)
    assert result["spy_price"] == pytest.approx(85.0)
    assert result["spy_ma20"] == pytest.approx(99.25)  # tail(20) includes the 85 bar


def test_spy_exactly_at_ma_treated_as_below():
    """SPY == MA is treated as NOT above (strictly greater check)."""
    closes = [100.0] * MA20_MIN_BARS
    df = _make_spy_df(closes)

    with patch("yfinance.download", return_value=df):
        result = _fetch_spy_ma20()

    assert result["spy_above_20d_ma"] is False
    assert result["market_regime_scalar"] == pytest.approx(MA20_BELOW_SCALAR)


def test_insufficient_bars_returns_defaults():
    """Fewer than MA20_MIN_BARS daily bars → defaults (fail-open, scalar=1.0)."""
    closes = [100.0] * (MA20_MIN_BARS - 1)
    df = _make_spy_df(closes)

    with patch("yfinance.download", return_value=df):
        result = _fetch_spy_ma20()

    assert result["spy_above_20d_ma"] is True
    assert result["market_regime_scalar"] == 1.0
    assert result["spy_price"] == 0.0


def test_empty_dataframe_returns_defaults():
    """Empty yfinance response → defaults (fail-open)."""
    with patch("yfinance.download", return_value=pd.DataFrame()):
        result = _fetch_spy_ma20()

    assert result["spy_above_20d_ma"] is True
    assert result["market_regime_scalar"] == 1.0


def test_yfinance_exception_returns_defaults():
    """yfinance error → defaults (fail-open, never blocks entries)."""
    with patch("yfinance.download", side_effect=RuntimeError("network error")):
        result = _fetch_spy_ma20()

    assert result["spy_above_20d_ma"] is True
    assert result["market_regime_scalar"] == 1.0
    assert result["spy_price"] == 0.0


def test_ma20_uses_last_20_bars_only():
    """The MA20 is computed from exactly the last 20 closes, not all bars.

    First 10 bars at 200 (old data), next 20 at 100, final bar at 105.
    tail(20) = [100.0]*19 + [105.0], MA20 = (19*100+105)/20 = 100.25.
    All-bars mean ≈ 132.4 — confirming old bars are excluded.
    """
    closes = [200.0] * 10 + [100.0] * 20 + [105.0]
    df = _make_spy_df(closes)

    with patch("yfinance.download", return_value=df):
        result = _fetch_spy_ma20()

    # MA20 is 100.25 (last 20 bars only), not ~132 (all 31 bars)
    assert result["spy_ma20"] == pytest.approx(100.25)
    assert result["spy_price"] == pytest.approx(105.0)
    assert result["spy_above_20d_ma"] is True


# ─── MarketRegimeSnapshot tests ──────────────────────────────────────────────

def test_snapshot_defaults_to_full_size():
    """Default snapshot has spy_above_20d_ma=True and scalar=1.0 (fail-open)."""
    snap = MarketRegimeSnapshot()
    assert snap.spy_above_20d_ma is True
    assert snap.market_regime_scalar == 1.0
    assert snap.spy_price == 0.0
    assert snap.spy_ma20 == 0.0


def test_snapshot_below_ma_fields():
    """Snapshot correctly stores below-MA state."""
    snap = MarketRegimeSnapshot(
        spy_price=450.0,
        spy_ma20=500.0,
        spy_above_20d_ma=False,
        market_regime_scalar=MA20_BELOW_SCALAR,
    )
    assert snap.spy_above_20d_ma is False
    assert snap.market_regime_scalar == pytest.approx(MA20_BELOW_SCALAR)


def test_to_dict_contains_ma_fields():
    """to_dict() must include all H2 overlay fields."""
    snap = MarketRegimeSnapshot(
        spy_price=500.0,
        spy_ma20=480.0,
        spy_above_20d_ma=True,
        market_regime_scalar=1.0,
    )
    d = snap.to_dict()
    assert "spy_price" in d
    assert "spy_ma20" in d
    assert "spy_above_20d_ma" in d
    assert "market_regime_scalar" in d
    assert d["spy_above_20d_ma"] is True
    assert d["market_regime_scalar"] == 1.0
    assert d["spy_price"] == pytest.approx(500.0)
    assert d["spy_ma20"] == pytest.approx(480.0)


def test_to_dict_below_ma_state():
    """to_dict() correctly serializes the below-MA / half-scalar state."""
    snap = MarketRegimeSnapshot(
        spy_price=450.0,
        spy_ma20=500.0,
        spy_above_20d_ma=False,
        market_regime_scalar=MA20_BELOW_SCALAR,
    )
    d = snap.to_dict()
    assert d["spy_above_20d_ma"] is False
    assert d["market_regime_scalar"] == pytest.approx(MA20_BELOW_SCALAR)


# ─── _classify_regime tests ──────────────────────────────────────────────────

def test_classify_risk_on():
    """Low VIX + strong up-trend → risk_on."""
    regime, score = _classify_regime(
        vix=12.0, vix_pct=0.1, spy_ret_5=0.003, spy_ret_15=0.007, qqq_spread=0.002
    )
    assert regime == "risk_on"
    assert score > 0.20


def test_classify_risk_off():
    """High VIX + strong down-trend → risk_off."""
    regime, score = _classify_regime(
        vix=35.0, vix_pct=0.9, spy_ret_5=-0.005, spy_ret_15=-0.010, qqq_spread=-0.002
    )
    assert regime == "risk_off"
    assert score < -0.20


def test_classify_neutral():
    """Flat market with moderate VIX → neutral."""
    regime, score = _classify_regime(
        vix=20.0, vix_pct=0.5, spy_ret_5=0.0, spy_ret_15=0.0, qqq_spread=0.0
    )
    assert regime == "neutral"
    assert -0.20 <= score <= 0.20


def test_classify_score_clamped():
    """Extreme inputs never produce score outside [-1, 1]."""
    _, score_high = _classify_regime(
        vix=5.0, vix_pct=0.0, spy_ret_5=0.1, spy_ret_15=0.5, qqq_spread=0.1
    )
    _, score_low = _classify_regime(
        vix=80.0, vix_pct=1.0, spy_ret_5=-0.1, spy_ret_15=-0.5, qqq_spread=-0.1
    )
    assert -1.0 <= score_high <= 1.0
    assert -1.0 <= score_low <= 1.0


# ─── get_market_regime tests ─────────────────────────────────────────────────

def test_get_market_regime_returns_snapshot():
    """get_market_regime() returns a MarketRegimeSnapshot instance."""
    snap = get_market_regime()
    assert isinstance(snap, MarketRegimeSnapshot)


def test_get_market_regime_default_state():
    """Initial snapshot defaults to full-size (spy_above_20d_ma=True)."""
    snap = get_market_regime()
    assert snap.spy_above_20d_ma is True
    assert snap.market_regime_scalar == 1.0


# ─── MA20_BELOW_SCALAR constant validation ───────────────────────────────────

def test_ma20_below_scalar_is_half():
    """The below-MA scalar must be 0.5 — anything else changes the hypothesis."""
    assert MA20_BELOW_SCALAR == pytest.approx(0.5)


def test_ma20_lookback_is_20():
    """20-day MA must use exactly 20 bars."""
    assert MA20_LOOKBACK_DAYS == 20
    assert MA20_MIN_BARS == MA20_LOOKBACK_DAYS + 1
