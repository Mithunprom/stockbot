"""Train/serve parity: the live feature path must reproduce the training path.

Production computes features from only the trailing `live.WARMUP_BARS` bars,
while training and the backtest compute them over full history. Both call the
same compute_indicators(), so any disagreement comes from a feature whose value
depends on WHERE THE SERIES STARTS rather than on the bar being scored.

That class of bug is invisible in ordinary unit tests — every feature computes
"successfully" on both paths, just to different numbers — and it is expensive:
on 2026-09-16 it put live predictions at a Spearman rank agreement of 0.36 with
the train-consistent ones, which placed production's entries near the median of
the model's own ranking instead of the top decile. See
reports/research/loss_diagnosis_2026-09-16.md.

These tests pin the invariant directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import compute_indicators
from src.features.live import WARMUP_BARS


def _synthetic_bars(n_sessions: int = 12, bars_per_session: int = 390) -> pd.DataFrame:
    """Deterministic multi-session 1m bars on an ET-aligned UTC index."""
    rng = np.random.default_rng(7)
    idx = []
    for d in range(n_sessions):
        day = pd.Timestamp("2026-06-01", tz="UTC") + pd.Timedelta(days=d)
        if day.weekday() >= 5:
            continue
        start = day + pd.Timedelta(hours=13, minutes=30)      # 09:30 ET
        idx.extend(start + pd.Timedelta(minutes=m) for m in range(bars_per_session))
    idx = pd.DatetimeIndex(idx)

    close = 100 + np.cumsum(rng.normal(0, 0.05, len(idx)))
    high = close + np.abs(rng.normal(0, 0.03, len(idx)))
    low = close - np.abs(rng.normal(0, 0.03, len(idx)))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.01, len(idx)),
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1_000, 50_000, len(idx)).astype(float),
            "vwap": close,
        },
        index=idx,
    )


def test_obv_is_anchored_to_the_session_not_the_window():
    """REGRESSION: raw cumulative OBV disagreed on 100% of rows at every warmup.

    `.cumsum()` over the whole series makes OBV a function of how much history
    the caller passed in. Anchoring it to the ET session makes it a function of
    the session only, so any window reaching back to the open reproduces it.
    """
    bars = _synthetic_bars()
    full = compute_indicators(bars, shift=True)

    # a bar late in the final session, scored from a window that starts
    # part-way through an EARLIER session
    i = len(bars) - 3
    ts = bars.index[i]
    window = bars.iloc[i - 1500 : i + 1]
    live = compute_indicators(window, shift=True)

    assert ts in live.index and ts in full.index
    assert full.loc[ts, "obv"] == pytest.approx(live.loc[ts, "obv"], rel=1e-9), (
        "OBV differs between the full-history and windowed paths — it is "
        "anchored to the window instead of the trading session."
    )


def test_obv_stays_bounded_by_one_session_of_volume():
    """OBV must not accumulate across sessions.

    The old `.cumsum()` grew without bound over the whole series, so its scale
    depended on how many days of history happened to be loaded — the property
    that made it unreproducible live. Session-anchoring bounds it by a single
    session's turnover no matter how long the input series is.
    """
    bars = _synthetic_bars(n_sessions=12)
    out = compute_indicators(bars, shift=True)

    et_dates = pd.Series(bars.index.tz_convert("America/New_York").date, index=bars.index)
    largest_session_volume = bars.groupby(et_dates)["volume"].sum().max()
    whole_series_volume = bars["volume"].sum()

    peak = out["obv"].abs().max()
    assert peak <= largest_session_volume, (
        f"OBV peaked at {peak:,.0f}, above one session's total volume "
        f"({largest_session_volume:,.0f}) — it is still accumulating across "
        f"sessions and its scale depends on how much history was loaded."
    )
    # and it must be far below the whole-series total the old version approached
    assert peak < 0.5 * whole_series_volume


@pytest.mark.parametrize("feature", ["obv", "vwap_dev", "atr_14", "macd", "adx"])
def test_live_warmup_reproduces_training_features(feature):
    """Every model feature must be identical on both code paths.

    Parametrised over a representative slice rather than all 30 so a failure
    names the offending feature directly.
    """
    bars = _synthetic_bars()
    full = compute_indicators(bars, shift=True)

    i = len(bars) - 3
    ts = bars.index[i]
    window = bars.iloc[max(0, i - WARMUP_BARS + 1) : i + 1]
    live = compute_indicators(window, shift=True)

    if feature not in full.columns or feature not in live.columns:
        pytest.skip(f"{feature} not produced by this pipeline build")

    a, b = full.loc[ts, feature], live.loc[ts, feature]
    if pd.isna(a) and pd.isna(b):
        return
    assert a == pytest.approx(b, rel=1e-4, abs=1e-9), (
        f"{feature} differs between training and live paths "
        f"(full={a!r} live={b!r}) — the live pipeline is serving the model an "
        f"input distribution it was not trained on."
    )


def test_warmup_is_long_enough_to_cover_a_session_anchor():
    """WARMUP_BARS must span multiple sessions or session-anchored features
    cannot be reconstructed for early-session bars."""
    assert WARMUP_BARS >= 390 * 2, (
        "WARMUP_BARS must cover at least two full sessions so that a bar near "
        "the open still sees its own session's start."
    )
