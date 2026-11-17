"""No feature at bar t may depend on data after t.

Seven of thirty features violated this until 2026-11-12, and it was the most
expensive bug in the project's history: the backtest read IC +0.181 while live
measured +0.025, and five of the seven leaking features were FFSA's top five.
Months of correct work on exits, sizing and ranking were evaluated against a
number inflated by look-ahead.

Two distinct mechanisms, both fixed:

  mtf_*  `df.resample("15min")` labels bins at their LEFT edge, so the bin
         labelled 09:30 closes at 09:44. Forward-filling from the label gave
         the 1m bar at 09:31 a value computed from the 09:44 print.
         (138 of 176 rows leaked on mtf_rsi_15m alone.)

  orb_*  `groupby(date).transform("max")` takes the max over the WHOLE opening
         range, so a bar at 09:40 — inside the entry window — saw 09:41–09:59.

These are fast synthetic checks. The full audit over real bars is
`scripts/point_in_time_audit.py`, which must report 0.000e+00 for every feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import compute_indicators
from src.features.multi_timeframe import _ffill_completed_bins


def _bars(n_sessions: int = 6) -> pd.DataFrame:
    """Deterministic 1m RTH bars on an ET-aligned UTC index."""
    rng = np.random.default_rng(11)
    idx = []
    for d in range(n_sessions):
        start = pd.Timestamp("2026-10-05", tz="UTC") + pd.Timedelta(days=d)
        # 09:30 ET == 13:30 UTC in October
        idx.extend(pd.date_range(start + pd.Timedelta(hours=13, minutes=30),
                                 periods=390, freq="1min"))
    idx = pd.DatetimeIndex(idx)
    close = 100 + np.cumsum(rng.normal(0, 0.05, len(idx)))
    high = close + np.abs(rng.normal(0, 0.05, len(idx)))
    low = close - np.abs(rng.normal(0, 0.05, len(idx)))
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close,
         "volume": rng.uniform(1e4, 1e5, len(idx))}, index=idx)


# ── The invariant, end to end ────────────────────────────────────────────────

@pytest.mark.parametrize("offset", [40, 45, 100, 300])
def test_no_feature_sees_the_future(offset):
    """Features at t must be identical whether or not bars after t exist.

    `offset` picks the scored bar within the final session. 40 and 45 land
    inside / just after the 09:30-10:00 opening range, which is where orb_*
    leaked and where real entries occur (the entry window opens 09:40 ET).
    """
    bars = _bars()
    ts = bars.index[-390 + offset]

    full = compute_indicators(bars, shift=True)
    pit = compute_indicators(bars[bars.index <= ts], shift=True)
    assert ts in full.index and ts in pit.index

    common = [c for c in full.columns if c in pit.columns]
    a, b = full.loc[ts, common], pit.loc[ts, common]
    leaking = []
    for c in common:
        x, y = a[c], b[c]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if abs(x - y) > 1e-9 * max(abs(x), abs(y), 1.0):
            leaking.append(f"{c}: full={x!r} pit={y!r}")
    assert not leaking, (
        f"{len(leaking)} feature(s) used data after {ts}:\n  " + "\n  ".join(leaking))


# ── mtf_*: only CLOSED bins may be visible ───────────────────────────────────

def test_resampled_bins_are_shifted_to_their_close():
    """A bin labelled at its left edge must not be visible from that label."""
    bins = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.DatetimeIndex(["2026-10-05 13:30", "2026-10-05 13:45",
                                "2026-10-05 14:00"], tz="UTC"))
    target = pd.date_range("2026-10-05 13:30", periods=46, freq="1min", tz="UTC")
    out = _ffill_completed_bins(bins, target)

    # Inside the first bin nothing has closed yet.
    assert out.loc["2026-10-05 13:31"] == 0.0
    assert out.loc["2026-10-05 13:44"] == 0.0
    # At 13:45 the 13:30-13:44 bin (value 1.0) has closed — and only that one.
    assert out.loc["2026-10-05 13:45"] == 1.0
    assert out.loc["2026-10-05 13:59"] == 1.0
    assert out.loc["2026-10-05 14:00"] == 2.0


def test_the_old_ffill_would_fail_this():
    """Proof the test has teeth: the pre-fix one-liner leaks here."""
    bins = pd.Series(
        [1.0, 2.0],
        index=pd.DatetimeIndex(["2026-10-05 13:30", "2026-10-05 13:45"], tz="UTC"))
    target = pd.date_range("2026-10-05 13:30", periods=16, freq="1min", tz="UTC")
    old = bins.reindex(target, method="ffill").fillna(0.0)
    assert old.loc["2026-10-05 13:31"] == 1.0, "old behaviour changed; update this test"
    assert _ffill_completed_bins(bins, target).loc["2026-10-05 13:31"] == 0.0


# ── orb_*: running extremes, not whole-window extremes ───────────────────────

def test_orb_high_is_a_running_max_during_the_opening_range():
    """Inside 09:30-10:00 the ORB high may only reflect bars seen so far."""
    bars = _bars()
    last = bars.index[-390:]
    early, later = last[10], last[25]          # both inside the 30-min ORB

    pit_early = compute_indicators(bars[bars.index <= early], shift=True)
    full = compute_indicators(bars, shift=True)
    # orb_dev is the model-facing feature; equality at `early` is the invariant.
    assert abs(full.loc[early, "orb_dev"] - pit_early.loc[early, "orb_dev"]) < 1e-9

    # And the range may only grow as the opening range fills in.
    r_early = full.loc[early, "orb_range_atr"]
    r_later = full.loc[later, "orb_range_atr"]
    if np.isfinite(r_early) and np.isfinite(r_later):
        assert r_later >= r_early - 1e-9


def test_orb_is_unchanged_after_the_opening_range_closes():
    """After 10:00 the ORB is complete, so the fix must not alter behaviour."""
    bars = _bars()
    ts = bars.index[-390 + 120]                # ~11:30 ET, well past the ORB
    full = compute_indicators(bars, shift=True)
    pit = compute_indicators(bars[bars.index <= ts], shift=True)
    for c in ("orb_dev", "orb_range_atr"):
        x, y = full.loc[ts, c], pit.loc[ts, c]
        if np.isfinite(x) and np.isfinite(y):
            assert abs(x - y) < 1e-9
