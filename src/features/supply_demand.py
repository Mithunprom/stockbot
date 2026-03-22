"""Supply & Demand zone detection — numeric features for ML models.

Based on Course 1 §8.2–§8.11:
  - Supply zone: price where sellers overwhelm buyers (sharp drop from level)
  - Demand zone: price where buyers overwhelm sellers (sharp rally from level)
  - Strongest SD: at least 2 Marubozu candles, immediate reversal, fresh zone

Zone detection algorithm:
  1. Find "impulse moves" — consecutive bars with large bodies in one direction
  2. The origin of an impulse move defines a zone (high/low of the last opposing candle)
  3. Track zone freshness (bars since creation) and touch count
  4. Compute proximity features: how far is current price from nearest zones

Features produced:
  sd_demand_dist_atr  : distance to nearest demand zone / ATR (positive = above zone)
  sd_supply_dist_atr  : distance to nearest supply zone / ATR (positive = below zone)
  sd_in_demand        : 1.0 if price is inside a demand zone
  sd_in_supply        : 1.0 if price is inside a supply zone
  sd_zone_strength    : strength score of nearest relevant zone [0, 1]

Also includes support/resistance features (§8.8):
  sr_level_dist_atr   : distance to nearest S/R level / ATR (signed)
  sr_touch_count      : how many times nearest level was touched

All features use only past data. No lookahead.
Shift-by-1 is handled at the pipeline level (indicators.py), NOT here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Configuration ───────────────────────────────────────────────────────────

_IMPULSE_BODY_THRESH = 0.6       # body/range > 0.6 = Marubozu-like (strong conviction)
_IMPULSE_MIN_BARS = 2            # at least 2 consecutive strong bars to form a zone
_ZONE_MAX_AGE_BARS = 500         # zones older than this are obsolete (§8.11)
_ZONE_MAX_TOUCHES = 3            # zone weakens after 3 touches (§8.11)
_SR_PIVOT_LOOKBACK = 20          # bars to look back for pivot highs/lows
_SR_TOLERANCE_ATR = 0.5          # price within 0.5× ATR of level = "at level"
_MAX_ZONES = 10                  # keep only the N most recent zones per side


def _detect_zones(
    o: pd.Series,
    h: pd.Series,
    l: pd.Series,
    c: pd.Series,
    atr: pd.Series,
) -> tuple[list[dict], list[dict]]:
    """Scan bars to detect supply and demand zones.

    Returns (demand_zones, supply_zones) where each zone is a dict:
        {high, low, bar_idx, strength, touches}
    """
    rng = (h - l).clip(lower=1e-9)
    body = (c - o).abs()
    body_ratio = body / rng
    bull = c > o
    bear = c < o

    demand_zones: list[dict] = []
    supply_zones: list[dict] = []

    n = len(c)
    i = 0

    while i < n - _IMPULSE_MIN_BARS:
        # Look for bullish impulse (demand zone at origin)
        if bull.iloc[i] and body_ratio.iloc[i] > _IMPULSE_BODY_THRESH:
            j = i + 1
            while j < n and bull.iloc[j] and body_ratio.iloc[j] > _IMPULSE_BODY_THRESH:
                j += 1
            impulse_len = j - i
            if impulse_len >= _IMPULSE_MIN_BARS:
                # Demand zone: from the low of the bar before impulse to the open of first impulse bar
                zone_low = l.iloc[max(0, i - 1):i + 1].min()
                zone_high = o.iloc[i]
                if zone_high < zone_low:
                    zone_high, zone_low = zone_low, zone_high
                strength = min(1.0, impulse_len / 5.0) * min(1.0, body_ratio.iloc[i:j].mean() / 0.8)
                demand_zones.append({
                    "high": zone_high, "low": zone_low,
                    "bar_idx": i, "strength": strength, "touches": 0,
                })
            i = j
            continue

        # Look for bearish impulse (supply zone at origin)
        if bear.iloc[i] and body_ratio.iloc[i] > _IMPULSE_BODY_THRESH:
            j = i + 1
            while j < n and bear.iloc[j] and body_ratio.iloc[j] > _IMPULSE_BODY_THRESH:
                j += 1
            impulse_len = j - i
            if impulse_len >= _IMPULSE_MIN_BARS:
                # Supply zone: from the high of the bar before impulse to the open of first impulse bar
                zone_high = h.iloc[max(0, i - 1):i + 1].max()
                zone_low = o.iloc[i]
                if zone_high < zone_low:
                    zone_high, zone_low = zone_low, zone_high
                strength = min(1.0, impulse_len / 5.0) * min(1.0, body_ratio.iloc[i:j].mean() / 0.8)
                supply_zones.append({
                    "high": zone_high, "low": zone_low,
                    "bar_idx": i, "strength": strength, "touches": 0,
                })
            i = j
            continue

        i += 1

    # Keep only the most recent zones
    demand_zones = demand_zones[-_MAX_ZONES:]
    supply_zones = supply_zones[-_MAX_ZONES:]

    return demand_zones, supply_zones


def _compute_zone_features(
    c: pd.Series,
    atr: pd.Series,
    demand_zones: list[dict],
    supply_zones: list[dict],
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute per-bar features from detected zones.

    Returns:
        sd_demand_dist_atr  — distance to nearest demand zone / ATR
        sd_supply_dist_atr  — distance to nearest supply zone / ATR
        sd_in_demand        — 1.0 if price is inside a demand zone
        sd_in_supply        — 1.0 if price is inside a supply zone
        sd_zone_strength    — strength of nearest active zone
    """
    n = len(c)
    demand_dist = pd.Series(np.nan, index=c.index)
    supply_dist = pd.Series(np.nan, index=c.index)
    in_demand = pd.Series(0.0, index=c.index)
    in_supply = pd.Series(0.0, index=c.index)
    zone_str = pd.Series(0.0, index=c.index)

    for i in range(n):
        price = c.iloc[i]
        atr_val = atr.iloc[i] if np.isfinite(atr.iloc[i]) else 1e-9

        # Find nearest demand zone (only zones created before this bar)
        best_d_dist = np.inf
        best_d_str = 0.0
        for z in demand_zones:
            if z["bar_idx"] >= i:
                continue
            age = i - z["bar_idx"]
            if age > _ZONE_MAX_AGE_BARS:
                continue
            mid = (z["high"] + z["low"]) / 2
            dist = (price - mid) / max(atr_val, 1e-9)
            if abs(dist) < abs(best_d_dist):
                best_d_dist = dist
                freshness = max(0, 1.0 - age / _ZONE_MAX_AGE_BARS)
                best_d_str = z["strength"] * freshness
            if z["low"] <= price <= z["high"]:
                in_demand.iloc[i] = 1.0

        # Find nearest supply zone
        best_s_dist = np.inf
        best_s_str = 0.0
        for z in supply_zones:
            if z["bar_idx"] >= i:
                continue
            age = i - z["bar_idx"]
            if age > _ZONE_MAX_AGE_BARS:
                continue
            mid = (z["high"] + z["low"]) / 2
            dist = (mid - price) / max(atr_val, 1e-9)
            if abs(dist) < abs(best_s_dist):
                best_s_dist = dist
                freshness = max(0, 1.0 - age / _ZONE_MAX_AGE_BARS)
                best_s_str = z["strength"] * freshness
            if z["low"] <= price <= z["high"]:
                in_supply.iloc[i] = 1.0

        if np.isfinite(best_d_dist):
            demand_dist.iloc[i] = best_d_dist
        if np.isfinite(best_s_dist):
            supply_dist.iloc[i] = best_s_dist

        # Zone strength = max of nearest demand and supply strength
        zone_str.iloc[i] = max(best_d_str, best_s_str)

    return demand_dist, supply_dist, in_demand, in_supply, zone_str


# ─── Support / Resistance (§8.8) ────────────────────────────────────────────

def _pivot_sr_levels(
    h: pd.Series,
    l: pd.Series,
    c: pd.Series,
    lookback: int = _SR_PIVOT_LOOKBACK,
) -> tuple[pd.Series, pd.Series]:
    """Compute distance to nearest rolling support/resistance levels.

    Support: rolling min of lows (price floor).
    Resistance: rolling max of highs (price ceiling).

    Returns (sr_support_dist, sr_resist_dist) as fraction of price.
    """
    support = l.rolling(lookback, min_periods=5).min()
    resistance = h.rolling(lookback, min_periods=5).max()

    sr_support_dist = (c - support) / c.clip(lower=1e-9)
    sr_resist_dist = (resistance - c) / c.clip(lower=1e-9)

    return sr_support_dist, sr_resist_dist


# ─── Public API ──────────────────────────────────────────────────────────────

def compute_supply_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add supply/demand and support/resistance features.

    Requires: open, high, low, close, atr_14 columns.
    Adds: sd_demand_dist_atr, sd_supply_dist_atr, sd_in_demand,
          sd_in_supply, sd_zone_strength, sr_support_dist, sr_resist_dist.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    atr = df["atr_14"] if "atr_14" in df.columns else (h - l).rolling(14).mean()

    # Detect zones
    demand_zones, supply_zones = _detect_zones(o, h, l, c, atr)

    # Compute zone proximity features
    dd, sd, ind, ins, zs = _compute_zone_features(c, atr, demand_zones, supply_zones)
    df["sd_demand_dist_atr"] = dd
    df["sd_supply_dist_atr"] = sd
    df["sd_in_demand"] = ind
    df["sd_in_supply"] = ins
    df["sd_zone_strength"] = zs

    # Support / Resistance
    df["sr_support_dist"], df["sr_resist_dist"] = _pivot_sr_levels(h, l, c)

    return df
