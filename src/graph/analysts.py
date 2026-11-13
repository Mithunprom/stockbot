"""Deterministic analyst nodes.

Each function maps a feature row (or column of a DataFrame) to a score in
[-1, +1] plus a confidence in [0, 1]. These are the quantitative stand-ins
for the LLM analysts in TradingAgents / NexusTrade / AlpacaTradingAgent /
PrimoAgent so the graph can be backtested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.graph.state import AnalystReport


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(min(hi, max(lo, x)))


def _tanh(x: float, scale: float) -> float:
    if not np.isfinite(x) or scale == 0:
        return 0.0
    return float(np.tanh(x / scale))


def score_technical(row: pd.Series) -> AnalystReport:
    """Market / Technical analyst (TradingAgents + Primo TA node)."""
    findings: list[str] = []
    score = 0.0

    rsi = float(row.get("rsi_14", 50.0) or 50.0)
    if rsi < 30:
        score += 0.35
        findings.append(f"RSI oversold {rsi:.1f}")
    elif rsi < 40:
        score += 0.12
    elif rsi > 70:
        score -= 0.35
        findings.append(f"RSI overbought {rsi:.1f}")
    elif rsi > 60:
        score -= 0.12
    else:
        score += (50.0 - rsi) / 200.0

    macd_h = float(row.get("macd_hist", 0.0) or 0.0)
    score += _tanh(macd_h, 0.5) * 0.25
    if macd_h > 0:
        findings.append("MACD hist > 0")

    ema_stack = float(row.get("ema_stack", 0.0) or 0.0)
    score += _clip(ema_stack) * 0.25

    adx = float(row.get("adx", 20.0) or 20.0)
    if adx < 15:
        score *= 0.65
        findings.append(f"weak ADX {adx:.1f}")
    elif adx > 25:
        findings.append(f"trending ADX {adx:.1f}")

    bb = float(row.get("bb_pct", 0.5) or 0.5)
    if bb < 0.1:
        score += 0.15
    elif bb > 0.9:
        score -= 0.15

    don = float(row.get("donchian_pos", 0.5) or 0.5)
    if don > 0.95:
        score += 0.10  # breakout
        findings.append("Donchian breakout")
    elif don < 0.05:
        score -= 0.10

    ich = float(row.get("ichimoku_cloud_dir", 0.0) or 0.0)
    score += 0.08 * np.sign(ich)

    conf = min(1.0, 0.35 + 0.02 * adx + 0.3 * abs(_clip(score)))
    return AnalystReport("technical", _clip(score), float(conf), tuple(findings))


def score_fundamental(row: pd.Series) -> AnalystReport:
    """Fundamentals analyst — price-path quality, no filing look-ahead.

    Real SEC/Yahoo fundamentals are dated by period-end, not filing date, so
    using them in a daily backtest would leak. We score 52-week location,
    relative trend quality, and drawdown instead.
    """
    findings: list[str] = []
    dist_high = float(row.get("dist_52w_high", 0.0) or 0.0)
    dd = float(row.get("drawdown_from_peak", 0.0) or 0.0)
    vs200 = float(row.get("close_vs_sma_200", 0.0) or 0.0)
    r21 = float(row.get("ret_21", 0.0) or 0.0)
    rvol = float(row.get("rvol_60", 0.2) or 0.2)

    # Quality trend: above 200d, not at a crash low, not a blow-off high.
    score = 0.0
    score += _tanh(vs200, 0.15) * 0.45
    score += _tanh(r21, 0.10) * 0.25
    # Deep drawdowns from 52w high are a yellow flag, not an automatic buy.
    if dist_high < -0.25:
        score -= 0.15
        findings.append("far from 52w high")
    elif dist_high > -0.05:
        score += 0.10
        findings.append("near 52w high")
    if dd < -0.20:
        score -= 0.20
        findings.append(f"drawdown {dd:.1%}")
    # Penalise chaotic names (quality filter).
    if rvol > 0.55:
        score *= 0.7
        findings.append("high 60d vol")
    conf = 0.45 + 0.25 * min(1.0, abs(vs200) / 0.12)
    return AnalystReport("fundamental", _clip(score), float(min(1.0, conf)), tuple(findings))


def score_sentiment(row: pd.Series) -> AnalystReport:
    """Sentiment / social-flow analyst (volume + short-horizon momentum)."""
    findings: list[str] = []
    r5 = float(row.get("ret_5", 0.0) or 0.0)
    vol_z = float(row.get("vol_z", 0.0) or 0.0)
    mfi = float(row.get("mfi", 50.0) or 50.0)
    cmf = float(row.get("cmf", 0.0) or 0.0)
    force = float(row.get("force_index_z", 0.0) or 0.0)

    score = _tanh(r5, 0.06) * 0.40
    score += _tanh(vol_z, 2.0) * np.sign(r5 + 1e-9) * 0.20
    score += _tanh((mfi - 50.0) / 50.0, 1.0) * 0.20
    score += _clip(cmf) * 0.15
    score += _tanh(force, 2.0) * 0.10
    if vol_z > 1.5:
        findings.append(f"volume z={vol_z:.1f}")
    if mfi > 80:
        findings.append("MFI hot")
    elif mfi < 20:
        findings.append("MFI washed out")
    conf = 0.40 + 0.15 * min(2.0, abs(vol_z)) / 2.0
    return AnalystReport("sentiment", _clip(score), float(min(1.0, conf)), tuple(findings))


def score_news(row: pd.Series) -> AnalystReport:
    """News analyst — PrimoAgent's 7 NLP features, price/volume proxied.

    Historical headline APIs are not point-in-time safe. Gaps, range
    expansion, and volume spikes are the public information set a news
    node would have reacted to the next morning.
    """
    gap = float(row.get("gap", 0.0) or 0.0)
    rng = float(row.get("high_low_range", 0.0) or 0.0)
    vol_z = float(row.get("vol_z", 0.0) or 0.0)
    atr = float(row.get("atr_pct", 0.015) or 0.015)
    r1 = float(row.get("ret_1", 0.0) or 0.0)
    r5 = float(row.get("ret_5", 0.0) or 0.0)
    vs20 = float(row.get("close_vs_sma_20", 0.0) or 0.0)

    news_relevance = _clip(abs(vol_z) / 3.0)  # 0..1, later mapped
    sentiment = _clip(gap / 0.03)
    price_impact = _clip((rng / max(atr, 1e-4) - 1.0) / 2.0)
    trend_direction = _clip(vs20 / 0.05)
    earnings_impact = _clip(gap / 0.04) if abs(gap) > 0.02 else 0.0
    investor_confidence = _clip(r5 / 0.08)
    risk_profile_change = _clip(-abs(vol_z) / 4.0)  # more vol → more risk

    # Average of the 7 Primo features, each in [-1, 1].
    feats = (
        np.sign(r1) * news_relevance,
        sentiment,
        np.sign(r1) * abs(price_impact),
        trend_direction,
        earnings_impact,
        investor_confidence,
        risk_profile_change,
    )
    score = float(np.mean(feats))
    conf = 0.35 + 0.25 * min(1.0, abs(vol_z) / 2.0 + abs(gap) / 0.03)
    findings = []
    if abs(gap) > 0.02:
        findings.append(f"gap {gap:.2%}")
    if vol_z > 2:
        findings.append("news-like volume")
    return AnalystReport("news", _clip(score), float(min(1.0, conf)), tuple(findings))


def score_macro(row: pd.Series) -> AnalystReport:
    """Macro analyst (AlpacaTradingAgent's fifth node) — SPY regime."""
    spy_vs = float(row.get("spy_vs_sma200", 0.0) or 0.0)
    spy_vol = float(row.get("spy_rvol_20", 0.15) or 0.15)
    rs = float(row.get("rs_spy_21", 0.0) or 0.0)
    regime = int(row.get("hmm_regime", 1) or 1)

    score = _tanh(spy_vs, 0.08) * 0.55
    score += _tanh(rs, 0.08) * 0.30
    if spy_vol > 0.28:
        score *= 0.6  # risk-off
    if regime == 2:
        score *= 0.5
    findings = []
    if spy_vs > 0:
        findings.append("SPY above 200d")
    else:
        findings.append("SPY below 200d")
    conf = 0.50 if abs(spy_vs) > 0.02 else 0.35
    return AnalystReport("macro", _clip(score), float(conf), tuple(findings))


def score_quant(row: pd.Series) -> AnalystReport:
    """Quant overlay — HMM + sequence predictor + swing geometry."""
    pred = float(row.get("seq_pred", 0.0) or 0.0)
    regime = int(row.get("hmm_regime", 1) or 1)
    atr = float(row.get("atr_pct", 0.015) or 0.015)
    don = float(row.get("donchian_pos", 0.5) or 0.5)
    rsi = float(row.get("rsi_14", 50.0) or 50.0)

    score = _tanh(pred, 0.03) * 0.55
    score += (don - 0.5) * 0.30
    score += (50.0 - rsi) / 200.0  # slight mean-reversion mix
    if regime == 2:
        score *= 0.45
    elif regime == 0:
        score *= 1.10
    findings = [f"seq_pred={pred:.3f}", f"regime={regime}", f"atr={atr:.2%}"]
    conf = min(1.0, 0.40 + 8.0 * abs(pred))
    return AnalystReport("quant", _clip(score), float(conf), tuple(findings))


ANALYST_FUNCS = {
    "technical": score_technical,
    "fundamental": score_fundamental,
    "sentiment": score_sentiment,
    "news": score_news,
    "macro": score_macro,
    "quant": score_quant,
}


def score_all(row: pd.Series, names: tuple[str, ...] | None = None) -> dict[str, AnalystReport]:
    """Run the requested analyst set on a single feature row."""
    use = names or tuple(ANALYST_FUNCS)
    return {n: ANALYST_FUNCS[n](row) for n in use if n in ANALYST_FUNCS}
