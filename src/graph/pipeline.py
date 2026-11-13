"""Graph topologies → daily cross-sectional signals.

``build_feature_store`` is the expensive step (indicators + HMM + predictor)
and is done once. ``build_signals`` is cheap and is called per candidate
strategy during search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.graph.analysts import score_all
from src.graph.catalog import StrategySpec
from src.graph.debate import risk_committee, run_debate, trader_plan
from src.graph.indicators import attach_quant_overlay, compute_daily_indicators
from src.graph.state import Action, GraphState


@dataclass(frozen=True)
class GraphTopology:
    """Which analyst / debate / risk nodes a topology enables."""

    name: str
    analysts: tuple[str, ...]
    debate: bool
    committee: bool


TOPOLOGIES: dict[str, GraphTopology] = {
    "primo_linear": GraphTopology("primo_linear", ("technical", "news"), False, False),
    "trading_agents": GraphTopology(
        "trading_agents", ("technical", "fundamental", "sentiment", "news"), True, False
    ),
    "nexus_trade": GraphTopology(
        "nexus_trade", ("technical", "fundamental", "sentiment", "news"), True, True
    ),
    "quant_swing": GraphTopology("quant_swing", ("technical", "quant", "macro"), False, False),
    "alpaca_five": GraphTopology(
        "alpaca_five",
        ("technical", "sentiment", "news", "fundamental", "macro"),
        True,
        False,
    ),
    "hybrid": GraphTopology(
        "hybrid",
        ("technical", "fundamental", "sentiment", "news", "macro", "quant"),
        True,
        True,
    ),
    "buy_hold": GraphTopology("buy_hold", (), False, False),
}


def build_feature_store(
    bars: dict[str, pd.DataFrame],
    spy: pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute causal daily features for every ticker. Shared across strategies."""
    spy_close = spy if spy is not None else (bars["SPY"]["close"] if "SPY" in bars else None)
    store: dict[str, pd.DataFrame] = {}
    for ticker, df in bars.items():
        feat = compute_daily_indicators(df, shift=True)
        feat = attach_quant_overlay(feat, spy_close)
        store[ticker] = feat
    return store


_PANEL_CACHE: dict[int, pd.DataFrame] = {}


def build_analyst_panel(store: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Score every analyst once per bar. Shared across all strategy candidates."""
    key = id(store)
    cached = _PANEL_CACHE.get(key)
    if cached is not None:
        return cached
    frames: list[pd.DataFrame] = []
    for ticker, df in store.items():
        if ticker == "SPY":
            continue
        use = df.iloc[60:] if len(df) > 80 else df
        if "rsi_14" in use.columns:
            use = use[np.isfinite(use["rsi_14"].to_numpy())]
        if use.empty:
            continue
        recs: list[dict] = []
        for ts, row in use.iterrows():
            rec: dict = {
                "time": ts,
                "ticker": ticker,
                "close": float(row["close"]),
                "atr_pct": float(row.get("atr_pct", 0.015) or 0.015),
                "hmm_regime": int(row.get("hmm_regime", 1) or 1),
                "spy_vs_sma200": float(row.get("spy_vs_sma200", 0.0) or 0.0),
                "spy_rvol_20": float(row.get("spy_rvol_20", 0.15) or 0.15),
            }
            for name, report in score_all(row).items():
                rec[f"{name}_score"] = report.score
                rec[f"{name}_conf"] = report.confidence
            recs.append(rec)
        if recs:
            frames.append(pd.DataFrame(recs))
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _PANEL_CACHE[key] = panel
    return panel


def build_signals(
    store: dict[str, pd.DataFrame],
    spec: StrategySpec,
    *,
    start: str | None = None,
    end: str | None = None,
    panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Daily signals. Analysts are scored once; weights/debate apply in vector form.

    Signal at timestamp t uses features already shifted to t-1 and is filled
    at t's open by the backtester.
    """
    tickers = [t for t in store if t != "SPY"]
    if spec.topology == "buy_hold":
        rows = []
        n = max(len(tickers), 1)
        for ticker in tickers:
            df = store[ticker]
            atr = df["atr_pct"] if "atr_pct" in df.columns else pd.Series(0.02, index=df.index)
            hmm = df["hmm_regime"] if "hmm_regime" in df.columns else pd.Series(1, index=df.index)
            chunk = pd.DataFrame({
                "time": df.index,
                "ticker": ticker,
                "action": "LONG",
                "score": 1.0,
                "confidence": 1.0,
                "size_pct": 1.0 / n,
                "stop_pct": 0.99,
                "target_pct": 9.99,
                "atr_pct": atr.fillna(0.02).to_numpy(),
                "close": df["close"].to_numpy(),
                "hmm_regime": hmm.fillna(1).astype(int).to_numpy(),
            })
            rows.append(chunk)
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        return _window(out, start, end)

    if panel is None:
        panel = build_analyst_panel(store)
    if panel.empty:
        return panel
    df = panel.copy()
    analysts = spec.analysts or TOPOLOGIES[spec.topology].analysts
    topo = TOPOLOGIES.get(spec.topology, TOPOLOGIES["hybrid"])
    use_debate = topo.debate or spec.debate_weight > 0

    score_cols = [f"{n}_score" for n in analysts if f"{n}_score" in df.columns]
    conf_cols = [f"{n}_conf" for n in analysts if f"{n}_conf" in df.columns]
    if not score_cols:
        return pd.DataFrame()

    weights = np.array([spec.weights.get(n.replace("_score", ""), 0.0) for n in [c[:-6] for c in score_cols]], dtype=float)
    wsum = float(weights.sum()) if float(weights.sum()) > 0 else 1.0
    scores = df[score_cols].to_numpy(dtype=float)
    confs = df[conf_cols].to_numpy(dtype=float) if conf_cols else np.full_like(scores, 0.5)
    confs = np.clip(confs, 0.15, 1.0)
    raw = (scores * confs * weights).sum(axis=1) / wsum

    if use_debate:
        pos = np.clip(scores, 0, None)
        neg = np.clip(-scores, 0, None)
        bull = pos.mean(axis=1)
        bear = neg.mean(axis=1)
        margin = bull - bear
        mean = scores.mean(axis=1, keepdims=True)
        var = ((scores - mean) ** 2).mean(axis=1)
        agreement = np.clip(1.0 - var, 0.0, 1.0)
        judge = np.clip(0.65 * margin + 0.35 * raw, -1.0, 1.0)
        ensemble = (1.0 - spec.debate_weight) * raw + spec.debate_weight * judge
        confidence = np.clip(0.30 + 0.50 * agreement + 0.20 * np.abs(margin), 0.0, 1.0)
    else:
        ensemble = raw
        confidence = confs.mean(axis=1)

    ensemble = np.clip(ensemble, -1.0, 1.0)
    atr = np.clip(df["atr_pct"].to_numpy(dtype=float), 0.005, 0.20)
    stop_pct = np.clip(atr * spec.stop_atr, 0.008, 0.12)
    target_pct = np.clip(atr * spec.tp_atr, 0.012, 0.25)
    long_mask = (ensemble >= spec.entry_threshold) & (confidence >= 0.30)
    if not spec.long_only:
        short_mask = (ensemble <= -spec.entry_threshold) & (confidence >= 0.30)
    else:
        short_mask = np.zeros(len(df), dtype=bool)

    # Ticker-local vetoes (portfolio heat / daily loss applied in the simulator).
    if spec.trend_filter:
        long_mask &= df["spy_vs_sma200"].to_numpy(dtype=float) >= 0
    if spec.hmm_filter:
        long_mask &= df["hmm_regime"].to_numpy(dtype=int) != 2
        short_mask &= df["hmm_regime"].to_numpy(dtype=int) != 2

    vix = df["spy_rvol_20"].to_numpy(dtype=float)
    long_mask &= vix <= 0.35
    short_mask &= vix <= 0.35

    if spec.risk_mode == "conservative":
        long_mask &= df["hmm_regime"].to_numpy(dtype=int) != 2
        long_mask &= df["spy_vs_sma200"].to_numpy(dtype=float) >= -0.03
        long_mask &= confidence >= 0.55
        stop_pct = stop_pct * 0.85
        target_pct = target_pct * 0.85
        size_mult = np.where(df["hmm_regime"].to_numpy(dtype=int) == 2, 0.25, 0.55)
    elif spec.risk_mode == "aggressive":
        long_mask &= confidence >= 0.22
        stop_pct = stop_pct * 1.10
        target_pct = target_pct * 1.20
        size_mult = np.where(df["hmm_regime"].to_numpy(dtype=int) == 2, 0.80, 1.25)
    elif spec.risk_mode == "committee":
        long_mask &= confidence >= 0.38
        hmm = df["hmm_regime"].to_numpy(dtype=int)
        # Median of 1.25 / 0.55 / 1.00, cut in high vol.
        size_mult = np.where(hmm == 2, 0.50, 1.00)
        weak = (confidence < 0.55) & (hmm == 2)
        long_mask &= ~weak
    else:
        long_mask &= confidence >= 0.35
        size_mult = np.where(df["hmm_regime"].to_numpy(dtype=int) == 2, 0.50, 1.00)

    sized = spec.size_pct * np.minimum(1.0, np.abs(ensemble) / max(spec.entry_threshold, 1e-6))
    sized = np.minimum(sized * size_mult, 0.20)
    action = np.where(long_mask, "LONG", np.where(short_mask, "SHORT", "FLAT"))
    sized = np.where(action == "FLAT", 0.0, sized)

    out = pd.DataFrame({
        "time": df["time"].to_numpy(),
        "ticker": df["ticker"].to_numpy(),
        "action": action,
        "score": ensemble,
        "confidence": confidence,
        "size_pct": sized,
        "stop_pct": stop_pct,
        "target_pct": target_pct,
        "atr_pct": atr,
        "close": df["close"].to_numpy(),
        "hmm_regime": df["hmm_regime"].to_numpy(),
    })
    # Drop flats so the simulator scans fewer candidates.
    out = out[out["action"] != "FLAT"]
    return _window(out, start, end)


def _window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if start:
        df = df[df["time"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["time"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    return df.reset_index(drop=True)


def explain_ticker(store: dict[str, pd.DataFrame], ticker: str, spec: StrategySpec, ts: pd.Timestamp) -> GraphState:
    """Full GraphState for one ticker/bar — used in reports / tests."""
    row = store[ticker].loc[ts]
    reports = score_all(row, spec.analysts)
    ensemble, debate = run_debate(reports, spec.weights, debate_weight=spec.debate_weight)
    plan = trader_plan(
        ensemble,
        debate.confidence,
        float(row.get("atr_pct", 0.015) or 0.015),
        entry_threshold=spec.entry_threshold,
        stop_atr=spec.stop_atr,
        tp_atr=spec.tp_atr,
        size_pct=spec.size_pct,
        holding_days=spec.holding_days,
        long_only=spec.long_only,
        close=float(row["close"]),
    )
    risk = risk_committee(
        plan,
        risk_mode=spec.risk_mode if spec.risk_mode != "none" else "balanced",
        hmm_regime=int(row.get("hmm_regime", 1) or 1),
        spy_vs_sma200=float(row.get("spy_vs_sma200", 0.0) or 0.0),
        portfolio_heat=0.0,
        heat_ceiling=spec.heat_ceiling,
        consecutive_losses=0,
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
        vix_proxy=float(row.get("spy_rvol_20", 0.15) or 0.15),
    )
    action: Action = plan.action if risk.decision != "rejected" else "FLAT"
    return GraphState(
        ticker=ticker,
        reports=reports,
        debate=debate,
        plan=plan,
        risk=risk,
        ensemble_score=ensemble,
        final_action=action,
        final_confidence=debate.confidence,
        final_size_pct=risk.size_pct if action != "FLAT" else 0.0,
        stop_pct=risk.stop_pct,
        target_pct=risk.target_pct,
    )
