"""Research debate + risk committee.

TradingAgents: Bull vs Bear researchers → Research Manager.
NexusTrade: 2-round debate + Research Judge, then Aggressive / Conservative /
Neutral risk analysts → CRO Judge.
"""

from __future__ import annotations

from src.graph.state import (
    Action,
    AnalystReport,
    DebateResult,
    RiskVerdict,
    TradePlan,
)


def run_debate(
    reports: dict[str, AnalystReport],
    weights: dict[str, float],
    *,
    debate_weight: float = 0.55,
) -> tuple[float, DebateResult]:
    """Combine analyst reports via a structured Bull/Bear debate.

    Returns ``(ensemble_score, debate)``. ``debate_weight`` blends the raw
    weighted analyst mean with the judge's score (0 = skip debate, 1 = judge
    only) — PrimoAgent is the debate_weight=0 extreme.
    """
    raw = 0.0
    wsum = 0.0
    bull_acc = 0.0
    bear_acc = 0.0
    bull_w = 0.0
    bear_w = 0.0
    for name, report in reports.items():
        w = float(weights.get(name, 0.0))
        if w <= 0:
            continue
        contrib = w * report.score * max(report.confidence, 0.15)
        raw += contrib
        wsum += w
        if report.score >= 0:
            bull_acc += contrib
            bull_w += w
        else:
            bear_acc += -contrib
            bear_w += w
    raw_score = raw / max(wsum, 1e-9)

    # Agreement: lower dispersion → higher judge confidence.
    scores = [r.score for r in reports.values()]
    if len(scores) > 1:
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        agreement = max(0.0, 1.0 - var)  # var of scores in [-1,1] is at most 1
    else:
        agreement = 0.5

    bull_score = bull_acc / max(bull_w, 1e-9) if bull_w else 0.0
    bear_score = bear_acc / max(bear_w, 1e-9) if bear_w else 0.0
    margin = bull_score - bear_score
    if margin > 0.12:
        verdict: str = "bullish"
    elif margin < -0.12:
        verdict = "bearish"
    else:
        verdict = "neutral"
    judge_score = max(-1.0, min(1.0, 0.65 * margin + 0.35 * raw_score))
    confidence = min(1.0, 0.30 + 0.50 * agreement + 0.20 * abs(margin))
    debate = DebateResult(
        bull_score=float(bull_score),
        bear_score=float(bear_score),
        judge_verdict=verdict,  # type: ignore[arg-type]
        judge_score=float(judge_score),
        confidence=float(confidence),
        thesis=f"{verdict} margin={margin:.3f} agreement={agreement:.2f}",
    )
    ensemble = (1.0 - debate_weight) * raw_score + debate_weight * judge_score
    return float(max(-1.0, min(1.0, ensemble))), debate


def trader_plan(
    ensemble_score: float,
    confidence: float,
    atr_pct: float,
    *,
    entry_threshold: float,
    stop_atr: float,
    tp_atr: float,
    size_pct: float,
    holding_days: int,
    long_only: bool,
    close: float,
) -> TradePlan:
    """Trader node — structured LONG/SHORT/FLAT plus swing geometry."""
    atr = max(float(atr_pct), 0.005)
    stop_pct = min(max(atr * stop_atr, 0.008), 0.12)
    target_pct = min(max(atr * tp_atr, 0.012), 0.25)
    action: Action = "FLAT"
    if ensemble_score >= entry_threshold and confidence >= 0.30:
        action = "LONG"
    elif (not long_only) and ensemble_score <= -entry_threshold and confidence >= 0.30:
        action = "SHORT"
    sized = size_pct * min(1.0, abs(ensemble_score) / max(entry_threshold, 1e-6))
    if action == "FLAT":
        sized = 0.0
    return TradePlan(
        action=action,
        score=float(ensemble_score),
        confidence=float(confidence),
        entry_ref=float(close),
        stop_pct=float(stop_pct),
        target_pct=float(target_pct),
        size_pct=float(sized),
        holding_days=int(holding_days),
    )


def risk_committee(
    plan: TradePlan,
    *,
    risk_mode: str,
    hmm_regime: int,
    spy_vs_sma200: float,
    portfolio_heat: float,
    heat_ceiling: float,
    consecutive_losses: int,
    daily_pnl_pct: float,
    drawdown_pct: float,
    daily_loss_limit: float = -0.03,
    max_drawdown: float = -0.08,
    vix_proxy: float = 0.15,
) -> RiskVerdict:
    """Five-role risk committee reduced to a deterministic CRO vote.

    Modes:
      aggressive   — Nexus aggressive analyst (size up, veto rarely)
      conservative — size down, veto high-vol / SPY-below-200d
      balanced     — AlpacaTradingAgent default caps
      committee    — median of the three, CRO can still veto hard limits
    """
    # Hard circuit breakers — never disabled (CLAUDE.md).
    if daily_pnl_pct <= daily_loss_limit:
        return RiskVerdict("rejected", 0.0, plan.stop_pct, plan.target_pct, 0.0, "daily_loss_limit")
    if drawdown_pct <= max_drawdown:
        return RiskVerdict("rejected", 0.0, plan.stop_pct, plan.target_pct, 0.0, "max_drawdown")
    if plan.action == "FLAT":
        return RiskVerdict("rejected", 0.0, plan.stop_pct, plan.target_pct, 0.0, "flat")
    if portfolio_heat >= heat_ceiling:
        return RiskVerdict("rejected", 0.0, plan.stop_pct, plan.target_pct, 0.0, "heat_ceiling")

    agg_mult, cons_mult, neu_mult = 1.25, 0.55, 1.00
    if hmm_regime == 2:
        agg_mult, cons_mult, neu_mult = 0.80, 0.25, 0.50
    if spy_vs_sma200 < 0 and plan.action == "LONG":
        agg_mult *= 0.85
        cons_mult *= 0.40
        neu_mult *= 0.70
    if consecutive_losses >= 5:
        agg_mult *= 0.50
        cons_mult *= 0.50
        neu_mult *= 0.50
    if vix_proxy > 0.35:
        return RiskVerdict("rejected", 0.0, plan.stop_pct, plan.target_pct, 0.0, "vol_spike")

    if risk_mode == "aggressive":
        mult = agg_mult
        veto_conf = 0.22
    elif risk_mode == "conservative":
        mult = cons_mult
        veto_conf = 0.55
        if hmm_regime == 2 or spy_vs_sma200 < -0.03:
            return RiskVerdict(
                "rejected", 0.0, plan.stop_pct, plan.target_pct, 0.0, "conservative_regime"
            )
    elif risk_mode == "committee":
        ordered = sorted([agg_mult, cons_mult, neu_mult])
        mult = ordered[1]  # median
        veto_conf = 0.38
        # Conservative still gets a veto vote on weak names.
        if cons_mult < 0.30 and plan.confidence < 0.55:
            return RiskVerdict(
                "rejected", 0.0, plan.stop_pct, plan.target_pct, 0.0, "cro_conservative_veto"
            )
    else:  # balanced — Alpaca-style
        mult = neu_mult
        veto_conf = 0.35

    if plan.confidence < veto_conf:
        return RiskVerdict("rejected", 0.0, plan.stop_pct, plan.target_pct, 0.0, "low_confidence")

    size = min(plan.size_pct * mult, 0.20)  # Alpaca 20% notional cap
    size = min(size, max(0.0, heat_ceiling - portfolio_heat))
    if size < 0.01:
        return RiskVerdict("rejected", 0.0, plan.stop_pct, plan.target_pct, 0.0, "size_floor")

    stop_pct = plan.stop_pct
    target_pct = plan.target_pct
    if risk_mode == "conservative":
        stop_pct *= 0.85
        target_pct *= 0.85
    elif risk_mode == "aggressive":
        stop_pct *= 1.10
        target_pct *= 1.20

    decision = "modified" if abs(mult - 1.0) > 0.05 else "approved"
    return RiskVerdict(
        decision=decision,  # type: ignore[arg-type]
        size_pct=float(size),
        stop_pct=float(stop_pct),
        target_pct=float(target_pct),
        size_mult=float(mult),
    )
