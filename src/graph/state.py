"""Typed state objects that flow through the hybrid trading graph.

Mirrors the LangGraph state machines in TradingAgents / NexusTrade /
AlpacaTradingAgent without taking a LangGraph dependency: the backtester
needs to evaluate thousands of candidate graphs, which LLM-checkpointed
graphs cannot do.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Action = Literal["LONG", "SHORT", "FLAT"]
Verdict = Literal["bullish", "bearish", "neutral"]
RiskDecision = Literal["approved", "modified", "rejected"]


@dataclass(frozen=True)
class AnalystReport:
    """One analyst node's output for one ticker on one bar."""

    name: str
    score: float  # [-1, +1]  bullish positive
    confidence: float  # [0, 1]
    findings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DebateResult:
    """Bull/Bear researcher output plus the Research Judge verdict."""

    bull_score: float
    bear_score: float
    judge_verdict: Verdict
    judge_score: float  # [-1, +1]
    confidence: float
    thesis: str = ""


@dataclass(frozen=True)
class TradePlan:
    """Trader-node output: structured action plus swing geometry."""

    action: Action
    score: float
    confidence: float
    entry_ref: float
    stop_pct: float
    target_pct: float
    size_pct: float
    holding_days: int


@dataclass(frozen=True)
class RiskVerdict:
    """Risk Committee (CRO) output."""

    decision: RiskDecision
    size_pct: float
    stop_pct: float
    target_pct: float
    size_mult: float
    veto_reason: str = ""


@dataclass
class GraphState:
    """Per-ticker, per-bar graph state (explainability payload)."""

    ticker: str
    reports: dict[str, AnalystReport] = field(default_factory=dict)
    debate: DebateResult | None = None
    plan: TradePlan | None = None
    risk: RiskVerdict | None = None
    ensemble_score: float = 0.0
    final_action: Action = "FLAT"
    final_confidence: float = 0.0
    final_size_pct: float = 0.0
    stop_pct: float = 0.0
    target_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "reports": {k: v.to_dict() for k, v in self.reports.items()},
            "debate": asdict(self.debate) if self.debate else None,
            "plan": asdict(self.plan) if self.plan else None,
            "risk": asdict(self.risk) if self.risk else None,
            "ensemble_score": round(self.ensemble_score, 6),
            "final_action": self.final_action,
            "final_confidence": round(self.final_confidence, 4),
            "final_size_pct": round(self.final_size_pct, 4),
            "stop_pct": round(self.stop_pct, 4),
            "target_pct": round(self.target_pct, 4),
        }
