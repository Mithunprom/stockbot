"""Named strategy seeds distilled from the five public repos.

Each ``StrategySpec`` is a complete graph configuration: topology (which
analysts run, whether debate/committee fire), analyst weights, and the
risk / exit geometry the backtester consumes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

TopologyName = Literal[
    "primo_linear",
    "trading_agents",
    "nexus_trade",
    "quant_swing",
    "alpaca_five",
    "hybrid",
    "buy_hold",
]


@dataclass(frozen=True)
class StrategySpec:
    """Fully specified candidate the search agent can mutate and backtest."""

    name: str
    topology: TopologyName
    source: str
    analysts: tuple[str, ...]
    weights: dict[str, float]
    debate_weight: float
    risk_mode: str  # aggressive | conservative | balanced | committee | none
    entry_threshold: float
    stop_atr: float
    tp_atr: float
    holding_days: int
    max_positions: int
    size_pct: float
    long_only: bool = True
    trend_filter: bool = False  # long only if SPY > SMA200
    hmm_filter: bool = False  # skip high-vol regime
    heat_ceiling: float = 0.80
    cost_bps: float = 5.0  # one-way

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["analysts"] = list(self.analysts)
        return d

    def mutated(self, **kwargs: Any) -> "StrategySpec":
        return replace(self, **kwargs)


def _renorm(weights: dict[str, float], analysts: tuple[str, ...]) -> dict[str, float]:
    w = {k: max(0.0, float(v)) for k, v in weights.items() if k in analysts}
    s = sum(w.values())
    if s <= 0:
        even = 1.0 / max(len(analysts), 1)
        return {k: even for k in analysts}
    return {k: v / s for k, v in w.items()}


def seed_strategies() -> list[StrategySpec]:
    """The six architectural families plus a buy-and-hold baseline."""
    primo = StrategySpec(
        name="primo_linear",
        topology="primo_linear",
        source="ivebotunac/PrimoAgent",
        analysts=("technical", "news"),
        weights=_renorm({"technical": 0.60, "news": 0.40}, ("technical", "news")),
        debate_weight=0.0,
        risk_mode="balanced",
        entry_threshold=0.18,
        stop_atr=1.4,
        tp_atr=2.4,
        holding_days=8,
        max_positions=6,
        size_pct=0.10,
    )
    tauric = StrategySpec(
        name="trading_agents",
        topology="trading_agents",
        source="TauricResearch/TradingAgents",
        analysts=("technical", "fundamental", "sentiment", "news"),
        weights=_renorm(
            {"technical": 0.30, "fundamental": 0.25, "sentiment": 0.20, "news": 0.25},
            ("technical", "fundamental", "sentiment", "news"),
        ),
        debate_weight=0.60,
        risk_mode="balanced",
        entry_threshold=0.22,
        stop_atr=1.5,
        tp_atr=2.8,
        holding_days=10,
        max_positions=5,
        size_pct=0.10,
        trend_filter=True,
    )
    nexus = StrategySpec(
        name="nexus_trade",
        topology="nexus_trade",
        source="NastyRunner13/StockTradingAgents",
        analysts=("technical", "fundamental", "sentiment", "news"),
        weights=_renorm(
            {"technical": 0.28, "fundamental": 0.22, "sentiment": 0.25, "news": 0.25},
            ("technical", "fundamental", "sentiment", "news"),
        ),
        debate_weight=0.70,
        risk_mode="committee",
        entry_threshold=0.25,
        stop_atr=1.6,
        tp_atr=3.0,
        holding_days=12,
        max_positions=4,
        size_pct=0.12,
        hmm_filter=True,
    )
    quant = StrategySpec(
        name="quant_swing",
        topology="quant_swing",
        source="kasyap1234/stock_bot_langgraph",
        analysts=("technical", "quant", "macro"),
        weights=_renorm(
            {"technical": 0.35, "quant": 0.45, "macro": 0.20},
            ("technical", "quant", "macro"),
        ),
        debate_weight=0.20,
        risk_mode="balanced",
        entry_threshold=0.16,
        stop_atr=1.2,
        tp_atr=2.2,
        holding_days=7,
        max_positions=6,
        size_pct=0.10,
        hmm_filter=True,
        trend_filter=True,
    )
    alpaca = StrategySpec(
        name="alpaca_five",
        topology="alpaca_five",
        source="huygiatrng/AlpacaTradingAgent",
        analysts=("technical", "sentiment", "news", "fundamental", "macro"),
        weights=_renorm(
            {
                "technical": 0.22,
                "sentiment": 0.18,
                "news": 0.18,
                "fundamental": 0.20,
                "macro": 0.22,
            },
            ("technical", "sentiment", "news", "fundamental", "macro"),
        ),
        debate_weight=0.50,
        risk_mode="balanced",
        entry_threshold=0.20,
        stop_atr=1.8,
        tp_atr=2.5,
        holding_days=5,
        max_positions=5,
        size_pct=0.08,
        trend_filter=True,
        heat_ceiling=0.80,
    )
    hybrid = StrategySpec(
        name="hybrid_best_of",
        topology="hybrid",
        source="hybrid: TradingAgents debate + Nexus committee + kasyap quant + Alpaca caps + Primo NLP",
        analysts=("technical", "fundamental", "sentiment", "news", "macro", "quant"),
        weights=_renorm(
            {
                "technical": 0.22,
                "fundamental": 0.14,
                "sentiment": 0.12,
                "news": 0.12,
                "macro": 0.16,
                "quant": 0.24,
            },
            ("technical", "fundamental", "sentiment", "news", "macro", "quant"),
        ),
        debate_weight=0.55,
        risk_mode="committee",
        entry_threshold=0.20,
        stop_atr=1.4,
        tp_atr=2.6,
        holding_days=8,
        max_positions=5,
        size_pct=0.10,
        trend_filter=True,
        hmm_filter=True,
        heat_ceiling=0.75,
    )
    hybrid_defensive = StrategySpec(
        name="hybrid_defensive",
        topology="hybrid",
        source="hybrid: small-size committee + trend/HMM filters",
        analysts=("technical", "fundamental", "sentiment", "news", "macro", "quant"),
        weights=_renorm(
            {
                "technical": 0.20,
                "fundamental": 0.16,
                "sentiment": 0.10,
                "news": 0.10,
                "macro": 0.22,
                "quant": 0.22,
            },
            ("technical", "fundamental", "sentiment", "news", "macro", "quant"),
        ),
        debate_weight=0.60,
        risk_mode="conservative",
        entry_threshold=0.24,
        stop_atr=1.2,
        tp_atr=2.4,
        holding_days=10,
        max_positions=3,
        size_pct=0.04,
        trend_filter=True,
        hmm_filter=True,
        heat_ceiling=0.40,
    )
    buy_hold = StrategySpec(
        name="buy_hold_equal",
        topology="buy_hold",
        source="baseline",
        analysts=(),
        weights={},
        debate_weight=0.0,
        risk_mode="none",
        entry_threshold=0.0,
        stop_atr=99.0,
        tp_atr=99.0,
        holding_days=10_000,
        max_positions=20,
        size_pct=0.0,
        long_only=True,
    )
    # Locked search winner (2026-W45 walk-forward). Same topology as
    # hybrid_defensive with a slightly tighter stop discovered by mutation.
    # Staged only — not live. See reports/research/hybrid_strategy_search_2026-W45.json.
    hybrid_defensive_search = hybrid_defensive.mutated(
        name="hybrid_defensive_search",
        stop_atr=1.08,
    )
    return [primo, tauric, nexus, quant, alpaca, hybrid, hybrid_defensive, hybrid_defensive_search, buy_hold]


def quant_sub_strategies() -> list[StrategySpec]:
    """kasyap's five named styles as additional search seeds."""
    base = [s for s in seed_strategies() if s.name == "quant_swing"][0]
    return [
        base.mutated(
            name="quant_trend",
            weights=_renorm({"technical": 0.55, "quant": 0.25, "macro": 0.20}, base.analysts),
            trend_filter=True,
            hmm_filter=True,
            entry_threshold=0.14,
        ),
        base.mutated(
            name="quant_mean_reversion",
            weights=_renorm({"technical": 0.70, "quant": 0.20, "macro": 0.10}, base.analysts),
            trend_filter=False,
            hmm_filter=True,
            entry_threshold=0.12,
            stop_atr=1.0,
            tp_atr=1.6,
        ),
        base.mutated(
            name="quant_breakout",
            weights=_renorm({"technical": 0.40, "quant": 0.45, "macro": 0.15}, base.analysts),
            entry_threshold=0.18,
            stop_atr=1.3,
            tp_atr=2.8,
            holding_days=6,
        ),
        base.mutated(
            name="quant_sentiment",
            analysts=("technical", "sentiment", "news", "quant"),
            weights=_renorm(
                {"technical": 0.20, "sentiment": 0.30, "news": 0.25, "quant": 0.25},
                ("technical", "sentiment", "news", "quant"),
            ),
            debate_weight=0.30,
        ),
        base.mutated(
            name="quant_ensemble",
            analysts=("technical", "fundamental", "sentiment", "news", "macro", "quant"),
            weights=_renorm(
                {
                    "technical": 1,
                    "fundamental": 1,
                    "sentiment": 1,
                    "news": 1,
                    "macro": 1,
                    "quant": 1,
                },
                ("technical", "fundamental", "sentiment", "news", "macro", "quant"),
            ),
            debate_weight=0.40,
        ),
    ]
