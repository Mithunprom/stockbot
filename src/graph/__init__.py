"""Hybrid multi-agent trading graph.

Combines the public architectures we studied into one StockBot-native DAG:

* TauricResearch/TradingAgents — parallel Fundamental / Sentiment / News /
  Technical analysts, Bull/Bear debate, Trader, Risk team.
* NastyRunner13/StockTradingAgents (NexusTrade) — 13-agent desk with a
  three-agent research debate and a five-agent Risk Committee.
* kasyap1234/stock_bot_langgraph — 40+ indicators, HMM regime, sequence
  predictor, swing entry / stop / target levels.
* huygiatrng/AlpacaTradingAgent — fifth Macro analyst, LONG/NEUTRAL/SHORT
  (or BUY/HOLD/SELL) structured action, Alpaca-style risk caps.
* ivebotunac/PrimoAgent — linear Data → Technical → 7-feature News NLP →
  Portfolio Manager baseline.

LLM calls are optional and off by default. Every node has a deterministic
quantitative stand-in so the graph can be backtested bar-by-bar without
look-ahead or token cost. Strategy search lives in
``src.agents.strategy_search_agent``.
"""

from src.graph.backtest import BacktestResult, simulate_portfolio
from src.graph.catalog import StrategySpec, seed_strategies
from src.graph.pipeline import GraphTopology, build_signals

__all__ = [
    "BacktestResult",
    "GraphTopology",
    "StrategySpec",
    "build_signals",
    "seed_strategies",
    "simulate_portfolio",
]
