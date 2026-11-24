"""Run the hybrid-graph strategy search end-to-end.

Usage:
    python scripts/agentic_strategy_search.py
    python scripts/agentic_strategy_search.py --synthetic   # offline tests
    python scripts/agentic_strategy_search.py --quick       # 1 generation
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.strategy_search_agent import SEARCH_UNIVERSE, StrategySearchAgent
from src.graph.data import load_universe_bars, synthetic_ohlcv


def _synthetic_universe(n: int = 520) -> dict:
    tickers = ["SPY", "AAPL", "MSFT", "NVDA", "JPM", "XOM"]
    drifts = [0.00028, 0.00040, 0.00035, 0.00055, 0.00022, 0.00018]
    vols = [0.011, 0.016, 0.015, 0.024, 0.014, 0.017]
    out = {}
    for i, t in enumerate(tickers):
        out[t] = synthetic_ohlcv(n=n, seed=10 + i, drift=drifts[i], vol=vols[i], start="2022-01-03")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-11-01")
    args = ap.parse_args()

    generations = 1 if args.quick else 3
    children = 4 if args.quick else 8
    agent = StrategySearchAgent(generations=generations, children_per_gen=children, elite_k=5)

    if args.synthetic:
        bars = _synthetic_universe()
        # Shrink windows to the synthetic calendar.
        idx = bars["SPY"].index
        agent.windows = {
            "tune": (str(idx[20].date()), str(idx[len(idx) // 2].date())),
            "val": (str(idx[len(idx) // 2 + 1].date()), str(idx[int(len(idx) * 0.8)].date())),
            "holdout": (str(idx[int(len(idx) * 0.8) + 1].date()), str(idx[-2].date())),
        }
    else:
        bars = load_universe_bars(
            SEARCH_UNIVERSE,
            date.fromisoformat(args.start),
            date.fromisoformat(args.end),
        )
        print(f"loaded {len(bars)} tickers: {sorted(bars)}")
        if len(bars) < 4:
            print("ERROR: not enough market data — retry or pass --synthetic")
            sys.exit(1)

    report = agent.search(bars)
    staging, path = agent.persist(report)
    winner = report["winner"]
    print("\n=== Strategy Search Winner ===")
    print(json.dumps({
        "name": winner["spec"]["name"],
        "topology": winner["spec"]["topology"],
        "source": winner["spec"]["source"],
        "tune": winner["tune"],
        "validation": winner["validation"],
        "holdout": winner["holdout"],
        "ready_for_paper": report["ready_for_paper"],
        "escalations": report["escalations"],
    }, indent=2, default=str))
    print(f"\nstaging → {staging}")
    print(f"report  → {path}")


if __name__ == "__main__":
    main()
