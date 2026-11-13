"""Opportunity / Strategy Search Agent — agentic search over hybrid graphs.

Scope (CLAUDE.md Opportunity Agent):
  Discover new alpha sources. A/B test in paper / backtest only.
  Must NOT modify the live ensemble. All proposals go to config/staging/.

The agent:
  1. Seeds a population from the five public architectures + hybrid.
  2. Backtests each candidate on the TUNE window.
  3. Mutates / crosses the elite set for a few generations.
  4. Re-ranks survivors on an untouched VAL window (both legs must hold).
  5. Scores the winner on a final HOLDOUT window and writes the proposal.

Escalate if no candidate beats buy-and-hold on validation with DD ≤ 12%.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.graph.backtest import BacktestResult, simulate_portfolio
from src.graph.catalog import StrategySpec, quant_sub_strategies, seed_strategies
from src.graph.pipeline import build_analyst_panel, build_feature_store, build_signals

logger = structlog.get_logger(__name__)

AGENT_NAME = "Strategy Search Agent"
STAGING_PATH = Path("config/staging/strategy_search.json")
REPORT_DIR = Path("reports/opportunities")

DEFAULT_WINDOWS = {
    "tune": ("2022-01-01", "2024-12-31"),
    "val": ("2025-01-01", "2025-12-31"),
    "holdout": ("2026-01-01", "2026-11-01"),
}

SEARCH_UNIVERSE = [
    "SPY", "QQQ",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "JPM", "XOM", "AVGO", "AMD", "TSLA", "V", "MA", "UNH", "HD", "PG",
]


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def mutate_spec(spec: StrategySpec, rng: np.random.Generator) -> StrategySpec:
    """Perturb one lever of a spec — the search's exploration step."""
    if spec.topology == "buy_hold":
        return spec
    choice = int(rng.integers(0, 9))
    w = dict(spec.weights)
    if choice == 0 and w:
        # Dirichlet-like jitter on weights.
        keys = list(w)
        noise = rng.normal(0.0, 0.08, size=len(keys))
        for k, n in zip(keys, noise):
            w[k] = max(0.02, w[k] + n)
        s = sum(w.values())
        w = {k: v / s for k, v in w.items()}
        return spec.mutated(name=f"{spec.name}~w", weights=w)
    if choice == 1:
        return spec.mutated(
            name=f"{spec.name}~thr",
            entry_threshold=_clamp(spec.entry_threshold + float(rng.normal(0, 0.04)), 0.08, 0.40),
        )
    if choice == 2:
        return spec.mutated(
            name=f"{spec.name}~stop",
            stop_atr=_clamp(spec.stop_atr + float(rng.normal(0, 0.25)), 0.7, 2.8),
        )
    if choice == 3:
        return spec.mutated(
            name=f"{spec.name}~tp",
            tp_atr=_clamp(spec.tp_atr + float(rng.normal(0, 0.4)), 1.2, 4.5),
        )
    if choice == 4:
        hold = int(_clamp(spec.holding_days + int(rng.integers(-3, 4)), 3, 21))
        return spec.mutated(name=f"{spec.name}~hold", holding_days=hold)
    if choice == 5:
        modes = ["aggressive", "conservative", "balanced", "committee"]
        modes = [m for m in modes if m != spec.risk_mode] or modes
        return spec.mutated(name=f"{spec.name}~risk", risk_mode=str(rng.choice(modes)))
    if choice == 6:
        return spec.mutated(
            name=f"{spec.name}~sz",
            size_pct=_clamp(spec.size_pct * float(rng.uniform(0.5, 1.15)), 0.02, 0.12),
            max_positions=int(_clamp(spec.max_positions + int(rng.integers(-1, 2)), 2, 6)),
        )
    if choice == 7:
        return spec.mutated(
            name=f"{spec.name}~heat",
            heat_ceiling=_clamp(spec.heat_ceiling + float(rng.choice([-0.15, -0.10, 0.0, 0.05])), 0.30, 0.80),
        )
    # Toggle a filter.
    if rng.random() < 0.5:
        return spec.mutated(name=f"{spec.name}~trend", trend_filter=not spec.trend_filter)
    return spec.mutated(name=f"{spec.name}~hmm", hmm_filter=not spec.hmm_filter)


def crossover(a: StrategySpec, b: StrategySpec, rng: np.random.Generator) -> StrategySpec:
    """Mix analyst weights from A with exits from B (or vice versa)."""
    if a.topology == "buy_hold" or b.topology == "buy_hold":
        return a
    parent_w, parent_x = (a, b) if rng.random() < 0.5 else (b, a)
    analysts = tuple(sorted(set(parent_w.analysts) | set(parent_x.analysts)))
    w: dict[str, float] = {}
    for k in analysts:
        wa = parent_w.weights.get(k, 0.0)
        wb = parent_x.weights.get(k, 0.0)
        w[k] = 0.5 * (wa + wb) if (wa + wb) > 0 else 0.0
    s = sum(w.values())
    w = {k: v / s for k, v in w.items() if v > 0} if s > 0 else dict(parent_w.weights)
    return replace(
        parent_w,
        name=f"x_{parent_w.name[:12]}_{parent_x.name[:12]}",
        analysts=tuple(w),
        weights=w,
        entry_threshold=parent_x.entry_threshold,
        stop_atr=parent_x.stop_atr,
        tp_atr=parent_x.tp_atr,
        holding_days=parent_x.holding_days,
        risk_mode=parent_x.risk_mode,
        debate_weight=0.5 * (parent_w.debate_weight + parent_x.debate_weight),
        trend_filter=parent_w.trend_filter or parent_x.trend_filter,
        hmm_filter=parent_w.hmm_filter or parent_x.hmm_filter,
    )


class StrategySearchAgent:
    """Evolutionary search over hybrid multi-agent graphs via backtest."""

    def __init__(
        self,
        *,
        generations: int = 3,
        elite_k: int = 5,
        children_per_gen: int = 8,
        seed: int = 7,
        windows: dict[str, tuple[str, str]] | None = None,
        capital: float = 100_000.0,
    ) -> None:
        self.generations = generations
        self.elite_k = elite_k
        self.children_per_gen = children_per_gen
        self.rng = np.random.default_rng(seed)
        self.windows = windows or DEFAULT_WINDOWS
        self.capital = capital

    def evaluate(
        self,
        spec: StrategySpec,
        store: dict[str, pd.DataFrame],
        bars: dict[str, pd.DataFrame],
        start: str,
        end: str,
        cache: dict[str, pd.DataFrame] | None = None,
    ) -> BacktestResult:
        key = spec.name
        sig_cache = cache if cache is not None else {}
        if key not in sig_cache:
            sig_cache[key] = build_signals(
                store, spec, start=start, end=end, panel=getattr(self, "_panel", None)
            )
        return simulate_portfolio(
            bars, sig_cache[key], spec, start=start, end=end, capital=self.capital
        )

    def search(
        self,
        bars: dict[str, pd.DataFrame],
        *,
        store: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, Any]:
        """Run the full tune → evolve → validate → holdout loop."""
        started = datetime.now(timezone.utc)
        if store is None:
            spy = bars["SPY"]["close"] if "SPY" in bars else None
            store = build_feature_store(bars, spy)
        self._panel = build_analyst_panel(store)

        population: list[StrategySpec] = seed_strategies() + quant_sub_strategies()
        tune_s, tune_e = self.windows["tune"]
        val_s, val_e = self.windows["val"]
        hold_s, hold_e = self.windows["holdout"]

        def rank(specs: list[StrategySpec], start: str, end: str) -> list[tuple[StrategySpec, BacktestResult]]:
            scored: list[tuple[StrategySpec, BacktestResult]] = []
            cache: dict[str, pd.DataFrame] = {}
            for spec in specs:
                try:
                    res = self.evaluate(spec, store, bars, start, end, cache)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("strategy_eval_failed", name=spec.name, error=str(exc))
                    continue
                scored.append((spec, res))
            scored.sort(key=lambda x: x[1].score, reverse=True)
            return scored

        logger.info("strategy_search_tune_start", n=len(population), window=f"{tune_s}→{tune_e}")
        ranked = rank(population, tune_s, tune_e)
        history: list[dict[str, Any]] = [
            {"gen": 0, "results": [r.to_dict() | {"spec": s.name} for s, r in ranked]}
        ]

        for gen in range(1, self.generations + 1):
            elites = [s for s, _ in ranked[: self.elite_k]]
            children: list[StrategySpec] = []
            for _ in range(self.children_per_gen):
                parent = elites[int(self.rng.integers(0, max(len(elites), 1)))] if elites else population[0]
                if len(elites) >= 2 and self.rng.random() < 0.35:
                    other = elites[int(self.rng.integers(0, len(elites)))]
                    child = crossover(parent, other, self.rng)
                else:
                    child = mutate_spec(parent, self.rng)
                children.append(child)
            ranked_c = rank(children, tune_s, tune_e)
            merged = ranked + ranked_c
            # Unique by name, keep best score.
            best: dict[str, tuple[StrategySpec, BacktestResult]] = {}
            for s, r in merged:
                prev = best.get(s.name)
                if prev is None or r.score > prev[1].score:
                    best[s.name] = (s, r)
            ranked = sorted(best.values(), key=lambda x: x[1].score, reverse=True)
            history.append(
                {"gen": gen, "results": [r.to_dict() | {"spec": s.name} for s, r in ranked[:12]]}
            )
            logger.info(
                "strategy_search_generation",
                gen=gen,
                best=ranked[0][0].name if ranked else None,
                best_score=ranked[0][1].score if ranked else None,
            )

        # Validation: re-evaluate top-K on untouched window.
        elites = [s for s, _ in ranked[: self.elite_k]]
        if not any(s.topology == "buy_hold" for s in elites):
            elites.append(next(s for s in seed_strategies() if s.topology == "buy_hold"))
        val_ranked = rank(elites, val_s, val_e)
        tune_by_name = {s.name: r for s, r in ranked}

        def both_legs_ok(spec: StrategySpec, val_res: BacktestResult) -> bool:
            tune_res = tune_by_name.get(spec.name)
            if tune_res is None:
                return False
            if spec.topology == "buy_hold":
                return True
            # Must not collapse on the untouched leg, and DD stays livable.
            return (
                val_res.n_days > 30
                and val_res.max_dd_pct <= 16.0
                and val_res.sharpe > -0.5
                and tune_res.score > -1.0
            )

        viable = [(s, r) for s, r in val_ranked if both_legs_ok(s, r)]
        buy_hold_val = next((r for s, r in val_ranked if s.topology == "buy_hold"), None)

        def pick_key(item: tuple[StrategySpec, BacktestResult]) -> float:
            spec, val_res = item
            tune_res = tune_by_name.get(spec.name)
            tune_score = tune_res.score if tune_res else -9.0
            # Prefer candidates that beat buy-hold on validation Sharpe.
            bh_bonus = 0.0
            if buy_hold_val is not None and val_res.sharpe > buy_hold_val.sharpe:
                bh_bonus = 0.35
            return 0.45 * tune_score + 0.55 * val_res.score + bh_bonus

        viable.sort(key=pick_key, reverse=True)
        if viable:
            winner_spec, winner_val = viable[0]
        else:
            winner_spec, winner_val = val_ranked[0] if val_ranked else ranked[0]

        holdout = self.evaluate(winner_spec, store, bars, hold_s, hold_e)
        winner_tune = tune_by_name.get(winner_spec.name)

        needs_review = True
        ready_for_paper = (
            winner_spec.topology != "buy_hold"
            and winner_val.sharpe >= 0.8
            and winner_val.max_dd_pct <= 10.0
            and (winner_tune is not None and winner_tune.sharpe >= 0.8)
            and holdout.max_dd_pct <= 12.0
        )
        escalations: list[dict[str, str]] = []
        if winner_spec.topology == "buy_hold":
            escalations.append({
                "level": "WARNING",
                "trigger": "No multi-agent graph beat buy-and-hold on validation",
                "recommendation": "Keep paper pipeline unchanged; do not promote a graph topology",
            })
        if not ready_for_paper:
            escalations.append({
                "level": "INFO",
                "trigger": "Winner does not yet meet paper-promotion gates (Sharpe≥0.8 both legs, DD≤10% val)",
                "recommendation": "Leave live.yaml untouched; review staging proposal only",
            })

        report = {
            "agent": AGENT_NAME,
            "timestamp": started.isoformat(),
            "windows": self.windows,
            "universe": sorted(t for t in bars if t != "SPY"),
            "n_tickers": len([t for t in bars if t != "SPY"]),
            "generations": self.generations,
            "winner": {
                "spec": winner_spec.to_dict(),
                "tune": winner_tune.to_dict() if winner_tune else None,
                "validation": winner_val.to_dict(),
                "holdout": holdout.to_dict(),
            },
            "buy_hold_validation": buy_hold_val.to_dict() if buy_hold_val else None,
            "validation_leaderboard": [
                {"spec": s.to_dict(), "metrics": r.to_dict()} for s, r in val_ranked
            ],
            "tune_top": [
                {"name": s.name, "source": s.source, "metrics": r.to_dict()}
                for s, r in ranked[:8]
            ],
            "history": history,
            "ready_for_paper": ready_for_paper,
            "needs_review": needs_review,
            "escalations": escalations,
            "notes": (
                "Proposal only — never writes config/live.yaml. "
                "Paper promotion still requires the Phase 4 criteria in PROJECT.md."
            ),
        }
        return report

    def persist(self, report: dict[str, Any]) -> tuple[Path, Path]:
        """Write staging proposal + weekly opportunity report. Never touches live.yaml."""
        STAGING_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        staging = {
            "agent": AGENT_NAME,
            "timestamp": report["timestamp"],
            "needs_review": True,
            "ready_for_paper": report.get("ready_for_paper", False),
            "do_not_apply_to_live": True,
            "winner": report["winner"],
            "escalations": report.get("escalations", []),
        }
        with open(STAGING_PATH, "w") as f:
            json.dump(staging, f, indent=2, default=str)
        week = datetime.now(timezone.utc).strftime("%Y-W%W")
        report_path = REPORT_DIR / f"{week}.json"
        slim = dict(report)
        # Equity curves are for the dashboard; drop them from the committed copy.
        def _strip_equity(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _strip_equity(v) for k, v in obj.items() if k != "equity"}
            if isinstance(obj, list):
                return [_strip_equity(x) for x in obj]
            return obj

        slim = _strip_equity(slim)
        with open(report_path, "w") as f:
            json.dump(slim, f, indent=2, default=str)
        # reports/opportunities is gitignored; keep a reviewable copy in research/.
        research_dir = REPORT_DIR.parent / "research"
        research_dir.mkdir(parents=True, exist_ok=True)
        research_path = research_dir / f"hybrid_strategy_search_{week}.json"
        with open(research_path, "w") as f:
            json.dump(slim, f, indent=2, default=str)
        logger.info(
            "strategy_search_persisted",
            staging=str(STAGING_PATH),
            report=str(report_path),
            research=str(research_path),
        )
        return STAGING_PATH, report_path

    async def run(self, bars: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
        """Scheduler entry point. If ``bars`` is None the caller should fetch."""
        if bars is None:
            from datetime import date as date_cls

            from src.graph.data import load_universe_bars

            start = date_cls.fromisoformat(self.windows["tune"][0])
            end = date_cls.fromisoformat(self.windows["holdout"][1])
            bars = load_universe_bars(SEARCH_UNIVERSE, start, end)
            if len(bars) < 4:
                return {"error": "not enough price history to search", "n_tickers": len(bars)}
        report = self.search(bars)
        self.persist(report)
        return report
