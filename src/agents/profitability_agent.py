"""Profitability Diagnosis Agent — detects WHY the bot isn't making money.

Scope: Read-only forensic analysis of closed trades. Compares live trading
       behavior against the strategy's *design assumptions* (holding horizon,
       exit mix, signal horizon) and ranks root causes with evidence.
Must NOT: Modify any config, models, or positions. Writes findings to
       reports/profitability/ and suggestions to config/staging/ only.
Output: reports/profitability/YYYY-MM-DD.json + .md
Escalate if: profit factor < 1.0 over the lookback window, or any CRITICAL
       design-conformance violation is detected.

The core idea: most "no profit" situations are not a missing alpha problem —
they are a *design-conformance* problem: the live execution layer does not
trade the strategy that was validated. This agent measures that gap.

CLI:
    python -m src.agents.profitability_agent --api https://<host> --days 60
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

AGENT_NAME = "Profitability Diagnosis Agent"
REPORT_DIR = Path("reports/profitability")
STAGING_PATH = Path("config/staging/profitability_suggestions.json")

# Severity ordering for ranking findings
_SEVERITY_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}


@dataclass
class StrategyDesign:
    """The strategy the system is SUPPOSED to be trading.

    Defaults mirror the v0.3.3 daily-vol swing design in signal_loop.py:
    LightGBM predicts ~1-day-ahead returns, entries are gated on pred_return
    clearing a cost threshold, and exits were backtest-validated at a
    1-trading-day holding horizon (SIZING_MAX_HOLD_BARS = 390 one-minute bars).
    """

    designed_hold_minutes: float = 390.0        # 1 trading day of 1m bars
    signal_horizon_minutes: float = 390.0       # LGBM label horizon (1 day)
    designed_exit_reasons: tuple[str, ...] = (
        "take_profit", "trailing_stop", "stop_loss", "max_hold",
    )
    round_trip_cost_pct: float = 0.0012         # spread+slippage estimate
    min_sample_size: int = 100                  # below this, stats are noise


@dataclass
class Finding:
    """One diagnosed root cause with supporting evidence."""

    code: str
    severity: str
    title: str
    evidence: str
    recommendation: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "title": self.title,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "metrics": self.metrics,
        }


# ---------------------------------------------------------------------------
# Metric primitives (pure functions — unit tested)
# ---------------------------------------------------------------------------

def hold_minutes(trade: dict[str, Any]) -> float | None:
    """Minutes between entry and exit; None if the trade is still open."""
    if not trade.get("exit_time") or not trade.get("entry_time"):
        return None
    entry = datetime.fromisoformat(trade["entry_time"])
    exit_ = datetime.fromisoformat(trade["exit_time"])
    return (exit_ - entry).total_seconds() / 60.0


def core_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Win rate, profit factor, expectancy, and hold-time distribution.

    Never reports PnL alone — always alongside risk stats (CLAUDE.md rule).
    """
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    holds = [h for h in (hold_minutes(t) for t in trades) if h is not None]
    return {
        "n_trades": len(trades),
        "total_pnl": round(sum(pnls), 2),
        "win_rate": round(len(wins) / len(pnls), 4) if pnls else 0.0,
        "profit_factor": (
            round(gross_profit / gross_loss, 3) if gross_loss > 0 else math.inf
        ),
        "avg_win": round(statistics.mean(wins), 2) if wins else 0.0,
        "avg_loss": round(statistics.mean(losses), 2) if losses else 0.0,
        "expectancy_per_trade": round(statistics.mean(pnls), 2) if pnls else 0.0,
        "median_hold_minutes": round(statistics.median(holds), 1) if holds else 0.0,
        "mean_hold_minutes": round(statistics.mean(holds), 1) if holds else 0.0,
    }


def daily_sharpe(trades: list[dict[str, Any]], portfolio_value: float) -> dict[str, Any]:
    """Annualized Sharpe and max drawdown from daily aggregated trade PnL."""
    by_day: dict[str, float] = defaultdict(float)
    for t in trades:
        by_day[t["exit_time"][:10]] += t["pnl"]
    if len(by_day) < 2 or portfolio_value <= 0:
        return {"sharpe": None, "max_drawdown_pct": None, "n_days": len(by_day)}
    daily_returns = [pnl / portfolio_value for _, pnl in sorted(by_day.items())]
    mean_r = statistics.mean(daily_returns)
    std_r = statistics.stdev(daily_returns)
    sharpe = (mean_r / std_r) * math.sqrt(252) if std_r > 0 else None
    # Max drawdown on the cumulative equity curve
    equity, peak, max_dd = 0.0, 0.0, 0.0
    for r in daily_returns:
        equity += r
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return {
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_days": len(by_day),
    }


def attribution_by(trades: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """PnL attribution grouped by a trade field (exit_reason, ticker, ...)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        groups[str(t.get(key, "unknown"))].append(t["pnl"])
    return {
        k: {
            "n": len(v),
            "total_pnl": round(sum(v), 2),
            "avg_pnl": round(statistics.mean(v), 2),
            "win_rate": round(sum(1 for p in v if p > 0) / len(v), 3),
        }
        for k, v in sorted(groups.items(), key=lambda kv: sum(kv[1]))
    }


# ---------------------------------------------------------------------------
# Root-cause detectors — each returns a Finding or None
# ---------------------------------------------------------------------------

def detect_horizon_mismatch(
    trades: list[dict[str, Any]], design: StrategyDesign
) -> Finding | None:
    """Live holding time far below the horizon the signal was validated on."""
    holds = [h for h in (hold_minutes(t) for t in trades) if h is not None]
    if not holds:
        return None
    median_hold = statistics.median(holds)
    ratio = median_hold / design.designed_hold_minutes
    if ratio >= 0.5:
        return None
    return Finding(
        code="HORIZON_MISMATCH",
        severity="CRITICAL",
        title="Trades exit long before the signal's validated horizon",
        evidence=(
            f"Median hold = {median_hold:.0f} min vs designed "
            f"{design.designed_hold_minutes:.0f} min ({ratio:.0%} of design). "
            f"The model predicts {design.signal_horizon_minutes:.0f}-minute-ahead "
            "returns; exiting earlier means the validated edge never has time "
            "to materialize — the live system trades a DIFFERENT strategy than "
            "the one backtested."
        ),
        recommendation=(
            "Disable or drastically slow any exit that can fire inside the "
            "signal horizon (e.g., require reversal confirmation over 30-60 "
            "bars, or gate reversal exits on a fresh opposite-direction "
            "pred_return that clears the cost threshold — not ensemble flicker)."
        ),
        metrics={"median_hold_minutes": round(median_hold, 1), "hold_ratio": round(ratio, 3)},
    )


def detect_undesigned_exit_dominance(
    trades: list[dict[str, Any]], design: StrategyDesign
) -> Finding | None:
    """An exit path outside the validated design dominates the trade flow."""
    attribution = attribution_by(trades, "exit_reason")
    n_total = len(trades)
    for reason, stats in attribution.items():
        if reason in design.designed_exit_reasons:
            continue
        share = stats["n"] / n_total
        if share >= 0.4:
            return Finding(
                code="UNDESIGNED_EXIT_DOMINANT",
                severity="CRITICAL",
                title=f"'{reason}' exits dominate ({share:.0%} of trades) but were never backtested",
                evidence=(
                    f"{stats['n']}/{n_total} trades exit via '{reason}' "
                    f"(total PnL {stats['total_pnl']:+.2f}, win rate "
                    f"{stats['win_rate']:.0%}). This exit path is not part of "
                    "the validated design — the backtest edge does not apply "
                    "to these trades."
                ),
                recommendation=(
                    f"Either backtest '{reason}' exits explicitly or remove "
                    "them. An unvalidated exit path invalidates the whole "
                    "strategy's expectancy estimate."
                ),
                metrics={"reason": reason, "share": round(share, 3), **stats},
            )
    return None


def detect_stop_hemorrhage(trades: list[dict[str, Any]]) -> Finding | None:
    """Stops + trailing stops destroying more than the winners earn."""
    attribution = attribution_by(trades, "exit_reason")
    stop_loss = attribution.get("stop_loss", {"total_pnl": 0, "n": 0})
    trailing = attribution.get("trailing_stop", {"total_pnl": 0, "n": 0})
    take_profit = attribution.get("take_profit", {"total_pnl": 0, "n": 0})
    stop_damage = stop_loss["total_pnl"] + trailing["total_pnl"]
    if stop_damage >= 0 or take_profit["total_pnl"] + stop_damage >= 0:
        return None
    return Finding(
        code="STOP_HEMORRHAGE",
        severity="HIGH",
        title="Risk exits destroy more than profit exits earn",
        evidence=(
            f"stop_loss: {stop_loss['n']} trades / {stop_loss['total_pnl']:+.2f}; "
            f"trailing_stop: {trailing['n']} trades / {trailing['total_pnl']:+.2f}; "
            f"take_profit: {take_profit['n']} trades / {take_profit['total_pnl']:+.2f}. "
            "The win side cannot pay for the loss side — classic sign that "
            "targets are unreachable at the actual holding horizon while "
            "stops remain fully reachable."
        ),
        recommendation=(
            "Re-derive stop/target multiples from the realized holding-period "
            "volatility (not daily ATR applied to minute-scale holds), or fix "
            "the holding horizon first and re-measure."
        ),
        metrics={
            "stop_damage": round(stop_damage, 2),
            "take_profit_total": take_profit["total_pnl"],
        },
    )


def detect_ticker_bleeders(trades: list[dict[str, Any]]) -> Finding | None:
    """Tickers that consistently lose money and should be IC-gated."""
    attribution = attribution_by(trades, "ticker")
    bleeders = {
        k: v for k, v in attribution.items() if v["n"] >= 3 and v["total_pnl"] < 0
    }
    if not bleeders:
        return None
    total_bleed = sum(v["total_pnl"] for v in bleeders.values())
    names = ", ".join(f"{k} ({v['total_pnl']:+.0f})" for k, v in bleeders.items())
    return Finding(
        code="TICKER_BLEEDERS",
        severity="MEDIUM",
        title=f"{len(bleeders)} tickers repeatedly lose money ({total_bleed:+.2f} combined)",
        evidence=f"Tickers with ≥3 trades and negative total PnL: {names}.",
        recommendation=(
            "Audit why the per-ticker live-IC gate has not excluded these "
            "names (tickers_ic_blocked is empty). Tighten the gate: require "
            "positive rolling live IC before a ticker is tradeable, not just "
            "'not strongly negative'."
        ),
        metrics={"bleeders": bleeders},
    )


def detect_cost_drag(
    trades: list[dict[str, Any]], design: StrategyDesign
) -> Finding | None:
    """Per-trade edge too small relative to round-trip transaction costs."""
    pnl_pcts = [abs(t.get("pnl_pct", 0.0)) for t in trades if t.get("pnl_pct") is not None]
    if not pnl_pcts:
        return None
    # pnl_pct in this schema can be in percent units; normalize heuristically
    median_move = statistics.median(pnl_pcts)
    if median_move > 1.0:  # clearly percent units
        median_move /= 100.0
    if median_move > design.round_trip_cost_pct * 3:
        return None
    return Finding(
        code="COST_DRAG",
        severity="HIGH",
        title="Median per-trade move is within ~3x of round-trip costs",
        evidence=(
            f"Median |pnl%| = {median_move:.3%} vs assumed round-trip cost "
            f"{design.round_trip_cost_pct:.3%}. At this ratio, costs and "
            "noise consume most of any real edge."
        ),
        recommendation=(
            "Trade less often at longer horizons where the expected move is "
            "a large multiple of cost, or require pred_return > 3x cost."
        ),
        metrics={"median_abs_pnl_pct": round(median_move, 5)},
    )


def detect_insufficient_sample(
    trades: list[dict[str, Any]], design: StrategyDesign, window_days: int
) -> Finding | None:
    """Not enough trades to distinguish edge from noise."""
    if len(trades) >= design.min_sample_size:
        return None
    return Finding(
        code="SMALL_SAMPLE",
        severity="INFO",
        title=f"Only {len(trades)} closed trades in {window_days} days — expectancy estimates are noisy",
        evidence=(
            f"With n={len(trades)}, a true 55% win-rate strategy can easily "
            "show <50% observed. Any strategy verdict needs "
            f"≥{design.min_sample_size} trades at a FIXED configuration — "
            "constant parameter churn resets the sample."
        ),
        recommendation=(
            "Freeze one configuration for a full evaluation window. Stop "
            "changing entry/exit parameters mid-sample."
        ),
        metrics={"n_trades": len(trades), "window_days": window_days},
    )


def detect_negative_expectancy(trades: list[dict[str, Any]]) -> Finding | None:
    """The bottom line: PF < 1 with losses larger than wins."""
    m = core_metrics(trades)
    if m["profit_factor"] >= 1.0:
        return None
    return Finding(
        code="NEGATIVE_EXPECTANCY",
        severity="HIGH",
        title=f"Profit factor {m['profit_factor']:.2f} — the book loses money as configured",
        evidence=(
            f"n={m['n_trades']}, win rate {m['win_rate']:.1%}, "
            f"avg win {m['avg_win']:+.2f} vs avg loss {m['avg_loss']:+.2f}, "
            f"total PnL {m['total_pnl']:+.2f}. Losses run larger than wins "
            "with a sub-50% hit rate: there is no positive expectancy at the "
            "current (execution-truncated) horizon."
        ),
        recommendation=(
            "Do not tune around this — fix the design-conformance findings "
            "first, then re-measure on a frozen configuration."
        ),
        metrics=m,
    )


ALL_DETECTORS = (
    detect_horizon_mismatch,
    detect_undesigned_exit_dominance,
    detect_stop_hemorrhage,
    detect_negative_expectancy,
    detect_cost_drag,
    detect_ticker_bleeders,
)


def diagnose(
    trades: list[dict[str, Any]],
    design: StrategyDesign | None = None,
    window_days: int = 60,
) -> list[Finding]:
    """Run all detectors and return findings ranked by severity."""
    design = design or StrategyDesign()
    findings: list[Finding] = []
    for detector in ALL_DETECTORS:
        try:
            result = (
                detector(trades, design)  # type: ignore[call-arg]
                if detector.__code__.co_argcount >= 2
                else detector(trades)  # type: ignore[call-arg]
            )
        except Exception:
            logger.exception("detector_failed", detector=detector.__name__)
            continue
        if result:
            findings.append(result)
    sample = detect_insufficient_sample(trades, design, window_days)
    if sample:
        findings.append(sample)
    findings.sort(key=lambda f: _SEVERITY_RANK.get(f.severity, 9))
    return findings


# ---------------------------------------------------------------------------
# Fetch + report
# ---------------------------------------------------------------------------

def fetch_live_trades(api_base: str, limit: int = 500) -> list[dict[str, Any]]:
    """Pull closed trades from the deployed API."""
    import httpx

    resp = httpx.get(f"{api_base.rstrip('/')}/trades", params={"limit": limit}, timeout=30)
    resp.raise_for_status()
    trades = resp.json().get("trades", [])
    return [t for t in trades if t.get("exit_time")]


def fetch_portfolio_value(api_base: str) -> float:
    """Pull current portfolio value for Sharpe normalization."""
    import httpx

    resp = httpx.get(f"{api_base.rstrip('/')}/portfolio/summary", timeout=30)
    resp.raise_for_status()
    return float(resp.json().get("portfolio_value", 0.0))


def build_report(
    trades: list[dict[str, Any]],
    portfolio_value: float,
    findings: list[Finding],
    window_days: int,
) -> dict[str, Any]:
    """Assemble the full JSON report."""
    metrics = core_metrics(trades)
    risk = daily_sharpe(trades, portfolio_value)
    needs_review = any(f.severity in ("CRITICAL", "HIGH") for f in findings)
    return {
        "agent": AGENT_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window_days": window_days,
        "portfolio_value": portfolio_value,
        "metrics": {**metrics, **risk},
        "exit_reason_attribution": attribution_by(trades, "exit_reason"),
        "ticker_attribution": attribution_by(trades, "ticker"),
        "findings": [f.to_dict() for f in findings],
        "needs_review": needs_review,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable version of the report."""
    m = report["metrics"]
    lines = [
        f"# {report['agent']} — {report['timestamp'][:10]}",
        "",
        f"Portfolio value: ${report['portfolio_value']:,.2f} | "
        f"Window: last {report['window_days']} days | "
        f"Trades: {m['n_trades']}",
        "",
        "## Headline metrics",
        "",
        f"- Total PnL: **{m['total_pnl']:+.2f}** "
        f"(win rate {m['win_rate']:.1%}, PF {m['profit_factor']}, "
        f"Sharpe {m.get('sharpe')}, max DD {m.get('max_drawdown_pct')}%)",
        f"- Avg win {m['avg_win']:+.2f} vs avg loss {m['avg_loss']:+.2f} "
        f"(expectancy {m['expectancy_per_trade']:+.2f}/trade)",
        f"- Median hold: {m['median_hold_minutes']:.0f} min",
        "",
        "## Root causes (ranked)",
        "",
    ]
    for i, f in enumerate(report["findings"], 1):
        lines += [
            f"### {i}. [{f['severity']}] {f['title']}",
            "",
            f"**Evidence:** {f['evidence']}",
            "",
            f"**Fix:** {f['recommendation']}",
            "",
        ]
    lines += ["## Exit-reason attribution", ""]
    for reason, s in report["exit_reason_attribution"].items():
        lines.append(
            f"- `{reason}`: n={s['n']}, total {s['total_pnl']:+.2f}, "
            f"win rate {s['win_rate']:.0%}"
        )
    lines += ["", "## Ticker attribution (worst first)", ""]
    for ticker, s in report["ticker_attribution"].items():
        lines.append(f"- {ticker}: n={s['n']}, total {s['total_pnl']:+.2f}")
    lines += [
        "",
        "---",
        "*Read-only analysis. No config or model was modified. "
        "Suggestions written to config/staging/profitability_suggestions.json.*",
        "",
    ]
    return "\n".join(lines)


def run(api_base: str, window_days: int = 60) -> dict[str, Any]:
    """Full agent cycle: fetch → diagnose → write reports + staging."""
    logger.info("profitability_agent_start", api=api_base)
    trades = fetch_live_trades(api_base)
    portfolio_value = fetch_portfolio_value(api_base)
    findings = diagnose(trades, window_days=window_days)
    report = build_report(trades, portfolio_value, findings, window_days)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    json_path = REPORT_DIR / f"{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    md_path = REPORT_DIR / f"{stamp}.md"
    md_path.write_text(render_markdown(report))

    STAGING_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAGING_PATH.write_text(json.dumps(
        {
            "agent": AGENT_NAME,
            "timestamp": report["timestamp"],
            "needs_review": report["needs_review"],
            "proposed_actions": [
                {"code": f["code"], "severity": f["severity"], "action": f["recommendation"]}
                for f in report["findings"]
            ],
        },
        indent=2,
    ))

    if report["needs_review"]:
        logger.warning(
            "profitability_escalation",
            n_findings=len(findings),
            top=findings[0].code if findings else None,
        )
    logger.info("profitability_agent_done", report=str(json_path))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=AGENT_NAME)
    parser.add_argument(
        "--api",
        default="https://stockbot-production-cbde.up.railway.app",
        help="Base URL of the deployed StockBot API",
    )
    parser.add_argument("--days", type=int, default=60, help="Lookback window (days)")
    args = parser.parse_args()
    report = run(args.api, window_days=args.days)
    print(render_markdown(report))


if __name__ == "__main__":
    main()
