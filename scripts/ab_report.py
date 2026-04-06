"""A/B Pipeline Comparison Report.

Queries the trades table grouped by pipeline_id and computes per-pipeline
performance metrics: Sharpe, drawdown, win rate, profit factor, avg hold time.

Usage:
    python scripts/ab_report.py                     # last 7 days
    python scripts/ab_report.py --days 30           # last 30 days
    python scripts/ab_report.py --json              # output JSON to reports/ab/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def fetch_trades(
    pipeline_id: str | None = None,
    days: int = 7,
) -> list[dict]:
    """Fetch closed trades from the database."""
    from sqlalchemy import select, and_
    from src.data.db import Trade, get_session_factory

    sf = get_session_factory()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    async with sf() as session:
        query = (
            select(Trade)
            .where(
                and_(
                    Trade.entry_time >= cutoff,
                    Trade.exit_time.isnot(None),  # only closed trades
                )
            )
            .order_by(Trade.entry_time)
        )
        if pipeline_id:
            query = query.where(Trade.pipeline_id == pipeline_id)

        rows = await session.execute(query)
        trades = rows.scalars().all()

        return [
            {
                "id": t.id,
                "pipeline_id": t.pipeline_id or "pipeline_a",
                "ticker": t.ticker,
                "side": t.side,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason,
                "ensemble_signal": t.ensemble_signal,
            }
            for t in trades
        ]


def compute_metrics(trades: list[dict]) -> dict:
    """Compute performance metrics for a set of closed trades."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "total_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "avg_hold_minutes": 0.0,
            "calmar_ratio": 0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
            "exit_reasons": {},
        }

    pnls = [t["pnl"] or 0.0 for t in trades]
    pnl_pcts = [t["pnl_pct"] or 0.0 for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    # Sharpe ratio (annualized, assuming 1-minute bars → ~252 trading days)
    pnl_arr = np.array(pnl_pcts)
    mean_ret = float(np.mean(pnl_arr))
    std_ret = float(np.std(pnl_arr)) if len(pnl_arr) > 1 else 1e-6
    # Approximate annualization: sqrt(trades_per_year)
    # With ~8 trades/day × 252 days ≈ 2016 trades/year
    trades_per_year = max(len(trades) / max((trades[-1].get("exit_time", "") > trades[0].get("entry_time", "")), 1) * 365, 252)
    sharpe = (mean_ret / max(std_ret, 1e-8)) * np.sqrt(min(trades_per_year, 2016))

    # Max drawdown (cumulative PnL curve)
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    initial_value = 100_000.0  # approximate for percentage
    max_dd_pct = max_dd / max(initial_value, 1.0)

    # Profit factor
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 1e-6
    profit_factor = gross_profit / max(gross_loss, 1e-6)

    # Average hold time
    hold_minutes = []
    for t in trades:
        if t["entry_time"] and t["exit_time"]:
            try:
                entry = datetime.fromisoformat(t["entry_time"])
                exit_ = datetime.fromisoformat(t["exit_time"])
                hold_minutes.append((exit_ - entry).total_seconds() / 60)
            except Exception:
                pass
    avg_hold = float(np.mean(hold_minutes)) if hold_minutes else 0.0

    # Exit reasons breakdown
    exit_reasons: dict[str, int] = {}
    for t in trades:
        reason = t.get("exit_reason", "unknown") or "unknown"
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # Calmar ratio
    calmar = (mean_ret * len(trades)) / max(abs(max_dd_pct), 1e-8) if max_dd_pct != 0 else 0.0

    return {
        "total_trades": len(trades),
        "win_rate": len(wins) / max(len(trades), 1),
        "avg_pnl_pct": mean_ret,
        "total_pnl": sum(pnls),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": max_dd_pct,
        "profit_factor": profit_factor,
        "avg_hold_minutes": avg_hold,
        "calmar_ratio": float(calmar),
        "best_trade_pct": max(pnl_pcts) if pnl_pcts else 0.0,
        "worst_trade_pct": min(pnl_pcts) if pnl_pcts else 0.0,
        "exit_reasons": exit_reasons,
    }


def format_console_report(
    metrics_a: dict,
    metrics_b: dict,
    days: int,
) -> str:
    """Format side-by-side console comparison."""
    lines = []
    lines.append("=" * 62)
    lines.append("        A/B Pipeline Comparison Report")
    lines.append(f"        Period: last {days} days")
    lines.append("=" * 62)
    lines.append("")
    lines.append(f"{'Metric':<25} {'Pipeline A (ML)':<20} {'Pipeline B (Rules)':<20}")
    lines.append("-" * 62)

    rows = [
        ("Trades", f"{metrics_a['total_trades']}", f"{metrics_b['total_trades']}"),
        ("Win Rate", f"{metrics_a['win_rate']:.1%}", f"{metrics_b['win_rate']:.1%}"),
        ("Sharpe Ratio", f"{metrics_a['sharpe_ratio']:.2f}", f"{metrics_b['sharpe_ratio']:.2f}"),
        ("Max Drawdown", f"{metrics_a['max_drawdown_pct']:.2%}", f"{metrics_b['max_drawdown_pct']:.2%}"),
        ("Total PnL", f"${metrics_a['total_pnl']:,.2f}", f"${metrics_b['total_pnl']:,.2f}"),
        ("Avg PnL %", f"{metrics_a['avg_pnl_pct']:.4%}", f"{metrics_b['avg_pnl_pct']:.4%}"),
        ("Profit Factor", f"{metrics_a['profit_factor']:.2f}", f"{metrics_b['profit_factor']:.2f}"),
        ("Avg Hold (min)", f"{metrics_a['avg_hold_minutes']:.1f}", f"{metrics_b['avg_hold_minutes']:.1f}"),
        ("Calmar Ratio", f"{metrics_a['calmar_ratio']:.2f}", f"{metrics_b['calmar_ratio']:.2f}"),
        ("Best Trade", f"{metrics_a['best_trade_pct']:.4%}", f"{metrics_b['best_trade_pct']:.4%}"),
        ("Worst Trade", f"{metrics_a['worst_trade_pct']:.4%}", f"{metrics_b['worst_trade_pct']:.4%}"),
    ]

    for label, val_a, val_b in rows:
        lines.append(f"{label:<25} {val_a:<20} {val_b:<20}")

    lines.append("")

    # Exit reason breakdown
    all_reasons = set(list(metrics_a.get("exit_reasons", {}).keys()) + list(metrics_b.get("exit_reasons", {}).keys()))
    if all_reasons:
        lines.append("Exit Reasons:")
        lines.append(f"{'  Reason':<25} {'Pipeline A':<20} {'Pipeline B':<20}")
        lines.append("-" * 62)
        for reason in sorted(all_reasons):
            a_count = metrics_a.get("exit_reasons", {}).get(reason, 0)
            b_count = metrics_b.get("exit_reasons", {}).get(reason, 0)
            lines.append(f"  {reason:<23} {a_count:<20} {b_count:<20}")

    lines.append("")

    # Winner determination
    a_sharpe = metrics_a["sharpe_ratio"]
    b_sharpe = metrics_b["sharpe_ratio"]
    if metrics_a["total_trades"] < 10 or metrics_b["total_trades"] < 10:
        lines.append("** Insufficient trades for statistical significance **")
    elif abs(a_sharpe - b_sharpe) < 0.3:
        lines.append(f"** No clear winner yet (Sharpe diff = {abs(a_sharpe - b_sharpe):.2f}) **")
    elif a_sharpe > b_sharpe:
        lines.append(f"** Pipeline A (ML) leads by Sharpe {a_sharpe - b_sharpe:+.2f} **")
    else:
        lines.append(f"** Pipeline B (Rules) leads by Sharpe {b_sharpe - a_sharpe:+.2f} **")

    lines.append("=" * 62)
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="A/B Pipeline Comparison Report")
    parser.add_argument("--days", type=int, default=7, help="Lookback period in days")
    parser.add_argument("--json", action="store_true", help="Output JSON to reports/ab/")
    args = parser.parse_args()

    from src.data.db import init_db
    await init_db()

    # Fetch trades for both pipelines
    all_trades = await fetch_trades(days=args.days)

    trades_a = [t for t in all_trades if (t.get("pipeline_id") or "pipeline_a") == "pipeline_a"]
    trades_b = [t for t in all_trades if t.get("pipeline_id") == "pipeline_b"]

    metrics_a = compute_metrics(trades_a)
    metrics_b = compute_metrics(trades_b)

    # Console output
    report = format_console_report(metrics_a, metrics_b, args.days)
    print(report)

    # JSON output
    if args.json:
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period_days": args.days,
            "pipeline_a": {**metrics_a, "total_trades_in_period": len(trades_a)},
            "pipeline_b": {**metrics_b, "total_trades_in_period": len(trades_b)},
        }
        out_path = Path("reports/ab") / f"ab_report_{date.today().isoformat()}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nJSON report saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
