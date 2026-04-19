"""Daily Performance Report — Pipeline A vs Pipeline B.

Queries the trades table grouped by pipeline_id, generates matplotlib plots
comparing performance, and prints a text summary to the terminal.

Plots saved to reports/performance/YYYY-MM-DD/:
  1. cumulative_pnl.png          — Cumulative P&L curve (A, B, Combined)
  2. daily_pnl.png               — Side-by-side daily P&L bar chart
  3. winrate_pf.png              — Win rate + Profit Factor comparison
  4. drawdown.png                — Running drawdown % per pipeline
  5. per_ticker_pnl.png          — Per-ticker P&L bar chart by pipeline
  6. trade_distribution.png      — Histogram of P&L % per trade
  7. dashboard.png               — Combined grid of all charts

Usage:
    python scripts/daily_performance.py                     # last 7 days
    python scripts/daily_performance.py --days 30           # last 30 days
    python scripts/daily_performance.py --db-url "postgresql+asyncpg://..."
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

async def fetch_trades(days: int, db_url: str | None = None) -> list[dict]:
    """Fetch closed trades from the database for the given lookback period."""
    # Override DATABASE_URL before importing settings if provided
    if db_url:
        os.environ["DATABASE_URL"] = db_url

    from sqlalchemy import select, and_
    from src.data.db import Trade, get_session_factory, init_db

    await init_db()

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
        rows = await session.execute(query)
        trades = rows.scalars().all()

        return [
            {
                "pipeline_id": t.pipeline_id or "pipeline_a",
                "ticker": t.ticker,
                "side": t.side,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "pnl": t.pnl or 0.0,
                "pnl_pct": t.pnl_pct or 0.0,
                "exit_reason": t.exit_reason or "unknown",
                "ensemble_signal": t.ensemble_signal,
            }
            for t in trades
        ]


def split_by_pipeline(trades: list[dict]) -> dict[str, list[dict]]:
    """Split trades into per-pipeline lists. Always returns both keys."""
    result: dict[str, list[dict]] = {"pipeline_a": [], "pipeline_b": []}
    for t in trades:
        pid = t["pipeline_id"]
        if pid not in result:
            result[pid] = []
        result[pid].append(t)
    return result


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(trades: list[dict]) -> dict:
    """Compute performance metrics for a set of closed trades."""
    empty = {
        "total_trades": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "avg_pnl_pct": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "profit_factor": 0.0,
        "avg_hold_minutes": 0.0,
        "best_trade_pct": 0.0,
        "worst_trade_pct": 0.0,
    }
    if not trades:
        return empty

    pnls = np.array([t["pnl"] for t in trades])
    pnl_pcts = np.array([t["pnl_pct"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    # Sharpe (annualized, assuming ~2016 trades/year for intraday)
    mean_ret = float(np.mean(pnl_pcts))
    std_ret = float(np.std(pnl_pcts)) if len(pnl_pcts) > 1 else 1e-6
    trades_per_year = min(2016, max(len(trades), 1))
    sharpe = (mean_ret / max(std_ret, 1e-8)) * np.sqrt(trades_per_year)

    # Max drawdown (from cumulative P&L curve)
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    portfolio_value = 100_000.0
    max_dd_pct = max_dd / portfolio_value

    # Profit factor
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-6
    profit_factor = gross_profit / max(gross_loss, 1e-6)

    # Average hold time
    hold_mins = []
    for t in trades:
        if t["entry_time"] and t["exit_time"]:
            try:
                dt = (t["exit_time"] - t["entry_time"]).total_seconds() / 60
                hold_mins.append(dt)
            except Exception:
                pass

    return {
        "total_trades": len(trades),
        "win_rate": float(len(wins) / max(len(trades), 1)),
        "total_pnl": float(np.sum(pnls)),
        "avg_pnl_pct": mean_ret,
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": max_dd_pct,
        "profit_factor": profit_factor,
        "avg_hold_minutes": float(np.mean(hold_mins)) if hold_mins else 0.0,
        "best_trade_pct": float(np.max(pnl_pcts)),
        "worst_trade_pct": float(np.min(pnl_pcts)),
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(
    metrics_a: dict,
    metrics_b: dict,
    metrics_all: dict,
    days: int,
) -> None:
    """Print a formatted text summary table to terminal."""
    print()
    print("=" * 78)
    print("         Daily Performance Report — Pipeline A vs Pipeline B")
    print(f"         Period: last {days} days  |  Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    print("=" * 78)
    print()
    print(f"{'Metric':<22} {'Pipeline A (ML)':<20} {'Pipeline B (Rules)':<20} {'Combined':<16}")
    print("-" * 78)

    rows = [
        ("Total Trades",
         f"{metrics_a['total_trades']}",
         f"{metrics_b['total_trades']}",
         f"{metrics_all['total_trades']}"),
        ("Win Rate",
         f"{metrics_a['win_rate']:.1%}",
         f"{metrics_b['win_rate']:.1%}",
         f"{metrics_all['win_rate']:.1%}"),
        ("Total P&L",
         f"${metrics_a['total_pnl']:,.2f}",
         f"${metrics_b['total_pnl']:,.2f}",
         f"${metrics_all['total_pnl']:,.2f}"),
        ("Avg P&L %",
         f"{metrics_a['avg_pnl_pct']:.4%}",
         f"{metrics_b['avg_pnl_pct']:.4%}",
         f"{metrics_all['avg_pnl_pct']:.4%}"),
        ("Sharpe Ratio",
         f"{metrics_a['sharpe_ratio']:.2f}",
         f"{metrics_b['sharpe_ratio']:.2f}",
         f"{metrics_all['sharpe_ratio']:.2f}"),
        ("Max Drawdown",
         f"{metrics_a['max_drawdown_pct']:.2%}",
         f"{metrics_b['max_drawdown_pct']:.2%}",
         f"{metrics_all['max_drawdown_pct']:.2%}"),
        ("Profit Factor",
         f"{metrics_a['profit_factor']:.2f}",
         f"{metrics_b['profit_factor']:.2f}",
         f"{metrics_all['profit_factor']:.2f}"),
        ("Avg Hold (min)",
         f"{metrics_a['avg_hold_minutes']:.1f}",
         f"{metrics_b['avg_hold_minutes']:.1f}",
         f"{metrics_all['avg_hold_minutes']:.1f}"),
        ("Best Trade %",
         f"{metrics_a['best_trade_pct']:.4%}",
         f"{metrics_b['best_trade_pct']:.4%}",
         f"{metrics_all['best_trade_pct']:.4%}"),
        ("Worst Trade %",
         f"{metrics_a['worst_trade_pct']:.4%}",
         f"{metrics_b['worst_trade_pct']:.4%}",
         f"{metrics_all['worst_trade_pct']:.4%}"),
    ]

    for label, va, vb, vc in rows:
        print(f"{label:<22} {va:<20} {vb:<20} {vc:<16}")

    print()
    # Winner determination
    sa = metrics_a["sharpe_ratio"]
    sb = metrics_b["sharpe_ratio"]
    min_trades = min(metrics_a["total_trades"], metrics_b["total_trades"])
    if min_trades < 5:
        print("** Insufficient trades in one or both pipelines for comparison **")
    elif abs(sa - sb) < 0.3:
        print(f"** No clear winner yet (Sharpe diff = {abs(sa - sb):.2f}) **")
    elif sa > sb:
        print(f"** Pipeline A (ML) leads — Sharpe {sa:.2f} vs {sb:.2f} (+{sa - sb:.2f}) **")
    else:
        print(f"** Pipeline B (Rules) leads — Sharpe {sb:.2f} vs {sa:.2f} (+{sb - sa:.2f}) **")

    print("=" * 78)
    print()


# ---------------------------------------------------------------------------
# Helper: build daily P&L DataFrame
# ---------------------------------------------------------------------------

def _build_daily_pnl(trades: list[dict]) -> pd.DataFrame:
    """Return a DataFrame with columns [date, pnl] from trade exit_times."""
    if not trades:
        return pd.DataFrame(columns=["date", "pnl"])

    records = []
    for t in trades:
        exit_dt = t["exit_time"]
        if exit_dt is None:
            continue
        d = exit_dt.date() if hasattr(exit_dt, "date") else exit_dt
        records.append({"date": d, "pnl": t["pnl"]})

    if not records:
        return pd.DataFrame(columns=["date", "pnl"])

    df = pd.DataFrame(records)
    return df.groupby("date")["pnl"].sum().reset_index()


def _build_cum_pnl_series(trades: list[dict]) -> tuple[list[datetime], list[float]]:
    """Return sorted (times, cumulative_pnl) from a trade list."""
    if not trades:
        return [], []
    sorted_trades = sorted(trades, key=lambda t: t["exit_time"] or t["entry_time"])
    times = []
    cum = 0.0
    vals = []
    for t in sorted_trades:
        cum += t["pnl"]
        times.append(t["exit_time"] or t["entry_time"])
        vals.append(cum)
    return times, vals


def _build_drawdown_series(trades: list[dict]) -> tuple[list[datetime], list[float]]:
    """Return sorted (times, drawdown_pct) from a trade list."""
    if not trades:
        return [], []
    sorted_trades = sorted(trades, key=lambda t: t["exit_time"] or t["entry_time"])
    portfolio_value = 100_000.0
    times = []
    cum = 0.0
    peak = 0.0
    dd_pcts = []
    for t in sorted_trades:
        cum += t["pnl"]
        if cum > peak:
            peak = cum
        dd = (cum - peak) / portfolio_value * 100  # as percentage
        times.append(t["exit_time"] or t["entry_time"])
        dd_pcts.append(dd)
    return times, dd_pcts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "pipeline_a": "#00d4aa",    # teal
    "pipeline_b": "#ff6b6b",    # coral
    "combined":   "#ffd93d",    # gold
}
LABELS = {
    "pipeline_a": "Pipeline A (ML)",
    "pipeline_b": "Pipeline B (Rules)",
    "combined":   "Combined",
}


def _apply_style() -> None:
    """Apply dark background style with consistent aesthetics."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "axes.grid": True,
        "grid.color": "#2a2a4a",
        "grid.alpha": 0.5,
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "text.color": "#e0e0e0",
        "legend.facecolor": "#16213e",
        "legend.edgecolor": "#444",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })


def plot_cumulative_pnl(
    pipe_trades: dict[str, list[dict]],
    all_trades: list[dict],
    out_dir: Path,
) -> None:
    """Plot 1: Cumulative P&L over time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for pid in ["pipeline_a", "pipeline_b"]:
        times, vals = _build_cum_pnl_series(pipe_trades.get(pid, []))
        if times:
            ax.plot(times, vals, color=COLORS[pid], label=LABELS[pid],
                    linewidth=2, alpha=0.9)

    # Combined
    times, vals = _build_cum_pnl_series(all_trades)
    if times:
        ax.plot(times, vals, color=COLORS["combined"], label=LABELS["combined"],
                linewidth=2, linestyle="--", alpha=0.7)

    ax.axhline(y=0, color="#666", linewidth=0.8, linestyle="-")
    ax.set_title("Cumulative P&L Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_pnl.png", dpi=150)
    plt.close(fig)


def plot_daily_pnl(
    pipe_trades: dict[str, list[dict]],
    out_dir: Path,
) -> None:
    """Plot 2: Side-by-side daily P&L bar chart."""
    daily_a = _build_daily_pnl(pipe_trades.get("pipeline_a", []))
    daily_b = _build_daily_pnl(pipe_trades.get("pipeline_b", []))

    # Merge on date to get aligned bars
    all_dates = set()
    if not daily_a.empty:
        all_dates.update(daily_a["date"].tolist())
    if not daily_b.empty:
        all_dates.update(daily_b["date"].tolist())

    if not all_dates:
        # Create empty plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Daily P&L by Pipeline")
        ax.text(0.5, 0.5, "No trades in period", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#888")
        fig.tight_layout()
        fig.savefig(out_dir / "daily_pnl.png", dpi=150)
        plt.close(fig)
        return

    all_dates = sorted(all_dates)
    a_map = dict(zip(daily_a["date"], daily_a["pnl"])) if not daily_a.empty else {}
    b_map = dict(zip(daily_b["date"], daily_b["pnl"])) if not daily_b.empty else {}

    a_vals = [a_map.get(d, 0.0) for d in all_dates]
    b_vals = [b_map.get(d, 0.0) for d in all_dates]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(all_dates))
    width = 0.35

    bars_a = ax.bar(x - width / 2, a_vals, width, label=LABELS["pipeline_a"],
                    color=COLORS["pipeline_a"], alpha=0.85, edgecolor="none")
    bars_b = ax.bar(x + width / 2, b_vals, width, label=LABELS["pipeline_b"],
                    color=COLORS["pipeline_b"], alpha=0.85, edgecolor="none")

    ax.axhline(y=0, color="#666", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%m-%d") for d in all_dates], rotation=45, ha="right")
    ax.set_title("Daily P&L by Pipeline")
    ax.set_ylabel("P&L ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "daily_pnl.png", dpi=150)
    plt.close(fig)


def plot_winrate_pf(
    metrics_a: dict,
    metrics_b: dict,
    out_dir: Path,
) -> None:
    """Plot 3: Win Rate + Profit Factor comparison bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Win Rate
    pipelines = ["Pipeline A\n(ML)", "Pipeline B\n(Rules)"]
    win_rates = [metrics_a["win_rate"] * 100, metrics_b["win_rate"] * 100]
    colors = [COLORS["pipeline_a"], COLORS["pipeline_b"]]

    bars = ax1.bar(pipelines, win_rates, color=colors, alpha=0.85, width=0.5, edgecolor="none")
    ax1.axhline(y=50, color="#888", linewidth=1, linestyle="--", label="50% baseline")
    ax1.set_ylabel("Win Rate (%)")
    ax1.set_title("Win Rate Comparison")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right", fontsize=9)
    for bar, val in zip(bars, win_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

    # Profit Factor
    pf_vals = [metrics_a["profit_factor"], metrics_b["profit_factor"]]
    bars = ax2.bar(pipelines, pf_vals, color=colors, alpha=0.85, width=0.5, edgecolor="none")
    ax2.axhline(y=1.0, color="#888", linewidth=1, linestyle="--", label="Breakeven (1.0)")
    ax2.set_ylabel("Profit Factor")
    ax2.set_title("Profit Factor Comparison")
    ax2.legend(loc="upper right", fontsize=9)
    for bar, val in zip(bars, pf_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    fig.suptitle("Win Rate & Profit Factor", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "winrate_pf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(
    pipe_trades: dict[str, list[dict]],
    out_dir: Path,
) -> None:
    """Plot 4: Running drawdown % for each pipeline."""
    fig, ax = plt.subplots(figsize=(12, 5))

    has_data = False
    for pid in ["pipeline_a", "pipeline_b"]:
        times, dd_pcts = _build_drawdown_series(pipe_trades.get(pid, []))
        if times:
            ax.fill_between(times, dd_pcts, 0, alpha=0.3, color=COLORS[pid])
            ax.plot(times, dd_pcts, color=COLORS[pid], label=LABELS[pid],
                    linewidth=1.5)
            has_data = True

    if not has_data:
        ax.text(0.5, 0.5, "No trades in period", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#888")

    ax.axhline(y=0, color="#666", linewidth=0.8)
    ax.set_title("Running Drawdown by Pipeline")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "drawdown.png", dpi=150)
    plt.close(fig)


def plot_per_ticker_pnl(
    pipe_trades: dict[str, list[dict]],
    out_dir: Path,
) -> None:
    """Plot 5: Per-ticker P&L bar chart, grouped by pipeline."""
    # Aggregate per ticker per pipeline
    ticker_pnl: dict[str, dict[str, float]] = defaultdict(lambda: {"pipeline_a": 0.0, "pipeline_b": 0.0})
    for pid, trades in pipe_trades.items():
        for t in trades:
            ticker_pnl[t["ticker"]][pid] += t["pnl"]

    if not ticker_pnl:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.set_title("P&L by Ticker and Pipeline")
        ax.text(0.5, 0.5, "No trades in period", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#888")
        fig.tight_layout()
        fig.savefig(out_dir / "per_ticker_pnl.png", dpi=150)
        plt.close(fig)
        return

    # Sort tickers by combined P&L (descending)
    tickers = sorted(ticker_pnl.keys(),
                     key=lambda tk: ticker_pnl[tk]["pipeline_a"] + ticker_pnl[tk]["pipeline_b"],
                     reverse=True)

    # Limit to top 20 tickers by absolute P&L if there are too many
    if len(tickers) > 20:
        tickers = sorted(tickers,
                         key=lambda tk: abs(ticker_pnl[tk]["pipeline_a"]) + abs(ticker_pnl[tk]["pipeline_b"]),
                         reverse=True)[:20]
        tickers = sorted(tickers,
                         key=lambda tk: ticker_pnl[tk]["pipeline_a"] + ticker_pnl[tk]["pipeline_b"],
                         reverse=True)

    a_vals = [ticker_pnl[tk]["pipeline_a"] for tk in tickers]
    b_vals = [ticker_pnl[tk]["pipeline_b"] for tk in tickers]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(tickers))
    width = 0.35

    ax.bar(x - width / 2, a_vals, width, label=LABELS["pipeline_a"],
           color=COLORS["pipeline_a"], alpha=0.85, edgecolor="none")
    ax.bar(x + width / 2, b_vals, width, label=LABELS["pipeline_b"],
           color=COLORS["pipeline_b"], alpha=0.85, edgecolor="none")

    ax.axhline(y=0, color="#666", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=9)
    ax.set_title("P&L by Ticker and Pipeline")
    ax.set_ylabel("P&L ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_ticker_pnl.png", dpi=150)
    plt.close(fig)


def plot_trade_distribution(
    pipe_trades: dict[str, list[dict]],
    out_dir: Path,
) -> None:
    """Plot 6: Histogram of P&L % per trade for each pipeline."""
    fig, ax = plt.subplots(figsize=(12, 5))

    has_data = False
    for pid in ["pipeline_a", "pipeline_b"]:
        trades = pipe_trades.get(pid, [])
        if not trades:
            continue
        pnl_pcts = [t["pnl_pct"] * 100 for t in trades]  # convert to percentage
        ax.hist(pnl_pcts, bins=30, alpha=0.6, color=COLORS[pid],
                label=f"{LABELS[pid]} (n={len(trades)})", edgecolor="none")
        has_data = True

    if not has_data:
        ax.text(0.5, 0.5, "No trades in period", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#888")

    ax.axvline(x=0, color="#888", linewidth=1, linestyle="--")
    ax.set_title("Trade P&L Distribution")
    ax.set_xlabel("P&L per Trade (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "trade_distribution.png", dpi=150)
    plt.close(fig)


def plot_dashboard(
    pipe_trades: dict[str, list[dict]],
    all_trades: list[dict],
    metrics_a: dict,
    metrics_b: dict,
    days: int,
    out_dir: Path,
) -> None:
    """Plot 7: Combined 3x2 grid dashboard with all charts."""
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle(
        f"Daily Performance Dashboard -- Pipeline A vs B  |  Last {days} days  |  {date.today().isoformat()}",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # ── (0,0) Cumulative P&L ──
    ax = axes[0, 0]
    for pid in ["pipeline_a", "pipeline_b"]:
        times, vals = _build_cum_pnl_series(pipe_trades.get(pid, []))
        if times:
            ax.plot(times, vals, color=COLORS[pid], label=LABELS[pid], linewidth=1.5)
    times, vals = _build_cum_pnl_series(all_trades)
    if times:
        ax.plot(times, vals, color=COLORS["combined"], label=LABELS["combined"],
                linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axhline(y=0, color="#666", linewidth=0.5)
    ax.set_title("Cumulative P&L")
    ax.set_ylabel("P&L ($)")
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    # ── (0,1) Daily P&L ──
    ax = axes[0, 1]
    daily_a = _build_daily_pnl(pipe_trades.get("pipeline_a", []))
    daily_b = _build_daily_pnl(pipe_trades.get("pipeline_b", []))
    all_dates_set: set = set()
    if not daily_a.empty:
        all_dates_set.update(daily_a["date"].tolist())
    if not daily_b.empty:
        all_dates_set.update(daily_b["date"].tolist())
    if all_dates_set:
        all_dates_sorted = sorted(all_dates_set)
        a_map = dict(zip(daily_a["date"], daily_a["pnl"])) if not daily_a.empty else {}
        b_map = dict(zip(daily_b["date"], daily_b["pnl"])) if not daily_b.empty else {}
        x = np.arange(len(all_dates_sorted))
        w = 0.35
        ax.bar(x - w / 2, [a_map.get(d, 0) for d in all_dates_sorted], w,
               color=COLORS["pipeline_a"], label=LABELS["pipeline_a"], alpha=0.85)
        ax.bar(x + w / 2, [b_map.get(d, 0) for d in all_dates_sorted], w,
               color=COLORS["pipeline_b"], label=LABELS["pipeline_b"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([d.strftime("%m-%d") for d in all_dates_sorted],
                           rotation=45, ha="right", fontsize=7)
    else:
        ax.text(0.5, 0.5, "No trades", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#888")
    ax.axhline(y=0, color="#666", linewidth=0.5)
    ax.set_title("Daily P&L")
    ax.set_ylabel("P&L ($)")
    ax.legend(fontsize=8)

    # ── (1,0) Win Rate ──
    ax = axes[1, 0]
    pipelines_short = ["A (ML)", "B (Rules)"]
    wr = [metrics_a["win_rate"] * 100, metrics_b["win_rate"] * 100]
    colors_list = [COLORS["pipeline_a"], COLORS["pipeline_b"]]
    bars = ax.bar(pipelines_short, wr, color=colors_list, alpha=0.85, width=0.4)
    ax.axhline(y=50, color="#888", linewidth=1, linestyle="--")
    ax.set_title("Win Rate (%)")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, wr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ── (1,1) Profit Factor ──
    ax = axes[1, 1]
    pf = [metrics_a["profit_factor"], metrics_b["profit_factor"]]
    bars = ax.bar(pipelines_short, pf, color=colors_list, alpha=0.85, width=0.4)
    ax.axhline(y=1.0, color="#888", linewidth=1, linestyle="--")
    ax.set_title("Profit Factor")
    for bar, val in zip(bars, pf):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ── (2,0) Drawdown ──
    ax = axes[2, 0]
    has_dd = False
    for pid in ["pipeline_a", "pipeline_b"]:
        times, dd = _build_drawdown_series(pipe_trades.get(pid, []))
        if times:
            ax.fill_between(times, dd, 0, alpha=0.25, color=COLORS[pid])
            ax.plot(times, dd, color=COLORS[pid], label=LABELS[pid], linewidth=1.2)
            has_dd = True
    if not has_dd:
        ax.text(0.5, 0.5, "No trades", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#888")
    ax.axhline(y=0, color="#666", linewidth=0.5)
    ax.set_title("Running Drawdown (%)")
    ax.set_ylabel("Drawdown %")
    ax.legend(fontsize=8, loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # ── (2,1) Trade Distribution ──
    ax = axes[2, 1]
    has_hist = False
    for pid in ["pipeline_a", "pipeline_b"]:
        trades = pipe_trades.get(pid, [])
        if trades:
            pnl_pcts = [t["pnl_pct"] * 100 for t in trades]
            ax.hist(pnl_pcts, bins=25, alpha=0.55, color=COLORS[pid],
                    label=f"{LABELS[pid]} (n={len(trades)})", edgecolor="none")
            has_hist = True
    if not has_hist:
        ax.text(0.5, 0.5, "No trades", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#888")
    ax.axvline(x=0, color="#888", linewidth=1, linestyle="--")
    ax.set_title("Trade P&L Distribution (%)")
    ax.set_xlabel("P&L %")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily Performance Report — Pipeline A vs Pipeline B",
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Lookback period in trading days (default: 7)",
    )
    parser.add_argument(
        "--db-url", type=str, default=None,
        help="Database URL (default: reads DATABASE_URL env var)",
    )
    args = parser.parse_args()

    # ── Fetch trades ──
    print(f"Fetching closed trades from the last {args.days} days ...")
    all_trades = await fetch_trades(days=args.days, db_url=args.db_url)
    print(f"  Found {len(all_trades)} closed trades.")

    if not all_trades:
        print("\nNo closed trades in the specified period. Nothing to plot.")
        return

    # ── Split by pipeline ──
    pipe_trades = split_by_pipeline(all_trades)
    for pid, trades in pipe_trades.items():
        print(f"  {LABELS.get(pid, pid)}: {len(trades)} trades")

    # ── Compute metrics ──
    metrics_a = compute_metrics(pipe_trades.get("pipeline_a", []))
    metrics_b = compute_metrics(pipe_trades.get("pipeline_b", []))
    metrics_all = compute_metrics(all_trades)

    # ── Print console summary ──
    print_summary(metrics_a, metrics_b, metrics_all, args.days)

    # ── Prepare output directory ──
    out_dir = Path("reports/performance") / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to {out_dir}/ ...")

    # ── Apply matplotlib style ──
    _apply_style()

    # ── Generate individual plots ──
    plot_cumulative_pnl(pipe_trades, all_trades, out_dir)
    print("  [1/7] cumulative_pnl.png")

    plot_daily_pnl(pipe_trades, out_dir)
    print("  [2/7] daily_pnl.png")

    plot_winrate_pf(metrics_a, metrics_b, out_dir)
    print("  [3/7] winrate_pf.png")

    plot_drawdown(pipe_trades, out_dir)
    print("  [4/7] drawdown.png")

    plot_per_ticker_pnl(pipe_trades, out_dir)
    print("  [5/7] per_ticker_pnl.png")

    plot_trade_distribution(pipe_trades, out_dir)
    print("  [6/7] trade_distribution.png")

    # ── Generate combined dashboard ──
    plot_dashboard(pipe_trades, all_trades, metrics_a, metrics_b, args.days, out_dir)
    print("  [7/7] dashboard.png")

    print(f"\nDone. All plots saved to {out_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
