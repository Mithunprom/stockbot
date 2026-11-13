"""Daily portfolio backtester for hybrid-graph signals.

Fill rule: signal at timestamp t uses features already shifted to t-1, so
the order is placed at t's open (standard overnight decision). Intrabar
stops/targets use that day's high/low. If both would hit, the stop is
assumed to fill first (pessimistic).

Costs: ``cost_bps`` one-way on every fill. Circuit breakers match
``src.risk.circuit_breakers`` defaults and cannot be disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.graph.catalog import StrategySpec


@dataclass
class BacktestResult:
    """Walk-forward-safe performance snapshot. Never report PnL alone."""

    label: str
    start: str
    end: str
    total_return_pct: float
    sharpe: float
    sortino: float
    max_dd_pct: float
    profit_factor: float
    win_rate: float
    n_trades: int
    avg_hold_days: float
    avg_win_pct: float
    avg_loss_pct: float
    calmar: float
    score: float
    n_days: int
    halted: bool = False
    exit_reasons: dict[str, int] = field(default_factory=dict)
    equity: list[float] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "total_return_pct": self.total_return_pct,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_dd_pct": self.max_dd_pct,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "n_trades": self.n_trades,
            "avg_hold_days": self.avg_hold_days,
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "calmar": self.calmar,
            "score": self.score,
            "n_days": self.n_days,
            "halted": self.halted,
            "exit_reasons": self.exit_reasons,
            "notes": self.notes,
        }
        return d


def composite_score(
    sharpe: float,
    max_dd_pct: float,
    profit_factor: float,
    win_rate: float,
    n_trades: int,
    total_return_pct: float,
) -> float:
    """Single ranking number. Sharpe-led, with hard-KPI penalties.

    Live promotion targets (PROJECT.md): Sharpe ≥ 1.5, DD ≤ 8%, PF ≥ 1.4,
    win rate ≥ 52%. Search uses the same shape so the winner is not a
    high-PnL, high-DD lottery ticket.
    """
    if n_trades < 8 and n_trades != 0:
        # Buy-hold reports 0 trades; don't kill it here — caller special-cases.
        sample_pen = 0.6
    else:
        sample_pen = 0.0
    dd_pen = 0.0
    if max_dd_pct > 8.0:
        dd_pen = 0.35 * (max_dd_pct - 8.0)
    pf_pen = max(0.0, 1.4 - (profit_factor if np.isfinite(profit_factor) else 0.0)) * 0.4
    wr_pen = max(0.0, 0.52 - win_rate) * 2.0
    ret_bonus = 0.01 * np.tanh(total_return_pct / 30.0)
    return float(sharpe - dd_pen - pf_pen - wr_pen - sample_pen + ret_bonus)


def _metrics(equity: list[float], trades: list[dict], label: str, start: str, end: str, halted: bool) -> BacktestResult:
    eq = pd.Series(equity, dtype=float)
    if len(eq) < 3 or float(eq.iloc[0]) <= 0:
        return BacktestResult(
            label=label, start=start, end=end,
            total_return_pct=0.0, sharpe=0.0, sortino=0.0, max_dd_pct=0.0,
            profit_factor=0.0, win_rate=0.0, n_trades=0, avg_hold_days=0.0,
            avg_win_pct=0.0, avg_loss_pct=0.0, calmar=0.0, score=-9.0,
            n_days=len(eq), halted=halted, notes="insufficient equity curve",
        )
    rets = eq.pct_change().dropna()
    vol = float(rets.std())
    sharpe = float(rets.mean() / vol * np.sqrt(252)) if vol > 0 else 0.0
    downside = rets[rets < 0]
    dvol = float(downside.std()) if len(downside) else 0.0
    sortino = float(rets.mean() / dvol * np.sqrt(252)) if dvol > 0 else 0.0
    dd = float(((eq.cummax() - eq) / eq.cummax()).max())
    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = max(len(eq) / 252.0, 1e-6)
    calmar = float((total_ret / years) / dd) if dd > 1e-9 else 0.0

    tdf = pd.DataFrame(trades)
    if tdf.empty or "pnl" not in tdf.columns:
        wr = 0.0
        pf = 0.0
        n_tr = 0
        avg_hold = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        reasons: dict[str, int] = {}
    else:
        wins = tdf[tdf.pnl > 0].pnl.sum()
        losses = -tdf[tdf.pnl < 0].pnl.sum()
        pf = float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0)
        wr = float((tdf.pnl > 0).mean())
        n_tr = int(len(tdf))
        avg_hold = float(tdf["days"].mean()) if "days" in tdf else 0.0
        avg_win = float(tdf[tdf.pnl > 0].pnl_pct.mean()) if (tdf.pnl > 0).any() else 0.0
        avg_loss = float(tdf[tdf.pnl < 0].pnl_pct.mean()) if (tdf.pnl < 0).any() else 0.0
        reasons = tdf["reason"].value_counts().to_dict() if "reason" in tdf else {}

    pf_score = pf if np.isfinite(pf) else 3.0
    score = composite_score(sharpe, dd * 100.0, pf_score, wr, n_tr, total_ret * 100.0)
    return BacktestResult(
        label=label,
        start=start,
        end=end,
        total_return_pct=round(total_ret * 100.0, 2),
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        max_dd_pct=round(dd * 100.0, 2),
        profit_factor=round(pf_score, 3) if np.isfinite(pf) else 99.0,
        win_rate=round(wr, 3),
        n_trades=n_tr,
        avg_hold_days=round(avg_hold, 1),
        avg_win_pct=round(avg_win * 100.0, 2),
        avg_loss_pct=round(avg_loss * 100.0, 2),
        calmar=round(calmar, 3),
        score=round(score, 3),
        n_days=int(len(eq)),
        halted=halted,
        exit_reasons={str(k): int(v) for k, v in reasons.items()},
        equity=[float(x) for x in eq.iloc[:: max(len(eq) // 250, 1)].tolist()],
    )


def _stop_target_fill(
    pos: dict,
    high: float,
    low: float,
    close: float,
    holding_days: int,
) -> tuple[str | None, float | None]:
    """Pessimistic same-bar exit: stop before target, then max-hold at close."""
    sl = pos["entry_px"] * (1.0 - pos["stop_pct"])
    tp = pos["entry_px"] * (1.0 + pos["target_pct"])
    if low <= sl:
        return "stop_loss", sl
    if high >= tp:
        return "take_profit", tp
    if pos["days"] >= holding_days:
        return "max_hold", close
    return None, None


def simulate_portfolio(
    bars: dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    spec: StrategySpec,
    *,
    start: str,
    end: str,
    capital: float = 100_000.0,
) -> BacktestResult:
    """Event-driven daily simulation. ``signals`` may be empty (flat book)."""
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    cost = spec.cost_bps / 10_000.0

    # Align a common calendar from SPY if present, else union of closes.
    if "SPY" in bars:
        calendar = bars["SPY"].loc[(bars["SPY"].index >= lo) & (bars["SPY"].index < hi)].index
    else:
        calendar = pd.DatetimeIndex([])
        for df in bars.values():
            calendar = calendar.union(df.loc[(df.index >= lo) & (df.index < hi)].index)
        calendar = calendar.sort_values()
    if len(calendar) < 5:
        return _metrics([], [], spec.name, start, end, False)

    sig = signals.copy() if signals is not None and not signals.empty else pd.DataFrame()
    if not sig.empty:
        sig["time"] = pd.to_datetime(sig["time"], utc=True)
        sig = sig[(sig.time >= lo) & (sig.time < hi)]
        by_day: dict[pd.Timestamp, list] = {}
        for r in sig.itertuples(index=False):
            by_day.setdefault(r.time, []).append(r)
    else:
        by_day = {}

    cash = capital
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity: list[float] = []
    consecutive_losses = 0
    day_start_value = capital
    peak = capital
    halted = False
    halt_reason = ""
    halt_count = 0
    pause_left = 0
    resume_size_scalar = 1.0
    trades_today = 0
    cur_day: date | None = None
    terminal_halt = False

    def mark(ts: pd.Timestamp) -> float:
        pv = cash
        for t, pos in positions.items():
            df = bars.get(t)
            if df is None or ts not in df.index:
                pv += pos["qty"] * pos["last"]
            else:
                px = float(df.at[ts, "close"])
                pos["last"] = px
                pv += pos["qty"] * px
        return pv

    def close_pos(ticker: str, fill: float, ts: pd.Timestamp, reason: str) -> None:
        nonlocal cash, consecutive_losses
        pos = positions.pop(ticker)
        fpx = fill * (1 - cost if pos["qty"] > 0 else 1 + cost)
        pnl = pos["qty"] * (fpx - pos["entry_px"])
        cash += pos["qty"] * fpx
        pnl_pct = fpx / pos["entry_px"] - 1.0 if pos["qty"] > 0 else pos["entry_px"] / fpx - 1.0
        trades.append(
            {
                "ticker": ticker,
                "entry": pos["entry_ts"],
                "exit": ts,
                "entry_px": pos["entry_px"],
                "exit_px": fpx,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "days": pos["days"],
                "reason": reason,
            }
        )
        if pnl < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

    buy_hold = spec.topology == "buy_hold"
    allocated_bh = False

    for ts in calendar:
        today = ts.date()
        if today != cur_day:
            cur_day = today
            # Prior session close — not today's close — so a gap/move can
            # trip the daily −3% breaker.
            day_start_value = equity[-1] if equity else capital
            trades_today = 0
            if pause_left > 0:
                pause_left -= 1
                if pause_left == 0 and not terminal_halt:
                    halted = False
                    halt_reason = ""
            elif halt_reason == "daily_loss_limit":
                halted = False
                halt_reason = ""

        # ── 1. Manage open positions against today's OHLC ──
        for t in list(positions):
            df = bars.get(t)
            if df is None or ts not in df.index:
                continue
            pos = positions[t]
            h = float(df.at[ts, "high"])
            l = float(df.at[ts, "low"])
            c = float(df.at[ts, "close"])
            pos["days"] += 1
            pos["peak"] = max(pos["peak"], h)
            reason, fill = _stop_target_fill(pos, h, l, c, spec.holding_days)
            if reason is not None:
                close_pos(t, float(fill), ts, reason)

        pv = mark(ts)
        peak = max(peak, pv)
        dd = (pv - peak) / peak if peak else 0.0
        daily_pnl = (pv - day_start_value) / day_start_value if day_start_value else 0.0

        # Buy-and-hold is the unmanaged benchmark: it does not run circuit
        # breakers. Active graphs do. Production needs a human to resume after
        # an 8% DD halt; the research sim models that as a 21-session pause,
        # half-size restart, and a fresh high-water mark — capped at 3 halts
        # per window so a reckless book still dies.
        if (not buy_hold) and daily_pnl <= -0.03:
            halted = True
            halt_reason = "daily_loss_limit"
        if (not buy_hold) and dd <= -0.08 and halt_reason != "drawdown_pause":
            for t in list(positions):
                df = bars.get(t)
                px = float(df.at[ts, "close"]) if df is not None and ts in df.index else positions[t]["last"]
                close_pos(t, px, ts, "drawdown_halt")
            halt_count += 1
            halted = True
            if halt_count >= 3:
                terminal_halt = True
                halt_reason = "max_drawdown"
            else:
                pause_left = 21
                resume_size_scalar = 0.5
                peak = mark(ts)
                halt_reason = "drawdown_pause"

        # ── 2. Entries ──
        if buy_hold and not allocated_bh:
            names = [t for t in bars if t != "SPY"]
            n = max(len(names), 1)
            per = cash / n
            for t in names:
                df = bars[t]
                if ts not in df.index:
                    continue
                o = float(df.at[ts, "open"]) * (1 + cost)
                qty = per / o
                cash -= qty * o
                positions[t] = {
                    "qty": qty, "entry_px": o, "entry_ts": ts, "peak": o,
                    "days": 0, "last": o, "stop_pct": 0.99, "target_pct": 99.0,
                }
            allocated_bh = True
        elif not buy_hold and not halted:
            cands = [r for r in by_day.get(ts, []) if getattr(r, "action", "FLAT") == "LONG"]
            cands.sort(key=lambda r: abs(getattr(r, "score", 0.0)), reverse=True)
            size_scalar = resume_size_scalar * (0.5 if consecutive_losses >= 5 else 1.0)
            for r in cands:
                if len(positions) >= spec.max_positions:
                    break
                if r.ticker in positions:
                    continue
                df = bars.get(r.ticker)
                if df is None or ts not in df.index:
                    continue
                heat = (pv - cash) / max(pv, 1.0)
                if heat >= spec.heat_ceiling:
                    break
                o = float(df.at[ts, "open"]) * (1 + cost)
                notional = min(r.size_pct * size_scalar * pv, spec.heat_ceiling * pv - (pv - cash))
                if notional < 500 or notional > cash:
                    continue
                qty = notional / o
                cash -= qty * o
                positions[r.ticker] = {
                    "qty": qty,
                    "entry_px": o,
                    "entry_ts": ts,
                    "peak": o,
                    "days": 0,
                    "last": o,
                    "stop_pct": float(r.stop_pct),
                    "target_pct": float(r.target_pct),
                }
                trades_today += 1
                h = float(df.at[ts, "high"])
                l = float(df.at[ts, "low"])
                c = float(df.at[ts, "close"])
                positions[r.ticker]["peak"] = max(o, h)
                reason, fill = _stop_target_fill(
                    positions[r.ticker], h, l, c, spec.holding_days
                )
                if reason is not None:
                    close_pos(r.ticker, float(fill), ts, reason)
                pv = mark(ts)

        equity.append(mark(ts))

    # Liquidate leftovers at last close (counted as time-exit, not a forced loss).
    if positions and len(calendar):
        last = calendar[-1]
        for t in list(positions):
            df = bars.get(t)
            px = float(df.at[last, "close"]) if df is not None and last in df.index else positions[t]["last"]
            close_pos(t, px, last, "eod_liquidate")
        if equity:
            equity[-1] = cash

    result = _metrics(equity, trades, spec.name, start, end, halted=terminal_halt)
    if buy_hold:
        result.notes = "equal-weight buy-and-hold baseline"
        # Buy-hold has n_trades = N names liquidated at end; don't penalise.
        result.score = composite_score(
            result.sharpe, result.max_dd_pct, 1.0, 0.5, 50, result.total_return_pct
        )
    return result
