"""Ledger replay harness — score counterfactual risk rules against real fills.

Why this exists
---------------
Every claim about "this risk rule would have helped" must be produced by a
reproducible measurement, not by intuition. This harness loads the closed-trade
ledger and re-runs candidate risk rules over the fills that actually happened.

WHAT THIS HARNESS CAN DO
    * Remove fills that actually happened ("this entry would have been blocked").
    * Re-price a fill's outcome against a barrier that the fill's realized
      entry/exit return demonstrably crossed.
    * Recompute portfolio statistics over the surviving set.

WHAT THIS HARNESS CANNOT DO — read before quoting any number from it
    1. It cannot simulate fills that never occurred. Blocking one entry in
       reality frees a slot, a day-trade budget and portfolio heat, which would
       have admitted some OTHER entry the bot never took. Every counterfactual
       P&L below is therefore "the book minus these fills", not "the book the
       bot would have produced".
    2. There is no intra-hold price path — only entry and exit. A barrier can
       only be scored as "crossed" when the position was still past it AT EXIT.
       So:
         - barrier fire rates are a strict LOWER bound (excursions that pierced
           the barrier mid-hold and recovered are invisible and uncounted);
         - barrier P&L savings are an OPTIMISTIC UPPER bound (a stop that would
           have knocked a mid-hold dip out of a trade that recovered into a
           winner cannot be seen, so its cost is never charged).
    3. Volatility is a point-in-time daily ATR(14)/close from the day BEFORE
       entry (no lookahead), which is the same quantity production uses — but
       it is a proxy for sigma, not sigma.
    4. Slippage, partial fills and commissions are not re-modelled; a re-priced
       exit is assumed to fill exactly at the barrier.

Usage:
    python scripts/replay_ledger.py summary
    python scripts/replay_ledger.py exits
    python scripts/replay_ledger.py bursts
    python scripts/replay_ledger.py all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_LEDGER = "reports/research/ledger_m2.json"
DEFAULT_VOL_CACHE = "reports/research/daily_atr_cache.json"

# M2 == the v0.6.0 regime: trades whose EXIT landed on/after this date.
M2_START = date(2026, 8, 6)

# Bars in a regular US equity session; the unit that daily sigma is quoted in.
BARS_PER_SESSION = 390

# The sector table exactly as it stood while the M2 ledger was being produced
# (24 tickers, everything else falling through to a shared "other" bucket).
# Frozen here so the before/after comparison stays reproducible once the
# production SECTOR_MAP moves on.
_LEGACY_SECTOR_MAP: dict[str, str] = {
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "PLTR": "tech",
    "MSTR": "tech",
    "NVDA": "semis", "AVGO": "semis", "AMD": "semis", "ARM": "semis",
    "SNDK": "semis", "MU": "semis", "SMCI": "semis", "WDC": "semis",
    "JPM": "financials", "V": "financials", "MA": "financials",
    "AMZN": "consumer", "TSLA": "consumer", "COST": "consumer",
    "NFLX": "consumer",
    "XOM": "energy", "CVX": "energy",
    "LLY": "healthcare", "UNH": "healthcare",
}


# ─── Ledger model ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Fill:
    """One closed round-trip from the production ledger."""

    trade_id: int
    ticker: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    ensemble_signal: float

    @property
    def notional(self) -> float:
        """Entry notional in dollars."""
        return self.shares * self.entry_price

    @property
    def hold_minutes(self) -> float:
        """Wall-clock hold length in minutes."""
        return (self.exit_time - self.entry_time).total_seconds() / 60.0

    @property
    def direction(self) -> int:
        """+1 for a long, -1 for a short."""
        return -1 if self.side in ("sell", "short") else 1

    @property
    def trade_return(self) -> float:
        """Realized return of the POSITION, already signed by direction.

        The ledger stores `pnl_pct` in position terms, not price terms: it
        carries the same sign as `pnl` for longs and shorts alike (verified
        across all 233 non-void rows). Barrier comparisons therefore use it
        directly — multiplying by `direction` would double-count the sign.
        """
        return self.pnl_pct


def _parse_ts(raw: str) -> datetime:
    """Parse an ISO timestamp into an aware UTC datetime."""
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_ledger(path: str | Path = DEFAULT_LEDGER) -> list[Fill]:
    """Load closed trades, dropping rows voided by integrity repairs.

    Rows with a null `pnl` are artifacts of ledger-integrity cleanups; they are
    not trades and must never enter a statistic.

    Args:
        path: Path to the ledger JSON (`{"trades": [...]}`).

    Returns:
        Fills sorted by entry time.
    """
    payload = json.loads(Path(path).read_text())
    fills: list[Fill] = []
    for row in payload["trades"]:
        if row.get("pnl") is None or row.get("exit_time") is None:
            continue
        fills.append(
            Fill(
                trade_id=int(row["id"]),
                ticker=str(row["ticker"]),
                side=str(row["side"]),
                entry_time=_parse_ts(row["entry_time"]),
                exit_time=_parse_ts(row["exit_time"]),
                entry_price=float(row["entry_price"]),
                exit_price=float(row["exit_price"]),
                shares=float(row["shares"]),
                pnl=float(row["pnl"]),
                pnl_pct=float(row["pnl_pct"]),
                exit_reason=str(row.get("exit_reason") or "unknown"),
                ensemble_signal=float(row.get("ensemble_signal") or 0.0),
            )
        )
    return sorted(fills, key=lambda f: f.entry_time)


def filter_window(
    fills: Sequence[Fill],
    since: date | None = None,
    until: date | None = None,
) -> list[Fill]:
    """Select fills by EXIT date (the convention M2 reporting uses).

    Args:
        fills: Source fills.
        since: Inclusive lower bound on exit date.
        until: Inclusive upper bound on exit date.

    Returns:
        The matching fills, order preserved.
    """
    out = []
    for fill in fills:
        exit_day = fill.exit_time.date()
        if since is not None and exit_day < since:
            continue
        if until is not None and exit_day > until:
            continue
        out.append(fill)
    return out


# ─── Statistics ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Summary:
    """Portfolio statistics over a set of fills.

    CLAUDE.md forbids reporting P&L alone, so every field needed to judge risk
    travels together.
    """

    n: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    net_pnl: float
    gross_profit: float
    gross_loss: float
    expectancy: float
    max_drawdown: float
    sharpe_per_trade: float

    def format(self, label: str) -> str:
        """One-line human-readable rendering."""
        if self.n == 0:
            return f"{label:<34} n=0"
        profit_factor = "inf" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"
        return (
            f"{label:<34} n={self.n:<4} WR={self.win_rate * 100:5.1f}%  "
            f"PF={profit_factor:>5}  net=${self.net_pnl:>9.2f}  "
            f"exp=${self.expectancy:>7.2f}  maxDD=${self.max_drawdown:>8.2f}  "
            f"Sharpe/trade={self.sharpe_per_trade:>6.3f}"
        )


def summarize(fills: Sequence[Fill]) -> Summary:
    """Compute win rate, profit factor, net P&L, drawdown and Sharpe together.

    `max_drawdown` is the deepest peak-to-trough decline of the closed-trade
    equity curve in dollars, with trades ordered by exit time. `sharpe_per_trade`
    is the mean/std of per-trade `pnl_pct` — deliberately NOT annualized: the
    series is far too short for an honest annual figure.

    Args:
        fills: Fills to summarize.

    Returns:
        A Summary. An empty input yields an all-zero Summary.
    """
    if not fills:
        return Summary(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    pnls = [f.pnl for f in fills]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0
        else (math.inf if gross_profit > 0 else 0.0)
    )

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for fill in sorted(fills, key=lambda f: f.exit_time):
        equity += fill.pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    returns = [f.pnl_pct for f in fills]
    mean_ret = sum(returns) / len(returns)
    if len(returns) > 1:
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    sharpe = mean_ret / std if std > 0 else 0.0

    return Summary(
        n=len(fills),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(fills),
        profit_factor=profit_factor,
        net_pnl=sum(pnls),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        expectancy=sum(pnls) / len(fills),
        max_drawdown=max_drawdown,
        sharpe_per_trade=sharpe,
    )


# ─── Counterfactual 1: concurrency / correlation guards ─────────────────────

@dataclass(frozen=True)
class GuardResult:
    """Outcome of replaying a concurrency guard over the ledger."""

    admitted: list[Fill]
    blocked: list[Fill]

    @property
    def blocked_pnl(self) -> float:
        """Net P&L of the fills the guard would have prevented."""
        return sum(f.pnl for f in self.blocked)


def replay_concurrency_guard(
    fills: Sequence[Fill],
    bucket_of: Callable[[str], str],
    max_per_bucket: int,
) -> GuardResult:
    """Replay a "max N concurrent positions per bucket" guard over real fills.

    Walks fills in entry order, maintaining the set of positions that the guard
    admitted and that are still open at each new entry. A candidate is blocked
    when its bucket already holds `max_per_bucket` admitted, still-open
    positions.

    Args:
        fills: Fills to replay, any order (sorted internally by entry time).
        bucket_of: Maps a ticker to its correlation bucket.
        max_per_bucket: Maximum concurrent positions allowed per bucket.

    Returns:
        A GuardResult splitting the fills into admitted and blocked.

    Note:
        Blocking is the only operation available. Freed capacity cannot admit a
        replacement entry, because entries the bot never took have no recorded
        outcome. See the module docstring.
    """
    admitted: list[Fill] = []
    blocked: list[Fill] = []
    open_fills: list[Fill] = []

    for fill in sorted(fills, key=lambda f: f.entry_time):
        open_fills = [o for o in open_fills if o.exit_time > fill.entry_time]
        bucket = bucket_of(fill.ticker)
        concurrent = sum(1 for o in open_fills if bucket_of(o.ticker) == bucket)
        if concurrent >= max_per_bucket:
            blocked.append(fill)
            continue
        admitted.append(fill)
        open_fills.append(fill)

    return GuardResult(admitted=admitted, blocked=blocked)


# ─── Counterfactual 2: entry-burst concentration ────────────────────────────

def replay_entry_burst_guard(
    fills: Sequence[Fill],
    window_minutes: float,
    max_entries: int,
) -> GuardResult:
    """Replay a "max N entries per rolling W-minute window" guard.

    Args:
        fills: Fills to replay.
        window_minutes: Width of the rolling look-back window.
        max_entries: Maximum admitted entries allowed inside the window.

    Returns:
        A GuardResult splitting the fills into admitted and blocked.
    """
    admitted: list[Fill] = []
    blocked: list[Fill] = []
    recent: list[datetime] = []
    window = timedelta(minutes=window_minutes)

    for fill in sorted(fills, key=lambda f: f.entry_time):
        recent = [t for t in recent if fill.entry_time - t < window]
        if len(recent) >= max_entries:
            blocked.append(fill)
            continue
        admitted.append(fill)
        recent.append(fill.entry_time)

    return GuardResult(admitted=admitted, blocked=blocked)


# ─── Placebo control ────────────────────────────────────────────────────────

def block_count_placebo(
    fills: Sequence[Fill], n_blocked: int, trials: int = 2000, seed: int = 7
) -> tuple[float, float]:
    """Null distribution for "a rule that blocks `n_blocked` fills".

    In a book with negative expectancy, removing ANY subset of fills improves
    net P&L. A counterfactual that blocks fills therefore cannot be judged by
    its net P&L alone — it must beat a rule that blocks the same NUMBER of
    fills at random. This returns that null.

    Args:
        fills: The full book.
        n_blocked: How many fills the rule under test blocked.
        trials: Monte-Carlo sample size.
        seed: RNG seed, so results are reproducible.

    Returns:
        (mean_net, stdev_net) of the surviving book across random draws.
    """
    import random

    if n_blocked <= 0 or n_blocked >= len(fills):
        total = sum(f.pnl for f in fills) if n_blocked <= 0 else 0.0
        return total, 0.0

    rng = random.Random(seed)
    pnls = [f.pnl for f in fills]
    total = sum(pnls)
    nets = []
    for _ in range(trials):
        removed = sum(rng.sample(pnls, n_blocked))
        nets.append(total - removed)
    mean = sum(nets) / len(nets)
    variance = sum((x - mean) ** 2 for x in nets) / (len(nets) - 1)
    return mean, math.sqrt(variance)


def placebo_z(fills: Sequence[Fill], result: GuardResult) -> float:
    """How many null standard deviations a guard's result beats chance by.

    Args:
        fills: The full book the guard was replayed over.
        result: The guard's outcome.

    Returns:
        A z-score against the random-blocking null. |z| < ~2 means the rule is
        indistinguishable from removing that many trades at random — i.e. the
        improvement is an artifact of a losing book, not evidence for the rule.
    """
    mean, std = block_count_placebo(fills, len(result.blocked))
    actual = sum(f.pnl for f in result.admitted)
    return (actual - mean) / std if std > 0 else 0.0


# ─── Counterfactual 3: exit barriers ────────────────────────────────────────

class DailyVolLookup:
    """Point-in-time daily ATR(14)/close, read strictly before a trade's entry.

    Mirrors what `signal_loop._daily_vol_for` serves in production, but pinned
    to the date preceding each entry so no future information enters a
    counterfactual.
    """

    def __init__(self, cache_path: str | Path = DEFAULT_VOL_CACHE,
                 fallback: float = 0.02,
                 floor: float = 0.005, ceil: float = 0.15) -> None:
        """Load the cache built by `scripts/build_daily_atr_cache.py`.

        Args:
            cache_path: Path to the {ticker: {date: atr_ratio}} JSON.
            fallback: Daily vol used when a ticker/date is missing.
            floor: Lower clamp, mirroring signal_loop.DAILY_VOL_FLOOR.
            ceil: Upper clamp, mirroring signal_loop.DAILY_VOL_CEIL.
        """
        raw = json.loads(Path(cache_path).read_text())
        self._cache = {
            ticker: sorted((date.fromisoformat(d), v) for d, v in series.items())
            for ticker, series in raw.items()
        }
        self._fallback = fallback
        self._floor = floor
        self._ceil = ceil
        self.misses = 0

    def daily_vol(self, ticker: str, as_of: date) -> float:
        """Latest daily ATR ratio strictly BEFORE `as_of`, clamped as production does.

        Args:
            ticker: Ticker symbol.
            as_of: The entry date; only bars before this date are eligible.

        Returns:
            The clamped daily ATR ratio, or the fallback when unavailable.
        """
        series = self._cache.get(ticker)
        raw_value = self._fallback
        if series:
            prior = [v for d, v in series if d < as_of]
            if prior:
                raw_value = prior[-1]
            else:
                self.misses += 1
        else:
            self.misses += 1
        return max(self._floor, min(raw_value, self._ceil))

    def hold_window_vol(self, ticker: str, as_of: date, hold_bars: int) -> float:
        """Daily vol rescaled to a `hold_bars`-long holding window.

        Volatility scales with the square root of time, so a 30-bar window sees
        sqrt(30/390) = 0.277 of a session's sigma. This is the conversion the
        production `_atr_exits` omits.

        Args:
            ticker: Ticker symbol.
            as_of: Entry date.
            hold_bars: Length of the holding window in 1-minute bars.

        Returns:
            Sigma of the holding window as a return fraction.
        """
        return self.daily_vol(ticker, as_of) * math.sqrt(hold_bars / BARS_PER_SESSION)


@dataclass(frozen=True)
class BarrierScan:
    """Result of scoring one stop-loss level against the ledger."""

    sigma_mult: float
    fires: int
    fire_rate: float
    pnl_delta: float
    net_after: float
    profit_factor_after: float
    median_stop_pct: float


def replay_stop_barrier(
    fills: Sequence[Fill],
    vol: DailyVolLookup,
    sigma_mult: float,
    hold_bars: int,
) -> BarrierScan:
    """Score a stop placed at `sigma_mult` x sigma OF THE HOLDING WINDOW.

    A fill is counted as stopped only when its realized return AT EXIT was
    still worse than the barrier. Its P&L is then re-priced to the barrier.

    Args:
        fills: Fills to score.
        vol: Point-in-time volatility source.
        sigma_mult: Stop distance in sigmas of the holding window.
        hold_bars: Holding-window length used to scale sigma.

    Returns:
        A BarrierScan. `fires` is a LOWER bound and `pnl_delta` an OPTIMISTIC
        UPPER bound on the benefit — see the module docstring for why.
    """
    fires = 0
    delta = 0.0
    adjusted: list[float] = []
    stop_pcts: list[float] = []

    for fill in fills:
        stop_pct = sigma_mult * vol.hold_window_vol(
            fill.ticker, fill.entry_time.date(), hold_bars
        )
        stop_pcts.append(stop_pct)
        if fill.trade_return < -stop_pct:
            fires += 1
            capped_pnl = -stop_pct * fill.notional
            delta += capped_pnl - fill.pnl
            adjusted.append(capped_pnl)
        else:
            adjusted.append(fill.pnl)

    wins = sum(p for p in adjusted if p > 0)
    loss = -sum(p for p in adjusted if p < 0)
    profit_factor = wins / loss if loss > 0 else (math.inf if wins > 0 else 0.0)
    stop_pcts.sort()
    median_stop = stop_pcts[len(stop_pcts) // 2] if stop_pcts else 0.0

    return BarrierScan(
        sigma_mult=sigma_mult,
        fires=fires,
        fire_rate=fires / len(fills) if fills else 0.0,
        pnl_delta=delta,
        net_after=sum(adjusted),
        profit_factor_after=profit_factor,
        median_stop_pct=median_stop,
    )


def effective_sigma_multiple(
    nominal_mult: float, hold_bars: int, bars_per_session: int = BARS_PER_SESSION
) -> float:
    """Convert a daily-sigma multiple into sigmas of a shorter holding window.

    This is the unit bug at the heart of H14: a barrier quoted as `nominal_mult`
    daily sigmas is really `nominal_mult / sqrt(hold_bars / 390)` sigmas of the
    move a `hold_bars` trade can actually make.

    Args:
        nominal_mult: Barrier multiple expressed in daily sigmas.
        hold_bars: Holding-window length in 1-minute bars.
        bars_per_session: Bars in a full session.

    Returns:
        The effective multiple in sigmas of the holding window.
    """
    return nominal_mult / math.sqrt(hold_bars / bars_per_session)


# ─── Reporting helpers ──────────────────────────────────────────────────────

def _concurrency_episodes(
    fills: Sequence[Fill], bucket_of: Callable[[str], str], min_size: int
) -> list[tuple[str, datetime, list[Fill]]]:
    """Find moments where >= `min_size` same-bucket positions were open at once.

    Args:
        fills: Fills to scan.
        bucket_of: Ticker -> bucket mapping.
        min_size: Minimum simultaneous same-bucket positions to report.

    Returns:
        (bucket, timestamp, fills) tuples, one per distinct overlapping set.
    """
    episodes: list[tuple[str, datetime, list[Fill]]] = []
    seen: set[tuple[int, ...]] = set()
    ordered = sorted(fills, key=lambda f: f.entry_time)
    for fill in ordered:
        bucket = bucket_of(fill.ticker)
        group = [
            o for o in ordered
            if bucket_of(o.ticker) == bucket
            and o.entry_time <= fill.entry_time < o.exit_time
        ]
        if len(group) >= min_size:
            key = tuple(sorted(g.trade_id for g in group))
            if key not in seen:
                seen.add(key)
                episodes.append((bucket, fill.entry_time, group))
    return episodes


def _print(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


# ─── CLI commands ───────────────────────────────────────────────────────────

def cmd_summary(m2: list[Fill], allf: list[Fill]) -> None:
    """Print headline statistics for the whole ledger and the M2 regime."""
    print("== Ledger summary ==")
    print(summarize(allf).format("ALL (2026-05-14 -> 09-11)"))
    print(summarize(m2).format(f"M2 v0.6.0 (exits >= {M2_START})"))
    print(summarize(filter_window(m2, since=date(2026, 8, 19))).format("  M2 since 2026-08-19"))
    print(summarize(filter_window(m2, until=date(2026, 8, 18))).format("  M2 through 2026-08-18"))
    print()
    reasons: dict[str, int] = {}
    sides: dict[str, int] = {}
    for fill in m2:
        reasons[fill.exit_reason] = reasons.get(fill.exit_reason, 0) + 1
        sides[fill.side] = sides.get(fill.side, 0) + 1
    print(f"M2 exit reasons : {reasons}")
    print(f"M2 sides        : {sides}")
    print(f"M2 tickers      : {len(set(f.ticker for f in m2))} distinct")
    holds = sorted(f.hold_minutes for f in m2)
    print(f"M2 median hold  : {holds[len(holds) // 2]:.1f} min")


def cmd_sectors(m2: list[Fill]) -> None:
    """Score the pre-fix sector resolver against the fail-closed one."""
    from src.execution.position_sizer import (
        MAX_POSITIONS_PER_SECTOR_DEFAULT, SECTOR_MAP, max_positions_for_sector,
        sector_of,
    )

    def legacy_bucket(ticker: str) -> str:
        """The pre-fix resolver: every unknown ticker shared one 'other' bucket."""
        return _LEGACY_SECTOR_MAP.get(ticker, "other")

    tickers = sorted(set(f.ticker for f in m2))
    unmapped_then = [t for t in tickers if t not in _LEGACY_SECTOR_MAP]
    unmapped_now = [t for t in tickers if t not in SECTOR_MAP]
    print("== Sector / correlation guard ==")
    print(f"M2 distinct tickers          : {len(tickers)}")
    print(f"  absent from the OLD map    : {len(unmapped_then)}  {unmapped_then}")
    print(f"  absent from the NEW map    : {len(unmapped_now)}  {unmapped_now}")
    then_pnl = sum(f.pnl for f in m2 if f.ticker not in _LEGACY_SECTOR_MAP)
    mapped_pnl = sum(f.pnl for f in m2 if f.ticker in _LEGACY_SECTOR_MAP)
    print(f"  net on OLD-mapped names    : ${mapped_pnl:+.2f}")
    print(f"  net on OLD-unmapped names  : ${then_pnl:+.2f}")
    print()

    cap = MAX_POSITIONS_PER_SECTOR_DEFAULT
    legacy = replay_concurrency_guard(m2, legacy_bucket, cap)
    fixed = replay_concurrency_guard(m2, sector_of, cap)
    print(f"Guard replay at max {cap} concurrent positions per bucket")
    print(f"  OLD resolver blocks {len(legacy.blocked):>3} fills "
          f"(${legacy.blocked_pnl:+.2f} of P&L)")
    print(f"  NEW resolver blocks {len(fixed.blocked):>3} fills "
          f"(${fixed.blocked_pnl:+.2f} of P&L)   placebo-z={placebo_z(m2, fixed):+.2f}")
    print()
    print(summarize(m2).format("as traded"))
    print(summarize(legacy.admitted).format("OLD resolver"))
    print(summarize(fixed.admitted).format("NEW fail-closed resolver"))
    print()
    print("Concentration episodes the OLD resolver could not see "
          "(>=3 same-bucket positions open at once):")
    for bucket, when, group in _concurrency_episodes(m2, sector_of, 3):
        names = ",".join(g.ticker for g in group)
        net = sum(g.pnl for g in group)
        seen = len({legacy_bucket(g.ticker) for g in group})
        print(f"  {when:%Y-%m-%d %H:%M}Z {bucket:<11} x{len(group)} {names:<28} "
              f"net=${net:>9.2f}  (OLD map saw {seen} bucket(s))")
    print()
    print(f"Unmapped bucket cap now {max_positions_for_sector('unmapped')} position "
          f"(vs {cap} for a known sector): an unrecognized ticker gets no "
          f"diversification credit.")


def replay_shipped_exits(fills: Sequence[Fill], vol: DailyVolLookup) -> BarrierScan:
    """Score the ACTUAL `_atr_exits` production function, floors and caps included.

    The sigma scan below is idealized — it ignores the clamps. This runs the
    shipped code path per trade so the reported fire rate is what the deployed
    configuration would really have produced.

    Args:
        fills: Fills to score.
        vol: Point-in-time volatility source.

    Returns:
        A BarrierScan built from the live stop distances.
    """
    from src.agents.signal_loop import _atr_exits

    fires = 0
    delta = 0.0
    adjusted: list[float] = []
    distances: list[float] = []
    for fill in fills:
        daily = vol.daily_vol(fill.ticker, fill.entry_time.date())
        stop_pct, _trail, _tp = _atr_exits(daily)
        distances.append(stop_pct)
        if fill.trade_return < -stop_pct:
            fires += 1
            capped = -stop_pct * fill.notional
            delta += capped - fill.pnl
            adjusted.append(capped)
        else:
            adjusted.append(fill.pnl)

    wins = sum(p for p in adjusted if p > 0)
    loss = -sum(p for p in adjusted if p < 0)
    distances.sort()
    return BarrierScan(
        sigma_mult=float("nan"),
        fires=fires,
        fire_rate=fires / len(fills) if fills else 0.0,
        pnl_delta=delta,
        net_after=sum(adjusted),
        profit_factor_after=wins / loss if loss > 0 else (math.inf if wins > 0 else 0.0),
        median_stop_pct=distances[len(distances) // 2] if distances else 0.0,
    )


def cmd_exits(m2: list[Fill], vol: DailyVolLookup) -> None:
    """Scan candidate stop levels in sigmas of the holding window."""
    from src.agents.signal_loop import SIZING_MAX_HOLD_BARS

    hold_bars = SIZING_MAX_HOLD_BARS
    print("== Exit barrier calibration (H14) ==")
    print(f"Hold window = {hold_bars} bars; sqrt({hold_bars}/390) = "
          f"{math.sqrt(hold_bars / BARS_PER_SESSION):.3f} of a session sigma")
    print()
    print("The pre-fix defect — multiples quoted in DAILY sigma, applied to a "
          f"{hold_bars}-bar hold:")
    for name, mult in (("stop", 1.1), ("trail", 1.2), ("take_profit", 1.5)):
        print(f"  {name:<12} {mult:.1f} daily-sigma -> "
              f"{effective_sigma_multiple(mult, hold_bars):.2f} sigma of the hold window")
    print()

    realized = sorted(abs(f.pnl_pct) for f in m2)
    median_abs = realized[len(realized) // 2]
    print(f"Median |realized return| at exit: {median_abs * 100:.3f}%")
    print()
    print("Idealized stop scan (no floors/caps):")
    print(f"{'stop':>6} {'median dist':>12} {'fires':>6} {'fire rate':>10} "
          f"{'P&L delta':>11} {'net after':>11} {'PF after':>9}")
    for mult in (0.75, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 3.97):
        scan = replay_stop_barrier(m2, vol, mult, hold_bars)
        print(f"{mult:>5.2f}s {scan.median_stop_pct * 100:>11.3f}% {scan.fires:>6} "
              f"{scan.fire_rate * 100:>9.1f}% {scan.pnl_delta:>+11.2f} "
              f"{scan.net_after:>+11.2f} {scan.profit_factor_after:>9.2f}")
    print()
    print("fires = LOWER bound (still past the barrier at exit); "
          "P&L delta = OPTIMISTIC upper bound on benefit.")
    worst = min(m2, key=lambda f: f.pnl)
    print(f"Worst single M2 loss: {worst.ticker} {worst.pnl:+.2f} "
          f"({worst.pnl_pct * 100:+.2f}%) on {worst.exit_time:%Y-%m-%d}")
    print()

    shipped = replay_shipped_exits(m2, vol)
    print("SHIPPED configuration, via the real _atr_exits (floors/caps applied):")
    print(f"  median stop distance : {shipped.median_stop_pct * 100:.3f}%")
    print(f"  fires                : {shipped.fires} / {len(m2)} "
          f"({shipped.fire_rate * 100:.1f}%)")
    print(f"  P&L delta            : ${shipped.pnl_delta:+.2f}")
    print(f"  net / PF after       : ${shipped.net_after:+.2f} / "
          f"{shipped.profit_factor_after:.2f}")
    print("  The stop is tail insurance: near-zero fire rate, near-zero measured")
    print("  P&L effect. It does not and cannot make this book profitable.")


def cmd_bursts(m2: list[Fill]) -> None:
    """Quantify entry-time concentration and score burst caps."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:  # pragma: no cover - py<3.9 only
        from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

    eastern = ZoneInfo("America/New_York")
    print("== Entry concentration ==")
    buckets: dict[str, list[Fill]] = {}
    for fill in m2:
        key = f"{fill.entry_time.astimezone(eastern):%H}:00"
        buckets.setdefault(key, []).append(fill)
    print(f"{'ET hour':>9} {'n':>5} {'WR':>7} {'PF':>7} {'net':>11}")
    for key in sorted(buckets):
        stats = summarize(buckets[key])
        profit_factor = "inf" if math.isinf(stats.profit_factor) else f"{stats.profit_factor:.2f}"
        print(f"{key:>9} {stats.n:>5} {stats.win_rate * 100:>6.1f}% "
              f"{profit_factor:>7} {stats.net_pnl:>+11.2f}")
    print()

    early = [f for f in m2
             if (f.entry_time.astimezone(eastern).hour,
                 f.entry_time.astimezone(eastern).minute) < (9, 44)]
    print(f"Entries in the 09:40-09:43 ET burst: {len(early)}/{len(m2)} "
          f"({len(early) / len(m2) * 100:.0f}%), net ${sum(f.pnl for f in early):.2f}")
    print()
    print("Rolling burst-cap replay (block-only; cannot re-admit later entries).")
    print("placebo-z compares each rule against blocking the SAME NUMBER of fills at")
    print("random: |z| < 2 means the rule is indistinguishable from chance.")
    print(f"{'window':>9} {'cap':>5} {'blocked':>9} {'blocked P&L':>13} "
          f"{'net after':>11} {'PF after':>9} {'placebo-z':>11}")
    for window in (5.0, 15.0, 30.0):
        for cap in (1, 2, 3):
            result = replay_entry_burst_guard(m2, window, cap)
            stats = summarize(result.admitted)
            profit_factor = (
                "inf" if math.isinf(stats.profit_factor) else f"{stats.profit_factor:.2f}"
            )
            print(f"{window:>8.0f}m {cap:>5} {len(result.blocked):>9} "
                  f"{result.blocked_pnl:>+13.2f} {stats.net_pnl:>+11.2f} "
                  f"{profit_factor:>9} {placebo_z(m2, result):>+11.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay risk rules against the ledger.")
    parser.add_argument("command",
                        choices=["summary", "sectors", "exits", "bursts", "all"])
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--vol-cache", default=DEFAULT_VOL_CACHE)
    args = parser.parse_args()

    fills = load_ledger(args.ledger)
    m2 = filter_window(fills, since=M2_START)

    if args.command in ("summary", "all"):
        cmd_summary(m2, fills)
        print()
    if args.command in ("sectors", "all"):
        cmd_sectors(m2)
        print()
    if args.command in ("exits", "all"):
        cmd_exits(m2, DailyVolLookup(args.vol_cache))
        print()
    if args.command in ("bursts", "all"):
        cmd_bursts(m2)


if __name__ == "__main__":
    main()
