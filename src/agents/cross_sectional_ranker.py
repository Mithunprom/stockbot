"""Cross-sectional top-k ranker — H1 hypothesis.

At a configurable snapshot time each trading day, rank all universe tickers by
predicted return and return the top-K long candidates.  This replaces the
per-ticker absolute-threshold entry approach with a relative-rank approach: we
always deploy into the BEST K opportunities available rather than any ticker
that clears a static bar.

Motivation (from agent_state.json H1 entry):
    The per-ticker gate fires whenever |pred| > threshold AND dir_prob > 0.60.
    On a strong-signal day that may yield 8 candidates across 20 tickers and
    the bot gets stalled by the MAX_OPEN_POSITIONS=6 cap anyway.  Cross-
    sectional ranking concentrates capital into the *highest-conviction* 3–4
    names instead of the first K to clear the bar.

Design rules (CLAUDE.md / manifesto):
    - No lookahead: the snapshot only reads predictions already available at
      snapshot_time on bar close.  Nothing from the future.
    - Fail-open: if no tickers qualify, no entries fire.  Never forces a trade.
    - Deterministic: pred_return DESC → dir_prob DESC → ticker ASC.  Ties are
      reproducible in backtest and production.
    - Sector diversity: respects max_sector_slots so semis/tech can't fill all
      K slots when the market is risk-on.
    - Risk controls untouched: stop/trail/TP thresholds remain in signal_loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime
from typing import Any


# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_SNAPSHOT_HOUR: int = 14
DEFAULT_SNAPSHOT_MINUTE: int = 0

# Qualification floor — lower than per-ticker threshold because cross-sectional
# rank already acts as the primary filter; we just exclude flat/directionless.
CS_DIR_PROB_MIN: float = 0.55
CS_PRED_RETURN_FLOOR: float = 0.001   # must be positive (long only)
CS_DEFAULT_K: int = 4
CS_MAX_SECTOR_SLOTS: int = 2


# ─── Data types ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RankCandidate:
    """Snapshot of a ticker's signal at ranking time."""

    ticker: str
    pred_return: float
    dir_prob: float
    price: float
    atr_pct: float
    daily_vol: float
    regime: int = 1

    def __lt__(self, other: "RankCandidate") -> bool:
        # Primary sort key: pred_return DESC (so the highest ranks first)
        return self.pred_return > other.pred_return


# ─── Ranker ───────────────────────────────────────────────────────────────────

class CrossSectionalRanker:
    """Collects per-bar predictions and selects the top-K at a daily snapshot.

    Lifecycle (called from the production signal loop each bar):

        ranker = CrossSectionalRanker(k=4)

        # Every 1-minute bar, for every ticker in the universe:
        ranker.update(ticker, pred_return, dir_prob, price, atr_pct, daily_vol)

        # After updating all tickers, check if the snapshot has fired:
        candidates = ranker.get_ranked_candidates(ts_et, sector_map, already_open)
        if candidates is not None:
            for c in candidates:
                # enter position in c.ticker with c.pred_return / c.dir_prob
                ...

    get_ranked_candidates() returns:
        None    — snapshot has not fired yet this bar (normal case, ~389/390 bars)
        []      — snapshot fired but no tickers qualified (no entries today)
        [...]   — ranked list of up to K candidates; caller enters them all
    """

    def __init__(
        self,
        k: int = CS_DEFAULT_K,
        snapshot_hour: int = DEFAULT_SNAPSHOT_HOUR,
        snapshot_minute: int = DEFAULT_SNAPSHOT_MINUTE,
        dir_prob_min: float = CS_DIR_PROB_MIN,
        pred_return_floor: float = CS_PRED_RETURN_FLOOR,
        max_sector_slots: int = CS_MAX_SECTOR_SLOTS,
    ) -> None:
        self.k = k
        self._snapshot_time = dtime(snapshot_hour, snapshot_minute)
        self._dir_prob_min = dir_prob_min
        self._pred_return_floor = pred_return_floor
        self._max_sector_slots = max_sector_slots

        # Latest predictions: ticker → candidate (overwritten each bar)
        self._latest: dict[str, RankCandidate] = {}
        # Prevent the snapshot from firing twice on the same day
        self._last_snapshot_date: Any = None

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        ticker: str,
        pred_return: float,
        dir_prob: float,
        price: float,
        atr_pct: float,
        daily_vol: float,
        regime: int = 1,
    ) -> None:
        """Record the latest bar's prediction for one ticker.

        Must be called for every ticker before get_ranked_candidates() so the
        snapshot reflects all current predictions.
        """
        self._latest[ticker] = RankCandidate(
            ticker=ticker,
            pred_return=float(pred_return),
            dir_prob=float(dir_prob),
            price=float(price),
            atr_pct=float(atr_pct),
            daily_vol=float(daily_vol),
            regime=int(regime),
        )

    # ── Rank ──────────────────────────────────────────────────────────────────

    def get_ranked_candidates(
        self,
        ts_et: Any,  # datetime in ET timezone
        sector_map: dict[str, str] | None = None,
        already_open: set[str] | frozenset[str] | None = None,
    ) -> list[RankCandidate] | None:
        """Return top-K candidates at the snapshot bar; None at all other bars.

        Returns None (not empty list) when the snapshot time has not arrived,
        so callers can distinguish "not yet" from "fired with no qualifiers."

        Args:
            ts_et:       Current bar's timestamp in ET (tz-aware datetime).
            sector_map:  {ticker: sector_name} for diversity filtering.
                         Pass None to skip sector diversity enforcement.
            already_open: Tickers currently in a position — excluded from
                         candidates so we do not double-enter.

        Returns:
            None               — not the snapshot bar, or already fired today.
            [] (empty list)    — snapshot fired, zero qualifying tickers.
            [RankCandidate, …] — up to K ranked candidates, sector-diversified.
        """
        bar_time = dtime(ts_et.hour, ts_et.minute)
        today = ts_et.date()

        if bar_time != self._snapshot_time:
            return None
        if self._last_snapshot_date == today:
            return None  # already fired this trading day

        self._last_snapshot_date = today

        open_set: frozenset[str] = (
            frozenset(already_open) if already_open else frozenset()
        )

        qualifiers: list[RankCandidate] = [
            c for c in self._latest.values()
            if c.pred_return > self._pred_return_floor
            and c.dir_prob >= self._dir_prob_min
            and c.ticker not in open_set
        ]

        # Deterministic sort: pred_return DESC, dir_prob DESC, ticker ASC
        qualifiers.sort(
            key=lambda c: (-c.pred_return, -c.dir_prob, c.ticker)
        )

        if sector_map is None:
            return qualifiers[: self.k]

        return self._apply_sector_cap(qualifiers, sector_map)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_sector_cap(
        self,
        ranked: list[RankCandidate],
        sector_map: dict[str, str],
    ) -> list[RankCandidate]:
        """Walk the ranked list and select up to K, respecting sector slots."""
        selected: list[RankCandidate] = []
        sector_counts: dict[str, int] = {}
        for c in ranked:
            if len(selected) >= self.k:
                break
            sector = sector_map.get(c.ticker, "other")
            if sector_counts.get(sector, 0) >= self._max_sector_slots:
                continue
            selected.append(c)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        return selected

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def snapshot_summary(self) -> dict[str, Any]:
        """Return a dict suitable for structured logging."""
        return {
            "k": self.k,
            "snapshot_time": str(self._snapshot_time),
            "n_tickers_tracked": len(self._latest),
            "last_snapshot_date": str(self._last_snapshot_date),
            "dir_prob_min": self._dir_prob_min,
            "pred_return_floor": self._pred_return_floor,
        }
