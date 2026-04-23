"""Nightly Universe Screener — rotates trading universe based on momentum.

Runs every weekday at 18:00 ET (after market close).

Scoring model:
  score = 0.35 * mom_5d + 0.35 * mom_20d + 0.20 * vol_ratio + 0.10 * rs_vs_spy

Selection:
  1. Fetch S&P 500 constituents from Wikipedia
  2. Score every stock using the model above
  3. Filter on liquidity (avg vol > 500k, price $5-$2000)
  4. Always include ANCHOR_TICKERS (defense, momentum names, core large-caps)
  5. Fill remaining slots (up to MAX_UNIVERSE) with top scorers
  6. Write result to config/universe.json
  7. Hot-swap the signal loop universe (no restart needed)

Output:
  config/universe.json — read by main.py on startup and by the signal loop
  reports/opportunities/screener_YYYY-MM-DD.json — audit trail
"""

from __future__ import annotations

import asyncio
import json
import structlog
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any

logger = structlog.get_logger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────

MAX_UNIVERSE = 40          # max tickers in trading universe
MIN_AVG_VOLUME = 500_000   # daily avg volume floor
MIN_PRICE = 5.0
MAX_PRICE = 2000.0

# These tickers are ALWAYS in the universe regardless of momentum score.
ANCHOR_TICKERS: list[str] = [
    # Core mega-cap tech (high liquidity, tight spreads)
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA",
    # Semiconductors / storage
    "AVGO", "AMD", "MU", "SMCI", "SNDK", "WDC",
    # Financials
    "JPM", "V", "MA",
    # High-momentum / AI
    "PLTR", "ARM", "MSTR",
    # Energy
    "XOM", "CVX",
]

# Crypto tickers — excluded until a crypto-specific model is trained.
CRYPTO_TICKERS: frozenset[str] = frozenset({"BTC/USD", "ETH/USD", "SOL/USD"})

# These tickers are NEVER added to the universe even if S&P 500 momentum qualifies them.
EXCLUDE_TICKERS: set[str] = {"META"}

# ─── S&P 500 constituent fetch ─────────────────────────────────────────────────

def _fetch_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 tickers from Wikipedia."""
    try:
        import pandas as pd
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        tickers = tables[0]["Symbol"].tolist()
        # Wikipedia uses . for class shares (BRK.B) — Alpaca uses /
        return [t.replace(".", "/") for t in tickers]
    except Exception as exc:
        logger.warning("sp500_fetch_failed", error=str(exc))
        return []


# ─── Momentum scoring ──────────────────────────────────────────────────────────

def _score_tickers(tickers: list[str]) -> list[dict[str, Any]]:
    """Fetch 30d daily OHLCV for each ticker and compute momentum score.

    Returns list of dicts sorted by score desc.
    """
    try:
        import yfinance as yf
        import numpy as np
    except ImportError:
        logger.error("yfinance_not_installed")
        return []

    results = []

    # Batch download — much faster than individual calls
    # Split into chunks to avoid yfinance timeouts
    CHUNK = 100
    all_data: dict[str, Any] = {}
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i: i + CHUNK]
        try:
            raw = yf.download(
                chunk,
                period="30d",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            all_data.update(_parse_yf_batch(raw, chunk))
        except Exception as exc:
            logger.warning("yfinance_batch_failed", chunk_start=i, error=str(exc))

    # Also get SPY for relative strength
    try:
        spy_raw = yf.download("SPY", period="30d", interval="1d",
                               auto_adjust=True, progress=False)
        spy_closes = spy_raw["Close"].dropna().values
        spy_ret_5d = float((spy_closes[-1] / spy_closes[-5] - 1)) if len(spy_closes) >= 5 else 0.0
        spy_ret_20d = float((spy_closes[-1] / spy_closes[-20] - 1)) if len(spy_closes) >= 20 else 0.0
    except Exception:
        spy_ret_5d = spy_ret_20d = 0.0

    for ticker, closes_vols in all_data.items():
        closes = closes_vols["closes"]
        volumes = closes_vols["volumes"]

        if len(closes) < 22:
            continue

        price = float(closes[-1])
        if price < MIN_PRICE or price > MAX_PRICE:
            continue

        avg_vol_20d = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else 0.0
        if avg_vol_20d < MIN_AVG_VOLUME:
            continue

        mom_5d = float(closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0.0
        mom_20d = float(closes[-1] / closes[-20] - 1) if len(closes) >= 20 else 0.0
        vol_ratio = float(np.mean(volumes[-5:]) / max(avg_vol_20d, 1)) if len(volumes) >= 5 else 1.0

        # Relative strength vs SPY
        rs_5d = mom_5d - spy_ret_5d
        rs_20d = mom_20d - spy_ret_20d
        rs_vs_spy = 0.5 * rs_5d + 0.5 * rs_20d

        score = (
            0.35 * mom_5d
            + 0.35 * mom_20d
            + 0.20 * min(vol_ratio - 1.0, 2.0)   # cap vol surge at 3x
            + 0.10 * rs_vs_spy
        )

        results.append({
            "ticker": ticker,
            "score": round(score, 4),
            "price": round(price, 2),
            "avg_vol_20d": int(avg_vol_20d),
            "mom_5d": round(mom_5d * 100, 2),
            "mom_20d": round(mom_20d * 100, 2),
            "vol_ratio": round(vol_ratio, 2),
            "rs_vs_spy": round(rs_vs_spy * 100, 2),
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)


def _parse_yf_batch(raw: Any, tickers: list[str]) -> dict[str, dict]:
    """Parse yfinance multi-ticker download into {ticker: {closes, volumes}}."""
    import numpy as np
    result = {}
    if len(tickers) == 1:
        # Single ticker — flat columns
        try:
            closes = raw["Close"].dropna().values
            volumes = raw["Volume"].dropna().values
            if len(closes) > 0:
                result[tickers[0]] = {"closes": closes, "volumes": volumes}
        except Exception:
            pass
        return result

    for ticker in tickers:
        try:
            closes = raw[ticker]["Close"].dropna().values
            volumes = raw[ticker]["Volume"].dropna().values
            if len(closes) > 0:
                result[ticker] = {"closes": closes, "volumes": volumes}
        except Exception:
            continue
    return result


# ─── Screener agent ────────────────────────────────────────────────────────────

class ScreenerAgent:
    """Nightly universe screener — updates config/universe.json.

    Optionally holds a reference to the live SignalLoop so it can hot-swap
    the universe without a restart.
    """

    def __init__(self, signal_loop: Any | None = None) -> None:
        self._signal_loop = signal_loop   # set after signal loop starts

    def set_signal_loop(self, signal_loop: Any) -> None:
        self._signal_loop = signal_loop

    async def run(self) -> None:
        logger.info("screener_agent_run", time=datetime.now(timezone.utc).isoformat())
        loop = asyncio.get_event_loop()

        try:
            # Run in thread pool (yfinance is synchronous + network-heavy)
            new_universe = await loop.run_in_executor(None, self._screen_sync)
            if not new_universe:
                logger.warning("screener_no_results_keeping_existing")
                return

            # Write to config/universe.json
            self._write_universe(new_universe)

            # Hot-swap signal loop universe if running
            if self._signal_loop is not None:
                old = set(self._signal_loop._universe)
                new = set(new_universe)
                added = sorted(new - old)
                removed = sorted(old - new)
                self._signal_loop._universe = new_universe
                logger.info(
                    "universe_hot_swapped",
                    total=len(new_universe),
                    added=added,
                    removed=removed,
                )
            else:
                logger.info("screener_universe_written", total=len(new_universe))

        except Exception as exc:
            logger.exception("screener_agent_error", error=str(exc))

    def _screen_sync(self) -> list[str]:
        """Synchronous screening logic — runs in thread executor."""
        logger.info("screener_fetching_sp500")
        sp500 = _fetch_sp500_tickers()
        if not sp500:
            logger.warning("screener_sp500_empty_using_anchors_only")
            return ANCHOR_TICKERS[:MAX_UNIVERSE]

        logger.info("screener_scoring_tickers", count=len(sp500))
        scored = _score_tickers(sp500)
        logger.info("screener_scored", eligible=len(scored))

        # Build universe: anchors first, then top momentum scores
        universe: list[str] = []

        # Add anchors (always included, never excluded)
        for ticker in ANCHOR_TICKERS:
            if len(universe) >= MAX_UNIVERSE:
                break
            universe.append(ticker)

        # Fill remaining slots with top S&P 500 scorers, skipping excluded/crypto tickers
        # (Crypto anchors are already included above; they don't come from S&P 500 scoring)
        universe_set = set(universe)
        for entry in scored:
            if len(universe) >= MAX_UNIVERSE:
                break
            ticker = entry["ticker"]
            if ticker not in universe_set and ticker not in EXCLUDE_TICKERS and ticker not in CRYPTO_TICKERS:
                universe.append(ticker)
                universe_set.add(ticker)

        # Write detailed report
        self._write_report(scored, universe)
        return universe

    def _write_universe(self, universe: list[str]) -> None:
        path = Path("config/universe.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "source": "screener_agent",
                "count": len(universe),
                "symbols": universe,
            }, f, indent=2)
        logger.info("universe_written", path=str(path), count=len(universe))

    def _write_report(self, scored: list[dict], universe: list[str]) -> None:
        today = date.today().isoformat()
        path = Path(f"reports/opportunities/screener_{today}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "date": today,
                "universe": universe,
                "top_50_by_score": scored[:50],
                "anchor_tickers": ANCHOR_TICKERS,
            }, f, indent=2)
        logger.info("screener_report_written", path=str(path))
