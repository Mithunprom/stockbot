"""Market regime monitor — VIX + SPY/QQQ momentum.

Polls VIX via yfinance and reads SPY/QQQ from the existing ohlcv_1m table
to classify the macro regime as risk_on / neutral / risk_off.

Used by Pipeline B as a top-level directional bias.

The regime_score [-1, +1] quantifies how bullish (positive) or
bearish (negative) the broad market environment is.
"""

from __future__ import annotations

import asyncio
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np

logger = structlog.get_logger(__name__)

POLL_INTERVAL_SECONDS = 60  # 1 minute — aligned with signal loop


@dataclass
class MarketRegimeSnapshot:
    """Point-in-time macro regime reading."""

    vix: float = 20.0
    vix_percentile_30d: float = 0.5     # [0, 1] — where current VIX sits in 30d range
    spy_return_5m: float = 0.0          # SPY 5-bar return
    spy_return_15m: float = 0.0         # SPY 15-bar return
    qqq_return_5m: float = 0.0          # QQQ 5-bar return
    qqq_vs_spy_spread: float = 0.0      # QQQ ret - SPY ret (risk-on proxy)
    regime: str = "neutral"             # "risk_on" / "neutral" / "risk_off"
    regime_score: float = 0.0           # [-1, +1] — bearish to bullish
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "vix": round(self.vix, 2),
            "vix_percentile_30d": round(self.vix_percentile_30d, 3),
            "spy_return_5m": round(self.spy_return_5m, 6),
            "spy_return_15m": round(self.spy_return_15m, 6),
            "qqq_return_5m": round(self.qqq_return_5m, 6),
            "qqq_vs_spy_spread": round(self.qqq_vs_spy_spread, 6),
            "regime": self.regime,
            "regime_score": round(self.regime_score, 4),
            "timestamp": self.timestamp.isoformat(),
        }


# ─── VIX fetcher (synchronous — run in thread executor) ─────────────────────

def _fetch_vix() -> dict[str, float]:
    """Fetch current VIX level and 30-day percentile via yfinance.

    Returns {"vix": float, "vix_percentile_30d": float}.
    """
    try:
        import yfinance as yf

        vix = yf.download("^VIX", period="1mo", interval="1d", progress=False)
        if vix is None or vix.empty:
            return {"vix": 20.0, "vix_percentile_30d": 0.5}

        closes = vix["Close"].dropna().values.flatten()
        if len(closes) == 0:
            return {"vix": 20.0, "vix_percentile_30d": 0.5}

        current = float(closes[-1])
        percentile = float(np.sum(closes < current) / len(closes))

        return {"vix": current, "vix_percentile_30d": percentile}

    except Exception as exc:
        logger.warning("vix_fetch_error", error=str(exc))
        return {"vix": 20.0, "vix_percentile_30d": 0.5}


async def _fetch_spy_qqq_returns(session_factory: Any) -> dict[str, float]:
    """Fetch SPY/QQQ recent returns from ohlcv_1m table.

    Returns 5-bar and 15-bar returns for both tickers.
    """
    from sqlalchemy import text

    defaults = {
        "spy_return_5m": 0.0,
        "spy_return_15m": 0.0,
        "qqq_return_5m": 0.0,
        "qqq_vs_spy_spread": 0.0,
    }

    if session_factory is None:
        return defaults

    try:
        async with session_factory() as session:
            results = {}
            for ticker in ("SPY", "QQQ"):
                row = await session.execute(
                    text(
                        "SELECT close FROM ohlcv_1m "
                        "WHERE ticker = :ticker "
                        "ORDER BY time DESC LIMIT 16"
                    ),
                    {"ticker": ticker},
                )
                closes = [float(r[0]) for r in row.fetchall()]
                closes.reverse()  # oldest first

                if len(closes) >= 6:
                    ret_5 = (closes[-1] - closes[-6]) / closes[-6]
                else:
                    ret_5 = 0.0

                if len(closes) >= 16:
                    ret_15 = (closes[-1] - closes[0]) / closes[0]
                else:
                    ret_15 = 0.0

                results[ticker] = {"ret_5": ret_5, "ret_15": ret_15}

            spy = results.get("SPY", {"ret_5": 0.0, "ret_15": 0.0})
            qqq = results.get("QQQ", {"ret_5": 0.0, "ret_15": 0.0})

            return {
                "spy_return_5m": spy["ret_5"],
                "spy_return_15m": spy["ret_15"],
                "qqq_return_5m": qqq["ret_5"],
                "qqq_vs_spy_spread": qqq["ret_5"] - spy["ret_5"],
            }

    except Exception as exc:
        logger.warning("spy_qqq_fetch_error", error=str(exc))
        return defaults


def _classify_regime(
    vix: float,
    vix_pct: float,
    spy_ret_5: float,
    spy_ret_15: float,
    qqq_spread: float,
) -> tuple[str, float]:
    """Classify market regime and compute score [-1, +1].

    Combines VIX level, VIX percentile, and index momentum.
    """
    score = 0.0

    # VIX component: low VIX = bullish, high = bearish
    if vix < 15:
        score += 0.30
    elif vix < 20:
        score += 0.10
    elif vix < 25:
        score -= 0.10
    elif vix < 30:
        score -= 0.25
    else:
        score -= 0.40

    # VIX percentile: below 30th pct = calm, above 70th = fear
    if vix_pct < 0.30:
        score += 0.15
    elif vix_pct > 0.70:
        score -= 0.15

    # SPY momentum (5-bar)
    if spy_ret_5 > 0.002:
        score += 0.20
    elif spy_ret_5 > 0.0005:
        score += 0.10
    elif spy_ret_5 < -0.002:
        score -= 0.20
    elif spy_ret_5 < -0.0005:
        score -= 0.10

    # SPY momentum (15-bar, longer trend)
    if spy_ret_15 > 0.005:
        score += 0.15
    elif spy_ret_15 < -0.005:
        score -= 0.15

    # Risk-on proxy: QQQ outperforming SPY = risk-on
    if qqq_spread > 0.001:
        score += 0.10
    elif qqq_spread < -0.001:
        score -= 0.10

    # Clip to [-1, 1]
    score = max(-1.0, min(1.0, score))

    # Classify
    if score >= 0.20:
        regime = "risk_on"
    elif score <= -0.20:
        regime = "risk_off"
    else:
        regime = "neutral"

    return regime, score


# ─── In-memory latest snapshot ───────────────────────────────────────────────

_latest: MarketRegimeSnapshot = MarketRegimeSnapshot()


def get_market_regime() -> MarketRegimeSnapshot:
    """Return the latest market regime snapshot."""
    return _latest


# ─── Polling loop ────────────────────────────────────────────────────────────

class MarketRegimeMonitor:
    """Polls VIX + SPY/QQQ every minute to classify market regime.

    Usage:
        monitor = MarketRegimeMonitor(session_factory=sf)
        await monitor.start()
        snapshot = get_market_regime()
    """

    def __init__(
        self,
        session_factory: Any = None,
        poll_interval_seconds: int = POLL_INTERVAL_SECONDS,
    ) -> None:
        self._sf = session_factory
        self._poll_interval = poll_interval_seconds
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info("market_regime_monitor_started", interval=self._poll_interval)
        while self._running:
            try:
                await self._poll_once()
            except Exception as exc:
                logger.error("market_regime_poll_error", error=str(exc))
            await asyncio.sleep(self._poll_interval)

    async def stop(self) -> None:
        self._running = False

    async def poll_once(self) -> MarketRegimeSnapshot:
        """One-shot poll (public — used by backtest and startup)."""
        return await self._poll_once()

    async def _poll_once(self) -> MarketRegimeSnapshot:
        global _latest

        loop = asyncio.get_event_loop()

        # Fetch VIX in thread executor (yfinance is sync)
        vix_data = await loop.run_in_executor(None, _fetch_vix)

        # Fetch SPY/QQQ from DB
        spy_qqq = await _fetch_spy_qqq_returns(self._sf)

        # Classify
        regime, score = _classify_regime(
            vix=vix_data["vix"],
            vix_pct=vix_data["vix_percentile_30d"],
            spy_ret_5=spy_qqq["spy_return_5m"],
            spy_ret_15=spy_qqq["spy_return_15m"],
            qqq_spread=spy_qqq["qqq_vs_spy_spread"],
        )

        snapshot = MarketRegimeSnapshot(
            vix=vix_data["vix"],
            vix_percentile_30d=vix_data["vix_percentile_30d"],
            spy_return_5m=spy_qqq["spy_return_5m"],
            spy_return_15m=spy_qqq["spy_return_15m"],
            qqq_return_5m=spy_qqq["qqq_return_5m"],
            qqq_vs_spy_spread=spy_qqq["qqq_vs_spy_spread"],
            regime=regime,
            regime_score=score,
            timestamp=datetime.now(timezone.utc),
        )

        _latest = snapshot
        logger.debug(
            "market_regime_updated",
            vix=snapshot.vix,
            regime=snapshot.regime,
            score=round(snapshot.regime_score, 3),
        )
        return snapshot
