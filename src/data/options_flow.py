"""Options flow data via yfinance (free, no API key needed).

Polls yfinance options chains every 5 minutes for all universe tickers.
Computes per-ticker aggregates: put_call_ratio, vol_oi_ratio, net_gex
(approximated from call/put OI imbalance), smart_money_score,
iv_rank, and unusual_flow_flag.

Writes one summary row per ticker per poll cycle to the `options_flow` table.

This replaces the Unusual Whales integration with a fully free alternative.
For production-grade data (flow alerts, dark pool, etc.) consider upgrading
to Unusual Whales ($30/mo) or Tradier options feed.
"""

from __future__ import annotations

import asyncio
import structlog
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import insert

from src.data.db import OptionsFlow, get_session_factory

logger = structlog.get_logger(__name__)

# Poll 5 tickers per batch to stay well within yfinance rate limits.
# Full 24-ticker universe = 5 batches = ~10s total per cycle.
BATCH_SIZE = 5
POLL_INTERVAL_SECONDS = 300  # 5 minutes


# ─── yfinance options chain aggregator ───────────────────────────────────────

def _fetch_options_aggregate(ticker: str) -> dict[str, Any] | None:
    """Fetch and aggregate yfinance options chain for a single ticker.

    Returns a dict of aggregated metrics, or None on error.
    Note: yfinance is synchronous; run in a thread executor.
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None

        # Use the nearest 1–3 expirations for best liquidity signal
        target_expirations = expirations[:3]

        total_call_vol = 0
        total_put_vol = 0
        total_call_oi = 0
        total_put_oi = 0
        total_call_iv_sum = 0.0
        total_put_iv_sum = 0.0
        iv_count = 0
        smart_score_num = 0.0

        for exp in target_expirations:
            try:
                chain = stock.option_chain(exp)
            except Exception:
                continue

            calls = chain.calls
            puts = chain.puts

            call_vol = int(calls["volume"].fillna(0).sum())
            put_vol = int(puts["volume"].fillna(0).sum())
            call_oi = int(calls["openInterest"].fillna(0).sum())
            put_oi = int(puts["openInterest"].fillna(0).sum())

            total_call_vol += call_vol
            total_put_vol += put_vol
            total_call_oi += call_oi
            total_put_oi += put_oi

            # IV averages (weight by volume)
            if call_vol > 0 and "impliedVolatility" in calls.columns:
                iv_call = float(
                    (calls["impliedVolatility"].fillna(0) * calls["volume"].fillna(0)).sum()
                    / max(call_vol, 1)
                )
                total_call_iv_sum += iv_call
                iv_count += 1
            if put_vol > 0 and "impliedVolatility" in puts.columns:
                iv_put = float(
                    (puts["impliedVolatility"].fillna(0) * puts["volume"].fillna(0)).sum()
                    / max(put_vol, 1)
                )
                total_put_iv_sum += iv_put
                iv_count += 1

            # Smart money score: premium-weighted directional score
            # Use in-the-money volume as a proxy for institutional conviction
            try:
                spot = stock.fast_info.last_price or 0.0
            except Exception:
                spot = 0.0

            if spot > 0:
                itm_calls = calls[calls["strike"] <= spot]
                itm_puts = puts[puts["strike"] >= spot]
                itm_call_vol = int(itm_calls["volume"].fillna(0).sum())
                itm_put_vol = int(itm_puts["volume"].fillna(0).sum())
                total_traded = max(itm_call_vol + itm_put_vol, 1)
                smart_score_num += (itm_call_vol - itm_put_vol) / total_traded

        total_vol = max(total_call_vol + total_put_vol, 1)
        total_oi = max(total_call_oi + total_put_oi, 1)

        put_call_ratio = total_put_vol / max(total_call_vol, 1)
        vol_oi_ratio = total_vol / max(total_oi, 1)

        # GEX approximation: OI imbalance scaled by call OI
        # Positive = dealers net long gamma (call OI dominant → price damping)
        # Negative = dealers net short gamma (put OI dominant → price amplification)
        oi_imbalance = (total_call_oi - total_put_oi) / max(total_oi, 1)
        net_gex = oi_imbalance * total_oi * 100  # arbitrary scale

        # IV rank (skew): put IV vs call IV → positive = fear skew
        avg_call_iv = total_call_iv_sum / max(iv_count // 2, 1)
        avg_put_iv = total_put_iv_sum / max(iv_count // 2, 1)
        iv_rank = avg_put_iv - avg_call_iv  # positive = puts more expensive

        # Smart money score: [-1, +1]; calls=bullish, puts=bearish
        smart_money_score = float(smart_score_num / len(target_expirations))
        smart_money_score = max(-1.0, min(1.0, smart_money_score))

        # Unusual flow flag: vol/OI > 5× is considered unusual
        unusual_flow_flag = vol_oi_ratio >= 5.0

        return {
            "ticker": ticker,
            "time": datetime.now(timezone.utc),
            "contract": f"{ticker}_aggregate",
            "option_type": "aggr",
            "strike": 0.0,
            "expiry": None,
            "premium": float(total_vol),       # use total volume as proxy
            "volume": total_call_vol + total_put_vol,
            "open_interest": total_call_oi + total_put_oi,
            "vol_oi_ratio": round(vol_oi_ratio, 4),
            "implied_volatility": round(avg_call_iv, 4),
            "delta": 0.0,
            "gamma": 0.0,
            "net_gex": round(net_gex, 2),
            "smart_money_score": round(smart_money_score, 4),
            "unusual_flag": unusual_flow_flag,
            # Extra metrics stored for signal_loop consumption
            "put_call_ratio": round(put_call_ratio, 4),
            "iv_rank": round(iv_rank, 4),
        }

    except Exception as exc:
        logger.warning("yfinance_options_error", ticker=ticker, error=str(exc))
        return None


# ─── DB writer ────────────────────────────────────────────────────────────────

async def _write_flow_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    # Strip extra keys not in OptionsFlow schema before upsert
    _schema_keys = {
        "time", "ticker", "contract", "option_type", "strike", "expiry",
        "premium", "volume", "open_interest", "vol_oi_ratio", "implied_volatility",
        "delta", "gamma", "net_gex", "smart_money_score", "unusual_flag",
    }
    clean = [{k: v for k, v in r.items() if k in _schema_keys} for r in rows]

    session_factory = get_session_factory()
    async with session_factory() as session:
        await session.execute(insert(OptionsFlow).values(clean))
        await session.commit()
    logger.info("options_flow_written", count=len(clean))


# ─── In-memory cache for signal_loop consumption ──────────────────────────────

_latest_flow: dict[str, dict[str, Any]] = {}


def get_options_flow(ticker: str) -> dict[str, Any]:
    """Return the latest options flow metrics for a ticker (or defaults)."""
    return _latest_flow.get(ticker, {
        "put_call_ratio": 1.0,
        "vol_oi_ratio": 0.0,
        "net_gex": 0.0,
        "smart_money_score": 0.0,
        "unusual_flag": False,
        "iv_rank": 0.0,
    })


# ─── Polling loop ─────────────────────────────────────────────────────────────

class OptionsFlowPoller:
    """Polls yfinance options chains every 5 minutes for the full universe.

    Compatible with the original Unusual Whales interface:
      poller = OptionsFlowPoller(universe=tickers, poll_interval_seconds=300)
      await poller.start()
      await poller.stop()
    """

    def __init__(
        self,
        universe: list[str] | None = None,
        poll_interval_seconds: int = POLL_INTERVAL_SECONDS,
    ) -> None:
        self._universe: list[str] = universe or []
        self._poll_interval = poll_interval_seconds
        self._running = False

    def set_universe(self, tickers: list[str]) -> None:
        self._universe = tickers

    async def start(self) -> None:
        self._running = True
        logger.info(
            "options_flow_poller_started",
            backend="yfinance",
            interval=self._poll_interval,
            universe=len(self._universe),
        )
        while self._running:
            try:
                await self._poll_once()
            except Exception as exc:
                logger.error("options_flow_poll_error", error=str(exc))
            await asyncio.sleep(self._poll_interval)

    async def stop(self) -> None:
        self._running = False

    async def _poll_once(self) -> None:
        if not self._universe:
            return

        loop = asyncio.get_event_loop()
        rows: list[dict[str, Any]] = []

        # Process in batches to be yfinance-friendly
        for i in range(0, len(self._universe), BATCH_SIZE):
            batch = self._universe[i: i + BATCH_SIZE]
            tasks = [
                loop.run_in_executor(None, _fetch_options_aggregate, ticker)
                for ticker in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for ticker, result in zip(batch, results):
                if isinstance(result, dict):
                    rows.append(result)
                    _latest_flow[ticker] = result
                elif isinstance(result, Exception):
                    logger.debug("options_fetch_failed", ticker=ticker, error=str(result))

            # Brief pause between batches to avoid hammering yfinance
            await asyncio.sleep(1.0)

        if rows:
            await _write_flow_rows(rows)
            logger.info("options_flow_poll_done", tickers=len(rows))
