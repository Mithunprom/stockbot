"""Signal Loop B — Pipeline B execution loop.

Subclasses SignalLoop, overriding only _tick() to use PipelineBEngine
instead of the ML-based EnsembleEngine. All execution logic (_act_on_signal,
SmartPositionSizer, exit management, Kelly gate, IC tracking) is inherited
unchanged.

The key difference from the parent:
  - No torch tensor construction (Pipeline B is pure rules-based)
  - Feature rows are passed as dicts to PipelineBEngine.compute_signal()
  - Pipeline B signals are EnsembleSignal-compatible, so sizing/execution works
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import numpy as np
import structlog

from src.agents.signal_loop import SignalLoop
from src.execution.alpaca import AlpacaOrderRouter
from src.execution.position_manager import PositionManager
from src.models.ensemble import EnsembleEngine, EnsembleSignal
from src.models.pipeline_b import PipelineBEngine
from src.risk.circuit_breakers import CircuitBreakers, RiskState

logger = structlog.get_logger(__name__)


class SignalLoopB(SignalLoop):
    """Pipeline B signal loop — rules-based signals with shared execution.

    Inherits all execution logic from SignalLoop. Only overrides _tick()
    to replace ML inference with PipelineBEngine scoring.

    Args:
        universe: List of ticker symbols to trade.
        ensemble: EnsembleEngine (needed by parent __init__, not used in _tick).
        pipeline_b: PipelineBEngine instance for rules-based signal computation.
        alpaca: AlpacaOrderRouter for paper/live execution.
        circuit_breakers: Active CircuitBreakers instance.
        pos_manager: PositionManager tracking Pipeline B's positions.
        session_factory: SQLAlchemy async_sessionmaker.
        feature_cols: Ordered list of FFSA feature column names.
        broadcast_fn: Async callable to push data to dashboard WebSocket.
        pipeline_id: Pipeline identifier for trade attribution.
    """

    def __init__(
        self,
        universe: list[str],
        ensemble: EnsembleEngine,
        pipeline_b: PipelineBEngine,
        alpaca: AlpacaOrderRouter,
        circuit_breakers: CircuitBreakers,
        pos_manager: PositionManager,
        session_factory: Any,
        feature_cols: list[str],
        broadcast_fn: Callable[[dict[str, Any]], Coroutine] | None = None,
        pipeline_id: str = "pipeline_b",
    ) -> None:
        super().__init__(
            universe=universe,
            ensemble=ensemble,
            alpaca=alpaca,
            circuit_breakers=circuit_breakers,
            pos_manager=pos_manager,
            session_factory=session_factory,
            feature_cols=feature_cols,
            broadcast_fn=broadcast_fn,
            pipeline_id=pipeline_id,
        )
        self._pipeline_b = pipeline_b
        # Override sizing mode: Pipeline B always uses sizing mode
        # (the dir_prob/pred_return fields are mapped for the sizer)
        self._sizing_mode = True

    async def _tick(self) -> None:
        """Pipeline B tick: fetch features → rules-based scoring → execution.

        Same overall flow as parent _tick(), but replaces:
          - Tensor construction with dict extraction
          - EnsembleEngine.compute_universe() with PipelineBEngine.compute_universe()
        """
        market_open = self._is_market_hours()
        has_crypto = any(t in self.CRYPTO_TICKERS for t in self._universe)
        if not market_open and not has_crypto:
            return

        # Decrement per-ticker cooldowns
        expired = []
        for t, remaining in self._ticker_cooldown.items():
            if remaining <= 1:
                expired.append(t)
            else:
                self._ticker_cooldown[t] = remaining - 1
        for t in expired:
            del self._ticker_cooldown[t]

        # 1. Fetch features + prices from DB (inherited method)
        features_map, regime_map, latest_feature_time = await self._fetch_features()
        prices = await self._fetch_prices()
        self._pm.update_prices(prices)

        # Data freshness gate (inherited flag — used by _sizing_entry_gate_open)
        from src.agents.signal_loop import DATA_FRESHNESS_MAX_MINUTES
        self._data_fresh = True
        if latest_feature_time is not None:
            age_minutes = (datetime.now(timezone.utc) - latest_feature_time).total_seconds() / 60
            if age_minutes > DATA_FRESHNESS_MAX_MINUTES:
                self._data_fresh = False
                logger.warning(
                    "data_stale_skipping_entries_b",
                    age_minutes=round(age_minutes, 1),
                )

        if not features_map:
            logger.warning("signal_loop_b_no_features")
            return

        # 2. Extract latest feature row per ticker as dict (no tensors needed)
        universe_feature_rows: dict[str, dict[str, float]] = {}
        for ticker in self._universe:
            arr = features_map.get(ticker)
            if arr is None or len(arr) < 1:
                continue
            # Last row of the (SEQ_LEN, n_features) array → dict
            last_row = arr[-1]
            row_dict = {
                col: float(last_row[i])
                for i, col in enumerate(self._feature_cols)
                if i < len(last_row)
            }
            universe_feature_rows[ticker] = row_dict

        if not universe_feature_rows:
            logger.warning("signal_loop_b_insufficient_data", universe=len(self._universe))
            return

        # 3. Compute Pipeline B signals (rules-based)
        signals = await self._pipeline_b.compute_universe(universe_feature_rows)
        self._latest_signals = signals
        logger.info("signal_loop_b_tick", n_signals=len(signals), pipeline="B")

        # 3b. Record predictions for live IC tracking
        if self._ic_tracker is not None:
            for sig in signals:
                if sig.lgbm_pred_return != 0.0 or sig.lgbm_dir_prob != 0.5:
                    try:
                        await self._ic_tracker.record_prediction(
                            ticker=sig.ticker,
                            timestamp=sig.timestamp,
                            pred_return=sig.lgbm_pred_return,
                            dir_prob=sig.lgbm_dir_prob,
                            ensemble_signal=sig.ensemble_signal,
                        )
                    except Exception as exc:
                        logger.debug(
                            "ic_tracker_record_error",
                            ticker=sig.ticker,
                            error=str(exc),
                        )

        # 4. Act on signals (using inherited execution logic)
        sig_by_ticker: dict[str, EnsembleSignal] = {s.ticker: s for s in signals}

        # Sizing mode exit checks for all open positions
        for ticker in list(self._pm._positions.keys()):
            if not market_open and ticker not in self.CRYPTO_TICKERS:
                continue
            price = prices.get(ticker, 0.0)
            if price <= 0:
                continue
            sig = sig_by_ticker.get(ticker)
            if sig is None:
                sig = EnsembleSignal(
                    ticker=ticker,
                    timestamp=datetime.now(timezone.utc),
                )
            await self._act_on_signal(sig, price, None, regime=regime_map.get(ticker, 1))

        # Process new entry signals
        for sig in signals:
            if not market_open and sig.ticker not in self.CRYPTO_TICKERS:
                continue
            if sig.ticker in self._pm._positions:
                continue  # already handled by exit checks above
            regime = regime_map.get(sig.ticker, 1)
            from src.features.regime import REGIME_GATE
            threshold, _size_scale = REGIME_GATE.get(regime, (self.SIGNAL_ENTRY_THRESHOLD, 1.0))
            should_act = self._sizing_mode or abs(sig.ensemble_signal) >= threshold
            if should_act:
                price = prices.get(sig.ticker, 0.0)
                if price > 0:
                    await self._act_on_signal(sig, price, None, regime=regime)

        # 5. Sync positions from broker
        try:
            await self._pm.sync_from_broker(self._alpaca)
        except Exception as exc:
            logger.warning("position_sync_failed", error=str(exc), pipeline="B")

        # 6. Check circuit breakers
        from src.data.market_regime import get_market_regime
        regime_snapshot = get_market_regime()
        state = RiskState(
            portfolio_value=self._pm.portfolio_value,
            peak_portfolio=self._pm._peak_value,
            daily_start_value=self._daily_start_value,
            vix=regime_snapshot.vix,  # real VIX from regime monitor
            consecutive_losses=self._consecutive_losses,
            portfolio_heat=self._pm.portfolio_heat,
        )
        await self._cb.check(state)

        # 7. Reset daily start value at market open
        self._maybe_reset_daily_value()

        # 8. Broadcast to dashboard
        if self._broadcast:
            try:
                await self._broadcast({
                    "type": "signals_pipeline_b",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "signals": [s.to_dict() for s in signals[:10]],
                    "positions": self._pm.get_positions(),
                    "positions_detail": self.get_positions_detail(),
                    "portfolio_value": self._pm.portfolio_value,
                    "portfolio_heat": round(self._pm.portfolio_heat, 4),
                    "halted": self._cb.is_halted,
                    "halt_reason": self._cb.halt_reason,
                    "pipeline": "B",
                })
            except Exception as exc:
                logger.warning("broadcast_error", error=str(exc), pipeline="B")
