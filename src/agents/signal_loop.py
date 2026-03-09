"""Signal loop — Phase 5 execution pipeline.

Runs every 1 minute (aligned to bar close) during market hours:
  1. Fetch latest features from DB for all tickers in universe
  2. Build feature tensors (1m + 5m) and compute ensemble signals
  3. Augment signals with options flow metrics (yfinance)
  4. Build 27-dim RL observation; use PPO agent to decide action
  5. Execute action via AlpacaOrderRouter
  6. Check circuit breakers on every tick
  7. Broadcast signals + portfolio state to dashboard WebSocket

Market hours: Mon–Fri 09:30–15:59 ET.
Shorts disabled in Phase 5 (paper account needs margin enable).
All orders are limit orders within 0.1% of mid-price.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import numpy as np
import structlog

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from src.data.options_flow import get_options_flow
from src.execution.alpaca import AlpacaOrderRouter, OrderRequest
from src.execution.position_manager import PositionManager
from src.models.ensemble import EnsembleEngine, EnsembleSignal
from src.risk.circuit_breakers import CircuitBreakers, RiskState

# Inline constants from trading_env to avoid importing gymnasium at startup
ACTION_NAMES = [
    "hold", "buy_small", "buy_medium", "buy_large",
    "sell_25pct", "sell_50pct", "sell_all",
    "short_small", "short_large",
]
STATE_DIM = 27

logger = structlog.get_logger(__name__)


# ─── RL agent loader ──────────────────────────────────────────────────────────

def _load_rl_agent() -> Any | None:
    """Load the best PPO checkpoint from models/rl_agent/.

    Returns the loaded model or None if no checkpoint exists.
    """
    from pathlib import Path

    ckpt_dir = Path("models/rl_agent")
    # Prefer "best_ppo_*" checkpoints; fall back to periodic ones
    best_files = sorted(ckpt_dir.glob("best_ppo_*.zip"), reverse=True)
    periodic_files = sorted(ckpt_dir.glob("periodic/ppo_*.zip"), reverse=True)
    candidates = best_files or periodic_files
    if not candidates:
        logger.warning("rl_agent_no_checkpoint_found")
        return None

    ckpt_path = candidates[0]
    try:
        from stable_baselines3 import PPO  # optional dependency
        model = PPO.load(str(ckpt_path), device="cpu")
        logger.info("rl_agent_loaded", path=str(ckpt_path))
        return model
    except ImportError:
        logger.warning("rl_agent_stable_baselines3_not_installed")
        return None
    except Exception as exc:
        logger.warning("rl_agent_load_failed", path=str(ckpt_path), error=str(exc))
        return None


class SignalLoop:
    """Runs the 1m bar → ensemble → execution loop.

    Args:
        universe: List of ticker symbols to trade.
        ensemble: Loaded EnsembleEngine instance.
        alpaca: AlpacaOrderRouter for paper/live execution.
        circuit_breakers: Active CircuitBreakers instance.
        pos_manager: PositionManager tracking open positions.
        session_factory: SQLAlchemy async_sessionmaker.
        feature_cols: Ordered list of FFSA feature column names (top-30).
        broadcast_fn: Async callable to push data to dashboard WebSocket.
    """

    SIGNAL_ENTRY_THRESHOLD: float = 0.40   # "moderate" or stronger signal required
    BASE_SIZE_PCT: float = 0.05            # 5% base position before vol scaling
    SEQ_LEN: int = 60                      # 1m bars per inference window

    def __init__(
        self,
        universe: list[str],
        ensemble: EnsembleEngine,
        alpaca: AlpacaOrderRouter,
        circuit_breakers: CircuitBreakers,
        pos_manager: PositionManager,
        session_factory: Any,
        feature_cols: list[str],
        broadcast_fn: Callable[[dict[str, Any]], Coroutine] | None = None,
    ) -> None:
        self._universe = universe
        self._ensemble = ensemble
        self._alpaca = alpaca
        self._cb = circuit_breakers
        self._pm = pos_manager
        self._sf = session_factory
        self._feature_cols = feature_cols[:30]
        self._n_features = len(self._feature_cols)
        self._broadcast = broadcast_fn
        self._stopped = False
        self._latest_signals: list[EnsembleSignal] = []
        self._daily_start_value: float = pos_manager.portfolio_value
        self._consecutive_losses: int = 0
        self._open_trade_ids: dict[str, int] = {}  # ticker → Trade.id (for exit matching)

        # RL agent (optional — falls back to threshold logic if unavailable)
        self._rl_agent: Any | None = _load_rl_agent()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Run signal loop until stop() is called."""
        logger.info("signal_loop_started", universe=len(self._universe))
        while not self._stopped:
            try:
                await self._tick()
            except Exception:
                logger.exception("signal_loop_tick_error")
            await self._sleep_until_next_minute()

    async def stop(self) -> None:
        self._stopped = True
        logger.info("signal_loop_stopped")

    def get_latest_signals(self) -> list[dict[str, Any]]:
        """Return latest ensemble signals for API response."""
        return [s.to_dict() for s in self._latest_signals]

    # ── Main tick ────────────────────────────────────────────────────────────

    async def _tick(self) -> None:
        if not self._is_market_hours():
            return

        # 1. Fetch latest features + prices from DB
        features_map = await self._fetch_features()
        prices = await self._fetch_prices()
        self._pm.update_prices(prices)

        if not features_map:
            logger.warning("signal_loop_no_features")
            return

        # 2. Build tensors for tickers with enough data
        universe_features: dict[str, dict[str, torch.Tensor]] = {}
        for ticker in self._universe:
            arr = features_map.get(ticker)
            if arr is None or len(arr) < self.SEQ_LEN:
                continue
            feat_1m, feat_5m = self._to_tensors(arr)
            universe_features[ticker] = {"1m": feat_1m, "5m": feat_5m}

        if not universe_features:
            logger.warning("signal_loop_insufficient_data", universe=len(self._universe))
            return

        # 3. Compute ensemble signals
        signals = await self._ensemble.compute_universe(universe_features)
        self._latest_signals = signals
        logger.info("signal_loop_tick", n_signals=len(signals))

        # 4. Act on signals — RL agent uses obs from features; fallback uses threshold
        for sig in signals:
            if abs(sig.ensemble_signal) >= self.SIGNAL_ENTRY_THRESHOLD or self._rl_agent is not None:
                price = prices.get(sig.ticker, 0.0)
                if price > 0:
                    features_arr = universe_features.get(sig.ticker, {}).get("1m")
                    feat_np = (
                        features_arr.numpy() if features_arr is not None
                        else None
                    )
                    await self._act_on_signal(sig, price, feat_np)

        # 5. Sync position state from broker (catches fills we missed)
        try:
            await self._pm.sync_from_broker(self._alpaca)
        except Exception as exc:
            logger.warning("position_sync_failed", error=str(exc))

        # 6. Check circuit breakers
        state = RiskState(
            portfolio_value=self._pm.portfolio_value,
            peak_portfolio=self._pm._peak_value,
            daily_start_value=self._daily_start_value,
            vix=20.0,   # TODO: wire real VIX feed (Phase 6)
            consecutive_losses=self._consecutive_losses,
        )
        await self._cb.check(state)

        # 7. Reset daily start value at market open (09:30 ET)
        self._maybe_reset_daily_value()

        # 8. Broadcast to dashboard
        if self._broadcast:
            try:
                await self._broadcast({
                    "type": "signals",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "signals": [s.to_dict() for s in signals[:10]],
                    "positions": self._pm.get_positions(),
                    "portfolio_value": self._pm.portfolio_value,
                    "portfolio_heat": round(self._pm.portfolio_heat, 4),
                    "halted": self._cb.is_halted,
                    "halt_reason": self._cb.halt_reason,
                })
            except Exception as exc:
                logger.warning("broadcast_error", error=str(exc))

    # ── RL observation builder ────────────────────────────────────────────────

    def _build_rl_obs(
        self,
        sig: EnsembleSignal,
        ticker: str,
        features_arr: np.ndarray | None,
    ) -> np.ndarray:
        """Build a 27-dim RL observation from signal + position + FFSA features.

        Matches the state space defined in TradingEnv._build_obs():
          [ensemble_signal, transformer_conf, tcn_conf, sentiment_index,  # 4
           position_pct, unrealized_pnl, time_in_trade, portfolio_heat,   # 4
           vix_level, regime_label, recent_drawdown,                       # 3
           ffsa_features × 16]                                             # 16
        """
        pos = self._pm._positions.get(ticker)
        position_pct = 0.0
        unrealized_pnl = 0.0
        time_in_trade = 0.0

        if pos is not None:
            position_pct = (pos.qty * pos.entry_price) / max(self._pm.portfolio_value, 1.0)
            # Approximate unrealized PnL from position tracker
            unrealized_pnl = getattr(pos, "unrealized_pnl_pct", 0.0)
            time_in_trade = float(getattr(pos, "bars_held", 0)) / 100.0

        portfolio_heat = self._pm.portfolio_heat
        drawdown = max(0.0, 1.0 - self._pm.portfolio_value / max(self._daily_start_value, 1.0))

        # Options flow for VIX proxy and regime signal
        flow = get_options_flow(ticker)
        iv_rank = float(flow.get("iv_rank", 0.0))
        smart_money = float(flow.get("smart_money_score", 0.0))
        # Approximate VIX proxy from IV rank (clamp 0–1)
        vix_proxy = min(max(abs(iv_rank), 0.0), 1.0)
        # Regime: use smart_money_score magnitude as trending vs choppy proxy
        regime_proxy = float(abs(smart_money))

        state = [
            float(sig.ensemble_signal),
            float(sig.transformer_confidence),
            float(sig.tcn_confidence),
            float(sig.sentiment_index),
            float(position_pct),
            float(unrealized_pnl),
            float(time_in_trade),
            float(portfolio_heat),
            float(vix_proxy),
            float(regime_proxy),
            float(drawdown),
        ]

        # Top-10 FFSA features from the most recent row
        if features_arr is not None and len(features_arr) > 0:
            last_row = features_arr[-1][:10].tolist()
        else:
            last_row = []
        ffsa_padded = last_row + [0.0] * (16 - len(last_row))
        state.extend(ffsa_padded[:16])

        obs = np.array(state[:STATE_DIM], dtype=np.float32)
        # Replace NaN/Inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def _rl_action_to_side_and_size(
        self, action: int, ticker: str, price: float
    ) -> tuple[str | None, float, float]:
        """Map RL action index to (side, notional, qty).

        Returns (None, 0, 0) for hold or skip actions.
        Shorts (actions 7-8) are skipped in Phase 5 (no margin).
        """
        has_position = ticker in self._pm._positions
        portfolio = self._pm.portfolio_value

        # Map to base size %
        size_map = {
            0: None,    # hold
            1: 0.05,    # buy_small
            2: 0.10,    # buy_medium
            3: 0.20,    # buy_large
            4: None,    # sell_25pct (handled separately)
            5: None,    # sell_50pct (handled separately)
            6: None,    # sell_all  (handled separately)
            7: None,    # short_small — skip Phase 5
            8: None,    # short_large — skip Phase 5
        }

        # Sell actions: only valid if we have a position
        if action in (4, 5, 6) and has_position:
            pos = self._pm._positions[ticker]
            if action == 4:
                qty = round(pos.qty * 0.25, 2)
            elif action == 5:
                qty = round(pos.qty * 0.50, 2)
            else:
                qty = pos.qty
            return "sell", qty * price, qty

        size_pct = size_map.get(action)
        if size_pct is None or has_position:
            return None, 0.0, 0.0  # hold or already in position

        notional = self._pm.compute_position_size(
            ticker, base_size_pct=size_pct
        )
        qty = round(notional / max(price, 0.01), 2)
        return "buy", notional, qty

    # ── Order execution ───────────────────────────────────────────────────────

    async def _act_on_signal(
        self,
        sig: EnsembleSignal,
        price: float,
        features_arr: np.ndarray | None = None,
    ) -> None:
        """Use RL agent (or threshold fallback) to decide and execute an action."""
        if self._cb.is_halted:
            return

        ticker = sig.ticker
        has_position = ticker in self._pm._positions

        # ── RL-driven decision ────────────────────────────────────────────────
        if self._rl_agent is not None:
            obs = self._build_rl_obs(sig, ticker, features_arr)
            action, _ = self._rl_agent.predict(obs, deterministic=True)
            action = int(action)
            action_name = ACTION_NAMES[action]
            logger.debug(
                "rl_action",
                ticker=ticker,
                action=action_name,
                ensemble_signal=round(sig.ensemble_signal, 4),
            )

            side, notional, qty = self._rl_action_to_side_and_size(action, ticker, price)

            if side is None:
                return  # hold or skip
        else:
            # ── Threshold fallback (no RL model loaded) ───────────────────────
            if sig.ensemble_signal >= self.SIGNAL_ENTRY_THRESHOLD and not has_position:
                side = "buy"
            elif has_position:
                pos = self._pm._positions[ticker]
                if pos.side == "long" and sig.ensemble_signal < -0.20:
                    side = "sell"
                else:
                    return
            else:
                return

            if side == "buy":
                notional = self._pm.compute_position_size(
                    ticker, base_size_pct=self.BASE_SIZE_PCT
                )
                qty = round(notional / max(price, 0.01), 2)
            else:
                pos = self._pm._positions[ticker]
                qty = pos.qty
                notional = qty * price

        # ── Validation ────────────────────────────────────────────────────────
        if side == "buy":
            if notional < 10.0 or qty < 0.01:
                return
            proposed_pct = notional / max(self._pm.portfolio_value, 1.0)
            size_check = self._cb.check_position_size(proposed_pct)
            if size_check.triggered:
                logger.warning(
                    "order_rejected_oversized",
                    ticker=ticker,
                    proposed_pct=f"{proposed_pct:.1%}",
                )
                return

        # Get live quote for limit price
        quote = await self._alpaca.get_latest_quote(ticker)
        mid = quote.get("mid", price)
        if mid <= 0:
            mid = price

        if side == "buy":
            limit_price = round(mid * (1 + AlpacaOrderRouter.LIMIT_OFFSET_PCT), 2)
        else:
            limit_price = round(mid * (1 - AlpacaOrderRouter.LIMIT_OFFSET_PCT), 2)

        req = OrderRequest(
            ticker=ticker,
            side=side,
            qty=qty,
            limit_price=limit_price,
            reason=sig.plain_english(),
        )

        result = await self._alpaca.submit_order(req)

        if result.status in ("filled", "partially_filled"):
            fill_price = result.filled_avg_price or mid
            filled_at = result.filled_at or datetime.now(timezone.utc)
            if side == "buy":
                self._pm.open_position(
                    ticker=ticker,
                    side="long",
                    qty=result.filled_qty,
                    entry_price=fill_price,
                )
                await self._write_trade_entry(
                    ticker=ticker,
                    sig=sig,
                    fill_price=fill_price,
                    qty=result.filled_qty,
                    order_id=result.order_id,
                    entry_time=filled_at,
                )
            else:
                pnl = self._pm.close_position(ticker, fill_price)
                self._consecutive_losses = (
                    self._consecutive_losses + 1 if pnl < 0 else 0
                )
                self._pm.record_return(pnl / max(notional, 1.0))
                await self._write_trade_exit(
                    ticker=ticker,
                    fill_price=fill_price,
                    qty=result.filled_qty,
                    pnl=pnl,
                    exit_time=filled_at,
                    exit_reason="signal_reversal",
                )

            logger.info(
                "order_executed",
                ticker=ticker,
                side=side,
                qty=result.filled_qty,
                fill_price=fill_price,
                status=result.status,
                ensemble_signal=round(sig.ensemble_signal, 4),
            )
        else:
            logger.warning(
                "order_not_executed",
                ticker=ticker,
                side=side,
                status=result.status,
                error=result.error,
            )

    # ── Trade persistence ─────────────────────────────────────────────────────

    async def _write_trade_entry(
        self,
        ticker: str,
        sig: EnsembleSignal,
        fill_price: float,
        qty: float,
        order_id: str,
        entry_time: datetime,
    ) -> None:
        """Write an open trade entry to the trades table."""
        from src.config import get_settings
        from src.data.db import Trade

        mode = get_settings().alpaca_mode
        try:
            async with self._sf() as session:
                trade = Trade(
                    mode=mode,
                    ticker=ticker,
                    side="buy",
                    entry_time=entry_time,
                    entry_price=fill_price,
                    shares=qty,
                    transformer_direction=sig.transformer_direction,
                    transformer_confidence=sig.transformer_confidence,
                    tcn_direction=sig.tcn_direction,
                    tcn_confidence=sig.tcn_confidence,
                    sentiment_index=sig.sentiment_index,
                    ensemble_signal=sig.ensemble_signal,
                    alpaca_order_id=order_id,
                )
                session.add(trade)
                await session.flush()
                self._open_trade_ids[ticker] = trade.id
                await session.commit()
                logger.debug("trade_entry_written", ticker=ticker, trade_id=trade.id)
        except Exception as exc:
            logger.warning("trade_entry_write_failed", ticker=ticker, error=str(exc))

    async def _write_trade_exit(
        self,
        ticker: str,
        fill_price: float,
        qty: float,
        pnl: float,
        exit_time: datetime,
        exit_reason: str = "signal_reversal",
    ) -> None:
        """Update an existing trade row with exit price and PnL."""
        from sqlalchemy import update

        from src.data.db import Trade

        trade_id = self._open_trade_ids.pop(ticker, None)
        if trade_id is None:
            logger.debug("trade_exit_no_open_record", ticker=ticker)
            return

        notional = qty * fill_price
        pnl_pct = pnl / max(notional, 1.0)
        try:
            async with self._sf() as session:
                await session.execute(
                    update(Trade)
                    .where(Trade.id == trade_id)
                    .values(
                        exit_time=exit_time,
                        exit_price=fill_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                    )
                )
                await session.commit()
                logger.debug(
                    "trade_exit_written",
                    ticker=ticker,
                    trade_id=trade_id,
                    pnl=round(pnl, 2),
                )
        except Exception as exc:
            logger.warning("trade_exit_write_failed", ticker=ticker, error=str(exc))

    # ── Data fetching ─────────────────────────────────────────────────────────

    async def _fetch_features(self) -> dict[str, np.ndarray]:
        """Fetch last SEQ_LEN feature rows per ticker from DB (most recent first)."""
        from sqlalchemy import select

        from src.data.db import FeatureMatrix

        result: dict[str, np.ndarray] = {}
        async with self._sf() as session:
            for ticker in self._universe:
                rows = await session.execute(
                    select(FeatureMatrix.features)
                    .where(FeatureMatrix.ticker == ticker)
                    .order_by(FeatureMatrix.time.desc())
                    .limit(self.SEQ_LEN)
                )
                feat_rows = list(reversed(rows.scalars().all()))
                if not feat_rows:
                    continue
                arr = np.array(
                    [
                        [float((row or {}).get(f, 0.0) or 0.0) for f in self._feature_cols]
                        for row in feat_rows
                    ],
                    dtype=np.float32,
                )
                result[ticker] = arr
        return result

    async def _fetch_prices(self) -> dict[str, float]:
        """Fetch latest close price per ticker from the 1m OHLCV table."""
        from sqlalchemy import select

        from src.data.db import OHLCV1m

        prices: dict[str, float] = {}
        async with self._sf() as session:
            for ticker in self._universe:
                row = await session.execute(
                    select(OHLCV1m.close)
                    .where(OHLCV1m.ticker == ticker)
                    .order_by(OHLCV1m.time.desc())
                    .limit(1)
                )
                close = row.scalar_one_or_none()
                if close is not None:
                    prices[ticker] = float(close)
        return prices

    # ── Tensor construction ───────────────────────────────────────────────────

    def _to_tensors(
        self, arr: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert (n_rows, n_features) numpy array → (1m tensor, 5m tensor).

        1m tensor: last SEQ_LEN rows         → shape (SEQ_LEN, n_features)
        5m tensor: every 5th of 1m sequence  → shape (~12, n_features)
        """
        seq = arr[-self.SEQ_LEN :]                    # (60, n_features)
        seq_5m = seq[4::5]                            # every 5th bar (~12 bars)
        if _TORCH_AVAILABLE:
            return (
                torch.from_numpy(seq.copy()),
                torch.from_numpy(seq_5m.copy()),
            )
        # Return numpy arrays as-is if torch unavailable (ensemble will skip inference)
        return seq.copy(), seq_5m.copy()  # type: ignore[return-value]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_market_hours(self) -> bool:
        """Return True if currently within regular US market hours."""
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

        now = datetime.now(ZoneInfo("America/New_York"))
        if now.weekday() >= 5:
            return False
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=59, second=0, microsecond=0)
        return market_open <= now <= market_close

    def _maybe_reset_daily_value(self) -> None:
        """Reset daily start value at market open (9:30 ET)."""
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

        now = datetime.now(ZoneInfo("America/New_York"))
        if now.hour == 9 and now.minute == 30:
            self._daily_start_value = self._pm.portfolio_value
            logger.info(
                "daily_start_value_reset",
                value=self._daily_start_value,
            )

    async def _sleep_until_next_minute(self) -> None:
        """Sleep until the next 1m bar boundary (:00 seconds)."""
        now = datetime.now(timezone.utc)
        seconds_left = 60.0 - now.second - now.microsecond / 1_000_000
        await asyncio.sleep(max(seconds_left, 1.0))
