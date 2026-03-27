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
STATE_DIM = 29

# Position-sizing env constants (from position_sizing_env.py)
SIZING_ACTION_NAMES = ["skip", "tiny", "small", "medium", "large"]
SIZING_ACTION_PCTS = {0: 0.0, 1: 0.02, 2: 0.05, 3: 0.10, 4: 0.20}
SIZING_STATE_DIM = 18
SIZING_COST_THRESHOLD = 0.0015
SIZING_DIR_PROB_DEAD_ZONE = (0.45, 0.55)
SIZING_REVERSAL_BARS = 2

# Exit thresholds calibrated to equity intraday microstructure.
# Mega-cap 1m vol ≈ 0.02%, so 15-bar range ≈ 0.08%, 45-bar ≈ 0.13%.
# Old values (2%/2.5%/3.5%) were 15-25× expected range → 89% max_hold exits.
SIZING_STOP_LOSS = 0.004       # 0.4% — ~5× 1min vol → 3σ 15-bar move
SIZING_TRAILING_STOP = 0.005   # 0.5% — locks in gains within realistic range
SIZING_TAKE_PROFIT = 0.006     # 0.6% — reachable in 15 min on good signal
SIZING_MAX_HOLD_BARS = 15      # 15 min — matches LightGBM prediction horizon

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


def _load_sizing_agent() -> Any | None:
    """Load the position-sizing PPO model (best_sizing_ppo.zip).

    Returns the loaded model or None if not available.
    """
    from pathlib import Path

    ckpt_path = Path("models/rl_agent/best_sizing_ppo.zip")
    if not ckpt_path.exists():
        logger.info("sizing_agent_not_found")
        return None

    try:
        from stable_baselines3 import PPO
        model = PPO.load(str(ckpt_path), device="cpu")
        logger.info("sizing_agent_loaded", path=str(ckpt_path))
        return model
    except ImportError:
        logger.warning("sizing_agent_sb3_not_installed")
        return None
    except Exception as exc:
        logger.warning("sizing_agent_load_failed", error=str(exc))
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

    # Crypto tickers trade 24/7 — exempt from equity market-hours gate
    CRYPTO_TICKERS: frozenset[str] = frozenset({"BTC/USD", "ETH/USD", "SOL/USD"})

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
        self._feature_cols = feature_cols  # use all FFSA features (matches model)
        self._n_features = len(self._feature_cols)
        self._broadcast = broadcast_fn
        self._stopped = False
        self._latest_signals: list[EnsembleSignal] = []
        self._daily_start_value: float = pos_manager.portfolio_value
        self._consecutive_losses: int = 0
        self._open_trade_ids: dict[str, int] = {}  # ticker → Trade.id (for exit matching)

        # RL agent (optional — falls back to threshold logic if unavailable)
        self._rl_agent: Any | None = _load_rl_agent()

        # Position-sizing RL agent (preferred over full RL when available)
        self._sizing_agent: Any | None = _load_sizing_agent()
        self._sizing_mode = self._sizing_agent is not None

        # Sizing mode state tracking (per-ticker)
        self._entry_directions: dict[str, int] = {}      # +1 long, -1 short
        self._entry_prices: dict[str, float] = {}
        self._peak_prices: dict[str, float] = {}
        self._bars_held: dict[str, int] = {}
        self._reversal_counts: dict[str, int] = {}
        self._sizing_returns_history: list[float] = [0.0] * 20
        self._sizing_recent_outcomes: list[float] = []
        self._sizing_n_trades_today: int = 0
        self._pending_exit_reasons: dict[str, str] = {}  # ticker → exit reason

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

    def get_positions_detail(self) -> list[dict[str, Any]]:
        """Return enriched position data with exit levels for mobile app."""
        positions = []
        for ticker, p in self._pm._positions.items():
            entry_price = self._entry_prices.get(ticker, p.avg_entry_price)
            entry_dir = self._entry_directions.get(ticker, 1 if p.side == "long" else -1)
            peak = self._peak_prices.get(ticker, p.last_price)
            bars = self._bars_held.get(ticker, 0)

            if entry_dir > 0:
                stop_loss_price = round(entry_price * (1 - SIZING_STOP_LOSS), 2)
                take_profit_price = round(entry_price * (1 + SIZING_TAKE_PROFIT), 2)
                trailing_stop_price = round(peak * (1 - SIZING_TRAILING_STOP), 2)
            else:
                stop_loss_price = round(entry_price * (1 + SIZING_STOP_LOSS), 2)
                take_profit_price = round(entry_price * (1 - SIZING_TAKE_PROFIT), 2)
                trailing_stop_price = round(peak * (1 + SIZING_TRAILING_STOP), 2)

            positions.append({
                "ticker": ticker,
                "side": p.side,
                "qty": p.qty,
                "avg_entry_price": p.avg_entry_price,
                "last_price": p.last_price,
                "notional": round(p.notional, 2),
                "unrealized_pnl": round(p.unrealized_pnl, 2),
                "unrealized_pnl_pct": round(p.unrealized_pnl_pct, 4),
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "trailing_stop_pct": SIZING_TRAILING_STOP,
                "trailing_stop_price": trailing_stop_price,
                "peak_price": round(peak, 2),
                "bars_held": bars,
                "max_hold_bars": SIZING_MAX_HOLD_BARS,
                "bars_remaining": max(0, SIZING_MAX_HOLD_BARS - bars),
                "entry_direction": entry_dir,
            })
        return positions

    def get_actionable_signals(self) -> list[dict[str, Any]]:
        """Return signals that pass entry gate with recommended sizing."""
        if not self._sizing_mode:
            return []

        portfolio_value = self._pm.portfolio_value
        result = []
        for sig in self._latest_signals:
            if not self._sizing_entry_gate_open(sig):
                continue
            # Skip tickers with existing positions
            if sig.ticker in self._pm._positions:
                continue

            direction = 1.0 if float(sig.lgbm_pred_return) > 0 else -1.0
            side = "buy" if direction > 0 else "sell"

            # Estimate sizing action from RL agent or default
            size_pct = 0.02  # default tiny

            # Get current price from latest signal data
            sig_dict = sig.to_dict()
            result.append({
                **sig_dict,
                "actionable": True,
                "recommended_side": side,
                "recommended_size_pct": round(size_pct, 4),
                "recommended_notional": round(portfolio_value * size_pct, 2),
                "stop_loss_pct": SIZING_STOP_LOSS,
                "take_profit_pct": SIZING_TAKE_PROFIT,
                "trailing_stop_pct": SIZING_TRAILING_STOP,
                "max_hold_bars": SIZING_MAX_HOLD_BARS,
            })
        return result

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Return aggregated portfolio summary for mobile dashboard."""
        from src.config import get_settings

        pm = self._pm
        daily_pnl_pct = (
            (pm.portfolio_value / max(self._daily_start_value, 1) - 1) * 100
        )
        daily_pnl_dollar = pm.portfolio_value - self._daily_start_value

        return {
            "portfolio_value": round(pm.portfolio_value, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 3),
            "daily_pnl_dollar": round(daily_pnl_dollar, 2),
            "total_unrealized_pnl": round(pm.total_unrealized_pnl, 2),
            "portfolio_heat": round(pm.portfolio_heat, 4),
            "available_cash": round(pm.available_cash, 2),
            "drawdown_pct": round(pm.drawdown * 100, 3),
            "n_open_positions": len(pm._positions),
            "n_trades_today": self._sizing_n_trades_today,
            "consecutive_losses": self._consecutive_losses,
            "halted": self._cb.is_halted,
            "halt_reason": self._cb.halt_reason,
            "mode": get_settings().alpaca_mode,
            "market_open": self._is_market_hours(),
            "sizing_mode": self._sizing_mode,
        }

    # ── Main tick ────────────────────────────────────────────────────────────

    async def _tick(self) -> None:
        market_open = self._is_market_hours()
        # If market is closed and we have no crypto, skip entirely
        has_crypto = any(t in self.CRYPTO_TICKERS for t in self._universe)
        if not market_open and not has_crypto:
            return

        # 1. Fetch latest features + prices from DB
        features_map, regime_map = await self._fetch_features()
        prices = await self._fetch_prices()
        self._pm.update_prices(prices)

        # Check whether ML path is viable — LightGBM alone is sufficient
        # (it's the primary 60% signal model; Transformer/TCN are optional)
        _ml_viable = (
            self._n_features >= 10
            and (
                self._ensemble._lgbm is not None
                or (_TORCH_AVAILABLE and (
                    self._ensemble._transformer is not None
                    or self._ensemble._tcn is not None
                ))
            )
        )

        if not _ml_viable:
            # ── Rule-based fallback ───────────────────────────────────────────
            from src.models.ensemble import _LGBM_AVAILABLE
            logger.info(
                "signal_loop_using_rule_based_fallback",
                n_features=self._n_features,
                torch_available=_TORCH_AVAILABLE,
                lgbm_available=_LGBM_AVAILABLE,
                lgbm_loaded=self._ensemble._lgbm is not None,
                transformer_loaded=self._ensemble._transformer is not None,
                tcn_loaded=self._ensemble._tcn is not None,
            )
            signals = await self._rule_based_tick(prices)
            if not signals:
                logger.warning("signal_loop_rule_based_no_data")
                return
            self._latest_signals = signals
            logger.info("signal_loop_tick_rule_based", n_signals=len(signals))
        else:
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
        # universe_features only exists in the ML path; rule-based path has no tensors
        _uf: dict = locals().get("universe_features", {})

        # Build a signal lookup for sizing mode exit checks
        sig_by_ticker: dict[str, EnsembleSignal] = {s.ticker: s for s in signals}

        # Sizing mode: check exits for ALL open positions (even without signal threshold)
        if self._sizing_mode:
            for ticker in list(self._pm._positions.keys()):
                if not market_open and ticker not in self.CRYPTO_TICKERS:
                    continue
                price = prices.get(ticker, 0.0)
                if price <= 0:
                    continue
                sig = sig_by_ticker.get(ticker)
                if sig is None:
                    # Create a minimal signal for exit checking
                    sig = EnsembleSignal(
                        ticker=ticker,
                        timestamp=datetime.now(timezone.utc),
                    )
                features_arr = _uf.get(ticker, {}).get("1m") if _uf else None
                feat_np = (
                    features_arr.numpy() if features_arr is not None
                    else None
                )
                await self._act_on_signal(sig, price, feat_np, regime=regime_map.get(ticker, 1))

        for sig in signals:
            # Gate equity tickers on market hours; crypto runs 24/7
            if not market_open and sig.ticker not in self.CRYPTO_TICKERS:
                continue
            # Skip tickers already handled by sizing exit above
            if self._sizing_mode and sig.ticker in self._pm._positions:
                continue
            regime = regime_map.get(sig.ticker, 1)  # default: choppy
            # Regime-aware threshold: choppy/high-vol require stronger signal
            from src.features.regime import REGIME_GATE
            threshold, _size_scale = REGIME_GATE.get(regime, (self.SIGNAL_ENTRY_THRESHOLD, 1.0))
            # In sizing mode, use LightGBM entry gate instead of ensemble threshold
            should_act = (
                self._sizing_mode
                or abs(sig.ensemble_signal) >= threshold
            )
            if should_act:
                price = prices.get(sig.ticker, 0.0)
                if price > 0:
                    features_arr = _uf.get(sig.ticker, {}).get("1m") if _uf else None
                    feat_np = (
                        features_arr.numpy() if features_arr is not None
                        else None
                    )
                    await self._act_on_signal(sig, price, feat_np, regime=regime)

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
            portfolio_heat=self._pm.portfolio_heat,
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
                    "positions_detail": self.get_positions_detail(),
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
        regime: int = 0,
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
            position_pct = (pos.qty * pos.avg_entry_price) / max(self._pm.portfolio_value, 1.0)
            # Approximate unrealized PnL from position tracker
            unrealized_pnl = getattr(pos, "unrealized_pnl_pct", 0.0)
            time_in_trade = float(getattr(pos, "bars_held", 0)) / 100.0

        portfolio_heat = self._pm.portfolio_heat
        drawdown = max(0.0, 1.0 - self._pm.portfolio_value / max(self._daily_start_value, 1.0))

        # Options flow for VIX proxy
        flow = get_options_flow(ticker)
        iv_rank = float(flow.get("iv_rank", 0.0))
        # Approximate VIX proxy from IV rank (clamp 0–1)
        vix_proxy = min(max(abs(iv_rank), 0.0), 1.0)
        # Real regime from feature_matrix (0=trending,1=choppy,2=high_vol)
        regime_proxy = float(regime) / 2.0   # normalize to [0,1] for RL obs

        state = [
            float(sig.ensemble_signal),
            float(sig.transformer_confidence),
            float(sig.tcn_confidence),
            float(sig.sentiment_index),
            float(sig.lgbm_pred_return) * 100.0,  # scale up for RL
            float(sig.lgbm_dir_prob),
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

    # ── Sizing-mode observation builder ──────────────────────────────────────

    def _build_sizing_obs(
        self,
        sig: EnsembleSignal,
        ticker: str,
        features_arr: np.ndarray | None,
    ) -> np.ndarray:
        """Build 18-dim observation for position-sizing RL agent.

        Matches PositionSizingEnv._build_obs() state space.
        """
        pred_ret = float(sig.lgbm_pred_return)
        dir_prob = float(sig.lgbm_dir_prob)
        drawdown = max(0.0, 1.0 - self._pm.portfolio_value / max(self._daily_start_value, 1.0))
        rolling_vol = float(np.std(self._sizing_returns_history[-20:]) + 1e-9)

        # Position state for this ticker
        pos = self._pm._positions.get(ticker)
        position_pct = 0.0
        unrealized_pnl = 0.0
        bars_held = 0

        if pos is not None:
            position_pct = (pos.qty * pos.avg_entry_price) / max(self._pm.portfolio_value, 1.0)
            if pos.side == "short":
                position_pct = -position_pct
            unrealized_pnl = getattr(pos, "unrealized_pnl_pct", 0.0)
            bars_held = self._bars_held.get(ticker, 0)

        recent_wr = 0.5
        if self._sizing_recent_outcomes:
            recent_wr = sum(1 for x in self._sizing_recent_outcomes if x > 0) / len(self._sizing_recent_outcomes)

        state = [
            pred_ret * 100.0,                                       # [0] scaled pred return
            dir_prob,                                                # [1] direction probability
            abs(pred_ret) / max(SIZING_COST_THRESHOLD, 1e-6),       # [2] signal-to-cost ratio
            float(position_pct),                                     # [3] current position
            float(unrealized_pnl),                                   # [4] unrealized PnL
            float(bars_held) / 60.0,                                 # [5] normalized bars held
            float(abs(position_pct)),                                # [6] portfolio heat
            float(drawdown),                                         # [7] drawdown
            float(rolling_vol) * 100.0,                              # [8] scaled rolling vol
            20.0 / 100.0,                                            # [9] VIX placeholder
            0.0,                                                     # [10] regime
            float(recent_wr),                                        # [11] recent win rate
            float(self._sizing_n_trades_today) / 10.0,              # [12] trades today
        ]

        # Top-5 FFSA features from most recent row
        if features_arr is not None and len(features_arr) > 0:
            last_row = features_arr[-1][:5].tolist()
        else:
            last_row = []
        ffsa_padded = last_row + [0.0] * (5 - len(last_row))
        state.extend(ffsa_padded[:5])

        obs = np.array(state[:SIZING_STATE_DIM], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    # ── Sizing-mode entry/exit gating ────────────────────────────────────────

    def _sizing_entry_gate_open(self, sig: EnsembleSignal) -> bool:
        """Check if LightGBM signal warrants entry (sizing mode)."""
        pred_ret = float(sig.lgbm_pred_return)
        dir_prob = float(sig.lgbm_dir_prob)
        lo, hi = SIZING_DIR_PROB_DEAD_ZONE
        return abs(pred_ret) > SIZING_COST_THRESHOLD and not (lo < dir_prob < hi)

    def _sizing_signal_direction(self, sig: EnsembleSignal) -> int:
        """Get entry direction from LightGBM: +1 long, -1 short."""
        pred_ret = float(sig.lgbm_pred_return)
        if pred_ret > 0:
            return 1
        elif pred_ret < 0:
            return -1
        return 0

    def _check_sizing_exit(
        self, ticker: str, price: float, sig: EnsembleSignal
    ) -> str | None:
        """Check exit conditions for a position in sizing mode.

        Returns exit reason string or None to continue holding.
        """
        entry_price = self._entry_prices.get(ticker)
        entry_dir = self._entry_directions.get(ticker, 0)
        if entry_price is None or entry_dir == 0:
            # Reconstruct sizing state from broker position (survives restarts)
            pos = self._pm._positions.get(ticker)
            if pos is None:
                return None
            entry_price = pos.avg_entry_price
            entry_dir = 1 if pos.side == "long" else -1
            self._entry_prices[ticker] = entry_price
            self._entry_directions[ticker] = entry_dir
            self._peak_prices[ticker] = pos.last_price or entry_price
            # Position predates this deployment — force max_hold exit
            self._bars_held[ticker] = SIZING_MAX_HOLD_BARS

        # Unrealized PnL
        unrealized = (price - entry_price) / entry_price
        if entry_dir < 0:
            unrealized = -unrealized

        # Stop loss
        if unrealized < -SIZING_STOP_LOSS:
            return "stop_loss"

        # Take profit
        if unrealized > SIZING_TAKE_PROFIT:
            return "take_profit"

        # Trailing stop
        peak = self._peak_prices.get(ticker, price)
        if entry_dir > 0:
            drop = (peak - price) / peak if peak > 0 else 0.0
            if drop > SIZING_TRAILING_STOP:
                return "trailing_stop"
        else:
            rise = (price - peak) / peak if peak > 0 else 0.0
            if rise > SIZING_TRAILING_STOP:
                return "trailing_stop"

        # Max hold
        bars = self._bars_held.get(ticker, 0)
        if bars >= SIZING_MAX_HOLD_BARS:
            return "max_hold"

        # Signal reversal (2 consecutive bars of opposite direction)
        current_dir = self._sizing_signal_direction(sig)
        if current_dir != 0 and current_dir != entry_dir:
            self._reversal_counts[ticker] = self._reversal_counts.get(ticker, 0) + 1
        else:
            self._reversal_counts[ticker] = 0
        if self._reversal_counts.get(ticker, 0) >= SIZING_REVERSAL_BARS:
            return "signal_reversal"

        return None

    def _clear_sizing_state(self, ticker: str) -> None:
        """Clear per-ticker sizing state after position close."""
        self._entry_directions.pop(ticker, None)
        self._entry_prices.pop(ticker, None)
        self._peak_prices.pop(ticker, None)
        self._bars_held.pop(ticker, None)
        self._reversal_counts.pop(ticker, None)

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

    MAX_PORTFOLIO_HEAT: float = 0.80   # circuit breaker: no new entries above 80%

    async def _act_on_signal(
        self,
        sig: EnsembleSignal,
        price: float,
        features_arr: np.ndarray | None = None,
        regime: int = 0,
    ) -> None:
        """Use RL agent (or threshold fallback) to decide and execute an action."""
        if self._cb.is_halted:
            return

        ticker = sig.ticker
        has_position = ticker in self._pm._positions

        # Portfolio heat circuit breaker — no new entries when > 80% deployed.
        # Sell orders are always allowed (they reduce heat).
        if not has_position and self._pm.portfolio_heat > self.MAX_PORTFOLIO_HEAT:
            logger.warning(
                "new_entry_blocked_portfolio_heat",
                ticker=ticker,
                heat=round(self._pm.portfolio_heat, 3),
                limit=self.MAX_PORTFOLIO_HEAT,
            )
            return

        # Regime-based position size scale (0.5× in high_vol, 0.7× in choppy)
        from src.features.regime import REGIME_GATE, regime_label
        _, size_scale = REGIME_GATE.get(regime, (0.40, 1.0))
        if size_scale < 1.0 and not has_position:
            logger.info(
                "regime_size_scaled",
                ticker=ticker,
                regime=regime_label(regime),
                size_scale=size_scale,
            )

        # ── Position-sizing mode (LightGBM gates entry/exit, RL sizes) ────────
        if self._sizing_mode:
            if has_position:
                # Check exit conditions
                exit_reason = self._check_sizing_exit(ticker, price, sig)
                if exit_reason:
                    side = "sell"
                    pos = self._pm._positions[ticker]
                    qty = pos.qty
                    notional = qty * price
                    self._pending_exit_reasons[ticker] = exit_reason
                    logger.info(
                        "sizing_exit",
                        ticker=ticker,
                        reason=exit_reason,
                        bars_held=self._bars_held.get(ticker, 0),
                    )
                else:
                    # Still holding — update tracking state
                    self._bars_held[ticker] = self._bars_held.get(ticker, 0) + 1
                    if ticker in self._peak_prices:
                        entry_dir = self._entry_directions.get(ticker, 1)
                        if entry_dir > 0:
                            self._peak_prices[ticker] = max(self._peak_prices[ticker], price)
                        else:
                            self._peak_prices[ticker] = min(self._peak_prices[ticker], price)
                    self._sizing_returns_history.append(0.0)
                    return
            else:
                # Block crypto entries — LightGBM was trained on equities only.
                # Running it on crypto features produces noise predictions that
                # churn through stop-losses and bleed the portfolio.
                if ticker in self.CRYPTO_TICKERS:
                    logger.debug("sizing_skip_crypto", ticker=ticker)
                    return

                # Check entry gate
                if not self._sizing_entry_gate_open(sig):
                    return

                # Ask RL for position size
                obs = self._build_sizing_obs(sig, ticker, features_arr)
                action, _ = self._sizing_agent.predict(obs, deterministic=True)
                action = int(action)
                size_pct = SIZING_ACTION_PCTS.get(action, 0.0)
                action_name = SIZING_ACTION_NAMES[action]

                if size_pct == 0.0:
                    logger.debug("sizing_skip", ticker=ticker, action=action_name)
                    return

                # Apply regime scaling
                size_pct *= size_scale

                direction = self._sizing_signal_direction(sig)
                if direction == 0:
                    return
                # Phase 5: no shorts
                if direction < 0:
                    return

                side = "buy"
                notional = self._pm.compute_position_size(
                    ticker, base_size_pct=size_pct
                )
                qty = round(notional / max(price, 0.01), 2)

                # Track entry state
                self._entry_directions[ticker] = direction
                self._entry_prices[ticker] = price
                self._peak_prices[ticker] = price
                self._bars_held[ticker] = 0
                self._reversal_counts[ticker] = 0
                self._sizing_n_trades_today += 1

                logger.info(
                    "sizing_entry",
                    ticker=ticker,
                    action=action_name,
                    size_pct=round(size_pct, 3),
                    direction="long" if direction > 0 else "short",
                    lgbm_pred=round(sig.lgbm_pred_return, 5),
                )

        # ── Block crypto entries in non-sizing modes too ──────────────────────
        elif not has_position and ticker in self.CRYPTO_TICKERS:
            logger.debug("skip_crypto_entry", ticker=ticker)
            return

        # ── RL-driven decision (full 9-action mode) ──────────────────────────
        elif self._rl_agent is not None:
            obs = self._build_rl_obs(sig, ticker, features_arr, regime=regime)
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

        # Sell exits use market orders (Alpaca rejects limit orders with
        # fractional qty, and we want guaranteed fills on exits).
        # Buy entries use limit orders for price protection.
        if side == "sell":
            limit_price = None  # market order
        else:
            quote = await self._alpaca.get_latest_quote(ticker)
            mid = quote.get("mid", price)
            if mid <= 0:
                mid = price
            limit_price = round(mid * (1 + AlpacaOrderRouter.LIMIT_OFFSET_PCT), 2)

        req = OrderRequest(
            ticker=ticker,
            side=side,
            qty=qty,
            limit_price=limit_price,
            reason=sig.plain_english(),
        )

        result = await self._alpaca.submit_order(req)

        if result.status in ("filled", "partially_filled"):
            fill_price = result.filled_avg_price or price
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
                # Capture entry price BEFORE closing position
                pos = self._pm.get_position(ticker)
                entry_price = (
                    pos.avg_entry_price if pos else
                    self._entry_prices.get(ticker, fill_price)
                )
                entry_notional = qty * entry_price if entry_price else notional
                pnl = self._pm.close_position(ticker, fill_price)
                self._consecutive_losses = (
                    self._consecutive_losses + 1 if pnl < 0 else 0
                )
                self._pm.record_return(pnl / max(entry_notional, 1.0))
                # Determine exit reason for sizing mode
                exit_reason = "signal_reversal"
                if self._sizing_mode:
                    exit_reason = self._pending_exit_reasons.pop(ticker, "signal_reversal")
                    # Track outcome for recent win rate
                    self._sizing_recent_outcomes.append(pnl)
                    if len(self._sizing_recent_outcomes) > 10:
                        self._sizing_recent_outcomes = self._sizing_recent_outcomes[-10:]
                    self._clear_sizing_state(ticker)
                await self._write_trade_exit(
                    ticker=ticker,
                    fill_price=fill_price,
                    qty=result.filled_qty,
                    pnl=pnl,
                    exit_time=filled_at,
                    exit_reason=exit_reason,
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
            logger.error("trade_entry_write_failed", ticker=ticker, error=str(exc), exc_info=True)

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
            # Recover orphaned trade: find the most recent open trade for this
            # ticker in the DB. This handles restarts/redeploys where the
            # in-memory _open_trade_ids dict was lost.
            from sqlalchemy import select as _sel
            from src.data.db import Trade as _T
            try:
                async with self._sf() as session:
                    result = await session.execute(
                        _sel(_T.id)
                        .where(_T.ticker == ticker, _T.exit_time.is_(None))
                        .order_by(_T.entry_time.desc())
                        .limit(1)
                    )
                    row = result.scalar_one_or_none()
                    if row is not None:
                        trade_id = row
                        logger.info("trade_exit_recovered_orphan", ticker=ticker, trade_id=trade_id)
            except Exception as exc:
                logger.warning("trade_exit_orphan_recovery_failed", ticker=ticker, error=str(exc))

        if trade_id is None:
            logger.debug("trade_exit_no_open_record", ticker=ticker)
            return

        # Use entry notional for accurate pnl_pct (not exit notional)
        entry_notional = qty * fill_price - pnl  # entry_price * qty = exit_notional - pnl (for longs)
        pnl_pct = pnl / max(abs(entry_notional), 1.0)
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

    # ── Rule-based signal path ────────────────────────────────────────────────

    async def _rule_based_tick(self, prices: dict[str, float]) -> list[EnsembleSignal]:
        """Generate signals via MACD + RSI when ML models/features unavailable.

        Fetches last 50 1m closes from the OHLCV DB table (or uses yfinance as
        a secondary fallback).  Computes rule-based signal for each ticker with
        enough data.  Returns list sorted by |signal| desc.
        """
        from src.models.ensemble import EnsembleEngine

        signals: list[EnsembleSignal] = []
        for ticker in self._universe:
            closes = await self._fetch_ohlcv_closes(ticker, bars=50)
            if len(closes) < 27:
                # Secondary fallback: try yfinance for recent 1m bars
                closes = await self._yfinance_closes(ticker, bars=50)
            if len(closes) < 27:
                logger.debug("rule_based_skip_no_data", ticker=ticker, bars=len(closes))
                continue
            sig = EnsembleEngine.compute_signal_rule_based(ticker, closes)
            signals.append(sig)

        return sorted(signals, key=lambda s: abs(s.ensemble_signal), reverse=True)

    async def _fetch_ohlcv_closes(self, ticker: str, bars: int = 50) -> list[float]:
        """Fetch last `bars` close prices from ohlcv_1m table."""
        from sqlalchemy import select

        from src.data.db import OHLCV1m

        try:
            async with self._sf() as session:
                rows = await session.execute(
                    select(OHLCV1m.close)
                    .where(OHLCV1m.ticker == ticker)
                    .order_by(OHLCV1m.time.desc())
                    .limit(bars)
                )
                closes = list(reversed([float(r) for r in rows.scalars().all() if r is not None]))
            return closes
        except Exception as exc:
            logger.warning("ohlcv_closes_fetch_failed", ticker=ticker, error=str(exc))
            return []

    async def _yfinance_closes(self, ticker: str, bars: int = 50) -> list[float]:
        """Fetch recent 1m closes from yfinance as a last-resort fallback."""
        try:
            loop = asyncio.get_event_loop()

            def _fetch() -> list[float]:
                import yfinance as yf  # optional dependency
                df = yf.download(ticker, period="1d", interval="1m", progress=False)
                if df is None or df.empty:
                    return []
                return df["Close"].dropna().tolist()[-bars:]

            closes = await loop.run_in_executor(None, _fetch)
            if closes:
                logger.info("yfinance_closes_fetched", ticker=ticker, bars=len(closes))
            return closes
        except Exception as exc:
            logger.debug("yfinance_closes_failed", ticker=ticker, error=str(exc))
            return []

    # ── Data fetching ─────────────────────────────────────────────────────────

    async def _fetch_features(self) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        """Fetch last SEQ_LEN feature rows per ticker from DB.

        Returns:
            features_map: ticker → (SEQ_LEN, n_features) float32 array
            regime_map:   ticker → latest regime int (0=trending,1=choppy,2=high_vol)
        """
        from sqlalchemy import select

        from src.data.db import FeatureMatrix

        result: dict[str, np.ndarray] = {}
        regime_map: dict[str, int] = {}
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
                # Latest regime from most-recent row (last after reversing)
                latest_row = feat_rows[-1] or {}
                regime_map[ticker] = int(latest_row.get("regime", 1) or 1)
        return result, regime_map

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
        today = now.date()
        if now.hour == 9 and now.minute <= 31 and getattr(self, '_last_reset_date', None) != today:
            self._last_reset_date = today
            self._daily_start_value = self._pm.portfolio_value
            self._sizing_n_trades_today = 0
            logger.info(
                "daily_start_value_reset",
                value=self._daily_start_value,
            )

    async def _sleep_until_next_minute(self) -> None:
        """Sleep until the next 1m bar boundary (:00 seconds)."""
        now = datetime.now(timezone.utc)
        seconds_left = 60.0 - now.second - now.microsecond / 1_000_000
        await asyncio.sleep(max(seconds_left, 1.0))
