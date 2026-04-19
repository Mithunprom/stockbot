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
from src.execution.position_sizer import SmartPositionSizer, SECTOR_MAP
from src.models.ensemble import EnsembleEngine, EnsembleSignal
from src.risk.circuit_breakers import CircuitBreakers, RiskState

# Inline constants from trading_env to avoid importing gymnasium at startup
ACTION_NAMES = [
    "hold", "buy_small", "buy_medium", "buy_large",
    "sell_25pct", "sell_50pct", "sell_all",
    "short_small", "short_large",
]
STATE_DIM = 29

# Signal quality thresholds for entry gating
SIZING_COST_THRESHOLD = 0.0015  # min |pred_return| — only enter on stronger signals (was 0.001)
SIZING_DIR_PROB_DEAD_ZONE = (0.45, 0.55)  # dir_prob inside this range → skip
SIZING_REVERSAL_BARS = 2

# Anti-churn controls
SIZING_MAX_TRADES_PER_DAY = 5       # fewer, higher-quality trades (was 8)
SIZING_TICKER_COOLDOWN_BARS = 60    # 1 hour cooldown — stop re-entering noise (was 30)

# Data freshness gate — skip new entries when features are stale
DATA_FRESHNESS_MAX_MINUTES = 5      # max age of latest feature row before gating entries

# Exit thresholds — ATR-adaptive with floors.
# ATR here is computed on 1-minute bars (atr_14 / close), so values are small:
#   AAPL ≈ 0.07%, NVDA ≈ 0.09%, MSTR ≈ 0.14%, CVX ≈ 0.04%.
# Multipliers are scaled accordingly to produce meaningful intraday stops.
# R:R stays ~2.3:1 regardless of volatility.
SIZING_STOP_LOSS_ATR_MULT = 15     # stop = ATR × 15  (NVDA: 0.09%×15 = 1.35%)
SIZING_TRAILING_STOP_ATR_MULT = 20 # trailing = ATR × 20  (NVDA: 0.09%×20 = 1.8%)
SIZING_TAKE_PROFIT_ATR_MULT = 35   # target = ATR × 35  (NVDA: 0.09%×35 = 3.15%)
SIZING_STOP_LOSS_FLOOR = 0.005     # 0.5% minimum stop (ultra-low-vol stocks)
SIZING_TRAILING_STOP_FLOOR = 0.008 # 0.8% minimum trailing
SIZING_TAKE_PROFIT_FLOOR = 0.015   # 1.5% minimum take profit
SIZING_MAX_HOLD_BARS = 45      # 45 min — give trades time to develop momentum
DEFAULT_ATR_PCT = 0.001            # fallback when ATR unavailable (typical 1-min ATR)

logger = structlog.get_logger(__name__)


def _atr_exits(atr_pct: float) -> tuple[float, float, float]:
    """Compute (stop_loss, trailing_stop, take_profit) from per-ticker ATR%.

    Returns percentages scaled to the stock's volatility with hard floors
    so low-vol stocks still have meaningful thresholds.
    """
    sl = max(atr_pct * SIZING_STOP_LOSS_ATR_MULT, SIZING_STOP_LOSS_FLOOR)
    ts = max(atr_pct * SIZING_TRAILING_STOP_ATR_MULT, SIZING_TRAILING_STOP_FLOOR)
    tp = max(atr_pct * SIZING_TAKE_PROFIT_ATR_MULT, SIZING_TAKE_PROFIT_FLOOR)
    return sl, ts, tp


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
        pipeline_id: str = "pipeline_a",
    ) -> None:
        self._universe = universe
        self._ensemble = ensemble
        self._alpaca = alpaca
        self._cb = circuit_breakers
        self._pm = pos_manager
        self._sf = session_factory
        self._pipeline_id = pipeline_id
        self._feature_cols = feature_cols  # use all FFSA features (matches model)
        self._n_features = len(self._feature_cols)
        self._broadcast = broadcast_fn
        self._stopped = False
        self._latest_signals: list[EnsembleSignal] = []
        self._daily_start_value: float = pos_manager.portfolio_value
        self._consecutive_losses: int = 0
        # A/B testing: reference to the OTHER pipeline's PositionManager.
        # When set, prevents both pipelines from opening the same ticker.
        self._other_pm: PositionManager | None = None
        self._open_trade_ids: dict[str, int] = {}  # ticker → Trade.id (for exit matching)

        # RL agent (legacy — kept for rule-based / non-sizing fallback)
        self._rl_agent: Any | None = _load_rl_agent()

        # Sizing mode: LightGBM gates entry/exit, SmartPositionSizer sizes.
        # Activated when LightGBM is loaded.
        self._sizing_mode = self._ensemble._lgbm is not None

        # Smart Position Sizer — 6-stage pipeline replacing flat % / RL sizing
        from src.config import get_settings
        self._sizer = SmartPositionSizer(mode=get_settings().alpaca_mode)

        # Sizing mode state tracking (per-ticker)
        self._entry_directions: dict[str, int] = {}      # +1 long, -1 short
        self._entry_prices: dict[str, float] = {}
        self._peak_prices: dict[str, float] = {}
        self._bars_held: dict[str, int] = {}
        self._reversal_counts: dict[str, int] = {}
        self._sizing_returns_history: list[float] = [0.0] * 20
        self._sizing_recent_outcomes: list[float] = []
        self._sizing_n_trades_today: int = 0
        self._ticker_cooldown: dict[str, int] = {}  # ticker → bars remaining

        # Per-ticker ATR cache (refreshed each tick from feature_matrix)
        self._ticker_atr: dict[str, float] = {}

        # Data freshness flag — set False when features are stale to block new entries
        self._data_fresh: bool = True

        # Kelly gate — blocks new entries when rolling Kelly fraction < 0.
        # Updated after each closed trade. Starts at 0 (conservative: no trades
        # until enough history proves positive expected value).
        self._kelly_fraction: float = 0.0
        self._kelly_min_trades: int = 20  # need ≥20 closed trades to compute
        self._pending_exit_reasons: dict[str, str] = {}  # ticker → exit reason

        # Live IC Tracker — set via set_ic_tracker() after construction
        # (tracker is created after signal loop in main.py startup sequence)
        self._ic_tracker: Any | None = None

    def set_ic_tracker(self, tracker: Any) -> None:
        """Attach a LiveICTracker instance for prediction recording.

        Called from main.py after both the signal loop and tracker are created.
        """
        self._ic_tracker = tracker
        logger.info("ic_tracker_attached_to_signal_loop")

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Run signal loop until stop() is called."""
        # Seed Kelly gate from DB history so it activates immediately
        try:
            await self._seed_kelly_from_db()
        except Exception:
            logger.exception("kelly_seed_startup_failed_continuing")

        logger.info(
            "signal_loop_started",
            universe=len(self._universe),
            pipeline=self._pipeline_id,
            kelly=round(self._kelly_fraction, 4),
            kelly_n=len(self._sizing_recent_outcomes),
        )
        _tick_count = 0
        while not self._stopped:
            try:
                await self._tick()
                _tick_count += 1
                if _tick_count % 30 == 0:
                    logger.info(
                        "signal_loop_heartbeat",
                        pipeline=self._pipeline_id,
                        ticks=_tick_count,
                        positions=len(self._pm._positions),
                    )
                await self._sleep_until_next_minute()
            except asyncio.CancelledError:
                logger.warning("signal_loop_cancelled", pipeline=self._pipeline_id)
                raise
            except Exception:
                logger.exception("signal_loop_tick_error", pipeline=self._pipeline_id)
                # Avoid tight crash loop — sleep before retrying
                await asyncio.sleep(5)

    async def stop(self) -> None:
        self._stopped = True
        logger.info("signal_loop_stopped")

    async def _seed_kelly_from_db(self) -> None:
        """Load recent closed trade PnLs from DB to bootstrap Kelly gate.

        Without this, the Kelly gate would need 20 new trades before it
        can decide whether to block entries. By seeding from history, the
        gate activates immediately on startup.

        Filters by pipeline_id so each pipeline only sees its own history,
        and excludes crypto tickers (old crypto trades had noise signals).
        """
        from sqlalchemy import select as _sel
        from src.data.db import Trade as _T

        try:
            async with self._sf() as session:
                query = (
                    _sel(_T.pnl)
                    .where(
                        _T.exit_time.isnot(None),
                        _T.pnl.isnot(None),
                        ~_T.ticker.in_(list(self.CRYPTO_TICKERS)),
                    )
                    .order_by(_T.exit_time.desc())
                    .limit(50)
                )
                # Filter by pipeline_id if set (A/B mode)
                if self._pipeline_id:
                    query = query.where(_T.pipeline_id == self._pipeline_id)

                result = await session.execute(query)
                pnls = [float(r) for r in result.scalars().all() if r is not None]

            if pnls:
                self._sizing_recent_outcomes = list(reversed(pnls))
                self._update_kelly()
                logger.info(
                    "kelly_seeded_from_db",
                    n_trades=len(pnls),
                    kelly=round(self._kelly_fraction, 4),
                    pipeline=self._pipeline_id,
                )
            else:
                # No history for this pipeline — start fresh (no Kelly gate)
                logger.info(
                    "kelly_no_history",
                    pipeline=self._pipeline_id,
                    note="Kelly gate inactive until 20+ trades",
                )
        except Exception as exc:
            logger.warning("kelly_seed_failed", error=str(exc))

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

            atr_pct = self._ticker_atr.get(ticker, DEFAULT_ATR_PCT)
            sl, ts, tp = _atr_exits(atr_pct)

            if entry_dir > 0:
                stop_loss_price = round(entry_price * (1 - sl), 2)
                take_profit_price = round(entry_price * (1 + tp), 2)
                trailing_stop_price = round(peak * (1 - ts), 2)
            else:
                stop_loss_price = round(entry_price * (1 + sl), 2)
                take_profit_price = round(entry_price * (1 - tp), 2)
                trailing_stop_price = round(peak * (1 + ts), 2)

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
                "stop_loss_pct": round(sl, 4),
                "take_profit_price": take_profit_price,
                "take_profit_pct": round(tp, 4),
                "trailing_stop_pct": round(ts, 4),
                "trailing_stop_price": trailing_stop_price,
                "peak_price": round(peak, 2),
                "bars_held": bars,
                "max_hold_bars": SIZING_MAX_HOLD_BARS,
                "bars_remaining": max(0, SIZING_MAX_HOLD_BARS - bars),
                "entry_direction": entry_dir,
                "atr_pct": round(atr_pct, 4),
            })
        return positions

    def get_actionable_signals(self) -> list[dict[str, Any]]:
        """Return signals that pass entry gate with recommended sizing."""
        if not self._sizing_mode:
            return []

        sector_notionals = self._compute_sector_notionals()
        result = []
        for sig in self._latest_signals:
            if not self._sizing_entry_gate_open(sig):
                continue
            if sig.ticker in self._pm._positions:
                continue

            sizing = self._sizer.compute(
                ticker=sig.ticker,
                dir_prob=float(sig.lgbm_dir_prob),
                pred_return=float(sig.lgbm_pred_return),
                atr_pct=self._ticker_atr.get(sig.ticker, 0.01),
                price=sig.price if hasattr(sig, "price") else 0.0,
                portfolio_value=self._pm.portfolio_value,
                portfolio_heat=self._pm.managed_heat,
                sector_notionals=sector_notionals,
                kelly_fraction=self._kelly_fraction,
            )
            if sizing is None:
                continue

            atr_pct = self._ticker_atr.get(sig.ticker, DEFAULT_ATR_PCT)
            sl, ts, tp = _atr_exits(atr_pct)
            sig_dict = sig.to_dict()
            result.append({
                **sig_dict,
                "actionable": True,
                "recommended_side": sizing.side,
                "recommended_size_pct": round(sizing.size_pct, 4),
                "recommended_notional": round(sizing.notional, 2),
                "sizing_stages": sizing.to_dict()["stages"],
                "stop_loss_pct": round(sl, 4),
                "take_profit_pct": round(tp, 4),
                "trailing_stop_pct": round(ts, 4),
                "max_hold_bars": SIZING_MAX_HOLD_BARS,
                "atr_pct": round(atr_pct, 4),
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
            "managed_heat": round(pm.managed_heat, 4),
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
            "kelly_fraction": round(self._kelly_fraction, 4),
            "kelly_gate_active": (
                len(self._sizing_recent_outcomes) >= self._kelly_min_trades
            ),
            "kelly_entries_blocked": (
                len(self._sizing_recent_outcomes) >= self._kelly_min_trades
                and self._kelly_fraction <= 0
            ),
            "kelly_n_trades": len(self._sizing_recent_outcomes),
            "max_trades_per_day": SIZING_MAX_TRADES_PER_DAY,
            "tickers_on_cooldown": list(self._ticker_cooldown.keys()),
            "sector_notionals": self._compute_sector_notionals(),
            "data_fresh": self._data_fresh,
            "managed_heat": round(pm.managed_heat, 4),
            "exit_mode": "atr_adaptive",
            "atr_multipliers": {
                "stop_loss": SIZING_STOP_LOSS_ATR_MULT,
                "trailing_stop": SIZING_TRAILING_STOP_ATR_MULT,
                "take_profit": SIZING_TAKE_PROFIT_ATR_MULT,
            },
            "atr_floors": {
                "stop_loss": SIZING_STOP_LOSS_FLOOR,
                "trailing_stop": SIZING_TRAILING_STOP_FLOOR,
                "take_profit": SIZING_TAKE_PROFIT_FLOOR,
            },
            "ticker_atr": {t: round(a, 4) for t, a in self._ticker_atr.items()},
        }

    # ── Main tick ────────────────────────────────────────────────────────────

    async def _tick(self) -> None:
        market_open = self._is_market_hours()
        # If market is closed and we have no crypto, skip entirely
        has_crypto = any(t in self.CRYPTO_TICKERS for t in self._universe)
        if not market_open and not has_crypto:
            return

        # Decrement per-ticker cooldowns (1 bar = 1 minute)
        expired = []
        for t, remaining in self._ticker_cooldown.items():
            if remaining <= 1:
                expired.append(t)
            else:
                self._ticker_cooldown[t] = remaining - 1
        for t in expired:
            del self._ticker_cooldown[t]

        # 1. Fetch latest features + prices from DB
        features_map, regime_map, latest_feature_time = await self._fetch_features()
        prices = await self._fetch_prices()

        # Data freshness gate — if features are stale, manage exits only (no new entries)
        self._data_fresh = True
        if latest_feature_time is not None:
            age_minutes = (datetime.now(timezone.utc) - latest_feature_time).total_seconds() / 60
            if age_minutes > DATA_FRESHNESS_MAX_MINUTES:
                self._data_fresh = False
                logger.warning(
                    "data_stale_skipping_entries",
                    age_minutes=round(age_minutes, 1),
                    threshold=DATA_FRESHNESS_MAX_MINUTES,
                    latest_feature_time=str(latest_feature_time),
                )
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

        # 3b. Record predictions for live IC tracking (fire-and-forget)
        if self._ic_tracker is not None:
            for sig in signals:
                # Only record LightGBM predictions (skip rule-based fallback)
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

    # ── Sector notional computation ────────────────────────────────────────────

    def _compute_sector_notionals(self) -> dict[str, float]:
        """Compute total $ deployed per sector from open positions.

        Used by SmartPositionSizer stage 4 to enforce sector concentration limits.
        """
        sector_notionals: dict[str, float] = {}
        for ticker, pos in self._pm._positions.items():
            sector = SECTOR_MAP.get(ticker, "other")
            sector_notionals[sector] = sector_notionals.get(sector, 0.0) + pos.notional
        return sector_notionals

    # ── Sizing-mode entry/exit gating ────────────────────────────────────────

    def _sizing_entry_gate_open(self, sig: EnsembleSignal) -> bool:
        """Check if LightGBM signal warrants entry (sizing mode).

        Gates:
          0. Data freshness — features must be recent (WebSocket may be down)
          1. Daily trade cap not exceeded
          2. Per-ticker cooldown elapsed (prevents re-entry churn)
          3. Kelly gate: once 20+ trades confirm negative EV, STOP trading
          4. LightGBM pred_return exceeds cost threshold
          5. dir_prob is outside the dead zone (0.45-0.55)
        """
        ticker = sig.ticker

        # Gate 0: Data freshness — don't enter on stale features
        if not self._data_fresh:
            return False

        # Gate 1: Daily trade cap
        if self._sizing_n_trades_today >= SIZING_MAX_TRADES_PER_DAY:
            logger.debug("sizing_daily_cap_hit", n=self._sizing_n_trades_today)
            return False

        # Gate 2: Per-ticker cooldown
        cooldown_remaining = self._ticker_cooldown.get(ticker, 0)
        if cooldown_remaining > 0:
            logger.debug("sizing_cooldown_active", ticker=ticker, bars=cooldown_remaining)
            return False

        # Gate 3: Kelly stop — if 20+ trades prove negative EV, stop bleeding
        if (len(self._sizing_recent_outcomes) >= self._kelly_min_trades
                and self._kelly_fraction <= 0):
            logger.debug("sizing_kelly_stop", kelly=round(self._kelly_fraction, 4),
                         n=len(self._sizing_recent_outcomes))
            return False

        # Gate 4+5: Signal quality
        pred_ret = float(sig.lgbm_pred_return)
        dir_prob = float(sig.lgbm_dir_prob)
        lo, hi = SIZING_DIR_PROB_DEAD_ZONE
        signal_ok = abs(pred_ret) > SIZING_COST_THRESHOLD and not (lo < dir_prob < hi)
        return signal_ok

    def _update_kelly(self) -> None:
        """Recompute rolling Kelly fraction from recent trade outcomes.

        f* = (p * b - q) / b
        where p = win rate, b = avg_win/avg_loss, q = 1-p

        Uses half-Kelly for safety. Updated after every closed trade.
        """
        outcomes = self._sizing_recent_outcomes
        if len(outcomes) < self._kelly_min_trades:
            return

        wins = [o for o in outcomes if o > 0]
        losses = [o for o in outcomes if o < 0]

        if not wins or not losses:
            self._kelly_fraction = 0.0
            return

        import statistics
        p = len(wins) / len(outcomes)
        avg_win = statistics.mean(wins)
        avg_loss = abs(statistics.mean(losses))
        b = avg_win / max(avg_loss, 1e-9)
        q = 1 - p

        self._kelly_fraction = (p * b - q) / max(b, 1e-9)
        logger.info(
            "kelly_updated",
            kelly_full=round(self._kelly_fraction, 4),
            kelly_half=round(max(0, self._kelly_fraction / 2), 4),
            win_rate=round(p, 3),
            win_loss_ratio=round(b, 3),
            n_trades=len(outcomes),
        )

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

        # ATR-adaptive exit thresholds
        atr_pct = self._ticker_atr.get(ticker, DEFAULT_ATR_PCT)
        sl, ts, tp = _atr_exits(atr_pct)

        # Unrealized PnL
        unrealized = (price - entry_price) / entry_price
        if entry_dir < 0:
            unrealized = -unrealized

        # Stop loss (ATR-scaled)
        if unrealized < -sl:
            return "stop_loss"

        # Take profit (ATR-scaled)
        if unrealized > tp:
            return "take_profit"

        # Trailing stop (ATR-scaled)
        peak = self._peak_prices.get(ticker, price)
        if entry_dir > 0:
            drop = (peak - price) / peak if peak > 0 else 0.0
            if drop > ts:
                return "trailing_stop"
        else:
            rise = (price - peak) / peak if peak > 0 else 0.0
            if rise > ts:
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

    async def _act_on_signal(
        self,
        sig: EnsembleSignal,
        price: float,
        features_arr: np.ndarray | None = None,
        regime: int = 0,
    ) -> None:
        """Use SmartPositionSizer (or RL/threshold fallback) to decide and execute."""
        if self._cb.is_halted:
            return

        ticker = sig.ticker
        has_position = ticker in self._pm._positions

        # A/B conflict prevention: skip entry if the OTHER pipeline holds this ticker
        if not has_position and self._other_pm is not None:
            if ticker in self._other_pm._positions:
                logger.debug("ab_ticker_conflict_skip", ticker=ticker, pipeline=self._pipeline_id)
                return

        # Regime logging (sizing constraints are handled inside SmartPositionSizer)
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
                    # Start cooldown to prevent immediate re-entry churn
                    self._ticker_cooldown[ticker] = SIZING_TICKER_COOLDOWN_BARS
                    atr_pct = self._ticker_atr.get(ticker, DEFAULT_ATR_PCT)
                    _sl, _ts, _tp = _atr_exits(atr_pct)
                    logger.info(
                        "sizing_exit",
                        ticker=ticker,
                        reason=exit_reason,
                        bars_held=self._bars_held.get(ticker, 0),
                        cooldown_bars=SIZING_TICKER_COOLDOWN_BARS,
                        atr_pct=round(atr_pct, 4),
                        stop=round(_sl, 4),
                        trail=round(_ts, 4),
                        target=round(_tp, 4),
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
                if ticker in self.CRYPTO_TICKERS:
                    logger.debug("sizing_skip_crypto", ticker=ticker)
                    return

                # Check entry gate (includes Kelly check)
                if not self._sizing_entry_gate_open(sig):
                    return

                direction = self._sizing_signal_direction(sig)
                if direction == 0:
                    return
                # Phase 5: no shorts
                if direction < 0:
                    return

                # ── Smart Position Sizer: 6-stage pipeline ───────────────────
                sector_notionals = self._compute_sector_notionals()
                sizing = self._sizer.compute(
                    ticker=ticker,
                    dir_prob=float(sig.lgbm_dir_prob),
                    pred_return=float(sig.lgbm_pred_return),
                    atr_pct=self._ticker_atr.get(ticker, 0.01),
                    price=price,
                    portfolio_value=self._pm.portfolio_value,
                    portfolio_heat=self._pm.managed_heat,
                    sector_notionals=sector_notionals,
                    kelly_fraction=self._kelly_fraction,
                )
                if sizing is None:
                    return

                side = sizing.side
                notional = sizing.notional
                qty = sizing.shares

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
                    sizing="smart_pipeline",
                    size_pct=round(sizing.size_pct, 4),
                    stages=f"{sizing.stage1_base_pct:.3f}→{sizing.stage2_atr_pct:.3f}→{sizing.stage3_kelly_pct:.3f}→{sizing.stage4_constraint_pct:.3f}",
                    direction="long" if direction > 0 else "short",
                    lgbm_pred=round(sig.lgbm_pred_return, 5),
                    atr=round(self._ticker_atr.get(ticker, 0.01), 4),
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
                pos = self._pm._positions.get(ticker)
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
                    # Track outcome for recent win rate + Kelly computation
                    self._sizing_recent_outcomes.append(pnl)
                    if len(self._sizing_recent_outcomes) > 50:
                        self._sizing_recent_outcomes = self._sizing_recent_outcomes[-50:]
                    self._update_kelly()
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
                    pipeline_id=self._pipeline_id,
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

    async def _fetch_features(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, int], datetime | None]:
        """Fetch last SEQ_LEN feature rows per ticker from DB.

        Returns:
            features_map: ticker → (SEQ_LEN, n_features) float32 array
            regime_map:   ticker → latest regime int (0=trending,1=choppy,2=high_vol)
            latest_time:  most recent feature timestamp across all tickers (or None)
        """
        from sqlalchemy import select

        from src.data.db import FeatureMatrix

        result: dict[str, np.ndarray] = {}
        regime_map: dict[str, int] = {}
        latest_time: datetime | None = None
        async with self._sf() as session:
            for ticker in self._universe:
                rows = await session.execute(
                    select(FeatureMatrix.features, FeatureMatrix.time)
                    .where(FeatureMatrix.ticker == ticker)
                    .order_by(FeatureMatrix.time.desc())
                    .limit(self.SEQ_LEN)
                )
                raw_rows = list(reversed(rows.all()))
                if not raw_rows:
                    continue
                feat_rows = [r[0] for r in raw_rows]
                row_time = raw_rows[-1][1]  # most recent (last after reversing)
                if row_time is not None:
                    if latest_time is None or row_time > latest_time:
                        latest_time = row_time

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
                # Cache per-ticker ATR for SmartPositionSizer
                atr_val = latest_row.get("atr_pct")
                if atr_val is not None:
                    self._ticker_atr[ticker] = float(atr_val)
        return result, regime_map, latest_time

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
            self._ticker_cooldown.clear()
            # Auto-clear daily_loss halts (structural halts like max_drawdown stay)
            self._cb.try_daily_reset()
            logger.info(
                "daily_start_value_reset",
                value=self._daily_start_value,
            )

    async def _sleep_until_next_minute(self) -> None:
        """Sleep until the next 1m bar boundary (:00 seconds)."""
        now = datetime.now(timezone.utc)
        seconds_left = 60.0 - now.second - now.microsecond / 1_000_000
        await asyncio.sleep(max(seconds_left, 1.0))
