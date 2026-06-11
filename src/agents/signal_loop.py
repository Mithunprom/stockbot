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
SIZING_COST_THRESHOLD = 0.003   # min |pred_return| — multi-factor gate is the primary filter
SIZING_DIR_PROB_DEAD_ZONE = (0.45, 0.55)  # standard dead zone — multi-factor gate handles quality
SIZING_REVERSAL_BARS = 3

# Anti-churn / entry discipline
SIZING_MAX_TRADES_PER_DAY = 4       # swing cadence: few, high-conviction entries
SIZING_TICKER_COOLDOWN_BARS = 60    # 1-hour cooldown after any exit
MAX_ENTRIES_PER_TICK = 2            # prevents same-tick multi-entry blowups (2026-05-22)
MAX_OPEN_POSITIONS = 4              # hard cap on concurrent positions
PORTFOLIO_HEAT_CEILING = 0.60       # no new entries above 60% deployed
MAX_POSITIONS_PER_SECTOR = 2        # correlation guard: max 2 positions per sector
ENTRY_WINDOW_ET = ((9, 40), (15, 30))  # no entries in first 10 / last 29 minutes

# Data freshness gate — skip new entries when features are stale
DATA_FRESHNESS_MAX_MINUTES = 5      # max age of latest feature row before gating entries

# Exit thresholds — scaled to DAILY volatility (swing horizon, 1–3 day holds).
# atr_pct is computed on 1-minute bars; daily sigma ≈ atr_1m × sqrt(390).
# The old 30-bar max hold meant TP/stop almost never fired (max_hold hit ~100%
# of exits) and every round trip was a PDT day trade on a <$25k account.
DAILY_VOL_SQRT_BARS = 19.75        # sqrt(390 one-minute bars per session)
SIZING_STOP_LOSS_DVOL_MULT = 1.1   # stop ≈ 1.1× daily sigma
SIZING_TRAILING_DVOL_MULT = 0.8    # trail ≈ 0.8× daily sigma
SIZING_TAKE_PROFIT_DVOL_MULT = 2.2 # TP ≈ 2× stop → 2:1 reward:risk
SIZING_STOP_LOSS_FLOOR = 0.010     # 1.0% minimum stop
SIZING_STOP_LOSS_CAP = 0.025       # 2.5% maximum stop
SIZING_TRAILING_STOP_FLOOR = 0.008 # 0.8% minimum trailing
SIZING_TRAILING_STOP_CAP = 0.020   # 2.0% maximum trailing
SIZING_TAKE_PROFIT_FLOOR = 0.020   # 2.0% take profit floor
SIZING_TAKE_PROFIT_CAP = 0.050     # 5.0% take profit cap
SIZING_MAX_HOLD_BARS = 1170        # ~3 trading days of market-hours 1m bars
SIZING_STAGNATION_BARS = 780       # ~2 trading days
SIZING_STAGNATION_PNL = 0.004      # |PnL| < 0.4% after 2 days → dead trade, free capital
CATASTROPHIC_STOP_MULT = 2.0       # 2× stop = emergency same-day exit threshold
DEFAULT_ATR_PCT = 0.001            # fallback when ATR unavailable (typical 1-min ATR)

# Kelly governor — self-recovering (replaces the permanent Kelly stop).
# The old gate (kelly ≤ 0 with ≥20 all-time trades → block ALL entries) was a
# deadlock: no trades → window never refreshes → blocked forever. This stalled
# production from 2026-05-27 to 2026-06-11. Now: only recent trades count, and
# negative Kelly degrades to probation (1 small probe/day) instead of a halt,
# so the window keeps refreshing and size must be re-earned, never bricked.
KELLY_LOOKBACK_DAYS = 10           # only trades closed in the last N days count
KELLY_MIN_TRADES = 10              # need ≥N recent closed trades before acting
KELLY_PROBATION_NOTIONAL = 1200.0  # probe size while Kelly ≤ 0
KELLY_PROBATION_MIN_TICKER_IC = 0.05  # probes only on tickers where signal works

# Per-ticker live IC gate — stop trading names the model is provably wrong on
# (e.g. JPM live IC −0.32 on 2026-05-27 yet it was the top-ranked entry signal)
TICKER_IC_BLOCK_THRESHOLD = -0.02
TICKER_IC_MIN_N = 100              # require enough filled predictions to judge
TICKER_IC_REFRESH_TICKS = 60       # refresh per-ticker IC from tracker hourly

# PDT (Pattern Day Trader) protection — accounts under $25k get 3 day trades
# per rolling 5 business days. A same-day round trip is a day trade, so the
# default plan is to hold overnight; the day-trade budget is reserved for
# stop-losses (and rich take-profits) only.
PDT_EQUITY_THRESHOLD = 25_000.0
PDT_MAX_DAY_TRADES = 3
PDT_REFRESH_TICKS = 5              # refresh daytrade_count from broker every N ticks

logger = structlog.get_logger(__name__)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def _atr_exits(atr_pct: float) -> tuple[float, float, float]:
    """Compute (stop_loss, trailing_stop, take_profit) from per-ticker ATR%.

    atr_pct is the 1-minute ATR ratio; it is scaled by sqrt(390) to a daily
    sigma estimate so thresholds are reachable over a 1–3 day swing hold.
    Floors and caps keep low-vol names meaningful and high-vol names sane.
    """
    daily_vol = _clamp(atr_pct, 0.0002, 0.01) * DAILY_VOL_SQRT_BARS
    sl = _clamp(daily_vol * SIZING_STOP_LOSS_DVOL_MULT,
                SIZING_STOP_LOSS_FLOOR, SIZING_STOP_LOSS_CAP)
    ts = _clamp(daily_vol * SIZING_TRAILING_DVOL_MULT,
                SIZING_TRAILING_STOP_FLOOR, SIZING_TRAILING_STOP_CAP)
    tp = _clamp(daily_vol * SIZING_TAKE_PROFIT_DVOL_MULT,
                SIZING_TAKE_PROFIT_FLOOR, SIZING_TAKE_PROFIT_CAP)
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
        self._entry_dates: dict[str, Any] = {}           # ticker → ET date of entry (PDT)
        self._peak_prices: dict[str, float] = {}
        self._bars_held: dict[str, int] = {}
        self._reversal_counts: dict[str, int] = {}
        self._sizing_returns_history: list[float] = [0.0] * 20
        # Rolling Kelly window: (exit_time_utc, pnl_pct) — only recent trades count
        self._sizing_recent_outcomes: list[tuple[datetime, float]] = []
        self._sizing_n_trades_today: int = 0
        self._ticker_cooldown: dict[str, int] = {}  # ticker → bars remaining
        self._exit_fail_cooldown: dict[str, int] = {}  # ticker → bars before exit retry

        # Per-ticker ATR cache (refreshed each tick from feature_matrix)
        self._ticker_atr: dict[str, float] = {}

        # Data freshness flag — set False when features are stale to block new entries
        self._data_fresh: bool = True

        # Kelly governor — degrades sizing on negative recent expectancy instead
        # of blocking entries forever (the old behavior deadlocked the system).
        self._kelly_fraction: float = 0.0
        self._kelly_min_trades: int = KELLY_MIN_TRADES
        self._probation_entries_today: int = 0
        self._pending_exit_reasons: dict[str, str] = {}  # ticker → exit reason

        # Per-ticker live IC cache: ticker → (ic, n_predictions)
        self._ticker_ic: dict[str, tuple[float, int]] = {}
        self._ic_refresh_countdown: int = 0

        # PDT day-trade budget tracking (refreshed from broker)
        self._daytrade_count: int = 0
        self._pdt_refresh_countdown: int = 0
        self._pdt_deferred_logged: set[str] = set()  # throttle deferral logs

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
        """Load RECENT closed trade PnLs from DB to bootstrap the Kelly governor.

        Only trades closed within KELLY_LOOKBACK_DAYS count: trades made under
        an older exit/sizing config are not representative of the current
        system, and an all-time window once froze the bot permanently (stale
        losses from May kept Kelly negative with no way to refresh).

        Filters by pipeline_id so each pipeline only sees its own history,
        and excludes crypto tickers (old crypto trades had noise signals).
        """
        from datetime import timedelta

        from sqlalchemy import select as _sel
        from src.data.db import Trade as _T

        cutoff = datetime.now(timezone.utc) - timedelta(days=KELLY_LOOKBACK_DAYS)
        try:
            async with self._sf() as session:
                query = (
                    _sel(_T.exit_time, _T.pnl_pct)
                    .where(
                        _T.exit_time.isnot(None),
                        _T.exit_time >= cutoff,
                        _T.pnl_pct.isnot(None),
                        ~_T.ticker.in_(list(self.CRYPTO_TICKERS)),
                    )
                    .order_by(_T.exit_time.desc())
                    .limit(50)
                )
                # Filter by pipeline_id if set (A/B mode)
                if self._pipeline_id:
                    query = query.where(_T.pipeline_id == self._pipeline_id)

                result = await session.execute(query)
                rows = [
                    (ts, float(p)) for ts, p in result.all()
                    if ts is not None and p is not None
                ]

            if rows:
                self._sizing_recent_outcomes = list(reversed(rows))
                self._update_kelly()
                logger.info(
                    "kelly_seeded_from_db",
                    n_trades=len(rows),
                    lookback_days=KELLY_LOOKBACK_DAYS,
                    kelly=round(self._kelly_fraction, 4),
                    mode=self._kelly_mode(),
                    pipeline=self._pipeline_id,
                )
            else:
                logger.info(
                    "kelly_no_recent_history",
                    pipeline=self._pipeline_id,
                    lookback_days=KELLY_LOOKBACK_DAYS,
                    note=f"Kelly governor inactive until {KELLY_MIN_TRADES}+ recent trades",
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
                atr_pct=self._ticker_atr.get(sig.ticker, DEFAULT_ATR_PCT),
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
            "kelly_mode": self._kelly_mode(),
            "kelly_gate_active": self._kelly_mode() != "inactive",
            "kelly_entries_blocked": False,  # probation replaces hard block
            "kelly_n_trades": len(self._sizing_recent_outcomes),
            "kelly_lookback_days": KELLY_LOOKBACK_DAYS,
            "probation_entries_today": self._probation_entries_today,
            "max_trades_per_day": SIZING_MAX_TRADES_PER_DAY,
            "max_open_positions": MAX_OPEN_POSITIONS,
            "heat_ceiling": PORTFOLIO_HEAT_CEILING,
            "daytrade_count": self._daytrade_count,
            "pdt_budget_remaining": max(0, PDT_MAX_DAY_TRADES - self._daytrade_count),
            "ticker_ic_tracked": len(self._ticker_ic),
            "tickers_ic_blocked": [
                t for t, (ic, n) in self._ticker_ic.items()
                if n >= TICKER_IC_MIN_N and ic < TICKER_IC_BLOCK_THRESHOLD
            ],
            "tickers_on_cooldown": list(self._ticker_cooldown.keys()),
            "sector_notionals": self._compute_sector_notionals(),
            "data_fresh": self._data_fresh,
            "managed_heat": round(pm.managed_heat, 4),
            "exit_mode": "daily_vol_swing",
            "atr_multipliers": {
                "stop_loss_dvol": SIZING_STOP_LOSS_DVOL_MULT,
                "trailing_stop_dvol": SIZING_TRAILING_DVOL_MULT,
                "take_profit_dvol": SIZING_TAKE_PROFIT_DVOL_MULT,
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

        await self._tick_maintenance()

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

        if self._sizing_mode:
            # Ranked entry pass: best signals first, capped per tick, so a
            # burst of correlated signals can't open 6 positions in one bar.
            candidates = [
                s for s in signals
                if (market_open or s.ticker in self.CRYPTO_TICKERS)
                and s.ticker not in self._pm._positions
            ]
            await self._execute_entries(candidates, prices, regime_map, _uf)
        else:
            for sig in signals:
                # Gate equity tickers on market hours; crypto runs 24/7
                if not market_open and sig.ticker not in self.CRYPTO_TICKERS:
                    continue
                regime = regime_map.get(sig.ticker, 1)  # default: choppy
                # Regime-aware threshold: choppy/high-vol require stronger signal
                from src.features.regime import REGIME_GATE
                threshold, _size_scale = REGIME_GATE.get(regime, (self.SIGNAL_ENTRY_THRESHOLD, 1.0))
                if abs(sig.ensemble_signal) >= threshold:
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
            await self._recover_entry_state()
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

    # ── Per-tick maintenance ──────────────────────────────────────────────────

    async def _tick_maintenance(self) -> None:
        """Decrement cooldowns and refresh cached broker/IC state."""
        for cooldowns in (self._ticker_cooldown, self._exit_fail_cooldown):
            expired = []
            for t, remaining in cooldowns.items():
                if remaining <= 1:
                    expired.append(t)
                else:
                    cooldowns[t] = remaining - 1
            for t in expired:
                del cooldowns[t]

        await self._maybe_refresh_ticker_ic()
        await self._maybe_refresh_daytrade_count()

    async def _maybe_refresh_ticker_ic(self) -> None:
        """Refresh the per-ticker live IC cache from the IC tracker (hourly)."""
        if self._ic_tracker is None:
            return
        self._ic_refresh_countdown -= 1
        if self._ic_refresh_countdown > 0:
            return
        self._ic_refresh_countdown = TICKER_IC_REFRESH_TICKS
        try:
            by_ticker = await self._ic_tracker._compute_per_ticker_ic(window_days=7)
            self._ticker_ic = {
                t: (float(v.get("ic", 0.0)), int(v.get("n", 0)))
                for t, v in (by_ticker or {}).items()
            }
            blocked = [
                t for t, (ic, n) in self._ticker_ic.items()
                if n >= TICKER_IC_MIN_N and ic < TICKER_IC_BLOCK_THRESHOLD
            ]
            logger.info(
                "ticker_ic_refreshed",
                n_tickers=len(self._ticker_ic),
                ic_blocked=blocked or None,
            )
        except Exception as exc:
            # Fail open: missing IC data never blocks trading by itself
            logger.warning("ticker_ic_refresh_failed", error=str(exc))

    async def _maybe_refresh_daytrade_count(self) -> None:
        """Refresh the rolling 5-day day-trade count from the broker."""
        self._pdt_refresh_countdown -= 1
        if self._pdt_refresh_countdown > 0:
            return
        self._pdt_refresh_countdown = PDT_REFRESH_TICKS
        try:
            account = await self._alpaca.get_account()
            self._daytrade_count = int(account.get("daytrade_count", 0) or 0)
        except Exception as exc:
            logger.warning("daytrade_count_refresh_failed", error=str(exc))

    async def _recover_entry_state(self) -> None:
        """Rebuild entry dates/prices for broker positions we lost track of.

        After a restart/redeploy, sync_from_broker restores positions but the
        in-memory entry metadata (entry date for PDT, bars held) is gone.
        Recover entry_time from the open Trade row in the DB; if none exists,
        assume the position was opened on a previous day (allows normal exits).
        """
        missing = [t for t in self._pm._positions if t not in self._entry_dates]
        if not missing:
            return

        from zoneinfo import ZoneInfo
        from sqlalchemy import select as _sel
        from src.data.db import Trade as _T

        et = ZoneInfo("America/New_York")
        today = datetime.now(et).date()
        for ticker in missing:
            entry_date = None
            try:
                async with self._sf() as session:
                    result = await session.execute(
                        _sel(_T.entry_time)
                        .where(_T.ticker == ticker, _T.exit_time.is_(None))
                        .order_by(_T.entry_time.desc())
                        .limit(1)
                    )
                    row = result.scalar_one_or_none()
                if row is not None:
                    entry_date = row.astimezone(et).date()
            except Exception as exc:
                logger.warning("entry_state_recovery_failed", ticker=ticker, error=str(exc))

            if entry_date is None:
                # Unknown origin — treat as opened yesterday so exits work
                from datetime import timedelta as _td
                entry_date = today - _td(days=1)

            self._entry_dates[ticker] = entry_date
            days_held = max(0, np.busday_count(entry_date, today))
            self._bars_held.setdefault(ticker, int(days_held) * 390)
            pos = self._pm._positions.get(ticker)
            if pos is not None:
                self._entry_prices.setdefault(ticker, pos.avg_entry_price)
                self._entry_directions.setdefault(ticker, 1 if pos.side == "long" else -1)
                self._peak_prices.setdefault(ticker, pos.last_price or pos.avg_entry_price)
            logger.info(
                "entry_state_recovered",
                ticker=ticker,
                entry_date=str(entry_date),
                bars_held=self._bars_held.get(ticker, 0),
            )

    # ── Entry execution (ranked, capped) ─────────────────────────────────────

    async def _execute_entries(
        self,
        candidates: list[EnsembleSignal],
        prices: dict[str, float],
        regime_map: dict[str, int],
        uf: dict | None = None,
    ) -> int:
        """Execute up to MAX_ENTRIES_PER_TICK entries, best signal first.

        Processing in conviction order with a per-tick cap (and re-checking
        heat/sector gates after every fill) prevents the same-tick multi-entry
        race that opened 6 correlated positions at 150% heat on 2026-05-22.
        """
        ranked = sorted(
            candidates,
            key=lambda s: abs(float(s.lgbm_pred_return)),
            reverse=True,
        )
        entries = 0
        for sig in ranked:
            if entries >= MAX_ENTRIES_PER_TICK:
                break
            price = prices.get(sig.ticker, 0.0)
            if price <= 0:
                continue
            features_arr = uf.get(sig.ticker, {}).get("1m") if uf else None
            feat_np = features_arr.numpy() if features_arr is not None else None
            regime = regime_map.get(sig.ticker, 1)
            entered = await self._act_on_signal(sig, price, feat_np, regime=regime)
            if entered:
                entries += 1
        return entries

    # ── Sizing-mode entry/exit gating ────────────────────────────────────────

    def _kelly_mode(self) -> str:
        """Current Kelly governor mode: inactive / normal / probation."""
        self._prune_kelly_window()
        if len(self._sizing_recent_outcomes) < self._kelly_min_trades:
            return "inactive"
        return "normal" if self._kelly_fraction > 0 else "probation"

    def _prune_kelly_window(self) -> None:
        """Drop outcomes older than the lookback window."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=KELLY_LOOKBACK_DAYS)
        pruned = [
            (ts, p) for ts, p in self._sizing_recent_outcomes
            if ts is not None and ts >= cutoff
        ]
        if len(pruned) != len(self._sizing_recent_outcomes):
            self._sizing_recent_outcomes = pruned

    def _in_entry_window(self) -> bool:
        """True if current ET time is within the allowed entry window."""
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]
        now = datetime.now(ZoneInfo("America/New_York"))
        (h0, m0), (h1, m1) = ENTRY_WINDOW_ET
        start = now.replace(hour=h0, minute=m0, second=0, microsecond=0)
        end = now.replace(hour=h1, minute=m1, second=0, microsecond=0)
        return start <= now <= end

    def _sector_position_count(self, ticker: str) -> int:
        """Number of open positions in the same sector as `ticker`."""
        sector = SECTOR_MAP.get(ticker, "other")
        return sum(
            1 for t in self._pm._positions
            if SECTOR_MAP.get(t, "other") == sector
        )

    def _ticker_ic_blocked(self, ticker: str) -> bool:
        """True if live IC proves the signal is wrong on this ticker."""
        ic, n = self._ticker_ic.get(ticker, (0.0, 0))
        return n >= TICKER_IC_MIN_N and ic < TICKER_IC_BLOCK_THRESHOLD

    def _is_same_day_entry(self, ticker: str) -> bool:
        """True if the position was opened today (ET) — selling it would be a day trade."""
        entry_date = self._entry_dates.get(ticker)
        if entry_date is None:
            return True  # unknown → conservative until recovery fills it in
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]
        return entry_date == datetime.now(ZoneInfo("America/New_York")).date()

    def _pdt_exit_allowed(self, reason: str, unrealized: float, stop_pct: float) -> bool:
        """Decide whether a SAME-DAY exit may consume a PDT day trade.

        Accounts ≥ $25k are exempt. Below that, the rolling budget is
        3 day trades per 5 business days, reserved for risk control:
          - stop_loss: allowed while any budget remains
          - take_profit: allowed only with ≥2 budget left (locking real gains)
          - everything else (trailing/max_hold/stagnation/reversal): deferred
            to the next session, where the exit is PDT-free.
        """
        if self._pm.portfolio_value >= PDT_EQUITY_THRESHOLD:
            return True
        budget = PDT_MAX_DAY_TRADES - self._daytrade_count
        if reason == "stop_loss":
            return budget >= 1
        if reason == "take_profit":
            return budget >= 2
        return False

    def _sizing_entry_gate_open(self, sig: EnsembleSignal) -> bool:
        """Check if the signal warrants a new entry (sizing mode).

        Gates:
          0. Data freshness — features must be recent (WebSocket may be down)
          0b. Entry time window — skip open/close noise (9:40–15:30 ET)
          1. Daily trade cap not exceeded
          2. Per-ticker cooldown elapsed (prevents re-entry churn)
          3. Kelly governor — probation allows 1 small probe/day on a
             positive-IC ticker instead of blocking everything forever
          4. Per-ticker live IC — never enter names the model is wrong on
          5. Position-count cap + portfolio heat ceiling + sector cap
          6. Signal quality: pred_return above cost, dir_prob outside dead zone
        """
        ticker = sig.ticker

        # Gate 0: Data freshness — don't enter on stale features
        if not self._data_fresh:
            return False

        # Gate 0b: Entry time window (equities; crypto is blocked elsewhere)
        if not self._in_entry_window():
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

        # Gate 3: Kelly governor — negative recent expectancy → probation
        if self._kelly_mode() == "probation":
            ic, n = self._ticker_ic.get(ticker, (0.0, 0))
            probe_ok = (
                self._probation_entries_today < 1
                and n >= TICKER_IC_MIN_N
                and ic >= KELLY_PROBATION_MIN_TICKER_IC
            )
            if not probe_ok:
                logger.debug(
                    "sizing_kelly_probation_skip",
                    ticker=ticker,
                    kelly=round(self._kelly_fraction, 4),
                    probes_used=self._probation_entries_today,
                    ticker_ic=round(ic, 3),
                )
                return False

        # Gate 4: Per-ticker live IC — model demonstrably wrong on this name
        if self._ticker_ic_blocked(ticker):
            ic, n = self._ticker_ic.get(ticker, (0.0, 0))
            logger.debug("sizing_ticker_ic_block", ticker=ticker, ic=round(ic, 3), n=n)
            return False

        # Gate 5: Portfolio shape — position count, heat ceiling, sector cap
        if len(self._pm._positions) >= MAX_OPEN_POSITIONS:
            logger.debug("sizing_max_positions", n=len(self._pm._positions))
            return False
        if self._pm.managed_heat >= PORTFOLIO_HEAT_CEILING:
            logger.debug("sizing_heat_ceiling", heat=round(self._pm.managed_heat, 3))
            return False
        if self._sector_position_count(ticker) >= MAX_POSITIONS_PER_SECTOR:
            logger.debug(
                "sizing_sector_position_cap",
                ticker=ticker,
                sector=SECTOR_MAP.get(ticker, "other"),
            )
            return False

        # Gate 6: Signal quality
        pred_ret = float(sig.lgbm_pred_return)
        dir_prob = float(sig.lgbm_dir_prob)
        lo, hi = SIZING_DIR_PROB_DEAD_ZONE
        signal_ok = abs(pred_ret) > SIZING_COST_THRESHOLD and not (lo < dir_prob < hi)
        return signal_ok

    def _update_kelly(self) -> None:
        """Recompute rolling Kelly fraction from recent trade outcomes.

        f* = (p * b - q) / b
        where p = win rate, b = avg_win/avg_loss, q = 1-p

        Outcomes are pnl_pct (scale-invariant) within KELLY_LOOKBACK_DAYS.
        """
        self._prune_kelly_window()
        outcomes = [p for _, p in self._sizing_recent_outcomes]
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
            kelly=round(self._kelly_fraction, 4),
            mode=self._kelly_mode(),
            win_rate=round(p, 3),
            win_loss_ratio=round(b, 3),
            n_trades=len(outcomes),
            lookback_days=KELLY_LOOKBACK_DAYS,
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
            # Treat as one day old: normal exit logic applies (a forced
            # max_hold here used to dump every position on each redeploy)
            self._bars_held.setdefault(ticker, 390)

        # ATR-adaptive exit thresholds
        atr_pct = self._ticker_atr.get(ticker, DEFAULT_ATR_PCT)
        sl, ts, tp = _atr_exits(atr_pct)

        # Unrealized PnL
        unrealized = (price - entry_price) / entry_price
        if entry_dir < 0:
            unrealized = -unrealized

        reason: str | None = None

        # Stop loss (ATR-scaled)
        if unrealized < -sl:
            reason = "stop_loss"

        # Take profit (ATR-scaled)
        elif unrealized > tp:
            reason = "take_profit"

        else:
            # Trailing stop (ATR-scaled)
            peak = self._peak_prices.get(ticker, price)
            if entry_dir > 0:
                drop = (peak - price) / peak if peak > 0 else 0.0
                if drop > ts:
                    reason = "trailing_stop"
            else:
                rise = (price - peak) / peak if peak > 0 else 0.0
                if rise > ts:
                    reason = "trailing_stop"

        bars = self._bars_held.get(ticker, 0)
        if reason is None:
            # Max hold (~3 trading days)
            if bars >= SIZING_MAX_HOLD_BARS:
                reason = "max_hold"
            # Stagnation: dead trade going nowhere — free up the capital
            elif bars >= SIZING_STAGNATION_BARS and abs(unrealized) < SIZING_STAGNATION_PNL:
                reason = "stagnation"

        if reason is None:
            # Signal reversal (N consecutive bars of opposite direction)
            current_dir = self._sizing_signal_direction(sig)
            if current_dir != 0 and current_dir != entry_dir:
                self._reversal_counts[ticker] = self._reversal_counts.get(ticker, 0) + 1
            else:
                self._reversal_counts[ticker] = 0
            if self._reversal_counts.get(ticker, 0) >= SIZING_REVERSAL_BARS:
                reason = "signal_reversal"

        if reason is None:
            return None

        # PDT guard: a same-day round trip is a day trade. On a <$25k account
        # only stop-losses (and rich take-profits) may spend the budget; all
        # other exits wait for the next session, where they are PDT-free.
        if self._is_same_day_entry(ticker):
            catastrophic = unrealized < -(sl * CATASTROPHIC_STOP_MULT)
            if catastrophic:
                reason = "stop_loss"
            if not self._pdt_exit_allowed(reason, unrealized, sl):
                if ticker not in self._pdt_deferred_logged:
                    self._pdt_deferred_logged.add(ticker)
                    logger.info(
                        "exit_deferred_pdt",
                        ticker=ticker,
                        reason=reason,
                        unrealized=round(unrealized, 4),
                        daytrades_used=self._daytrade_count,
                    )
                return None

        return reason

    def _clear_sizing_state(self, ticker: str) -> None:
        """Clear per-ticker sizing state after position close."""
        self._entry_directions.pop(ticker, None)
        self._entry_prices.pop(ticker, None)
        self._entry_dates.pop(ticker, None)
        self._peak_prices.pop(ticker, None)
        self._bars_held.pop(ticker, None)
        self._reversal_counts.pop(ticker, None)
        self._pdt_deferred_logged.discard(ticker)

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
    ) -> bool:
        """Use SmartPositionSizer (or RL/threshold fallback) to decide and execute.

        Returns True only when a new entry order FILLED (used by the per-tick
        entry cap); exits and skips return False.
        """
        if self._cb.is_halted:
            return False

        ticker = sig.ticker
        has_position = ticker in self._pm._positions

        # A/B conflict prevention: skip entry if the OTHER pipeline holds this ticker
        if not has_position and self._other_pm is not None:
            if ticker in self._other_pm._positions:
                logger.debug("ab_ticker_conflict_skip", ticker=ticker, pipeline=self._pipeline_id)
                return False

        from src.features.regime import REGIME_GATE, regime_label
        _, size_scale = REGIME_GATE.get(regime, (0.40, 1.0))

        is_entry = False
        # ── Position-sizing mode (model gates entry/exit, sizer sizes) ────────
        if self._sizing_mode:
            if has_position:
                # Back off after a failed exit order instead of re-firing
                # every minute (the April XOM loop spammed hundreds of sells)
                if self._exit_fail_cooldown.get(ticker, 0) > 0:
                    self._bars_held[ticker] = self._bars_held.get(ticker, 0) + 1
                    return False

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
                    return False
            else:
                # Block crypto entries — LightGBM was trained on equities only.
                if ticker in self.CRYPTO_TICKERS:
                    logger.debug("sizing_skip_crypto", ticker=ticker)
                    return False

                # Check entry gate (Kelly governor, IC, heat, sector, quality)
                if not self._sizing_entry_gate_open(sig):
                    return False

                direction = self._sizing_signal_direction(sig)
                if direction == 0:
                    return False
                # Phase 5: no shorts
                if direction < 0:
                    return False

                # ── Smart Position Sizer: 6-stage pipeline ───────────────────
                sector_notionals = self._compute_sector_notionals()
                sizing = self._sizer.compute(
                    ticker=ticker,
                    dir_prob=float(sig.lgbm_dir_prob),
                    pred_return=float(sig.lgbm_pred_return),
                    atr_pct=self._ticker_atr.get(ticker, DEFAULT_ATR_PCT),
                    price=price,
                    portfolio_value=self._pm.portfolio_value,
                    portfolio_heat=self._pm.managed_heat,
                    sector_notionals=sector_notionals,
                    kelly_fraction=self._kelly_fraction,
                )
                if sizing is None:
                    return False

                side = sizing.side
                notional = sizing.notional

                # Conservative on uncertainty: shrink size in choppy/high-vol
                # regimes (REGIME_GATE size scale was previously log-only)
                if size_scale < 1.0:
                    notional *= size_scale
                    logger.info(
                        "regime_size_scaled",
                        ticker=ticker,
                        regime=regime_label(regime),
                        size_scale=size_scale,
                    )

                # Kelly probation: probe-sized entry to refresh the window
                if self._kelly_mode() == "probation":
                    notional = min(notional, KELLY_PROBATION_NOTIONAL)
                    logger.info(
                        "kelly_probation_probe",
                        ticker=ticker,
                        notional=round(notional, 2),
                        kelly=round(self._kelly_fraction, 4),
                    )

                qty = round(notional / max(price, 0.01), 2)
                if notional < 500.0 or qty < 0.01:
                    logger.debug("sizing_entry_too_small_after_scaling",
                                 ticker=ticker, notional=round(notional, 2))
                    return False
                is_entry = True

                logger.info(
                    "sizing_entry",
                    ticker=ticker,
                    sizing="smart_pipeline",
                    size_pct=round(notional / max(self._pm.portfolio_value, 1.0), 4),
                    stages=f"{sizing.stage1_base_pct:.3f}→{sizing.stage2_atr_pct:.3f}→{sizing.stage3_kelly_pct:.3f}→{sizing.stage4_constraint_pct:.3f}",
                    direction="long" if direction > 0 else "short",
                    pred=round(sig.lgbm_pred_return, 5),
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
                return False
            proposed_pct = notional / max(self._pm.portfolio_value, 1.0)
            size_check = self._cb.check_position_size(proposed_pct)
            if size_check.triggered:
                logger.warning(
                    "order_rejected_oversized",
                    ticker=ticker,
                    proposed_pct=f"{proposed_pct:.1%}",
                )
                return False

        # Sell exits use market orders (Alpaca rejects limit orders with
        # fractional qty, and we want guaranteed fills on exits).
        # Buy entries use limit orders for price protection.
        if side == "sell":
            limit_price = None  # market order
        else:
            quote = await self._alpaca.get_latest_quote(ticker)
            mid = quote.get("mid", 0.0)
            if mid <= 0:
                # No live quote — a limit computed from a stale DB price sits
                # below the market and gets canceled (May 26-27 cancel spam).
                logger.warning("entry_skipped_no_quote", ticker=ticker)
                return False
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
                if self._sizing_mode and is_entry:
                    # Track entry state only after a confirmed fill — canceled
                    # orders used to consume the daily cap and corrupt state
                    try:
                        from zoneinfo import ZoneInfo
                    except ImportError:
                        from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]
                    self._entry_directions[ticker] = 1
                    self._entry_prices[ticker] = fill_price
                    self._entry_dates[ticker] = datetime.now(ZoneInfo("America/New_York")).date()
                    self._peak_prices[ticker] = fill_price
                    self._bars_held[ticker] = 0
                    self._reversal_counts[ticker] = 0
                    self._sizing_n_trades_today += 1
                    if self._kelly_mode() == "probation":
                        self._probation_entries_today += 1
                await self._write_trade_entry(
                    ticker=ticker,
                    sig=sig,
                    fill_price=fill_price,
                    qty=result.filled_qty,
                    order_id=result.order_id,
                    entry_time=filled_at,
                )
            else:
                # Selling a same-day entry consumes a PDT day trade — keep the
                # local budget cache current between broker refreshes
                if self._is_same_day_entry(ticker):
                    self._daytrade_count += 1
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
                pnl_pct = pnl / max(entry_notional, 1.0)
                self._pm.record_return(pnl_pct)
                # Determine exit reason for sizing mode
                exit_reason = "signal_reversal"
                if self._sizing_mode:
                    exit_reason = self._pending_exit_reasons.pop(ticker, "signal_reversal")
                    # Track outcome for recent win rate + Kelly computation
                    self._sizing_recent_outcomes.append(
                        (filled_at if filled_at.tzinfo else filled_at.replace(tzinfo=timezone.utc), pnl_pct)
                    )
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
            return side == "buy" and is_entry
        else:
            logger.warning(
                "order_not_executed",
                ticker=ticker,
                side=side,
                status=result.status,
                error=result.error,
            )
            if side == "sell":
                # Don't re-fire the same exit every minute — back off, then
                # let sync_from_broker reconcile actual broker state
                self._exit_fail_cooldown[ticker] = 3
            else:
                self._ticker_cooldown[ticker] = 3
            return False

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
            self._probation_entries_today = 0
            self._ticker_cooldown.clear()
            self._pdt_deferred_logged.clear()
            self._prune_kelly_window()
            self._update_kelly()
            # Auto-clear daily_loss halts (structural halts like max_drawdown stay)
            self._cb.try_daily_reset()
            logger.info(
                "daily_start_value_reset",
                value=self._daily_start_value,
                kelly=round(self._kelly_fraction, 4),
                kelly_mode=self._kelly_mode(),
            )

    async def _sleep_until_next_minute(self) -> None:
        """Sleep until the next 1m bar boundary (:00 seconds)."""
        now = datetime.now(timezone.utc)
        seconds_left = 60.0 - now.second - now.microsecond / 1_000_000
        await asyncio.sleep(max(seconds_left, 1.0))
