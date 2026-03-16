"""Position-sizing Gymnasium environment for RL training.

LightGBM decides WHEN to trade (entry/exit gating).
RL agent decides HOW MUCH to allocate (position sizing).

State space (18-dim):
  lgbm_pred_return, lgbm_dir_prob, signal_to_cost_ratio,
  current_position, unrealized_pnl, bars_in_trade, portfolio_heat,
  drawdown, rolling_vol, vix, regime, recent_win_rate, n_trades_today,
  + top-5 FFSA features

Action space (5 discrete):
  0=skip, 1=tiny(2%), 2=small(5%), 3=medium(10%), 4=large(20%)

Entry gate: |lgbm_pred_return| > cost_threshold AND dir_prob outside [0.45, 0.55]
Exit gate: signal reversal (2 bars), hard rules (stop_loss, trailing_stop,
           take_profit, max_hold), or end-of-day.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import gymnasium as gym
import numpy as np

from src.rl.reward import SizingRewardConfig, compute_sizing_reward
from src.rl.trading_env import PDTTracker

logger = structlog.get_logger(__name__)

# ─── Actions ─────────────────────────────────────────────────────────────────

SIZING_ACTION_NAMES = ["skip", "tiny", "small", "medium", "large"]
SIZING_ACTION_PCTS = {0: 0.0, 1: 0.02, 2: 0.05, 3: 0.10, 4: 0.20}
SIZING_STATE_DIM = 18
SIZING_N_ACTIONS = len(SIZING_ACTION_NAMES)

# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class SizingEnvConfig:
    initial_portfolio: float = 100_000.0
    max_position_pct: float = 0.25
    slippage_pct: float = 0.00075          # 0.075% per side
    commission_per_share: float = 0.005
    pdt_max_round_trips: int = 3
    pdt_account_threshold: float = 25_000
    reward_cfg: SizingRewardConfig = field(default_factory=SizingRewardConfig)

    # Entry gating
    cost_threshold: float = 0.0015         # min |pred_return| to enter (= round-trip cost)
    dir_prob_dead_zone: tuple[float, float] = (0.45, 0.55)

    # Exit controls
    stop_loss_pct: float = 0.02            # wider: 2% (was 1%)
    trailing_stop_pct: float = 0.025       # wider: 2.5% (was 1.5%)
    take_profit_pct: float = 0.035         # wider: 3.5% (was 2.5%)
    max_hold_bars: int = 45                # ~45 minutes
    reversal_bars: int = 2                 # signal must reverse for 2 bars to trigger exit


# ─── PositionSizingEnv ──────────────────────────────────────────────────────

class PositionSizingEnv(gym.Env):
    """Position-sizing environment: LightGBM gates entry/exit, RL sizes.

    Args:
        bars: List of bar dicts (must include lgbm_pred_return, lgbm_dir_prob).
        cfg: Environment configuration.
        shuffle_days: Randomize day order on reset.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: list[dict[str, Any]],
        cfg: SizingEnvConfig | None = None,
        shuffle_days: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg or SizingEnvConfig()
        self._shuffle_days = shuffle_days

        self._days = self._split_days(bars)
        if not self._days:
            self._days = [bars]
        self._day_idx = 0
        self.bars = self._days[0]

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(SIZING_STATE_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(SIZING_N_ACTIONS)
        self._reset_state()

    @staticmethod
    def _split_days(bars: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Split bars into per-day chunks (min 50 bars/day)."""
        days: list[list[dict[str, Any]]] = []
        current_day: list[dict[str, Any]] = []
        current_date: date | None = None
        for bar in bars:
            try:
                bar_date = datetime.fromisoformat(str(bar["time"])).date()
            except (ValueError, TypeError):
                continue
            if current_date is None:
                current_date = bar_date
            if bar_date != current_date:
                if len(current_day) >= 50:
                    days.append(current_day)
                current_day = []
                current_date = bar_date
            current_day.append(bar)
        if len(current_day) >= 50:
            days.append(current_day)
        return days

    def _reset_state(self) -> None:
        self._step_idx = 0
        self._portfolio = self.cfg.initial_portfolio
        self._peak_portfolio = self.cfg.initial_portfolio
        self._position_pct = 0.0
        self._entry_direction: int = 0        # +1 long, -1 short
        self._entry_price: float | None = None
        self._peak_price_since_entry: float | None = None
        self._bars_held = 0
        self._returns_history: list[float] = [0.0] * 20
        self._pdt = PDTTracker(self.cfg.pdt_max_round_trips)
        self._n_trades_today = 0
        self._reversal_count = 0              # consecutive reversal bars
        self._recent_outcomes: list[float] = []  # last 10 trade PnLs
        self._done = False

    # ── Entry / exit gating ───────────────────────────────────────────────────

    def _entry_gate_open(self, bar: dict) -> bool:
        """Check if LightGBM signal warrants an entry."""
        pred_ret = float(bar.get("lgbm_pred_return", 0.0))
        dir_prob = float(bar.get("lgbm_dir_prob", 0.5))
        lo, hi = self.cfg.dir_prob_dead_zone
        return abs(pred_ret) > self.cfg.cost_threshold and not (lo < dir_prob < hi)

    def _signal_direction(self, bar: dict) -> int:
        """Get signal direction from LightGBM: +1 long, -1 short."""
        pred_ret = float(bar.get("lgbm_pred_return", 0.0))
        if pred_ret > 0:
            return 1
        elif pred_ret < 0:
            return -1
        return 0

    def _check_signal_reversal(self, bar: dict) -> bool:
        """Check if LightGBM signal has reversed vs entry direction."""
        if self._entry_direction == 0:
            return False
        current_dir = self._signal_direction(bar)
        if current_dir != 0 and current_dir != self._entry_direction:
            self._reversal_count += 1
        else:
            self._reversal_count = 0
        return self._reversal_count >= self.cfg.reversal_bars

    # ── Hard exit rules (reused from TradingEnv) ─────────────────────────────

    def _check_exit_rules(self, price: float) -> str | None:
        if self._position_pct == 0 or self._entry_price is None:
            return None
        unrealized = (price - self._entry_price) / self._entry_price
        if self._position_pct < 0:
            unrealized = -unrealized
        if unrealized < -self.cfg.stop_loss_pct:
            return "stop_loss"
        if unrealized > self.cfg.take_profit_pct:
            return "take_profit"
        if self._peak_price_since_entry is not None:
            if self._position_pct > 0:
                drop = (self._peak_price_since_entry - price) / self._peak_price_since_entry
                if drop > self.cfg.trailing_stop_pct:
                    return "trailing_stop"
            else:
                rise = (price - self._peak_price_since_entry) / self._peak_price_since_entry
                if rise > self.cfg.trailing_stop_pct:
                    return "trailing_stop"
        if self._bars_held >= self.cfg.max_hold_bars:
            return "max_hold"
        return None

    def _update_peak_price(self, price: float) -> None:
        if self._position_pct == 0:
            self._peak_price_since_entry = None
            return
        if self._peak_price_since_entry is None:
            self._peak_price_since_entry = price
        elif self._position_pct > 0:
            self._peak_price_since_entry = max(self._peak_price_since_entry, price)
        else:
            self._peak_price_since_entry = min(self._peak_price_since_entry, price)

    # ── Trade execution ──────────────────────────────────────────────────────

    def _open_position(self, size_pct: float, direction: int, price: float) -> float:
        """Open a position. Returns trade cost as fraction of portfolio."""
        self._position_pct = size_pct * direction
        self._entry_direction = direction
        self._entry_price = price
        self._peak_price_since_entry = price
        self._bars_held = 0
        self._reversal_count = 0
        self._n_trades_today += 1

        traded_notional = size_pct * self._portfolio
        shares = traded_notional / max(price, 0.01)
        cost = (
            traded_notional * self.cfg.slippage_pct
            + shares * self.cfg.commission_per_share
        ) / self._portfolio
        return cost

    def _close_position(self, price: float) -> tuple[float, float]:
        """Close position. Returns (realized_pnl, trade_cost)."""
        if self._entry_price is None or self._position_pct == 0:
            return 0.0, 0.0

        price_ret = (price - self._entry_price) / self._entry_price
        if self._position_pct < 0:
            price_ret = -price_ret
        realized_pnl = abs(self._position_pct) * price_ret

        traded_notional = abs(self._position_pct) * self._portfolio
        shares = traded_notional / max(price, 0.01)
        cost = (
            traded_notional * self.cfg.slippage_pct
            + shares * self.cfg.commission_per_share
        ) / self._portfolio

        # Record outcome
        self._recent_outcomes.append(realized_pnl - cost)
        if len(self._recent_outcomes) > 10:
            self._recent_outcomes = self._recent_outcomes[-10:]

        # PDT tracking
        try:
            today = datetime.fromisoformat(str(self.bars[self._step_idx]["time"])).date()
            self._pdt.add_round_trip(today)
        except (ValueError, TypeError, IndexError):
            pass

        self._position_pct = 0.0
        self._entry_direction = 0
        self._entry_price = None
        self._peak_price_since_entry = None
        self._bars_held = 0
        self._reversal_count = 0

        return realized_pnl - cost, cost

    # ── Observation ──────────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        bar = self.bars[self._step_idx]
        pred_ret = float(bar.get("lgbm_pred_return", 0.0))
        dir_prob = float(bar.get("lgbm_dir_prob", 0.5))
        drawdown = (self._peak_portfolio - self._portfolio) / self._peak_portfolio
        rolling_vol = float(np.std(self._returns_history[-20:]) + 1e-9)

        unrealized_pnl = 0.0
        if self._entry_price and self._position_pct != 0:
            price_change = (bar["close"] - self._entry_price) / self._entry_price
            if self._position_pct < 0:
                price_change = -price_change
            unrealized_pnl = abs(self._position_pct) * price_change

        recent_wr = 0.5
        if self._recent_outcomes:
            recent_wr = sum(1 for x in self._recent_outcomes if x > 0) / len(self._recent_outcomes)

        state = [
            pred_ret * 100.0,                                  # [0] scaled pred return
            dir_prob,                                          # [1] direction probability
            abs(pred_ret) / max(self.cfg.cost_threshold, 1e-6),  # [2] signal-to-cost ratio
            float(self._position_pct),                         # [3] current position
            float(unrealized_pnl),                             # [4] unrealized PnL
            float(self._bars_held) / 60.0,                     # [5] normalized bars held
            float(abs(self._position_pct)),                    # [6] portfolio heat
            float(drawdown),                                   # [7] drawdown
            float(rolling_vol) * 100.0,                        # [8] scaled rolling vol
            float(bar.get("vix", 20.0)) / 100.0,              # [9] VIX
            float(bar.get("regime", 0.0)) / 2.0,              # [10] regime normalized
            float(recent_wr),                                  # [11] recent win rate
            float(self._n_trades_today) / 10.0,               # [12] trades today
        ]

        # Top-5 FFSA features
        ffsa = bar.get("features", [])[:5]
        ffsa_padded = list(ffsa) + [0.0] * (5 - len(ffsa))
        state.extend(ffsa_padded[:5])

        return np.array(state[:SIZING_STATE_DIM], dtype=np.float32)

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if self._shuffle_days:
            self._day_idx = self.np_random.integers(0, len(self._days))
        else:
            self._day_idx = (self._day_idx + 1) % len(self._days)
        self.bars = self._days[self._day_idx]
        self._reset_state()
        return self._build_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("step() called on done episode — reset first")

        bar = self.bars[self._step_idx]
        price = bar["close"]
        step_pnl = 0.0
        trade_cost = 0.0
        is_entry_bar = False

        # ── Position open: check exits ────────────────────────────────────────
        if self._position_pct != 0:
            self._update_peak_price(price)

            exit_reason = self._check_exit_rules(price)
            if not exit_reason and self._check_signal_reversal(bar):
                exit_reason = "signal_reversal"

            if exit_reason:
                step_pnl, trade_cost = self._close_position(price)
                logger.debug("exit: %s at bar %d, pnl=%.4f", exit_reason, self._step_idx, step_pnl)
            else:
                # Mark-to-market PnL while holding
                next_idx = min(self._step_idx + 1, len(self.bars) - 1)
                next_price = self.bars[next_idx]["close"]
                price_ret = (next_price - price) / max(price, 0.01)
                if self._position_pct < 0:
                    price_ret = -price_ret
                step_pnl = abs(self._position_pct) * price_ret
                self._bars_held += 1

        # ── No position: check entry gate ─────────────────────────────────────
        elif self._entry_gate_open(bar):
            is_entry_bar = True
            size_pct = SIZING_ACTION_PCTS.get(action, 0.0)
            if size_pct > 0:
                direction = self._signal_direction(bar)
                if direction != 0:
                    trade_cost = self._open_position(size_pct, direction, price)
                    # Immediate PnL from first bar
                    next_idx = min(self._step_idx + 1, len(self.bars) - 1)
                    next_price = self.bars[next_idx]["close"]
                    price_ret = (next_price - price) / max(price, 0.01)
                    if direction < 0:
                        price_ret = -price_ret
                    step_pnl = size_pct * price_ret - trade_cost

        # ── Update portfolio ──────────────────────────────────────────────────
        self._portfolio *= 1 + step_pnl
        self._peak_portfolio = max(self._peak_portfolio, self._portfolio)
        self._returns_history.append(step_pnl)

        # ── Reward ────────────────────────────────────────────────────────────
        rolling_vol = float(np.std(self._returns_history[-20:]) + 1e-9)
        drawdown = (self._peak_portfolio - self._portfolio) / self._peak_portfolio
        reward = compute_sizing_reward(
            step_pnl=step_pnl,
            position_size_pct=abs(self._position_pct),
            lgbm_pred_return=float(bar.get("lgbm_pred_return", 0.0)),
            rolling_vol=rolling_vol,
            current_drawdown=drawdown,
            trade_cost=trade_cost,
            is_entry_bar=is_entry_bar,
            cfg=self.cfg.reward_cfg,
        )

        # ── Advance time ──────────────────────────────────────────────────────
        self._step_idx += 1
        done = self._step_idx >= len(self.bars) - 1

        # End-of-day: force close
        if done and self._position_pct != 0:
            eod_pnl, eod_cost = self._close_position(self.bars[self._step_idx]["close"])
            self._portfolio *= 1 + eod_pnl
            step_pnl += eod_pnl

        self._done = done
        obs = self._build_obs() if not done else np.zeros(SIZING_STATE_DIM, dtype=np.float32)
        info = {
            "portfolio": self._portfolio,
            "position_pct": self._position_pct,
            "action_name": SIZING_ACTION_NAMES[action] if is_entry_bar else "auto",
            "step_pnl": step_pnl,
            "is_entry_bar": is_entry_bar,
        }
        return obs, reward, done, False, info
