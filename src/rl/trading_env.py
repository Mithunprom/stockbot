"""Custom Gymnasium trading environment for RL training.

State space (29-dim):
  ensemble_signal, transformer_conf, tcn_conf, sentiment_index,
  lgbm_pred_return, lgbm_dir_prob,
  current_position, unrealized_pnl, time_in_trade, portfolio_heat,
  vix_level, regime_label, recent_drawdown + top-10 FFSA features (padded)

Action space (9 discrete):
  0=hold, 1=buy_small, 2=buy_medium, 3=buy_large,
  4=sell_25pct, 5=sell_50pct, 6=sell_all, 7=short_small, 8=short_large

Intraday: resets each trading day, no overnight positions.
PDT tracker: alerts when approaching 3 round-trip limit.
Slippage model: 0.075% per side + $0.005/share commission.
"""

from __future__ import annotations

import logging
import structlog
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import gymnasium as gym
import numpy as np

from src.rl.reward import RewardConfig, compute_reward

logger = structlog.get_logger(__name__)

# ─── Actions ─────────────────────────────────────────────────────────────────

ACTION_NAMES = [
    "hold",
    "buy_small",    # +5% position
    "buy_medium",   # +10% position
    "buy_large",    # +20% position
    "sell_25pct",
    "sell_50pct",
    "sell_all",
    "short_small",  # -5% position
    "short_large",  # -15% position
]

# Position delta as % of portfolio for each action
_ACTION_DELTAS: dict[int, float] = {
    0: 0.0,
    1: 0.05,
    2: 0.10,
    3: 0.20,
    4: -0.25,   # sell fraction of existing
    5: -0.50,
    6: -1.0,    # sell all
    7: -0.05,
    8: -0.15,
}

STATE_DIM = 29
N_ACTIONS = len(ACTION_NAMES)

# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    initial_portfolio: float = 100_000.0
    max_position_pct: float = 0.25          # 25% max per position
    min_hold_bars: int = 3                   # prevent overtrading
    slippage_pct: float = 0.00075           # 0.075% per side
    commission_per_share: float = 0.005
    pdt_max_round_trips: int = 3            # PDT rule limit per 5 days
    pdt_account_threshold: float = 25_000  # only applies if below this
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)

    # ── Exit controls (hard rules, override agent action) ────────────────────
    stop_loss_pct: float = 0.01        # force exit if unrealized loss > 1%
    trailing_stop_pct: float = 0.015   # force exit if price drops 1.5% from peak
    take_profit_pct: float = 0.025     # lock in gains at +2.5%
    max_hold_bars: int = 60            # force exit after 60 bars (1 hour)


# ─── PDT tracker ──────────────────────────────────────────────────────────────

class PDTTracker:
    """Tracks round-trip trades for the Pattern Day Trader rule."""

    def __init__(self, max_round_trips: int = 3) -> None:
        self.max_round_trips = max_round_trips
        self._round_trips: list[date] = []

    def add_round_trip(self, dt: date) -> None:
        self._round_trips.append(dt)

    def count_last_5_days(self, today: date) -> int:
        from datetime import timedelta

        cutoff = today - timedelta(days=5)
        return sum(1 for d in self._round_trips if d > cutoff)

    def is_blocked(self, today: date, portfolio_value: float, threshold: float) -> bool:
        """Return True if placing a new trade would violate PDT rules."""
        if portfolio_value >= threshold:
            return False
        return self.count_last_5_days(today) >= self.max_round_trips


# ─── TradingEnv ───────────────────────────────────────────────────────────────

class TradingEnv(gym.Env):
    """Intraday trading environment for one ticker.

    Episodes are split by trading day (~390 bars each).
    On each reset(), the env advances to the next day (or random day).

    Args:
        bars: List of bar dicts with keys:
              time, close, features (FFSA top-10 values), ensemble_signal,
              transformer_conf, tcn_conf, sentiment_index, vix, regime
        cfg: Environment configuration.
        shuffle_days: If True, sample days randomly. If False, cycle sequentially.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: list[dict[str, Any]],
        cfg: EnvConfig | None = None,
        shuffle_days: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self._shuffle_days = shuffle_days

        # Split bars into daily episodes
        self._days = self._split_days(bars)
        if not self._days:
            self._days = [bars]  # fallback: use all bars as one episode
        self._day_idx = 0
        self.bars = self._days[0]

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

        self._reset_state()

    @staticmethod
    def _split_days(bars: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Split bars into per-day chunks (min 50 bars/day to filter partial days)."""
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
        self._position_pct = 0.0       # current position as % of portfolio
        self._entry_price: float | None = None
        self._peak_price_since_entry: float | None = None   # for trailing stop
        self._bars_held = 0
        self._returns_history: list[float] = [0.0] * 20  # rolling vol window
        self._pdt = PDTTracker(self.cfg.pdt_max_round_trips)
        self._trades_today: int = 0
        self._done = False

    # ── Observation ──────────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        bar = self.bars[self._step_idx]
        drawdown = (self._peak_portfolio - self._portfolio) / self._peak_portfolio

        # Rolling vol (std of last 20 returns)
        rolling_vol = float(np.std(self._returns_history[-20:]) + 1e-9)

        unrealized_pnl = 0.0
        if self._entry_price and self._position_pct != 0:
            price_change = (bar["close"] - self._entry_price) / self._entry_price
            unrealized_pnl = price_change * self._position_pct

        # Core state features
        state = [
            float(bar.get("ensemble_signal", 0.0)),
            float(bar.get("transformer_conf", 0.0)),
            float(bar.get("tcn_conf", 0.0)),
            float(bar.get("sentiment_index", 0.0)),
            float(bar.get("lgbm_pred_return", 0.0)) * 100.0,  # scale up for RL
            float(bar.get("lgbm_dir_prob", 0.5)),
            float(self._position_pct),
            float(unrealized_pnl),
            float(self._bars_held) / 100.0,     # normalized
            float(abs(self._position_pct)),      # portfolio_heat
            float(bar.get("vix", 20.0)) / 100.0,
            float(bar.get("regime", 0.0)),       # 0=trending, 1=mean-rev, 2=high-vol
            float(drawdown),
        ]

        # Top-10 FFSA features (padded to 16 if fewer)
        ffsa = bar.get("features", [])[:10]
        ffsa_padded = list(ffsa) + [0.0] * (16 - len(ffsa))
        state.extend(ffsa_padded[:16])

        # Pad to STATE_DIM
        obs = np.array(state[:STATE_DIM], dtype=np.float32)
        return obs

    # ── Hard exit rules ──────────────────────────────────────────────────────

    def _check_exit_rules(self, price: float) -> str | None:
        """Check if a hard exit rule is triggered. Returns reason or None."""
        if self._position_pct == 0 or self._entry_price is None:
            return None

        unrealized = (price - self._entry_price) / self._entry_price

        # For short positions, unrealized gain/loss is inverted
        if self._position_pct < 0:
            unrealized = -unrealized

        # Stop loss: exit if loss exceeds threshold
        if unrealized < -self.cfg.stop_loss_pct:
            return "stop_loss"

        # Take profit: lock in gains
        if unrealized > self.cfg.take_profit_pct:
            return "take_profit"

        # Trailing stop: track peak price since entry, exit on pullback
        if self._peak_price_since_entry is not None:
            if self._position_pct > 0:
                drop_from_peak = (self._peak_price_since_entry - price) / self._peak_price_since_entry
                if drop_from_peak > self.cfg.trailing_stop_pct:
                    return "trailing_stop"
            else:  # short: track trough
                rise_from_trough = (price - self._peak_price_since_entry) / self._peak_price_since_entry
                if rise_from_trough > self.cfg.trailing_stop_pct:
                    return "trailing_stop"

        # Max hold duration
        if self._bars_held >= self.cfg.max_hold_bars:
            return "max_hold_exceeded"

        return None

    def _update_peak_price(self, price: float) -> None:
        """Update the peak (or trough for shorts) price since entry."""
        if self._position_pct == 0:
            self._peak_price_since_entry = None
            return
        if self._peak_price_since_entry is None:
            self._peak_price_since_entry = price
        elif self._position_pct > 0:
            self._peak_price_since_entry = max(self._peak_price_since_entry, price)
        else:
            self._peak_price_since_entry = min(self._peak_price_since_entry, price)

    # ── Reward ───────────────────────────────────────────────────────────────

    def _compute_step_reward(self, step_pnl: float, trade_cost: float) -> float:
        bar = self.bars[self._step_idx]
        rolling_vol = float(np.std(self._returns_history[-20:]) + 1e-9)
        drawdown = (self._peak_portfolio - self._portfolio) / self._peak_portfolio
        return compute_reward(
            step_pnl=step_pnl,
            rolling_vol=rolling_vol,
            current_drawdown=drawdown,
            trade_cost=trade_cost,
            bars_held=self._bars_held,
            position_pct=self._position_pct,
            lgbm_pred_return=float(bar.get("lgbm_pred_return", 0.0)),
            cfg=self.cfg.reward_cfg,
        )

    # ── Action execution ─────────────────────────────────────────────────────

    def _execute_action(self, action: int) -> tuple[float, float]:
        """Execute action, return (step_pnl, trade_cost)."""
        bar = self.bars[self._step_idx]
        price = bar["close"]
        delta = _ACTION_DELTAS[action]
        trade_cost = 0.0

        # Enforce minimum hold period (only while in a position — not for new entries)
        if self._position_pct != 0 and self._bars_held < self.cfg.min_hold_bars and action not in (0, 6):
            return 0.0, 0.0

        # PDT check
        today = datetime.fromisoformat(str(bar["time"])).date()
        if self._pdt.is_blocked(today, self._portfolio, self.cfg.pdt_account_threshold):
            if action not in (0,):  # block all trade actions
                logger.warning("pdt_block: action=%s", ACTION_NAMES[action])
                return 0.0, 0.0

        # Compute new position
        # Sell actions are relative to current position, not portfolio
        if action == 6:    # sell_all  → fully close
            new_pos = 0.0
        elif action == 4:  # sell_25pct → reduce position by 25%
            new_pos = self._position_pct * 0.75
        elif action == 5:  # sell_50pct → reduce position by 50%
            new_pos = self._position_pct * 0.50
        else:              # buy / short / hold — fixed delta
            new_pos = self._position_pct + delta
            new_pos = max(-self.cfg.max_position_pct, min(self.cfg.max_position_pct, new_pos))

        # Slippage + commission on the traded notional
        traded_pct = abs(new_pos - self._position_pct)
        if traded_pct > 0:
            traded_notional = traded_pct * self._portfolio
            shares_traded = traded_notional / max(price, 0.01)
            trade_cost = (
                traded_notional * self.cfg.slippage_pct
                + shares_traded * self.cfg.commission_per_share
            ) / self._portfolio

            # Track entry price for unrealized PnL
            if self._position_pct == 0 and new_pos != 0:
                self._entry_price = price
                self._bars_held = 0
            elif new_pos == 0:
                # Full exit — record round trip if we had a position
                if self._position_pct != 0:
                    self._pdt.add_round_trip(today)
                self._entry_price = None
                self._bars_held = 0

        self._position_pct = new_pos

        # PnL from price movement on the current position
        next_idx = min(self._step_idx + 1, len(self.bars) - 1)
        next_price = self.bars[next_idx]["close"]
        price_ret = (next_price - price) / max(price, 0.01)
        step_pnl = self._position_pct * price_ret - trade_cost

        return step_pnl, trade_cost

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Pick next day's bars
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

        # Hard exit rules override agent action
        bar = self.bars[self._step_idx]
        exit_reason = self._check_exit_rules(bar["close"])
        if exit_reason:
            action = 6  # sell_all (force close)
            logger.debug("exit_rule_triggered: %s at bar %d", exit_reason, self._step_idx)

        # Update peak price tracking before executing action
        self._update_peak_price(bar["close"])

        step_pnl, trade_cost = self._execute_action(action)

        # Update portfolio
        self._portfolio *= 1 + step_pnl
        self._peak_portfolio = max(self._peak_portfolio, self._portfolio)
        self._returns_history.append(step_pnl)

        if self._position_pct != 0:
            self._bars_held += 1

        reward = self._compute_step_reward(step_pnl, trade_cost)

        # Advance time
        self._step_idx += 1
        done = self._step_idx >= len(self.bars) - 1

        # End-of-day: force close any open position
        if done and self._position_pct != 0:
            bar = self.bars[self._step_idx]
            price_ret = 0.0   # already at close
            eod_pnl = self._position_pct * price_ret
            self._portfolio *= 1 + eod_pnl
            self._position_pct = 0.0

        self._done = done
        obs = self._build_obs() if not done else np.zeros(STATE_DIM, dtype=np.float32)
        info = {
            "portfolio": self._portfolio,
            "position_pct": self._position_pct,
            "action_name": ACTION_NAMES[action],
            "step_pnl": step_pnl,
        }
        return obs, reward, done, False, info

    def render(self) -> None:
        bar = self.bars[min(self._step_idx, len(self.bars) - 1)]
        print(
            f"Step {self._step_idx} | Portfolio: ${self._portfolio:,.0f} "
            f"| Position: {self._position_pct:.1%} | Bars held: {self._bars_held}"
        )
