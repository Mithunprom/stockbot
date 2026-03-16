"""Reward functions for RL trading agents.

compute_reward():        Full TradingEnv (9 actions, agent decides when + how much)
compute_sizing_reward(): PositionSizingEnv (5 actions, agent decides only how much)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    drawdown_penalty_factor: float = 2.0
    drawdown_free_zone: float = 0.05    # first 5% drawdown not penalized
    cost_penalty_factor: float = 0.05   # reduced: was 0.1, making trades too costly
    overhold_penalty: float = 0.0003    # reduced: was 0.001, accumulated too fast
    inactivity_penalty: float = 0.0005  # penalty for holding cash when signal is strong
    vol_floor: float = 0.001            # ~0.1% per bar — realistic 1m vol floor
    reward_clip: float = 5.0            # tighter clip for stability


def compute_reward(
    step_pnl: float,
    rolling_vol: float,
    current_drawdown: float,
    trade_cost: float,
    bars_held: int,
    position_pct: float = 0.0,
    lgbm_pred_return: float = 0.0,
    cfg: RewardConfig | None = None,
) -> float:
    """Compute step reward.

    Args:
        step_pnl: PnL this bar as a fraction of portfolio value.
        rolling_vol: Rolling daily volatility of portfolio returns.
        current_drawdown: Current drawdown from peak (positive = drawdown, e.g., 0.03 = 3%).
        trade_cost: Total round-trip cost as fraction of trade value.
        bars_held: Number of bars the current position has been held.
        position_pct: Current position as fraction of portfolio.
        lgbm_pred_return: LightGBM predicted return (used for inactivity penalty).
        cfg: Optional config override.

    Returns:
        Scalar reward.
    """
    cfg = cfg or RewardConfig()
    vol = max(rolling_vol, cfg.vol_floor)

    sharpe_component = step_pnl / vol
    drawdown_excess = max(current_drawdown - cfg.drawdown_free_zone, 0.0)
    drawdown_penalty = cfg.drawdown_penalty_factor * drawdown_excess
    cost_penalty = cfg.cost_penalty_factor * trade_cost
    overhold_penalty = cfg.overhold_penalty * bars_held

    # Inactivity penalty: penalize sitting in cash when LightGBM has a strong signal
    inactivity = 0.0
    if position_pct == 0.0 and abs(lgbm_pred_return) > 0.001:
        inactivity = cfg.inactivity_penalty * min(abs(lgbm_pred_return) * 100.0, 5.0)

    reward = sharpe_component - drawdown_penalty - cost_penalty - overhold_penalty - inactivity
    # Clip to prevent exploding gradients from vol denominator instability
    reward = max(-cfg.reward_clip, min(cfg.reward_clip, reward))
    return float(reward)


# ─── Position-Sizing Reward ──────────────────────────────────────────────────

@dataclass
class SizingRewardConfig:
    drawdown_penalty_factor: float = 1.5
    drawdown_free_zone: float = 0.05
    cost_penalty_factor: float = 0.02     # lower: entries are pre-filtered by signal gate
    vol_floor: float = 0.001
    reward_clip: float = 5.0
    conviction_bonus: float = 0.5         # bonus for sizing proportional to signal strength


def compute_sizing_reward(
    step_pnl: float,
    position_size_pct: float,
    lgbm_pred_return: float,
    rolling_vol: float,
    current_drawdown: float,
    trade_cost: float,
    is_entry_bar: bool,
    cfg: SizingRewardConfig | None = None,
) -> float:
    """Compute reward for position-sizing decisions.

    Key difference from compute_reward(): no overhold or inactivity penalties.
    Entry/exit timing is handled by LightGBM gates, so we only reward/penalize
    the sizing decision quality.
    """
    cfg = cfg or SizingRewardConfig()
    vol = max(rolling_vol, cfg.vol_floor)

    # Sharpe component: PnL normalized by volatility
    sharpe_component = step_pnl / vol

    # Size-aligned reward: scale by position size so larger correct bets
    # earn proportionally more (and larger wrong bets lose more)
    if position_size_pct > 0:
        sharpe_component *= (1.0 + position_size_pct * 5.0)  # amplify by size

    # Drawdown penalty
    drawdown_excess = max(current_drawdown - cfg.drawdown_free_zone, 0.0)
    drawdown_penalty = cfg.drawdown_penalty_factor * drawdown_excess

    # Cost penalty (light — entries are already filtered)
    cost_penalty = cfg.cost_penalty_factor * trade_cost

    # Conviction bonus: reward sizing proportional to signal strength on entry
    conviction = 0.0
    if is_entry_bar and position_size_pct > 0:
        signal_strength = min(abs(lgbm_pred_return) * 100.0, 5.0)  # cap at 5
        # Higher reward when size is proportional to signal
        size_signal_alignment = position_size_pct * signal_strength
        conviction = cfg.conviction_bonus * size_signal_alignment

    reward = sharpe_component - drawdown_penalty - cost_penalty + conviction
    reward = max(-cfg.reward_clip, min(cfg.reward_clip, reward))
    return float(reward)
