"""Reward function for the RL trading agent.

reward = (pnl / rolling_vol)          # Sharpe component
       - 2.0 * max(drawdown - 0.05, 0) # Drawdown penalty (ramps at 5%)
       - trade_cost * 0.1              # Cost penalty
       - 0.001 * holding_time          # Overhold penalty
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    drawdown_penalty_factor: float = 2.0
    drawdown_free_zone: float = 0.05    # first 5% drawdown not penalized
    cost_penalty_factor: float = 0.1
    overhold_penalty: float = 0.001     # per bar held
    vol_floor: float = 0.001            # ~0.1% per bar — realistic 1m vol floor
    reward_clip: float = 10.0           # clip reward to [-10, +10] for RL stability


def compute_reward(
    step_pnl: float,
    rolling_vol: float,
    current_drawdown: float,
    trade_cost: float,
    bars_held: int,
    cfg: RewardConfig | None = None,
) -> float:
    """Compute step reward.

    Args:
        step_pnl: PnL this bar as a fraction of portfolio value.
        rolling_vol: Rolling daily volatility of portfolio returns.
        current_drawdown: Current drawdown from peak (positive = drawdown, e.g., 0.03 = 3%).
        trade_cost: Total round-trip cost as fraction of trade value.
        bars_held: Number of bars the current position has been held.
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

    reward = sharpe_component - drawdown_penalty - cost_penalty - overhold_penalty
    # Clip to prevent exploding gradients from vol denominator instability
    reward = max(-cfg.reward_clip, min(cfg.reward_clip, reward))
    return float(reward)
