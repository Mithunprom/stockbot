"""Behavioral cloning pre-training for the RL policy.

Generates expert trajectories from a MACD-crossover rules-based strategy.
Pre-trains the PPO policy network to imitate expert actions before full RL.
This significantly accelerates convergence.

Usage:
    expert = MACDExpertPolicy()
    trajectories = expert.generate(bars)
    bc_trainer = BehavioralCloningTrainer(policy_net)
    bc_trainer.train(trajectories)
"""

from __future__ import annotations

import logging
import structlog
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = structlog.get_logger(__name__)


# ─── MACD crossover expert ────────────────────────────────────────────────────

class MACDExpertPolicy:
    """Simple MACD crossover rules-based strategy to generate expert trajectories.

    Rules:
      - Buy (buy_medium=2): MACD crosses above signal AND RSI < 70
      - Sell all (sell_all=6): MACD crosses below signal OR RSI > 80
      - Hold (0): otherwise
      - Short (short_small=7): MACD crosses below signal AND RSI > 30
    """

    def generate(self, bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Generate (obs, action) pairs from bar sequence.

        Args:
            bars: List of bar dicts with features including macd, macd_signal, rsi_14.

        Returns:
            List of {"obs": np.ndarray, "action": int} dicts.
        """
        from src.rl.trading_env import TradingEnv, EnvConfig

        env = TradingEnv(bars, cfg=EnvConfig())
        obs, _ = env.reset()

        trajectories: list[dict[str, Any]] = []
        done = False
        position = 0.0  # track position state locally

        for i, bar in enumerate(bars[:-1]):
            macd = float(bar.get("macd", 0) or 0)
            macd_sig = float(bar.get("macd_signal", 0) or 0)
            rsi = float(bar.get("rsi_14", 50) or 50)

            prev_bar = bars[i - 1] if i > 0 else bar
            prev_macd = float(prev_bar.get("macd", 0) or 0)
            prev_macd_sig = float(prev_bar.get("macd_signal", 0) or 0)

            macd_bull_cross = prev_macd <= prev_macd_sig and macd > macd_sig
            macd_bear_cross = prev_macd >= prev_macd_sig and macd < macd_sig

            if macd_bull_cross and rsi < 70:
                action = 2   # buy_medium
            elif (macd_bear_cross or rsi > 80) and position > 0:
                action = 6   # sell_all
            elif macd_bear_cross and rsi > 30:
                action = 7   # short_small
            else:
                action = 0   # hold

            trajectories.append({"obs": obs.copy(), "action": action})
            obs, _, done, _, info = env.step(action)
            position = info.get("position_pct", 0.0)

            if done:
                break

        logger.info("expert_trajectories_generated: %d", len(trajectories))
        return trajectories


# ─── Behavioral cloning trainer ───────────────────────────────────────────────

class PolicyActionWrapper(nn.Module):
    """Wraps SB3's mlp_extractor + action_net into a single forward pass
    that returns action logits from raw observations."""

    def __init__(self, policy: Any) -> None:
        super().__init__()
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent_pi, _ = self.mlp_extractor(obs)
        return self.action_net(latent_pi)


class BehavioralCloningTrainer:
    """Supervised pre-training of the RL policy network.

    Trains with cross-entropy loss on (obs, expert_action) pairs.
    Compatible with any nn.Module that accepts obs and outputs action logits.
    """

    def __init__(
        self,
        policy_net: nn.Module,
        lr: float = 3e-4,
        device: str | None = None,
    ) -> None:
        self.policy_net = policy_net
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(
        self,
        trajectories: list[dict[str, Any]],
        epochs: int = 20,
        batch_size: int = 256,
    ) -> list[float]:
        """Train the policy network on expert demonstrations.

        Returns list of per-epoch losses.
        """
        obs_list = [t["obs"] for t in trajectories]
        act_list = [t["action"] for t in trajectories]

        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32)
        act_tensor = torch.tensor(act_list, dtype=torch.long)

        dataset = TensorDataset(obs_tensor, act_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.policy_net = self.policy_net.to(self.device)
        self.policy_net.train()

        epoch_losses: list[float] = []

        for epoch in range(epochs):
            total_loss = 0.0
            for obs_batch, act_batch in loader:
                obs_batch = obs_batch.to(self.device)
                act_batch = act_batch.to(self.device)

                logits = self.policy_net(obs_batch)
                loss = self.loss_fn(logits, act_batch)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item() * len(obs_batch)

            avg_loss = total_loss / len(dataset)
            epoch_losses.append(avg_loss)
            logger.info("bc_epoch %d: loss=%.4f", epoch + 1, avg_loss)

        logger.info("bc_training_complete: final_loss=%.4f", epoch_losses[-1])
        return epoch_losses
