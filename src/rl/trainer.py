"""PPO trainer using Stable-Baselines3.

Walk-forward training: train on years 1-2, validate on year 3.
Checkpoints every 10k steps; keeps best by Sharpe ratio.

Usage:
    trainer = RLTrainer(bars_train, bars_val)
    trainer.run()
"""

from __future__ import annotations

import json
import logging
import structlog
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from src.rl.trading_env import EnvConfig, TradingEnv

logger = structlog.get_logger(__name__)

CHECKPOINT_DIR = Path("models/rl_agent")
REPORT_DIR = Path("reports")


# ─── Sharpe callback ──────────────────────────────────────────────────────────

class SharpeCheckpointCallback(BaseCallback):
    """Saves the best PPO model by Sharpe ratio on validation environment."""

    def __init__(
        self,
        val_bars: list[dict[str, Any]],
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.val_bars = val_bars
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_sharpe = -float("inf")
        self.best_model_path: Path | None = None

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            sharpe = self._evaluate_sharpe()
            logger.info(
                "rl_eval: step=%d sharpe=%.4f best=%.4f",
                self.n_calls, sharpe, self.best_sharpe,
            )
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                path = CHECKPOINT_DIR / f"best_ppo_step_{self.n_calls}_sharpe_{sharpe:.3f}.zip"
                CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                self.model.save(path)
                self.best_model_path = path
                logger.info("rl_best_saved: path=%s sharpe=%.4f", str(path), sharpe)
        return True

    def _evaluate_sharpe(self) -> float:
        """Run one full episode and compute annualized Sharpe from per-step PnL."""
        env = TradingEnv(self.val_bars)
        obs, _ = env.reset()
        done = False
        step_pnls: list[float] = []
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _reward, done, _, info = env.step(int(action))
            step_pnls.append(info.get("step_pnl", 0.0))

        if not step_pnls:
            return 0.0
        arr = np.array(step_pnls)
        # Annualized Sharpe (390 1m bars/day, 252 trading days)
        return float(arr.mean() / (arr.std() + 1e-9) * np.sqrt(252 * 390))


# ─── Walk-forward splitter ────────────────────────────────────────────────────

def walk_forward_split(
    all_bars: list[dict[str, Any]],
    train_frac: float = 0.67,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split bars into training and validation sets chronologically."""
    split_idx = int(len(all_bars) * train_frac)
    return all_bars[:split_idx], all_bars[split_idx:]


# ─── Main trainer ─────────────────────────────────────────────────────────────

class RLTrainer:
    """Wraps SB3 PPO with behavioral cloning pre-training and walk-forward eval.

    Args:
        bars_train: Training bars (years 1-2).
        bars_val: Validation bars (year 3).
        cfg: Optional EnvConfig override.
    """

    def __init__(
        self,
        bars_train: list[dict[str, Any]],
        bars_val: list[dict[str, Any]],
        cfg: EnvConfig | None = None,
    ) -> None:
        self.bars_train = bars_train
        self.bars_val = bars_val
        self.cfg = cfg or EnvConfig()
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        total_timesteps: int = 2_000_000,
        n_envs: int = 4,
        bc_epochs: int = 20,
    ) -> PPO:
        """Run behavioral cloning pre-train then full PPO training.

        Returns the best PPO model by validation Sharpe.
        """
        # ── Behavioral cloning ────────────────────────────────────────────────
        logger.info("behavioral_cloning_start")
        from src.rl.behavioral_cloning import BehavioralCloningTrainer, MACDExpertPolicy

        expert = MACDExpertPolicy()
        trajectories = expert.generate(self.bars_train[:5000])   # first 5k bars

        # Create PPO model first so we can BC-pretrain its policy
        env = TradingEnv(self.bars_train, cfg=self.cfg)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=None,
        )

        # Pre-train the policy network via the action-logit wrapper
        from src.rl.behavioral_cloning import PolicyActionWrapper
        policy_wrapper = PolicyActionWrapper(model.policy)
        bc_trainer = BehavioralCloningTrainer(policy_wrapper)
        bc_trainer.train(trajectories, epochs=bc_epochs)
        logger.info("behavioral_cloning_complete")

        # ── PPO training ──────────────────────────────────────────────────────
        sharpe_cb = SharpeCheckpointCallback(self.bars_val)
        checkpoint_cb = CheckpointCallback(
            save_freq=10_000,
            save_path=str(CHECKPOINT_DIR / "periodic"),
            name_prefix="ppo",
        )

        logger.info("ppo_training_start: timesteps=%d", total_timesteps)
        model.learn(
            total_timesteps=total_timesteps,
            callback=[sharpe_cb, checkpoint_cb],
            reset_num_timesteps=False,
        )

        # ── Load best by Sharpe ───────────────────────────────────────────────
        if sharpe_cb.best_model_path:
            best_model = PPO.load(sharpe_cb.best_model_path, env=env)
            logger.info(
                "ppo_best_loaded: path=%s sharpe=%.4f",
                str(sharpe_cb.best_model_path), sharpe_cb.best_sharpe,
            )
            return best_model

        return model

    def backtest(self, model: PPO) -> dict[str, float]:
        """Run a full backtest on validation bars and return performance metrics."""
        env = TradingEnv(self.bars_val, cfg=self.cfg)
        obs, _ = env.reset()
        done = False
        returns: list[float] = []
        portfolio_values: list[float] = [env.cfg.initial_portfolio]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(int(action))
            returns.append(info.get("step_pnl", 0.0))
            portfolio_values.append(info.get("portfolio", portfolio_values[-1]))

        if not returns:
            return {}

        ret_arr = np.array(returns)
        sharpe = float(ret_arr.mean() / (ret_arr.std() + 1e-9) * np.sqrt(252 * 390))
        win_rate = float((ret_arr > 0).mean())
        gross_profit = float(ret_arr[ret_arr > 0].sum())
        gross_loss = float(abs(ret_arr[ret_arr < 0].sum()))
        profit_factor = gross_profit / max(gross_loss, 1e-9)

        # Max drawdown
        peak = portfolio_values[0]
        max_dd = 0.0
        for v in portfolio_values:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)

        metrics = {
            "sharpe": round(sharpe, 3),
            "win_rate": round(win_rate, 3),
            "profit_factor": round(profit_factor, 3),
            "max_drawdown": round(max_dd, 3),
            "final_portfolio": round(portfolio_values[-1], 2),
            "total_return_pct": round(
                (portfolio_values[-1] / portfolio_values[0] - 1) * 100, 2
            ),
            "n_bars": len(returns),
        }
        logger.info("backtest_complete: %s", metrics)
        return metrics
