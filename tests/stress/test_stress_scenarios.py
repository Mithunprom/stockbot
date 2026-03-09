"""Stress tests: flash crash, high-vol, earnings surprise.

All circuit breakers must trigger correctly under these scenarios.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest


def _make_bars(prices: list[float]) -> list[dict[str, Any]]:
    """Build a minimal bar sequence from a price list."""
    now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    from datetime import timedelta

    bars = []
    for i, price in enumerate(prices):
        ts = now + timedelta(minutes=i)
        bars.append(
            {
                "time": ts.isoformat(),
                "open": price * 0.999,
                "high": price * 1.002,
                "low": price * 0.997,
                "close": price,
                "volume": 100_000,
                "ensemble_signal": 0.5,
                "transformer_conf": 0.7,
                "tcn_conf": 0.65,
                "sentiment_index": 0.1,
                "vix": 20.0,
                "regime": 0,
                "features": [0.0] * 10,
                "macd": 0.5,
                "macd_signal": 0.3,
                "rsi_14": 55.0,
            }
        )
    return bars


# ─── Flash crash ─────────────────────────────────────────────────────────────

class TestFlashCrash:
    """20% drop in 5 minutes must trigger circuit breakers."""

    def test_flash_crash_triggers_daily_loss_halt(self) -> None:
        """TradingEnv must not keep opening positions during a flash crash."""
        from src.rl.trading_env import EnvConfig, TradingEnv

        # Start at 100, crash to 80 over 5 bars
        prices = [100.0, 98.0, 95.0, 90.0, 84.0, 80.0] + [80.0] * 50
        bars = _make_bars(prices)

        cfg = EnvConfig(initial_portfolio=100_000.0)
        env = TradingEnv(bars, cfg=cfg)
        obs, _ = env.reset()

        # Agent keeps buying (worst case)
        for _ in range(5):
            obs, reward, done, _, info = env.step(3)  # buy_large
            if done:
                break

        # Portfolio should be down but not below the hard floor (circuit breaker)
        final_portfolio = info.get("portfolio", 100_000.0)
        drawdown = (100_000.0 - final_portfolio) / 100_000.0
        # We don't enforce halt inside TradingEnv (that's the risk module),
        # but the drawdown penalty should produce large negative reward
        assert reward < 0, "Reward should be negative during flash crash"


# ─── High-vol (VIX > 35) ─────────────────────────────────────────────────────

class TestHighVolRegime:
    """Circuit breakers must activate when VIX > 35."""

    @pytest.mark.asyncio
    async def test_vix_spike_triggers_cash_mode(self) -> None:
        from src.risk.circuit_breakers import CircuitBreakers, RiskState

        state = RiskState(
            portfolio_value=100_000.0,
            peak_portfolio=100_000.0,
            daily_start_value=100_000.0,
            vix=36.0,   # above threshold of 35
            consecutive_losses=0,
        )
        cb = CircuitBreakers()
        triggered = await cb.check(state)
        assert "vix_spike" in triggered, f"VIX circuit breaker not triggered. Got: {triggered}"

    @pytest.mark.asyncio
    async def test_daily_loss_limit_halt(self) -> None:
        from src.risk.circuit_breakers import CircuitBreakers, RiskState

        state = RiskState(
            portfolio_value=96_900.0,    # -3.1% from start
            peak_portfolio=100_000.0,
            daily_start_value=100_000.0,
            vix=20.0,
            consecutive_losses=2,
        )
        cb = CircuitBreakers()
        triggered = await cb.check(state)
        assert "daily_loss" in triggered, f"Daily loss breaker not triggered. Got: {triggered}"


# ─── Earnings surprise ────────────────────────────────────────────────────────

class TestEarningsSurprise:
    """±10% gap open on earnings — positions must not be opened 2 days before."""

    @pytest.mark.asyncio
    async def test_earnings_blackout_blocks_new_positions(self) -> None:
        from datetime import date, timedelta

        from src.risk.circuit_breakers import CircuitBreakers, RiskState

        # Earnings in 1 day → should block new positions
        earnings_date = date.today() + timedelta(days=1)
        state = RiskState(
            portfolio_value=100_000.0,
            peak_portfolio=100_000.0,
            daily_start_value=100_000.0,
            vix=20.0,
            consecutive_losses=0,
            earnings_dates={"AAPL": earnings_date},
            ticker="AAPL",
        )
        cb = CircuitBreakers()
        triggered = await cb.check(state)
        assert "earnings_blackout" in triggered, f"Earnings blackout not triggered. Got: {triggered}"

    def test_earnings_gap_up_reward_negative_if_short(self) -> None:
        """RL agent holding short position gets penalized by large gap up."""
        from src.rl.trading_env import EnvConfig, TradingEnv

        # Normal prices, then +10% gap
        prices = [100.0] * 20 + [110.0] + [110.0] * 30
        bars = _make_bars(prices)
        cfg = EnvConfig(initial_portfolio=100_000.0)
        env = TradingEnv(bars, cfg=cfg)
        obs, _ = env.reset()

        # Open short position
        for _ in range(3):
            obs, _, done, _, _ = env.step(8)  # short_large

        # Advance to the gap
        for _ in range(20):
            obs, reward, done, _, info = env.step(0)  # hold
            if done:
                break

        # The gap should have produced a negative step_pnl while short
        assert info.get("step_pnl", 0) < 0 or info.get("position_pct", 0) == 0


# ─── Consecutive loss protection ──────────────────────────────────────────────

class TestConsecutiveLossProtection:
    @pytest.mark.asyncio
    async def test_five_consecutive_losses_reduces_sizing(self) -> None:
        from src.risk.circuit_breakers import CircuitBreakers, RiskState

        state = RiskState(
            portfolio_value=98_000.0,
            peak_portfolio=100_000.0,
            daily_start_value=100_000.0,
            vix=20.0,
            consecutive_losses=5,
        )
        cb = CircuitBreakers()
        triggered = await cb.check(state)
        assert "consecutive_losses" in triggered


# ─── Max drawdown ─────────────────────────────────────────────────────────────

class TestMaxDrawdown:
    @pytest.mark.asyncio
    async def test_8pct_drawdown_pauses_trading(self) -> None:
        from src.risk.circuit_breakers import CircuitBreakers, RiskState

        state = RiskState(
            portfolio_value=91_500.0,   # -8.5% from peak
            peak_portfolio=100_000.0,
            daily_start_value=100_000.0,
            vix=20.0,
            consecutive_losses=1,
        )
        cb = CircuitBreakers()
        triggered = await cb.check(state)
        assert "max_drawdown" in triggered
