"""Unit tests for SignalLoop — Phase 5 paper trading pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.signal_loop import SignalLoop
from src.execution.position_manager import PositionManager
from src.models.ensemble import EnsembleSignal
from src.risk.circuit_breakers import CircuitBreakers


def _make_loop(broadcast=None) -> SignalLoop:
    """Create a SignalLoop with mocked dependencies."""
    ensemble = MagicMock()
    alpaca = MagicMock()
    cb = CircuitBreakers()
    pm = PositionManager(initial_portfolio=100_000.0)
    sf = MagicMock()
    feature_cols = [f"feat_{i}" for i in range(30)]
    return SignalLoop(
        universe=["AAPL", "MSFT"],
        ensemble=ensemble,
        alpaca=alpaca,
        circuit_breakers=cb,
        pos_manager=pm,
        session_factory=sf,
        feature_cols=feature_cols,
        broadcast_fn=broadcast,
    )


def test_signal_loop_instantiation():
    loop = _make_loop()
    assert loop._n_features == 30
    assert loop._universe == ["AAPL", "MSFT"]
    assert not loop._stopped


def test_to_tensors_shapes():
    loop = _make_loop()
    arr = np.random.randn(60, 30).astype(np.float32)
    feat_1m, feat_5m = loop._to_tensors(arr)
    assert feat_1m.shape == (60, 30)
    # Every 5th of 60 rows starting at index 4: indices 4,9,14,...,59 → 12 bars
    assert feat_5m.shape == (12, 30)


def test_to_tensors_chronological_order():
    """Verify 5m tensor is every 5th bar from the 1m tensor (not reversed)."""
    loop = _make_loop()
    arr = np.arange(60 * 30, dtype=np.float32).reshape(60, 30)
    feat_1m, feat_5m = loop._to_tensors(arr)
    # feat_5m[0] should be 1m row at index 4 (arr[4])
    assert torch.allclose(feat_5m[0], feat_1m[4])
    # feat_5m[1] should be 1m row at index 9 (arr[9])
    assert torch.allclose(feat_5m[1], feat_1m[9])


def test_get_latest_signals_empty():
    loop = _make_loop()
    assert loop.get_latest_signals() == []


def test_market_hours_weekend():
    """Saturday → not market hours."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    loop = _make_loop()
    # Patch datetime inside the method
    with patch("src.agents.signal_loop.datetime") as mock_dt:
        # Saturday
        fake_now = datetime(2026, 3, 7, 10, 30, tzinfo=ZoneInfo("America/New_York"))
        mock_dt.now.return_value = fake_now
        assert not loop._is_market_hours()


def test_market_hours_weekday_open():
    """Wednesday 10:30 ET → market hours."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    loop = _make_loop()
    with patch("src.agents.signal_loop.datetime") as mock_dt:
        fake_now = datetime(2026, 3, 4, 10, 30, tzinfo=ZoneInfo("America/New_York"))
        mock_dt.now.return_value = fake_now
        assert loop._is_market_hours()


def test_circuit_breaker_blocks_order():
    """When trading is halted, _act_on_signal should return immediately."""
    loop = _make_loop()
    loop._cb._halted = True

    # _act_on_signal with halted CB should return without calling alpaca
    import asyncio

    async def run():
        sig = MagicMock(spec=EnsembleSignal)
        sig.ensemble_signal = 0.8
        sig.ticker = "AAPL"
        await loop._act_on_signal(sig, 200.0)

    asyncio.run(run())
    loop._alpaca.get_latest_quote.assert_not_called()
    loop._alpaca.submit_order.assert_not_called()


def test_position_size_respects_cap():
    """Vol-scaled position size never exceeds max_position_pct."""
    pm = PositionManager(initial_portfolio=100_000.0, max_position_pct=0.25)
    # With very low realized vol, vol_scalar is capped at 2x
    # base 5% × 2 = 10% < 25% cap
    size = pm.compute_position_size("AAPL", recent_returns=[0.0001] * 30)
    assert size <= pm.portfolio_value * pm.max_position_pct
    assert size > 0
