"""Unit tests for the Watchdog Agent — detection, self-heal, alert dedupe."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.agents.watchdog_agent import (
    EMAIL_DEDUPE_HOURS,
    EXIT_PIPELINE_CRIT_PCT,
    EXIT_PIPELINE_MIN_N,
    EXIT_PIPELINE_WARN_PCT,
    WatchdogAgent,
    ZOMBIE_GRACE_BARS,
)


def _make_agent(market_open: bool = True) -> WatchdogAgent:
    loop = MagicMock()
    loop.last_tick_at = datetime.now(timezone.utc)
    loop.last_exit_at = None
    loop.tick_error_count = 0
    loop.last_tick_error = ""
    loop._data_fresh = True
    loop._bars_held = {}
    loop._last_reset_date = None
    loop._universe = ["T%d" % i for i in range(75)]   # healthy screened universe
    pm = MagicMock()
    pm._positions = {}
    cb = MagicMock()
    cb._halted = False
    cb._halt_reason = ""
    agent = WatchdogAgent(signal_loop=loop, pos_manager=pm, circuit_breakers=cb)
    agent._market_open = lambda: market_open  # type: ignore[method-assign]
    # Never send real email in tests
    agent._send_email = MagicMock()  # type: ignore[method-assign]
    return agent


class TestChecks:
    def test_all_ok_when_healthy(self):
        agent = _make_agent()
        # today's reset already done
        from zoneinfo import ZoneInfo
        agent._loop._last_reset_date = datetime.now(ZoneInfo("America/New_York")).date()
        report = asyncio.run(agent.run())
        assert report["status"] == "ok"
        agent._send_email.assert_not_called()

    def test_stale_tick_is_critical(self):
        agent = _make_agent()
        agent._loop.last_tick_at = datetime.now(timezone.utc) - timedelta(minutes=20)
        check = agent._check_loop_tick_fresh()
        assert check["status"] == "critical"

    def test_new_tick_errors_warn_then_clear(self):
        agent = _make_agent()
        agent._loop.tick_error_count = 3
        first = agent._check_tick_errors()
        assert first["status"] == "warn"
        # No NEW errors on the next run → ok
        second = agent._check_tick_errors()
        assert second["status"] == "ok"

    def test_many_tick_errors_critical(self):
        agent = _make_agent()
        agent._loop.tick_error_count = 25
        assert agent._check_tick_errors()["status"] == "critical"

    def test_zombie_detection_requires_grace(self):
        from src.agents.signal_loop import SIZING_MAX_HOLD_BARS
        agent = _make_agent()
        agent._pm._positions = {"GOOGL": MagicMock()}
        # At max_hold but within grace → not a zombie yet
        agent._loop._bars_held = {"GOOGL": SIZING_MAX_HOLD_BARS + 5}
        assert agent._check_zombie_positions()["status"] == "ok"
        # Past grace → critical
        agent._loop._bars_held = {"GOOGL": SIZING_MAX_HOLD_BARS + ZOMBIE_GRACE_BARS + 1}
        check = agent._check_zombie_positions()
        assert check["status"] == "critical"
        assert check["zombies"] == ["GOOGL"]

    def test_circuit_breaker_halt_never_auto_resumed(self):
        agent = _make_agent()
        agent._cb._halted = True
        agent._cb._halt_reason = "daily_loss"
        check = agent._check_circuit_breaker()
        assert check["status"] == "critical"
        assert check["healed"] is False
        # resume_trading must never be touched
        agent._cb.resume_trading.assert_not_called()

    def test_data_stale_warns(self):
        agent = _make_agent()
        agent._loop._data_fresh = False
        assert agent._check_data_fresh()["status"] == "warn"

    def test_market_closed_skips_positional_checks(self):
        agent = _make_agent(market_open=False)
        agent._loop._bars_held = {"GOOGL": 9999}
        agent._pm._positions = {"GOOGL": MagicMock()}
        assert agent._check_zombie_positions()["status"] == "ok"
        assert agent._check_daily_reset()["status"] == "ok"


class TestHealing:
    def test_daily_reset_heals(self):
        from zoneinfo import ZoneInfo
        agent = _make_agent()
        today = datetime.now(ZoneInfo("America/New_York")).date()

        def do_reset():
            agent._loop._last_reset_date = today

        agent._loop._maybe_reset_daily_value = do_reset
        with patch.object(WatchdogAgent, "_market_open", return_value=True):
            check = agent._check_daily_reset()
        # Outside the 9:35+ window this may be n/a — only assert when applicable
        if "not applicable" not in check["detail"]:
            assert check["healed"] is True
            assert check["status"] == "ok"

    def test_force_exit_only_in_paper_mode(self):
        agent = _make_agent()
        pos = MagicMock()
        pos.qty = 10
        pos.side = "long"
        agent._pm._positions = {"GOOGL": pos}
        with patch("src.config.get_settings") as gs:
            gs.return_value.alpaca_mode = "live"
            healed = asyncio.run(agent._heal_zombies(["GOOGL"]))
        assert healed == []
        agent._loop._alpaca.submit_order.assert_not_called()

    def test_force_exit_respects_env_kill_switch(self, monkeypatch):
        agent = _make_agent()
        pos = MagicMock()
        pos.qty = 10
        pos.side = "long"
        agent._pm._positions = {"GOOGL": pos}
        monkeypatch.setenv("WATCHDOG_FORCE_EXIT", "false")
        with patch("src.config.get_settings") as gs:
            gs.return_value.alpaca_mode = "paper"
            healed = asyncio.run(agent._heal_zombies(["GOOGL"]))
        assert healed == []

    def test_force_exit_submits_market_sell(self, monkeypatch):
        agent = _make_agent()
        pos = MagicMock()
        pos.qty = 10
        pos.side = "long"
        agent._pm._positions = {"GOOGL": pos}
        monkeypatch.setenv("WATCHDOG_FORCE_EXIT", "true")

        submitted = {}

        async def fake_submit(req):
            submitted["side"] = req.side
            submitted["qty"] = req.qty
            r = MagicMock()
            r.status = "filled"
            return r

        agent._loop._alpaca.submit_order = fake_submit
        with patch("src.config.get_settings") as gs:
            gs.return_value.alpaca_mode = "paper"
            healed = asyncio.run(agent._heal_zombies(["GOOGL"]))
        assert healed == ["GOOGL"]
        assert submitted == {"side": "sell", "qty": 10}


class TestAlerting:
    def test_email_sent_on_critical(self):
        agent = _make_agent()
        agent._loop.last_tick_at = datetime.now(timezone.utc) - timedelta(minutes=30)
        asyncio.run(agent.run(light=True))
        agent._send_email.assert_called_once()

    def test_identical_issue_set_deduped(self):
        agent = _make_agent()
        agent._loop.last_tick_at = datetime.now(timezone.utc) - timedelta(minutes=30)
        asyncio.run(agent.run(light=True))
        asyncio.run(agent.run(light=True))
        assert agent._send_email.call_count == 1

    def test_dedupe_expires(self):
        agent = _make_agent()
        agent._loop.last_tick_at = datetime.now(timezone.utc) - timedelta(minutes=30)
        asyncio.run(agent.run(light=True))
        agent._last_email_at = datetime.now(timezone.utc) - timedelta(
            hours=EMAIL_DEDUPE_HOURS + 1
        )
        asyncio.run(agent.run(light=True))
        assert agent._send_email.call_count == 2

    def test_new_issue_bypasses_dedupe(self):
        agent = _make_agent()
        agent._loop.last_tick_at = datetime.now(timezone.utc) - timedelta(minutes=30)
        asyncio.run(agent.run(light=True))
        # A different issue set appears → email again immediately
        agent._loop.tick_error_count = 3
        asyncio.run(agent.run(light=True))
        assert agent._send_email.call_count == 2


class TestUniverseCheck:
    def test_empty_universe_is_critical(self):
        agent = _make_agent()
        agent._loop._universe = []
        c = agent._check_universe_fresh()
        assert c["status"] == "critical"

    def test_anchors_only_universe_warns(self):
        # 52 == the anchor count: the S&P screen contributed nothing, which
        # is exactly the silent degradation that froze the universe for weeks
        agent = _make_agent()
        agent._loop._universe = ["T%d" % i for i in range(52)]
        c = agent._check_universe_fresh()
        assert c["status"] == "warn"

    def test_screened_universe_ok(self):
        agent = _make_agent()
        agent._loop._universe = ["T%d" % i for i in range(75)]
        assert agent._check_universe_fresh()["status"] == "ok"


class TestExitPipelineHealth:
    """H31 — exit pipeline health check."""

    def _make_sf(self, exit_reasons: list[str]):
        """Build a mock session_factory returning rows with exit_reason values."""
        from unittest.mock import AsyncMock

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(r,) for r in exit_reasons]

        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        sf = MagicMock()
        sf.return_value = mock_ctx
        return sf

    def test_no_session_factory_skips_check(self):
        agent = _make_agent()
        assert agent._sf is None
        result = asyncio.run(agent._check_exit_pipeline_health())
        assert result["status"] == "ok"
        assert "skipped" in result["detail"]

    def test_below_min_n_is_ok(self):
        """Too few trades — not enough data to alert."""
        agent = _make_agent()
        agent._sf = self._make_sf(
            ["integrity_broker_reconcile"] * (EXIT_PIPELINE_MIN_N - 1)
        )
        result = asyncio.run(agent._check_exit_pipeline_health())
        assert result["status"] == "ok"
        assert "insufficient" in result["detail"]

    def test_all_strategy_exits_is_ok(self):
        reasons = ["max_hold", "stop_loss", "take_profit", "trailing_stop", "session_close"]
        agent = _make_agent()
        agent._sf = self._make_sf(reasons * 4)
        result = asyncio.run(agent._check_exit_pipeline_health())
        assert result["status"] == "ok"
        assert "0/" in result["detail"]

    def test_sep30_incident_pattern_warns(self):
        """Sep-30 2026: 6/20 = 30% integrity_broker_reconcile — at warn threshold."""
        reasons = ["integrity_broker_reconcile"] * 6 + ["max_hold"] * 14
        agent = _make_agent()
        agent._sf = self._make_sf(reasons)
        result = asyncio.run(agent._check_exit_pipeline_health())
        assert result["status"] in ("warn", "critical")

    def test_majority_orphan_exits_is_critical(self):
        n = 20
        orphan = int(EXIT_PIPELINE_CRIT_PCT * n) + 1
        reasons = (["integrity_broker_reconcile"] * orphan
                   + ["max_hold"] * (n - orphan))
        agent = _make_agent()
        agent._sf = self._make_sf(reasons)
        result = asyncio.run(agent._check_exit_pipeline_health())
        assert result["status"] == "critical"
        assert "MAJORITY" in result["detail"]

    def test_constants_are_sane(self):
        """Threshold ordering: MIN_N < lookback, WARN < CRIT <= 1."""
        from src.agents.watchdog_agent import EXIT_PIPELINE_LOOKBACK
        assert EXIT_PIPELINE_MIN_N < EXIT_PIPELINE_LOOKBACK
        assert EXIT_PIPELINE_WARN_PCT < EXIT_PIPELINE_CRIT_PCT
        assert EXIT_PIPELINE_CRIT_PCT <= 1.0
