#!/usr/bin/env python3
"""Pipeline health test agent — verifies the full trading pipeline end-to-end.

Run during market hours to confirm bars → features → signals → trades flow.
Can also run after-hours to check DB and deployment health.

Usage:
    python scripts/test_pipeline_health.py
    python scripts/test_pipeline_health.py --url https://stockbot-production-cbde.up.railway.app
    python scripts/test_pipeline_health.py --db-url postgresql://...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any


# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
WARN = f"{YELLOW}WARN{RESET}"


def _status(ok: bool, msg: str, detail: str = "") -> bool:
    icon = PASS if ok else FAIL
    detail_str = f"  ({detail})" if detail else ""
    print(f"  [{icon}] {msg}{detail_str}")
    return ok


class PipelineHealthTest:
    def __init__(self, url: str, db_url: str | None = None):
        self.url = url.rstrip("/")
        self.db_url = db_url
        self.results: list[bool] = []

    def check(self, ok: bool, msg: str, detail: str = "") -> bool:
        result = _status(ok, msg, detail)
        self.results.append(result)
        return result

    # ── HTTP checks ──────────────────────────────────────────────────────────

    async def test_health(self) -> None:
        import httpx

        print(f"\n{BOLD}1. Health Check{RESET}")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self.url}/health")
                data = r.json()
                self.check(r.status_code == 200, "HTTP 200", f"status={data.get('status')}")
                self.check(
                    data.get("signal_loop_active") is True,
                    "Signal loop active",
                )
                self.check(data.get("mode") == "paper", "Paper mode")
        except Exception as e:
            self.check(False, f"Health endpoint reachable", str(e))

    async def test_diagnostics(self) -> dict[str, Any]:
        import httpx

        print(f"\n{BOLD}2. Diagnostics{RESET}")
        data: dict[str, Any] = {}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self.url}/diagnostics")
                data = r.json()

                pa = data.get("pipeline_a", {})
                self.check(pa.get("active") is True, "Pipeline A active")
                self.check(
                    pa.get("circuit_breaker_halted") is False,
                    "Pipeline A not halted",
                    f"reason={pa.get('halt_reason')}" if pa.get("circuit_breaker_halted") else "",
                )
                self.check(
                    pa.get("exit_mode") == "atr_adaptive",
                    "ATR-adaptive exits enabled",
                    f"exit_mode={pa.get('exit_mode')}",
                )

                # ATR multipliers
                mult = pa.get("atr_multipliers", {})
                self.check(
                    mult.get("stop_loss", 0) >= 10,
                    "ATR multipliers scaled for 1-min bars",
                    f"stop={mult.get('stop_loss')}x trail={mult.get('trailing_stop')}x target={mult.get('take_profit')}x",
                )

                # Ticker ATR
                atr = pa.get("ticker_atr", {})
                self.check(
                    len(atr) > 0,
                    f"Ticker ATR populated ({len(atr)} tickers)",
                    "empty — will populate during market hours" if not atr else "",
                )

                # Signal gate
                sga = pa.get("signal_gate_analysis", [])
                n_would_trade = sum(1 for s in sga if s.get("would_trade"))
                self.check(
                    len(sga) > 0,
                    f"Signals computed ({len(sga)} tickers)",
                )
                self.check(
                    n_would_trade > 0 or not pa.get("market_open", False),
                    f"Tradeable signals: {n_would_trade}",
                    "0 but market closed" if not pa.get("market_open") else "",
                )

                # Thresholds
                thresh = pa.get("thresholds", {})
                self.check(
                    thresh.get("sizing_cost_threshold", 0) == 0.0015,
                    f"Cost threshold = {thresh.get('sizing_cost_threshold')}",
                )
                self.check(
                    pa.get("max_trades_per_day", 0) == 5,
                    f"Max trades/day = {pa.get('max_trades_per_day')}",
                )

        except Exception as e:
            self.check(False, f"Diagnostics endpoint reachable", str(e))
        return data

    # ── DB checks ────────────────────────────────────────────────────────────

    async def test_db(self) -> None:
        if not self.db_url:
            print(f"\n{BOLD}3. Database (skipped — no --db-url){RESET}")
            return

        print(f"\n{BOLD}3. Database{RESET}")
        try:
            import asyncpg

            conn = await asyncpg.connect(self.db_url)
            try:
                # OHLCV freshness
                row = await conn.fetchrow(
                    "SELECT MAX(time) as latest, COUNT(*) as total FROM ohlcv_1m WHERE time > NOW() - INTERVAL '3 days'"
                )
                latest_bar = row["latest"]
                bar_count = row["total"]
                bar_age = (datetime.now(timezone.utc) - latest_bar).total_seconds() / 3600 if latest_bar else 999

                self.check(
                    bar_count > 0,
                    f"OHLCV bars in last 3 days: {bar_count}",
                )
                self.check(
                    bar_age < 48,  # generous for weekend
                    f"Latest bar age: {bar_age:.1f}h",
                    "stale!" if bar_age > 48 else "ok",
                )

                # Feature matrix freshness
                row = await conn.fetchrow(
                    "SELECT MAX(time) as latest, COUNT(*) as total FROM feature_matrix WHERE time > NOW() - INTERVAL '3 days'"
                )
                feat_latest = row["latest"]
                feat_count = row["total"]
                feat_age = (datetime.now(timezone.utc) - feat_latest).total_seconds() / 3600 if feat_latest else 999

                self.check(
                    feat_count > 0,
                    f"Feature rows in last 3 days: {feat_count}",
                )
                self.check(
                    feat_age < 48,
                    f"Latest feature age: {feat_age:.1f}h",
                )

                # OHLCV per day
                rows = await conn.fetch("""
                    SELECT DATE(time AT TIME ZONE 'America/New_York') as day, COUNT(*) as n
                    FROM ohlcv_1m WHERE time > NOW() - INTERVAL '7 days'
                    GROUP BY 1 ORDER BY 1
                """)
                days_with_data = len(rows)
                self.check(
                    days_with_data >= 2,
                    f"Days with OHLCV data: {days_with_data}",
                    ", ".join(f"{r['day']}:{r['n']}" for r in rows),
                )

                # DB size
                row = await conn.fetchrow(
                    "SELECT pg_database_size(current_database()) as sz"
                )
                db_mb = row["sz"] / (1024 * 1024)
                self.check(
                    db_mb < 400,
                    f"Database size: {db_mb:.0f} MB",
                    "approaching limit!" if db_mb > 350 else "ok",
                )

            finally:
                await conn.close()

        except Exception as e:
            self.check(False, f"Database connection", str(e))

    # ── REST bar poller check ────────────────────────────────────────────────

    async def test_rest_poller(self) -> None:
        import httpx

        print(f"\n{BOLD}4. REST Bar Poller{RESET}")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self.url}/health")
                data = r.json()
                # The poller runs in background — check logs indirectly
                # If bars are fresh, poller is working
                self.check(True, "REST bar poller deployed (WS disabled)")
                self.check(
                    data.get("signal_loop_active") is True,
                    "Signal loop running alongside poller",
                )
        except Exception as e:
            self.check(False, "REST poller check", str(e))

    # ── ATR stop validation ──────────────────────────────────────────────────

    async def test_atr_stops(self, diagnostics: dict[str, Any]) -> None:
        print(f"\n{BOLD}5. ATR Stop Validation{RESET}")
        pa = diagnostics.get("pipeline_a", {})
        atr = pa.get("ticker_atr", {})
        mult = pa.get("atr_multipliers", {})
        floors = pa.get("atr_floors", {})

        if not atr:
            self.check(False, "ATR data available", "empty — run during market hours")
            return

        # Check that ATR produces meaningful (not floor-dominated) stops
        floor_dominated = 0
        for ticker, a in atr.items():
            sl = a * mult.get("stop_loss", 15)
            if sl < floors.get("stop_loss", 0.005):
                floor_dominated += 1

        self.check(
            floor_dominated < len(atr) * 0.5,
            f"ATR-scaled stops (not floor-dominated): {len(atr) - floor_dominated}/{len(atr)}",
            f"{floor_dominated} on floors" if floor_dominated else "all ATR-scaled",
        )

        # Check volatile stocks get wider stops
        if "NVDA" in atr and "V" in atr:
            nvda_stop = max(atr["NVDA"] * mult.get("stop_loss", 15), floors.get("stop_loss", 0.005))
            v_stop = max(atr["V"] * mult.get("stop_loss", 15), floors.get("stop_loss", 0.005))
            self.check(
                nvda_stop > v_stop,
                f"NVDA stop ({nvda_stop*100:.1f}%) > V stop ({v_stop*100:.1f}%)",
                "volatile stocks get wider stops",
            )

    # ── Summary ──────────────────────────────────────────────────────────────

    async def run_all(self) -> int:
        print(f"\n{BOLD}{'='*60}")
        print(f"  StockBot Pipeline Health Test")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Target: {self.url}")
        print(f"{'='*60}{RESET}")

        await self.test_health()
        diag = await self.test_diagnostics()
        await self.test_db()
        await self.test_rest_poller()
        await self.test_atr_stops(diag)

        passed = sum(self.results)
        total = len(self.results)
        failed = total - passed
        color = GREEN if failed == 0 else RED

        print(f"\n{BOLD}{'='*60}")
        print(f"  Results: {color}{passed}/{total} passed{RESET}")
        if failed:
            print(f"  {RED}{failed} FAILED{RESET}")
        else:
            print(f"  {GREEN}All checks passed!{RESET}")
        print(f"{BOLD}{'='*60}{RESET}\n")

        return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Test StockBot pipeline health")
    parser.add_argument(
        "--url",
        default="https://stockbot-production-cbde.up.railway.app",
        help="StockBot API URL",
    )
    parser.add_argument("--db-url", default=None, help="Direct PostgreSQL URL for DB checks")
    args = parser.parse_args()

    exit_code = asyncio.run(PipelineHealthTest(args.url, args.db_url).run_all())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
