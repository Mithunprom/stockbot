#!/usr/bin/env python3
"""Trade roundtrip test — buys and sells 1 share to verify execution pipeline.

Uses Alpaca paper trading to place a real buy order, waits for fill,
then immediately sells. Confirms the full path:
  Alpaca API → order submit → fill → position manager → close

Usage:
    python scripts/test_trade_roundtrip.py
    python scripts/test_trade_roundtrip.py --ticker AAPL --qty 1
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone

# ── ANSI colors ──────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"
PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"


def _status(ok: bool, msg: str, detail: str = "") -> bool:
    icon = PASS if ok else FAIL
    d = f"  ({detail})" if detail else ""
    print(f"  [{icon}] {msg}{d}")
    return ok


async def run_test(ticker: str, qty: int) -> int:
    from src.config import get_settings
    from src.risk.circuit_breakers import CircuitBreakers
    from src.execution.alpaca import AlpacaOrderRouter, OrderRequest

    settings = get_settings()
    results: list[bool] = []

    print(f"\n{BOLD}{'='*60}")
    print(f"  Trade Roundtrip Test")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Ticker: {ticker}  Qty: {qty}  Mode: {settings.alpaca_mode}")
    print(f"{'='*60}{RESET}")

    # ── 1. Verify paper mode ─────────────────────────────────────────────────
    print(f"\n{BOLD}1. Safety Checks{RESET}")
    results.append(_status(
        settings.alpaca_mode == "paper",
        "Paper mode confirmed",
        f"mode={settings.alpaca_mode}",
    ))
    if settings.alpaca_mode != "paper":
        print(f"  {RED}ABORTING — not in paper mode!{RESET}")
        return 1

    # ── 2. Check Alpaca connectivity ─────────────────────────────────────────
    print(f"\n{BOLD}2. Alpaca Connectivity{RESET}")
    cb = CircuitBreakers(pipeline_id="test_roundtrip")
    router = AlpacaOrderRouter(cb)
    try:
        client = router._get_client()
        account = client.get_account()
        cash = float(account.cash)
        portfolio = float(account.portfolio_value)
        results.append(_status(True, "Alpaca connected", f"cash=${cash:,.0f} portfolio=${portfolio:,.0f}"))
    except Exception as e:
        results.append(_status(False, "Alpaca connection", str(e)))
        return 1

    # ── 3. Get current price ─────────────────────────────────────────────────
    print(f"\n{BOLD}3. Market Data{RESET}")
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestBarRequest
        data_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        latest = data_client.get_stock_latest_bar(StockLatestBarRequest(symbol_or_symbols=ticker))
        bar = latest[ticker]
        price = float(bar.close)
        results.append(_status(True, f"Latest price: ${price:.2f}", f"vol={bar.volume}"))
    except Exception as e:
        results.append(_status(False, f"Get price for {ticker}", str(e)))
        return 1

    notional = price * qty
    results.append(_status(
        notional < cash * 0.05,
        f"Order size ${notional:.2f} < 5% of cash",
        f"{notional/cash*100:.1f}% of ${cash:,.0f}",
    ))

    # ── 4. Place BUY order ───────────────────────────────────────────────────
    print(f"\n{BOLD}4. BUY Order{RESET}")
    buy_req = OrderRequest(
        ticker=ticker,
        side="buy",
        qty=qty,
        limit_price=None,  # market order for test
    )
    t0 = time.monotonic()
    buy_result = await router.submit_order(buy_req)
    buy_time = time.monotonic() - t0

    results.append(_status(
        buy_result.status == "filled",
        f"BUY {qty} {ticker}",
        f"status={buy_result.status} price=${buy_result.filled_avg_price} time={buy_time:.1f}s",
    ))
    if buy_result.status != "filled":
        print(f"  {RED}BUY failed: {buy_result.error}{RESET}")
        # Still try to check if position exists
        if buy_result.status == "error" and "market is not open" in buy_result.error.lower():
            print(f"  {YELLOW}Market is closed — cannot execute live trades{RESET}")
            print(f"  {YELLOW}Run this script during market hours (9:30-16:00 ET){RESET}")
        return 1

    buy_price = buy_result.filled_avg_price
    print(f"  Filled: {buy_result.filled_qty} shares @ ${buy_price:.2f}")

    # ── 5. Verify position exists ────────────────────────────────────────────
    print(f"\n{BOLD}5. Position Check{RESET}")
    await asyncio.sleep(1)  # brief settle
    try:
        positions = client.get_all_positions()
        pos = next((p for p in positions if p.symbol == ticker), None)
        results.append(_status(
            pos is not None,
            f"Position visible in Alpaca",
            f"qty={pos.qty} avg_entry=${float(pos.avg_entry_price):.2f}" if pos else "not found",
        ))
    except Exception as e:
        results.append(_status(False, "Position check", str(e)))

    # ── 6. Place SELL order ──────────────────────────────────────────────────
    print(f"\n{BOLD}6. SELL Order{RESET}")
    sell_req = OrderRequest(
        ticker=ticker,
        side="sell",
        qty=qty,
        limit_price=None,
    )
    t0 = time.monotonic()
    sell_result = await router.submit_order(sell_req)
    sell_time = time.monotonic() - t0

    results.append(_status(
        sell_result.status == "filled",
        f"SELL {qty} {ticker}",
        f"status={sell_result.status} price=${sell_result.filled_avg_price} time={sell_time:.1f}s",
    ))
    if sell_result.status == "filled":
        sell_price = sell_result.filled_avg_price
        pnl = (sell_price - buy_price) * qty
        pnl_pct = (sell_price / buy_price - 1) * 100
        print(f"  Filled: {sell_result.filled_qty} shares @ ${sell_price:.2f}")
        print(f"  P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
    else:
        print(f"  {RED}SELL failed: {sell_result.error}{RESET}")

    # ── 7. Verify position closed ────────────────────────────────────────────
    print(f"\n{BOLD}7. Position Closed{RESET}")
    await asyncio.sleep(1)
    try:
        positions = client.get_all_positions()
        still_open = any(p.symbol == ticker for p in positions)
        results.append(_status(
            not still_open,
            f"Position closed (no open {ticker})",
        ))
    except Exception as e:
        results.append(_status(False, "Close check", str(e)))

    # ── Summary ──────────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    failed = total - passed
    color = GREEN if failed == 0 else RED

    print(f"\n{BOLD}{'='*60}")
    print(f"  Results: {color}{passed}/{total} passed{RESET}")
    if failed == 0:
        print(f"  {GREEN}Execution pipeline is working!{RESET}")
    else:
        print(f"  {RED}{failed} checks failed{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Test trade roundtrip (paper mode)")
    parser.add_argument("--ticker", default="AAPL", help="Stock to test with")
    parser.add_argument("--qty", type=int, default=1, help="Number of shares")
    args = parser.parse_args()
    sys.exit(asyncio.run(run_test(args.ticker, args.qty)))


if __name__ == "__main__":
    main()
