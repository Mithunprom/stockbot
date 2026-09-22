"""Build a point-in-time daily-ATR cache for the ledger replay harness.

`scripts/replay_ledger.py` needs a per-ticker, per-date daily volatility figure
to score counterfactual exit barriers. Production gets that from
`signal_loop._compute_daily_vols` (yfinance daily bars → ATR(14)/close). This
script snapshots the same quantity for every date in the ledger window so the
harness is reproducible and runs offline.

The cached value for date D is ATR(14)/close computed from bars up to and
including D. Callers must look up the date STRICTLY BEFORE a trade's entry so
no lookahead leaks into a counterfactual.

Usage:
    python scripts/build_daily_atr_cache.py [--out reports/research/daily_atr_cache.json]
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

DEFAULT_LEDGER = "reports/research/ledger_m2.json"
DEFAULT_OUT = "reports/research/daily_atr_cache.json"


def build_cache(tickers: list[str], start: str, end: str) -> dict[str, dict[str, float]]:
    """Fetch daily bars and return {ticker: {YYYY-MM-DD: atr14_over_close}}.

    Args:
        tickers: Ticker symbols to fetch.
        start: ISO start date (inclusive).
        end: ISO end date (exclusive).

    Returns:
        Nested dict of point-in-time daily ATR ratios. Tickers that fail to
        download are omitted rather than raising.
    """
    import yfinance as yf

    warnings.filterwarnings("ignore")
    frame = yf.download(
        tickers, start=start, end=end, interval="1d", progress=False,
        auto_adjust=True, group_by="ticker", threads=True,
    )
    multi = len(tickers) > 1
    out: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        try:
            sub = frame[ticker] if multi else frame
            high = sub["High"].astype(float)
            low = sub["Low"].astype(float)
            close = sub["Close"].astype(float)
            true_range = np.maximum(
                high - low,
                np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()),
            )
            ratio = (true_range.rolling(14).mean() / close).dropna()
            if ratio.empty:
                continue
            out[ticker] = {
                day.strftime("%Y-%m-%d"): round(float(val), 6)
                for day, val in ratio.items()
            }
        except Exception:  # noqa: BLE001 - a missing ticker must not abort the build
            continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--start", default="2026-02-01")
    parser.add_argument("--end", default="2026-09-16")
    args = parser.parse_args()

    trades = json.loads(Path(args.ledger).read_text())["trades"]
    tickers = sorted({t["ticker"] for t in trades})
    cache = build_cache(tickers, args.start, args.end)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cache, indent=0, sort_keys=True))
    rows = sum(len(v) for v in cache.values())
    print(f"cached {len(cache)}/{len(tickers)} tickers, {rows} ticker-days -> {out_path}")


if __name__ == "__main__":
    main()
